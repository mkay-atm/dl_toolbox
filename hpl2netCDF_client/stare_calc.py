# stare_calc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr


KOLMOGOROV_A_1D = 0.55


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def mdiff(x: np.ndarray) -> float:
    """Median step size of a monotonic vector."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 1.0
    return float(np.median(np.diff(x)))



def transpose(x: np.ndarray) -> np.ndarray:
    """Cloudnet-like transpose helper for 1D profile arrays."""
    x = np.asarray(x)
    return x[:, None] if x.ndim == 1 else x.T



def cumsumr(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Cumulative run length of True values along axis, resetting at False."""
    a = np.asarray(a, dtype=bool)
    out = np.zeros_like(a, dtype=int)

    if axis == 1:
        for i in range(a.shape[0]):
            run = 0
            for j in range(a.shape[1]):
                run = run + 1 if a[i, j] else 0
                out[i, j] = run
    else:
        for j in range(a.shape[1]):
            run = 0
            for i in range(a.shape[0]):
                run = run + 1 if a[i, j] else 0
                out[i, j] = run
    return out



def _calc_var_from_top_gates(data: np.ndarray) -> ma.MaskedArray:
    """Variance estimated from top 10% of gates."""
    fraction = 0.1
    n_gates = max(1, round(data.shape[1] * fraction))
    return ma.var(data[:, -n_gates:], axis=1)



def _calc_mean_from_top_gates(data: np.ndarray) -> ma.MaskedArray:
    """Mean estimated from top 10% of gates."""
    fraction = 0.1
    n_gates = max(1, round(data.shape[1] * fraction))
    return ma.mean(data[:, -n_gates:], axis=1)



def _estimate_background_noise(data: np.ndarray) -> np.ndarray:
    return np.sqrt(_calc_var_from_top_gates(data))



def _estimate_background_mean(data: np.ndarray) -> ma.MaskedArray:
    return _calc_mean_from_top_gates(data)



def calc_sigma_units(
    time_vector: np.ndarray,
    range_los: np.ndarray,
    sigma_minutes: float = 1,
    sigma_metres: float = 10,
) -> tuple[float, float]:
    """2D Gaussian smoothing widths in pixel units."""
    time_step = mdiff(time_vector) * 60
    alt_step = mdiff(range_los)
    return sigma_minutes / time_step, sigma_metres / alt_step



def _rename_timebin_dim(obj: xr.Dataset | xr.DataArray, old_dim: str = "time_bins", new_dim: str = "time"):
    if old_dim in obj.dims:
        obj = obj.rename({old_dim: new_dim})
    if old_dim in obj.coords:
        obj = obj.rename({old_dim: new_dim})
    return obj



def _bin_midpoints_as_time(bins: Iterable[pd.Timestamp]) -> xr.DataArray:
    bins = pd.DatetimeIndex(bins)
    return xr.DataArray(
        bins[:-1] + (bins[1:] - bins[:-1]) / 2,
        dims=("time",),
        name="time",
    )



def _replace_time_bins_with_midpoints(
    obj: xr.Dataset | xr.DataArray,
    bins: Iterable[pd.Timestamp],
    old_dim: str = "time_bins",
    new_dim: str = "time",
):
    """Replace groupby_bins Interval coordinate with datetime midpoints."""
    obj = _rename_timebin_dim(obj, old_dim=old_dim, new_dim=new_dim)
    time_mid = _bin_midpoints_as_time(bins)

    if new_dim not in obj.dims:
        raise ValueError(f"{new_dim!r} is not a dimension in the object.")

    if obj.sizes[new_dim] != time_mid.sizes["time"]:
        raise ValueError(
            f"Time size mismatch: object has {obj.sizes[new_dim]} bins, "
            f"but provided bins define {time_mid.sizes['time']} intervals."
        )

    return obj.assign_coords({new_dim: time_mid})



def _ensure_height_dim(obj: xr.Dataset | xr.DataArray, elevation_deg: float = 75.0):
    """
    Convert a vertical dimension `range` to geophysical `height` using
    height = range * sin(elevation_deg).
    """
    if "height" in obj.dims:
        return obj

    if "range" not in obj.dims:
        raise ValueError("Object has neither 'height' nor 'range' as vertical dimension.")

    if "range" not in obj.coords:
        raise ValueError("Cannot derive height because no 'range' coordinate is present.")

    height = obj["range"] * np.sin(np.deg2rad(elevation_deg))
    height = xr.DataArray(height.data, dims=("range",), coords={"range": obj["range"]}, name="height")

    obj = obj.assign_coords(height=height)
    obj = obj.swap_dims({"range": "height"})
    obj = obj.drop_vars("range", errors="ignore")
    return obj



def _finalize_time_height_dataset(ds: xr.Dataset, bins: Iterable[pd.Timestamp], elevation_deg: float = 75.0) -> xr.Dataset:
    """Force final dataset to contain only time and height as coords/dims."""
    ds = _ensure_height_dim(ds, elevation_deg=elevation_deg)

    cleaned: dict[str, xr.DataArray] = {}
    for name, da in ds.data_vars.items():
        da = _ensure_height_dim(da, elevation_deg=elevation_deg)
        drop_coords = [c for c in da.coords if c not in ("time", "height")]
        if drop_coords:
            da = da.drop_vars(drop_coords, errors="ignore")
        unexpected_dims = [d for d in da.dims if d not in ("time", "height")]
        if unexpected_dims:
            raise ValueError(f"{name}: unexpected dimensions remaining: {unexpected_dims}")
        cleaned[name] = da.reset_coords(drop=True)

    out = xr.Dataset(cleaned)
    if "time" not in ds.coords:
        out = out.assign_coords(time=_bin_midpoints_as_time(bins))
    else:
        out = out.assign_coords(time=ds["time"])
    out = out.assign_coords(height=ds["height"])
    return out.transpose("time", "height")


# -----------------------------------------------------------------------------
# Cloudnet-like screening
# -----------------------------------------------------------------------------


@dataclass
class NoiseParam:
    noise_min: float = 1e-12
    noise_smooth_min: float = 1e-12


class NoisyData:
    """Subset of Cloudnet-like noise screening logic."""

    def __init__(self, data: dict[str, np.ndarray], noise_param: NoiseParam, *, range_corrected: bool = True, instrument=None):
        self.data = data
        self.noise_param = noise_param
        self.range_corrected = range_corrected
        self.instrument = instrument

    def screen_data(
        self,
        data_in: np.ndarray,
        snr_limit: float = 5.0,
        n_negatives: int = 5,
        *,
        is_smoothed: bool = False,
        keep_negative: bool = False,
        filter_fog: bool = False,
        filter_negatives: bool = True,
        filter_snr: bool = True,
        return_snr: bool = False,
    ):
        data = ma.copy(data_in)

        self._calc_range_uncorrected(data)
        mean = _estimate_background_mean(data)
        noise = _estimate_background_noise(data)
        noise = self._adjust_noise(noise, is_smoothed=is_smoothed)

        if filter_negatives:
            is_negative = self._mask_low_values_above_consequent_negatives(data, n_negatives=n_negatives)
            noise[is_negative] = 1e-12

        if filter_snr:
            data, snr = self._remove_noise(data, noise, keep_negative=keep_negative, snr_limit=snr_limit)
        else:
            snr = None
            mean = None

        self._calc_range_corrected(data)

        if return_snr:
            return data, snr, noise, mean
        return data

    def _adjust_noise(self, noise: np.ndarray, *, is_smoothed: bool) -> np.ndarray:
        noise_min = self.noise_param.noise_smooth_min if is_smoothed else self.noise_param.noise_min
        noise = np.asarray(noise, dtype=float)
        noise[noise < noise_min] = noise_min
        return noise

    @staticmethod
    def _mask_low_values_above_consequent_negatives(
        data: ma.MaskedArray,
        n_negatives: int = 5,
        threshold: float = 8e-6,
        n_gates: int = 95,
        n_skip_lowest: int = 5,
    ) -> np.ndarray:
        if data.shape[1] <= n_skip_lowest:
            return np.array([], dtype=int)

        end_gate = min(data.shape[1], n_skip_lowest + n_gates)
        neg = data[:, n_skip_lowest:end_gate] < 0
        runs = cumsumr(neg, axis=1)
        t_idx, z_idx = np.where(runs > n_negatives)
        z_idx += n_skip_lowest

        for t, z in zip(t_idx, z_idx):
            profile = data[t, z:]
            profile[profile < threshold] = ma.masked

        return np.unique(t_idx)

    def _remove_noise(self, array: ma.MaskedArray, noise: np.ndarray, *, keep_negative: bool, snr_limit: float):
        snr = array / transpose(noise)
        if not ma.isMaskedArray(array):
            array = ma.masked_array(array)
        if keep_negative:
            array[np.abs(snr) < snr_limit] = ma.masked
        else:
            array[snr < snr_limit] = ma.masked
        return array, snr

    def _calc_range_uncorrected(self, data: ma.MaskedArray) -> None:
        r2 = self._get_range_squared()
        data[:] = data / r2

    def _calc_range_corrected(self, data: ma.MaskedArray) -> None:
        r2 = self._get_range_squared()
        data[:] = data * r2

    def _get_range_squared(self) -> np.ndarray:
        return (self.data["range"] * 0.001) ** 2



def screen_stare_dv_with_snr(
    ds: xr.Dataset,
    *,
    dv_name: str = "dv",
    snr_name: str = "snr",
    range_name: str = "range",
    time_name: str = "time",
    snr_limit: float = 5.0,
    n_negatives: int = 5,
    noise_min: float = 1e-12,
    noise_smooth_min: float = 1e-12,
) -> xr.Dataset:
    """Screen stare Doppler velocity using Cloudnet-like SNR masking."""
    ds_out = ds.copy()

    dv = np.asarray(ds[dv_name].values, dtype=float)
    snr_in = np.asarray(ds[snr_name].values, dtype=float)

    if dv.ndim == 1:
        dv = dv[None, :]
    if snr_in.ndim == 1:
        snr_in = snr_in[None, :]

    rng = np.asarray(ds[range_name].values, dtype=float)

    noise_param = NoiseParam(noise_min=noise_min, noise_smooth_min=noise_smooth_min)
    data_dict = {"beta_raw": snr_in.copy(), "range": rng}
    noisy = NoisyData(data=data_dict, noise_param=noise_param, range_corrected=True)

    snr_scr, snr_ratio, noise, mean = noisy.screen_data(
        snr_in,
        snr_limit=snr_limit,
        n_negatives=n_negatives,
        return_snr=True,
    )

    snr_centered = (snr_ratio * transpose(noise) - transpose(mean)) / transpose(noise)
    mask = ~np.isfinite(snr_scr.filled(np.nan)) | ma.getmaskarray(snr_scr)

    dv_screened = ma.masked_where(mask, dv)
    snr_screened = ma.masked_where(mask, snr_in)

    dims = ds[dv_name].dims
    coords = {d: ds[d] for d in dims}

    ds_out["dv_screened"] = xr.DataArray(dv_screened.filled(np.nan), dims=dims, coords=coords)
    ds_out["snr_screened"] = xr.DataArray(snr_screened.filled(np.nan), dims=dims, coords=coords)
    ds_out["screen_mask"] = xr.DataArray((~mask).astype(np.int8), dims=dims, coords=coords)
    ds_out["noise_estimate"] = xr.DataArray(noise, dims=(dims[0],), coords={dims[0]: ds[dims[0]]})
    ds_out["snr_centered"] = xr.DataArray(snr_centered, dims=dims, coords=coords)
    return ds_out


# -----------------------------------------------------------------------------
# Wind-speed support field for epsilon
# -----------------------------------------------------------------------------


def build_u_for_epsilon_xr(
    U: xr.DataArray,
    *,
    min_u: float = 0.1,
    time_smooth: int = 5,
    height_smooth: int = 3,
    use_median: bool = True,
    interp_time: bool = True,
    interp_height: bool = True,
    nearest_fallback: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Build a smooth, gap-filled wind-speed field for epsilon estimation."""
    if "time" not in U.dims or "height" not in U.dims:
        raise ValueError("U must have dimensions ('time', 'height').")

    U0 = U.where(np.isfinite(U) & (U > min_u))
    originally_valid = U0.notnull()

    roller = U0.rolling(time=time_smooth, center=True, min_periods=1)
    U_time = roller.median() if use_median else roller.mean()

    roller = U_time.rolling(height=height_smooth, center=True, min_periods=1)
    U_s = roller.median() if use_median else roller.mean()

    # Track values created by smoothing as well as by interpolation.
    # Previously, a missing original sample that gained a finite value during
    # rolling smoothing retained flag 5 ("missing"), even though U_s was used
    # for epsilon. This also suppressed its propagated uncertainty.
    u_fill_flag = xr.full_like(U_s, 5, dtype=np.int8)
    u_fill_flag = xr.where(originally_valid & U_s.notnull(), 1, u_fill_flag)
    u_fill_flag = xr.where(~originally_valid & U_time.notnull(), 2, u_fill_flag)
    u_fill_flag = xr.where(~U_time.notnull() & U_s.notnull(), 3, u_fill_flag)
    if time_smooth == 1 and height_smooth == 1:
        u_fill_flag = xr.where(originally_valid, 0, u_fill_flag)

    U_work = U_s

    if interp_time:
        U_t = U_work.interpolate_na(dim="time", method="linear")
        u_fill_flag = xr.where(~U_work.notnull() & U_t.notnull(), 2, u_fill_flag)
        U_work = U_t

    if interp_height:
        U_h = U_work.interpolate_na(dim="height", method="linear")
        u_fill_flag = xr.where(~U_work.notnull() & U_h.notnull(), 3, u_fill_flag)
        U_work = U_h

    if nearest_fallback:
        U_n = U_work.interpolate_na(dim="time", method="nearest")
        U_n = U_n.interpolate_na(dim="height", method="nearest")
        u_fill_flag = xr.where(~U_work.notnull() & U_n.notnull(), 4, u_fill_flag)
        U_work = U_n

    u_fill_flag = xr.where(U_work.notnull(), u_fill_flag, 5).astype(np.int8)
    U_work.name = "horizontal_wind_speed_for_epsilon"
    u_fill_flag.name = "u_fill_flag"
    return U_work, u_fill_flag


# -----------------------------------------------------------------------------
# Variance / autocovariance / Lenschow correction
# -----------------------------------------------------------------------------


def lagged_autocovariance(x: xr.DataArray, lag: int = 0, dim: str = "time") -> xr.DataArray:
    """Pairwise-valid lagged autocovariance along `dim`."""
    x0 = x
    x1 = x.shift({dim: -lag})

    valid = x0.notnull() & x1.notnull()
    n = valid.sum(dim)

    x0m = x0.where(valid)
    x1m = x1.where(valid)

    mean0 = x0m.mean(dim)
    mean1 = x1m.mean(dim)

    cov = ((x0m - mean0) * (x1m - mean1)).sum(dim) / n
    return cov.where(n > 0)



def binned_autocovariances(
    da: xr.DataArray,
    bins: Iterable[pd.Timestamp],
    lags: Iterable[int],
    dim: str = "time",
    right: bool = True,
    include_lowest: bool = False,
) -> xr.DataArray:
    """Compute lagged autocovariances for each time bin."""
    grouped = da.groupby_bins(dim, bins, right=right, include_lowest=include_lowest)

    acov_list = []
    for lag in lags:
        out = grouped.apply(lambda x: lagged_autocovariance(x, lag=lag, dim=dim))
        out = out.expand_dims({"lag": [lag]})
        acov_list.append(out)

    return xr.concat(acov_list, dim="lag")



def lenschow_noise_correction(acov: xr.DataArray, fit_lags=slice(1, 5), lag_dim: str = "lag") -> xr.Dataset:
    """Estimate total, atmospheric, and noise variance from small-lag ACF fit."""
    acov_fit = acov.sel({lag_dim: fit_lags})
    fit = acov_fit.polyfit(dim=lag_dim, deg=1)

    coeff = fit["polyfit_coefficients"]
    slope = coeff.sel(degree=1, drop=True).reset_coords(drop=True)
    intercept = coeff.sel(degree=0, drop=True).reset_coords(drop=True)

    total_var = acov.sel({lag_dim: 0}, drop=True).reset_coords(drop=True)
    atmospheric_var = intercept
    noise_var = (total_var - atmospheric_var).where((total_var - atmospheric_var) >= 0)

    return xr.Dataset(
        {
            "total_variance": total_var,
            "atmospheric_variance": atmospheric_var,
            "noise_variance": noise_var,
            "acov_slope": slope,
        }
    )


# -----------------------------------------------------------------------------
# O'Connor epsilon retrieval
# -----------------------------------------------------------------------------


def oconnor_length_scales(
    U: xr.DataArray,
    t_seconds: float,
    N: int,
    z: Optional[xr.DataArray] = None,
    beam_divergence_half_angle_rad: float = 0.0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """O'Connor et al. length scales L1 and L."""
    if z is None:
        z_term = 0.0
    else:
        z_term = 2.0 * z * np.sin(beam_divergence_half_angle_rad / 2.0)

    L1 = U * t_seconds + z_term
    L = N * U * t_seconds
    return L1, L



def epsilon_from_oconnor_variance(
    sigma2_atm: xr.DataArray,
    U: xr.DataArray,
    t_seconds: float,
    N: int,
    z: Optional[xr.DataArray] = None,
    beam_divergence_half_angle_rad: float = 0.0,
    a: float = KOLMOGOROV_A_1D,
    min_u: float = 0.5,
    min_sigma2: float = 0.0,
) -> xr.DataArray:
    """Retrieve epsilon from atmospheric variance using O'Connor et al. (2010)."""
    sigma2_atm = sigma2_atm.where(sigma2_atm > min_sigma2)
    U = U.where(U > min_u)

    L1, L = oconnor_length_scales(
        U=U,
        t_seconds=t_seconds,
        N=N,
        z=z,
        beam_divergence_half_angle_rad=beam_divergence_half_angle_rad,
    )

    delta = L ** (2.0 / 3.0) - L1 ** (2.0 / 3.0)
    delta = delta.where(delta > 0)

    epsilon = ((2.0 * np.pi) / (1.5 * a)) ** 1.5 * (sigma2_atm ** 1.5) * (delta ** (-1.5))
    epsilon.name = "epsilon"
    return epsilon



def estimate_epsilon_from_vertical_stare(
    dv: xr.DataArray,
    U: xr.DataArray,
    bins: Iterable[pd.Timestamp],
    lags: Iterable[int] = np.arange(0, 6),
    fit_lags=slice(1, 5),
    dim: str = "time",
    sample_spacing_seconds: float = 4.0,
    beam_divergence_half_angle_rad: float = 0.0,
    min_u: float = 0.5,
    min_sigma2: float = 0.0,
    elevation_deg: float = 75.0,
) -> xr.Dataset:
    """End-to-end epsilon retrieval from screened vertical-stare velocities."""
    dv = _ensure_height_dim(dv, elevation_deg=elevation_deg)
    if "height" not in U.dims and "range" in U.dims:
        U = _ensure_height_dim(U, elevation_deg=elevation_deg)

    acov = binned_autocovariances(dv, bins=bins, lags=lags, dim=dim)
    mom = lenschow_noise_correction(acov, fit_lags=fit_lags, lag_dim="lag")
    mom = _replace_time_bins_with_midpoints(mom, bins=bins, old_dim="time_bins", new_dim="time")
    mom = _ensure_height_dim(mom, elevation_deg=elevation_deg)

    bins = pd.DatetimeIndex(bins)
    if len(bins) < 2:
        raise ValueError("Need at least two bin edges to estimate averaging length N.")

    bin_seconds = (pd.Timestamp(bins[1]) - pd.Timestamp(bins[0])).total_seconds()
    N = int(round(bin_seconds / sample_spacing_seconds))
    if N < 2:
        raise ValueError("Estimated N < 2; check bins and sample_spacing_seconds.")

    sigma2_aligned = mom["atmospheric_variance"]
    U_aligned = U.interp(
        time=sigma2_aligned["time"],
        height=sigma2_aligned["height"],
        kwargs={"fill_value": np.nan},
    )

    z = sigma2_aligned["height"]
    eps = epsilon_from_oconnor_variance(
        sigma2_atm=sigma2_aligned,
        U=U_aligned,
        t_seconds=sample_spacing_seconds,
        N=N,
        z=z,
        beam_divergence_half_angle_rad=beam_divergence_half_angle_rad,
        min_u=min_u,
        min_sigma2=min_sigma2,
    ).reset_coords(drop=True)

    out = xr.Dataset(
        {
            "total_variance": mom["total_variance"].reset_coords(drop=True),
            "atmospheric_variance": mom["atmospheric_variance"].reset_coords(drop=True),
            "noise_variance": mom["noise_variance"].reset_coords(drop=True),
            "horizontal_wind_speed": U_aligned.reset_coords(drop=True),
            "epsilon": eps,
        }
    )

    q = (
        out["atmospheric_variance"].notnull()
        & (out["atmospheric_variance"] > min_sigma2)
        & out["horizontal_wind_speed"].notnull()
        & (out["horizontal_wind_speed"] > min_u)
        & out["epsilon"].notnull()
    )
    out["qepsilon"] = q.astype(np.int8)

    out = _finalize_time_height_dataset(out, bins=bins, elevation_deg=elevation_deg)
    out.attrs["sample_spacing_seconds"] = float(sample_spacing_seconds)
    out.attrs["N_samples_per_bin"] = int(N)
    out.attrs["kolmogorov_constant_1d"] = float(KOLMOGOROV_A_1D)
    return out



def estimate_sigma2_uncertainty(sigma2_atm: xr.DataArray, n_samples: xr.DataArray, min_samples: int = 3) -> xr.DataArray:
    """Estimate sampling uncertainty of atmospheric variance."""
    n = n_samples.where(n_samples >= min_samples)
    rel = np.sqrt(2.0 / (n - 1.0))
    sigma2_err = sigma2_atm * rel
    sigma2_err.name = "atmospheric_variance_uncertainty"
    return sigma2_err



def estimate_u_uncertainty_from_fill_flag(U: xr.DataArray, u_fill_flag: xr.DataArray) -> xr.DataArray:
    """Pragmatic uncertainty model for U used in epsilon based on fill provenance."""
    rel = xr.full_like(U, np.nan, dtype=float)
    rel = xr.where(u_fill_flag <= 1, 0.10, rel)
    rel = xr.where((u_fill_flag == 2) | (u_fill_flag == 3), 0.20, rel)
    rel = xr.where(u_fill_flag == 4, 0.35, rel)
    u_err = U * rel
    u_err.name = "horizontal_wind_speed_for_epsilon_uncertainty"
    return u_err



def epsilon_uncertainty_oconnor(
    epsilon: xr.DataArray,
    sigma2_atm: xr.DataArray,
    sigma2_err: xr.DataArray,
    U: xr.DataArray,
    U_err: xr.DataArray,
    t_seconds: float,
    N: int,
    z: xr.DataArray,
    beam_divergence_half_angle_rad: float = 0.0,
) -> xr.DataArray:
    """First-order propagated uncertainty for epsilon."""
    L1, L = oconnor_length_scales(
        U=U,
        t_seconds=t_seconds,
        N=N,
        z=z,
        beam_divergence_half_angle_rad=beam_divergence_half_angle_rad,
    )
    delta = L ** (2.0 / 3.0) - L1 ** (2.0 / 3.0)

    dU = xr.where(U_err.notnull() & (U_err > 0), U_err, 0.01 * U)
    U_plus = U + dU
    U_minus = xr.where(U - dU > 0, U - dU, 0.5 * U)

    L1_p, L_p = oconnor_length_scales(U_plus, t_seconds, N, z, beam_divergence_half_angle_rad)
    L1_m, L_m = oconnor_length_scales(U_minus, t_seconds, N, z, beam_divergence_half_angle_rad)

    delta_p = L_p ** (2.0 / 3.0) - L1_p ** (2.0 / 3.0)
    delta_m = L_m ** (2.0 / 3.0) - L1_m ** (2.0 / 3.0)

    ddelta_dU = (delta_p - delta_m) / (U_plus - U_minus)
    delta_err = np.abs(ddelta_dU) * U_err

    rel_sigma2 = sigma2_err / sigma2_atm
    rel_delta = delta_err / delta
    rel_eps = np.sqrt((1.5 * rel_sigma2) ** 2 + (1.5 * rel_delta) ** 2)

    eps_err = epsilon * rel_eps
    eps_err = eps_err.where(np.isfinite(eps_err))
    eps_err.name = "epsilon_uncertainty"
    return eps_err

def add_epsilon_uncertainty(
    eps_ds,
    *,
    binsize,
    sample_spacing_seconds,
    beam_divergence_half_angle_rad=0.0,
    min_samples=3,
):
    out = eps_ds.copy()

    sigma2_err = estimate_sigma2_uncertainty(
        out["atmospheric_variance"],
        out["n_samples"],
        min_samples=min_samples,
    )

    u_err = estimate_u_uncertainty_from_fill_flag(
        out["horizontal_wind_speed_for_epsilon"],
        out["u_fill_flag"],
    )

    bin_seconds = pd.to_timedelta(binsize).total_seconds()
    N = int(round(bin_seconds / sample_spacing_seconds))

    eps_err = epsilon_uncertainty_oconnor(
        epsilon=out["epsilon"],
        sigma2_atm=out["atmospheric_variance"],
        sigma2_err=sigma2_err,
        U=out["horizontal_wind_speed_for_epsilon"],
        U_err=u_err,
        t_seconds=sample_spacing_seconds,
        N=N,
        z=out["height"],
        beam_divergence_half_angle_rad=beam_divergence_half_angle_rad,
    )

    out["atmospheric_variance_uncertainty"] = sigma2_err
    out["horizontal_wind_speed_for_epsilon_uncertainty"] = u_err
    out["epsilon_uncertainty"] = eps_err

    return out