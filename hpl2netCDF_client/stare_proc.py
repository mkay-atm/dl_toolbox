# stare_proc.py
from __future__ import annotations

import os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from hpl2netCDF_client.stare_calc import (
    _ensure_height_dim,
    _replace_time_bins_with_midpoints,
    add_epsilon_uncertainty,
    build_u_for_epsilon_xr,
    estimate_epsilon_from_vertical_stare,
    screen_stare_dv_with_snr,
)


# -----------------------------------------------------------------------------
# Small config helpers
# -----------------------------------------------------------------------------


def _parse_bool(confDict, key, default=False):
    return str(confDict.get(key, int(default))).strip().lower() in {"1", "true", "yes", "y"}


def _parse_int(confDict, key, default):
    return int(confDict.get(key, default))


def _parse_float(confDict, key, default):
    return float(confDict.get(key, default))


def _parse_str(confDict, key, default):
    return str(confDict.get(key, default))


def _parse_list_of_ints(confDict, key, default):
    raw = str(confDict.get(key, default))
    return np.array([int(x.strip()) for x in raw.split(",") if x.strip() != ""], dtype=int)


def _parse_fit_lags(confDict, key="STARE_FIT_LAGS", default="1,3"):
    a, b = [int(x.strip()) for x in str(confDict.get(key, default)).split(",")]
    return slice(a, b)


def _get_daily_timerange(date_chosen):
    start = pd.Timestamp(date_chosen).floor("D")
    end = start + pd.Timedelta(days=1)
    return start, end


# -----------------------------------------------------------------------------
# Path / filename helpers
# -----------------------------------------------------------------------------


def _dated_subdir(base_path: str, date_chosen: dt.datetime, *, create_dir=True) -> str:
    year = pd.Timestamp(date_chosen).strftime("%Y")
    yearmonth = pd.Timestamp(date_chosen).strftime("%Y%m")
    outdir = os.path.join(base_path, year, yearmonth)
    if create_dir:
        os.makedirs(outdir, exist_ok=True)
    return outdir


def _infer_instrument_prefix(confDict):
    """
    Minimal filename prefix helper aligned with the toolbox basename style.
    Adjust if your project already has a central basename builder.
    """
    system = _parse_str(confDict, "SYSTEM", "dl")
    scan_type = _parse_str(confDict, "SCAN_TYPE", "stare")
    sn = _parse_str(confDict, "SYSTEM_ID", _parse_str(confDict, "SYSTEM_SN", "unknown"))
    return f"{system}_{scan_type}_{sn}"


def build_stare_l2_filename(confDict, date_chosen, *, create_dir=True):
    """
    Build final stare L2 filename.

    Preferred config keys:
      NC_STARE_L2_PATH
      NC_STARE_L2_BASENAME

    Fallback:
      NC_L2_PATH with generic basename
    """
    out_base = _parse_str(confDict, "NC_STARE_L2_PATH", confDict.get("NC_L2_PATH", "."))
    basename = confDict.get("NC_STARE_L2_BASENAME", None)

    if basename is None:
        basename = _infer_instrument_prefix(confDict) + "_l2_stare_"

    outdir = _dated_subdir(out_base, date_chosen, create_dir=create_dir)
    fname = f"{basename}{pd.Timestamp(date_chosen).strftime('%Y%m%d')}.nc"
    return os.path.join(outdir, fname)


def build_stare_l1_filename(confDict, date_chosen):
    """
    Build input stare L1 filename.

    Preferred config keys:
      NC_STARE_L1_PATH
      NC_STARE_L1_BASENAME

    Fallback:
      NC_L1_PATH with generic basename
    """
    in_base = _parse_str(confDict, "NC_STARE_L1_PATH", confDict.get("NC_L1_PATH", "."))
    basename = confDict.get("NC_STARE_L1_BASENAME", None)

    if basename is None:
        basename = _infer_instrument_prefix(confDict) + "_l1_stare_"

    indir = _dated_subdir(in_base, date_chosen, create_dir=False)
    fname = f"{basename}{pd.Timestamp(date_chosen).strftime('%Y%m%d')}.nc"
    return os.path.join(indir, fname)


def build_wind_l2_filename(confDict, date_chosen):
    """
    Build input wind L2 filename used as support field for epsilon.

    Preferred config keys:
      NC_WIND_L2_PATH
      NC_WIND_L2_BASENAME

    Fallback:
      NC_L2_PATH with generic basename
    """
    in_base = _parse_str(confDict, "NC_WIND_L2_PATH", confDict.get("NC_L2_PATH", "."))
    basename = confDict.get("NC_WIND_L2_BASENAME", None)

    if basename is None:
        basename = _infer_instrument_prefix(confDict) + "_l2_"

    indir = _dated_subdir(in_base, date_chosen, create_dir=False)
    fname = f"{basename}{pd.Timestamp(date_chosen).strftime('%Y%m%d')}.nc"
    return os.path.join(indir, fname)


# -----------------------------------------------------------------------------
# Core retrieval orchestration
# -----------------------------------------------------------------------------


def lvl2_stare_epsilon(
    ds_stare: xr.Dataset,
    ds_wind: xr.Dataset,
    *,
    date_start=None,
    date_end=None,
    binsize="10min",
    dv_name="dv",
    intensity_name="intensity",
    range_name="range",
    time_name="time",
    wind_name="wspeed",
    stare_elevation_deg=90.0,
    wind_support_elevation_deg=75.0,
    snr_limit=5.0,
    n_negatives=5,
    noise_min=1e-12,
    noise_smooth_min=1e-12,
    u_min=0.1,
    u_time_smooth=5,
    u_height_smooth=3,
    u_use_median=True,
    lags=np.arange(0, 4),
    fit_lags=slice(1, 3),
    sample_spacing_seconds=6.0,
    beam_divergence_half_angle_rad=0.0,
    min_sigma2=1e-7,
):
    """
    Full in-memory stare Level-2 retrieval.

    Returns an internal dataset with diagnostics.
    """
    t = pd.to_datetime(ds_stare[time_name].values)
    if date_start is None:
        date_start = pd.Timestamp(t.min()).floor("D")
    if date_end is None:
        date_end = pd.Timestamp(t.max()).ceil("D")

    bins = pd.date_range(start=date_start, end=date_end, freq=binsize)
    if len(bins) < 2:
        raise ValueError("Not enough time bins created. Check date_start/date_end/binsize.")

    stare_ds = xr.Dataset(
        {
            "dv": ds_stare[dv_name],
            "snr": ds_stare[intensity_name] - 1,
        },
        coords={
            "time": ds_stare[time_name],
            "range": ds_stare[range_name],
        },
    )

    stare_scr = screen_stare_dv_with_snr(
        stare_ds,
        dv_name="dv",
        snr_name="snr",
        range_name="range",
        time_name="time",
        snr_limit=snr_limit,
        n_negatives=n_negatives,
        noise_min=noise_min,
        noise_smooth_min=noise_smooth_min,
    )

    U_in = ds_wind[wind_name]
    if "height" not in U_in.dims and "range" in U_in.dims:
        U_in = _ensure_height_dim(U_in, elevation_deg=wind_support_elevation_deg)

    U_eps, U_eps_fill_flag = build_u_for_epsilon_xr(
        U=U_in.load(),
        min_u=u_min,
        time_smooth=u_time_smooth,
        height_smooth=u_height_smooth,
        use_median=u_use_median,
        interp_time=True,
        interp_height=True,
        nearest_fallback=True,
    )

    eps_ds = estimate_epsilon_from_vertical_stare(
        dv=stare_scr["dv_screened"].load(),
        U=U_eps,
        bins=bins,
        lags=lags,
        fit_lags=fit_lags,
        dim="time",
        sample_spacing_seconds=sample_spacing_seconds,
        beam_divergence_half_angle_rad=beam_divergence_half_angle_rad,
        min_u=u_min,
        min_sigma2=min_sigma2,
        elevation_deg=stare_elevation_deg,
    )

    snr_scr = _ensure_height_dim(stare_scr["snr_screened"], elevation_deg=stare_elevation_deg)
    screen_mask = _ensure_height_dim(stare_scr["screen_mask"], elevation_deg=stare_elevation_deg)

    snr_binned = snr_scr.groupby_bins("time", bins, right=True, include_lowest=False).median("time")
    snr_binned = _replace_time_bins_with_midpoints(snr_binned, bins=bins, old_dim="time_bins", new_dim="time")
    snr_binned = _ensure_height_dim(snr_binned, elevation_deg=stare_elevation_deg)

    screen_fraction = screen_mask.groupby_bins("time", bins, right=True, include_lowest=False).mean("time")
    screen_fraction = _replace_time_bins_with_midpoints(screen_fraction, bins=bins, old_dim="time_bins", new_dim="time")
    screen_fraction = _ensure_height_dim(screen_fraction, elevation_deg=stare_elevation_deg)

    w_mean = _ensure_height_dim(stare_scr["dv_screened"], elevation_deg=stare_elevation_deg)
    w_mean = w_mean.groupby_bins("time", bins, right=True, include_lowest=False).mean("time")
    w_mean = _replace_time_bins_with_midpoints(w_mean, bins=bins, old_dim="time_bins", new_dim="time")
    w_mean = _ensure_height_dim(w_mean, elevation_deg=stare_elevation_deg)

    n_samples = _ensure_height_dim(stare_scr["dv_screened"].notnull(), elevation_deg=stare_elevation_deg)
    n_samples = n_samples.groupby_bins("time", bins, right=True, include_lowest=False).sum("time")
    n_samples = _replace_time_bins_with_midpoints(n_samples, bins=bins, old_dim="time_bins", new_dim="time")
    n_samples = _ensure_height_dim(n_samples, elevation_deg=stare_elevation_deg)

    eps_ds["horizontal_wind_speed_for_epsilon"] = U_eps.interp(
        time=eps_ds["time"],
        height=eps_ds["height"],
        kwargs={"fill_value": np.nan},
    )

    eps_ds["u_fill_flag"] = U_eps_fill_flag.interp(
        time=eps_ds["time"],
        height=eps_ds["height"],
        method="nearest",
        kwargs={"fill_value": 5},
    ).astype(np.int8)

    eps_ds["snr_median"] = snr_binned.interp(
        time=eps_ds["time"],
        height=eps_ds["height"],
        kwargs={"fill_value": np.nan},
    )

    eps_ds["screen_fraction"] = screen_fraction.interp(
        time=eps_ds["time"],
        height=eps_ds["height"],
        kwargs={"fill_value": np.nan},
    )

    eps_ds["w_mean"] = w_mean.interp(
        time=eps_ds["time"],
        height=eps_ds["height"],
        kwargs={"fill_value": np.nan},
    )

    eps_ds["n_samples"] = n_samples.interp(
        time=eps_ds["time"],
        height=eps_ds["height"],
        method="nearest",
        kwargs={"fill_value": np.nan},
    )

    # Negative noise-corrected variance is not a physical standard deviation.
    eps_ds["w_std"] = np.sqrt(
        eps_ds["atmospheric_variance"].where(eps_ds["atmospheric_variance"] >= 0)
    )

    eps_ds.attrs["level2_product"] = "stare_epsilon"
    eps_ds.attrs["binsize"] = str(binsize)
    eps_ds.attrs["elevation_deg"] = float(stare_elevation_deg)
    eps_ds.attrs["snr_limit"] = float(snr_limit)
    eps_ds.attrs["sample_spacing_seconds"] = float(sample_spacing_seconds)
    eps_ds.attrs["stare_elevation_deg"] = float(stare_elevation_deg)
    eps_ds.attrs["wind_support_elevation_deg"] = float(wind_support_elevation_deg)

    return eps_ds


def process_stare_l2_from_ds(ds_stare, ds_wind, date_chosen, confDict):
    """
    Config-driven in-memory stare processing.
    """
    binsize = f"{_parse_int(confDict, 'STARE_AVG_MIN', confDict.get('AVG_MIN', 10))}min"
    date_start, date_end = _get_daily_timerange(date_chosen)

    eps_internal = lvl2_stare_epsilon(
        ds_stare=ds_stare,
        ds_wind=ds_wind,
        date_start=date_start,
        date_end=date_end,
        binsize=binsize,
        dv_name=_parse_str(confDict, "STARE_DV_NAME", "dv"),
        intensity_name=_parse_str(confDict, "STARE_INTENSITY_NAME", "intensity"),
        range_name=_parse_str(confDict, "STARE_RANGE_NAME", "range"),
        time_name=_parse_str(confDict, "STARE_TIME_NAME", "time"),
        wind_name=_parse_str(confDict, "STARE_WIND_NAME", "wspeed"),
        stare_elevation_deg = _parse_float(confDict, "STARE_ELEVATION_DEG", 90.0),
        wind_support_elevation_deg = _parse_float(confDict, "WIND_SUPPORT_ELEVATION_DEG",
                                                  _parse_float(confDict, "STARE_ELEVATION_DEG", 90.0),
                                                  ),
        snr_limit=_parse_float(confDict, "STARE_SNR_LIMIT", 5.0),
        n_negatives=_parse_int(confDict, "STARE_N_NEGATIVES", 5),
        noise_min=_parse_float(confDict, "STARE_NOISE_MIN", 1e-12),
        noise_smooth_min=_parse_float(confDict, "STARE_NOISE_SMOOTH_MIN", 1e-12),
        u_min=_parse_float(confDict, "STARE_U_MIN", 0.1),
        u_time_smooth=_parse_int(confDict, "STARE_U_TIME_SMOOTH", 6),
        u_height_smooth=_parse_int(confDict, "STARE_U_HEIGHT_SMOOTH", 3),
        u_use_median=_parse_bool(confDict, "STARE_U_USE_MEDIAN", True),
        lags=_parse_list_of_ints(confDict, "STARE_LAGS", "0,1,2,3"),
        fit_lags=_parse_fit_lags(confDict, "STARE_FIT_LAGS", "1,3"),
        sample_spacing_seconds=_parse_float(confDict, "STARE_SAMPLE_SPACING_SECONDS", 6.0),
        beam_divergence_half_angle_rad=_parse_float(confDict, "STARE_BEAM_DIVERGENCE_RAD", 0.0),
        min_sigma2=_parse_float(confDict, "STARE_MIN_SIGMA2", 1e-7),
    )

    if _parse_bool(confDict, "STARE_ADD_EPSILON_UNCERTAINTY", False):
        eps_internal = add_epsilon_uncertainty(
            eps_internal,
            binsize=binsize,
            sample_spacing_seconds=_parse_float(confDict, "STARE_SAMPLE_SPACING_SECONDS", 6.0),
            beam_divergence_half_angle_rad=_parse_float(confDict, "STARE_BEAM_DIVERGENCE_RAD", 0.0),
            min_samples=_parse_int(confDict, "STARE_EPSILON_MIN_SAMPLES", 3),
        )

    return stare_output_to_xarray(eps_internal, confDict=confDict)


def process_stare_l2_file(stare_l1_file, wind_l2_file, date_chosen, confDict):
    """
    File-based stare processing.
    """
    ds_stare = xr.open_dataset(stare_l1_file)
    ds_wind = xr.open_dataset(wind_l2_file)
    try:
        ds_out = process_stare_l2_from_ds(ds_stare, ds_wind, date_chosen, confDict)
    finally:
        ds_stare.close()
        ds_wind.close()
    return ds_out


def write_stare_l2_file(stare_l1_file, wind_l2_file, date_chosen, confDict, overwrite=True):
    """
    Run stare L2 processing and write output netCDF.
    """
    outfile = build_stare_l2_filename(confDict, date_chosen)

    if os.path.exists(outfile) and not overwrite:
        return outfile

    ds_out = process_stare_l2_file(stare_l1_file, wind_l2_file, date_chosen, confDict)
    encoding = {var: {"zlib": True, "complevel": 1} for var in ds_out.data_vars}
    ds_out.to_netcdf(outfile, encoding=encoding)
    return outfile


def run_stare_l2_daily(date_chosen, confDict, overwrite=True):
    """
    High-level daily driver aligned with wind_proc-style config usage.
    """
    if "NC_WIND_L2_PATH" not in confDict:
        raise KeyError("Missing required config key: NC_WIND_L2_PATH")
    if "NC_STARE_L1_PATH" not in confDict:
        raise KeyError("Missing required config key: NC_STARE_L1_PATH")
    if "NC_STARE_L2_PATH" not in confDict:
        raise KeyError("Missing required config key: NC_STARE_L2_PATH")
    
    stare_l1_file = build_stare_l1_filename(confDict, date_chosen)
    wind_l2_file = build_wind_l2_filename(confDict, date_chosen)

    if not os.path.isfile(stare_l1_file):
        raise FileNotFoundError(f"Missing stare L1 input file: {stare_l1_file}")
    if not os.path.isfile(wind_l2_file):
        raise FileNotFoundError(f"Missing wind L2 input file: {wind_l2_file}")

    return write_stare_l2_file(
        stare_l1_file=stare_l1_file,
        wind_l2_file=wind_l2_file,
        date_chosen=date_chosen,
        confDict=confDict,
        overwrite=overwrite,
    )


def stare_output_to_xarray(eps_ds, confDict=None):
    """
    Convert internal stare dataset into final Level-2 style dataset.
    """
    out = xr.Dataset(
        {
            "epsilon": (
                ["time", "height"],
                np.float32(eps_ds["epsilon"].values),
                {"units": "m2 s-3", "long_name": "TKE dissipation rate", "_FillValue": -999.0},
            ),
            "qepsilon": (
                ["time", "height"],
                eps_ds["qepsilon"].values.astype(np.int8),
                {
                    "long_name": "quality_flag_epsilon",
                    "flag_values": np.arange(0, 2).astype(np.int8),
                    "flag_meanings": "quality_bad quality_good",
                    "_FillValue": np.array(-128).astype(np.int8),
                },
            ),
            "w_mean": (
                ["time", "height"],
                np.float32(eps_ds["w_mean"].values),
                {
                    "units": "m s-1",
                    "standard_name": "upward_air_velocity",
                    "long_name": "Mean vertical velocity from screened stare measurements",
                    "_FillValue": -999.0,
                },
            ),
            "w_std": (
                ["time", "height"],
                np.float32(eps_ds["w_std"].values),
                {
                    "units": "m s-1",
                    "long_name": "Noise-corrected standard deviation of vertical velocity",
                    "_FillValue": -999.0,
                },
            ),
            "n_samples": (
                ["time", "height"],
                np.float32(eps_ds["n_samples"].values),
                {
                    "units": "1",
                    "long_name": "Number of screened stare samples in averaging interval",
                    "_FillValue": -999.0,
                },
            ),
            "total_variance": (
                ["time", "height"],
                np.float32(eps_ds["total_variance"].values),
                {"units": "m2 s-2", "long_name": "Total variance of vertical velocity", "_FillValue": -999.0},
            ),
            "atmospheric_variance": (
                ["time", "height"],
                np.float32(eps_ds["atmospheric_variance"].values),
                {"units": "m2 s-2", "long_name": "Noise-corrected atmospheric variance of vertical velocity", "_FillValue": -999.0},
            ),
            "noise_variance": (
                ["time", "height"],
                np.float32(eps_ds["noise_variance"].values),
                {"units": "m2 s-2", "long_name": "Noise variance of vertical velocity", "_FillValue": -999.0},
            ),
            "horizontal_wind_speed_for_epsilon": (
                ["time", "height"],
                np.float32(eps_ds["horizontal_wind_speed_for_epsilon"].values),
                {"units": "m s-1", "long_name": "Horizontal wind speed used for epsilon estimation", "_FillValue": -999.0},
            ),
            "u_fill_flag": (
                ["time", "height"],
                eps_ds["u_fill_flag"].values.astype(np.int8),
                {
                    "long_name": "fill_flag_for_wind_speed_used_in_epsilon",
                    "flag_values": np.arange(0, 6).astype(np.int8),
                    "flag_meanings": "original_valid smoothed_original interpolated_time interpolated_height nearest_fallback missing",
                    "_FillValue": np.array(-128).astype(np.int8),
                },
            ),
            "snr_median": (
                ["time", "height"],
                np.float32(eps_ds["snr_median"].values),
                {"units": "1", "long_name": "Median screened SNR", "_FillValue": -999.0},
            ),
            "screen_fraction": (
                ["time", "height"],
                np.float32(eps_ds["screen_fraction"].values),
                {"units": "1", "long_name": "Fraction of valid screened stare samples", "_FillValue": -999.0},
            ),
        },
        coords={
            "time": eps_ds["time"],
            "height": eps_ds["height"],
        },
    )

    optional_float_vars = [
        "epsilon_uncertainty",
        "atmospheric_variance_uncertainty",
        "horizontal_wind_speed_for_epsilon_uncertainty",
    ]
    for name in optional_float_vars:
        if name in eps_ds:
            out[name] = (
                ["time", "height"],
                np.float32(eps_ds[name].values),
                {"_FillValue": -999.0},
            )

    for name in out.data_vars:
        if np.issubdtype(out[name].dtype, np.floating):
            arr = out[name].values
            arr[~np.isfinite(arr)] = -999.0
            out[name].values[:] = arr

    if confDict is not None:
        configuration = ""
        for dd in confDict:
            if dd not in ["PROC_PATH", "NC_L1_PATH", "NC_L2_PATH", "NC_L2_QL_PATH"]:
                configuration += dd + "=" + str(confDict[dd]) + "\n"
        out["config"] = ([], configuration, {"standard_name": "configuration_file"})

    return out