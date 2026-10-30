# import datetime
import datetime

import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import diagsvd

from hpl2netCDF_client.signal_calc import in_db, CN_est
# import hpl2netCDF_client.wind_calc
from hpl2netCDF_client.wind_calc import (
    parse_float_list,
    set_azi_dir,
    calc_sigma_single,
    consensus,
    build_Amatrix,
    uvw_2_spd,
    uvw_2_dir,
    nanmax_where_valid_2d,
    nanmin_where_valid_2d,
    nanmedian_where_valid_2d,
    nanmean_where_valid_2d
)


MIN_BEAMS_FOR_RETRIEVAL = 4


def _build_time_chunks(time_ds, date_chosen, confDict):
    time_delta = int(confDict.get("UTC_OFFSET", 0))
    avg_min = int(confDict["AVG_MIN"])

    time_vec = np.arange(
        date_chosen - datetime.timedelta(hours=time_delta),
        date_chosen + datetime.timedelta(days=1) - datetime.timedelta(hours=time_delta)
        + datetime.timedelta(minutes=avg_min),
        datetime.timedelta(minutes=avg_min),
    )

    calc_idx = [
        np.where((ii <= time_ds) * (time_ds < iip1))
        for ii, iip1 in zip(time_vec[:-1], time_vec[1:])
    ]

    time_start = np.array(
        [
            int(
                pd.to_datetime(time_ds[t[0][-1]])
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
            )
            if len(t[0]) != 0
            else int(
                pd.to_datetime(time_vec[ii + 1])
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
            )
            for ii, t in enumerate(calc_idx)
        ]
    )

    time_bnds = np.array(
        [
            [
                int(
                    pd.to_datetime(time_ds[t[0][0]])
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                ),
                int(
                    pd.to_datetime(time_ds[t[0][-1]])
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                ),
            ]
            if len(t[0]) != 0
            else [
                int(
                    pd.to_datetime(time_vec[ii])
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                ),
                int(
                    pd.to_datetime(time_vec[ii + 1])
                    .replace(tzinfo=datetime.timezone.utc)
                    .timestamp()
                ),
            ]
            for ii, t in enumerate(calc_idx)
        ]
    ).T

    return calc_idx, time_start, time_bnds


def process_time_chunk(
    kk, n_rays, confDict, calc_idx, azimuth, dv, snr, beta, delv, elevation, time_start,
    M, n, B, mode,
    UVW, UVWunc, SPEED, SPEEDunc, DIREC, DIRECunc, R2, CN, n_good, binary_dir,
    SNR_tot, SNR_max, SNR_min, SPEC_tot, BETA_tot
):
    idx_all = calc_idx[kk][0]
    ele_all = elevation[idx_all]
    azi_all = azimuth[idx_all]

    slanted_mask = ele_all < 89
    vertical_mask = ele_all > 89
    has_vertical = np.any(vertical_mask)

    # Number of azimuthal directions used for set_azi_dir / beam spacing.
    # Prefer explicit horizontal count if provided.
    if 'N_HORI_DIRECTIONS' in confDict:
        n_rays_h = int(confDict['N_HORI_DIRECTIONS'])
    else:
        n_rays_h = n_rays - int(has_vertical)

    if n_rays_h < 1:
        print('invalid horizontal direction count', n_rays_h, time_start[kk])
        return

    # Build azimuth grouping only from slanted beams
    calc_idx_slanted = [np.where(slanted_mask)]
    indicator, azi_mean = set_azi_dir(n_rays_h, calc_idx_slanted, azi_all[slanted_mask], 0, confDict)
    r_phi = 360 / n_rays_h / 2
    
    if 'EX_AZI' in confDict:
        azi_ex = parse_float_list(confDict['EX_AZI'])
        azi_mean = np.setdiff1d(azi_mean, azi_ex, assume_unique=False)

    if not indicator:
        print('some issue with the data', n_rays_h, len(azi_mean), time_start[kk])
        return

    # Keep full chunk data available so vertical stares can be processed too
    VR_all = dv[idx_all]
    SNR_all = snr[idx_all]
    BETA_all = beta[idx_all]
    SPEC_all = delv[idx_all]

    # Slanted subset for the azimuthal VAD/DBS part
    VR = VR_all[slanted_mask]
    SNR = SNR_all[slanted_mask]
    BETA = BETA_all[slanted_mask]
    SPEC = SPEC_all[slanted_mask]
    azi = azi_all[slanted_mask]
    ele = ele_all[slanted_mask]

    n_dirs = len(azi_mean)
    n_gates = VR.shape[1]

    SNR_CNS = np.full((n_dirs, n_gates), np.nan)
    SNR_CNS_max = np.full((n_dirs, n_gates), np.nan)
    SNR_CNS_min = np.full((n_dirs, n_gates), np.nan)
    SPEC_CNS = np.full((n_dirs, n_gates), np.nan)
    BETA_CNS = np.full((n_dirs, n_gates), np.nan)
    VR_CNSmax = np.full((n_dirs, n_gates), np.nan)
    VR_CNSunc = np.full((n_dirs, n_gates), np.nan)
    SIGMA_CNS = np.full((n_dirs, n_gates), np.nan)
    ele_cns = np.full((n_dirs,), np.nan)

    for ii, azi_i in enumerate(azi_mean):
        azi_idx = (np.mod(360 - np.mod(np.mod(azi - azi_i, 360) - r_phi, 360), 360) < 2 * r_phi)

        if not np.any(azi_idx):
            continue

        ele_cns[ii] = np.median(ele[azi_idx])

        VR_CNSmax[ii, :], idx_tmp, VR_CNSunc[ii, :] = consensus(
            VR[azi_idx], SNR[azi_idx], BETA[azi_idx],
            int(confDict['CNS_RANGE']),
            int(confDict['CNS_PERCENTAGE']),
            int(confDict['SNR_THRESHOLD']),
            B
        )

        if mode == "halo":
            sigma_tmp = calc_sigma_single(in_db(SNR[azi_idx]), M, n, 2 * B, 1.316)
            valid_counts = np.sum(idx_tmp, axis=0)
            SIGMA_CNS[ii, :] = np.divide(
                np.sqrt(np.nansum(np.where(idx_tmp, sigma_tmp ** 2, np.nan), axis=0)),
                valid_counts,
                out=np.full(n_gates, np.nan),
                where=valid_counts > 0
            )
        else:
            SIGMA_CNS[ii, :] = VR_CNSunc[ii, :]

        masked_snr = np.where(idx_tmp, SNR[azi_idx], np.nan)
        masked_spec = np.where(idx_tmp, SPEC[azi_idx], np.nan)
        masked_beta = np.where(idx_tmp, BETA[azi_idx], np.nan)

        SNR_CNS[ii, :] = nanmean_where_valid_2d(masked_snr)
        SPEC_CNS[ii, :] = nanmean_where_valid_2d(masked_spec)
        BETA_CNS[ii, :] = nanmean_where_valid_2d(masked_beta)

        valid_any = np.any(idx_tmp, axis=0)
        with np.errstate(all='ignore'):
            SNR_CNS_max[ii, :] = nanmax_where_valid_2d(masked_snr)
            SNR_CNS_min[ii, :] = nanmin_where_valid_2d(masked_snr)

    # ------------------------------------------------------------
    # Add vertical stare as one extra retrieval beam, if present
    # ------------------------------------------------------------
    if has_vertical:
        WR = VR_all[vertical_mask]
        SNR_WR = SNR_all[vertical_mask]
        BETA_WR = BETA_all[vertical_mask]
        SPEC_WR = SPEC_all[vertical_mask]

        # w_cns, idx_tmp_wr, WR_unc = consensus(
        #     WR,
        #     np.ones(WR.shape),
        #     np.ones(WR.shape),
        #     int(confDict.get('CNS_RANGE_VERTICAL', 2)),
        #     int(confDict.get('CNS_PERCENTAGE_VERTICAL', 30)),
        #     int(confDict.get('SNR_THRESHOLD_VERTICAL', 0)),
        #     B
        # )
        w_cns, idx_tmp_wr, WR_unc = consensus(
            WR,
            SNR_WR,
            BETA_WR,
            int(confDict.get('CNS_RANGE_VERTICAL', 2)),
            int(confDict.get('CNS_PERCENTAGE_VERTICAL', 30)),
            int(confDict.get('SNR_THRESHOLD_VERTICAL', 0)),
            B
        )

        WR_SNR = nanmean_where_valid_2d(np.where(idx_tmp_wr, SNR_WR, np.nan))
        WR_BETA = nanmean_where_valid_2d(np.where(idx_tmp_wr, BETA_WR, np.nan))
        WR_SPEC = nanmean_where_valid_2d(np.where(idx_tmp_wr, SPEC_WR, np.nan))

        valid_any_wr = np.any(idx_tmp_wr, axis=0)
        with np.errstate(all='ignore'):
            WR_SNR_max = nanmax_where_valid_2d(np.where(idx_tmp_wr, SNR_WR, np.nan))
            WR_SNR_min = nanmin_where_valid_2d(np.where(idx_tmp_wr, SNR_WR, np.nan))

        VR_CNSmax = np.vstack([VR_CNSmax, w_cns])
        SIGMA_CNS = np.vstack([SIGMA_CNS, WR_unc])

        SNR_CNS = np.vstack([SNR_CNS, WR_SNR])
        SNR_CNS_max = np.vstack([SNR_CNS_max, WR_SNR_max])
        SNR_CNS_min = np.vstack([SNR_CNS_min, WR_SNR_min])
        SPEC_CNS = np.vstack([SPEC_CNS, WR_SPEC])
        BETA_CNS = np.vstack([BETA_CNS, WR_BETA])

        azi_mean = np.hstack([azi_mean, 0.0])
        ele_cns = np.hstack([ele_cns, 90.0])

    n_dirs_eff = len(azi_mean)

    n_good_kk = (~np.isnan(VR_CNSmax)).sum(axis=0)
    n_good[kk, :] = n_good_kk

    binary_dir[kk, :] = (~np.isnan(VR_CNSmax)).T.dot(1 << np.arange(n_dirs_eff - 1, -1, -1))

    V_r = np.ma.masked_where(np.isnan(VR_CNSmax), VR_CNSmax).T[..., None]
    mask_V_in = (np.isnan(VR_CNSmax)) | (np.tile(n_good_kk, (n_dirs_eff, 1)) < MIN_BEAMS_FOR_RETRIEVAL)
    V_in = np.ma.masked_where(mask_V_in, VR_CNSmax)
    A = np.round(build_Amatrix(azi_mean, ele_cns), 6)
    SIGMA_r = np.ma.masked_where(np.isnan(VR_CNSmax), SIGMA_CNS).T

    condi = np.isnan(VR_CNSmax)
    U, S, Vh = [], [], []
    A_masked_list = []

    for c_nn in condi.T:
        A_masked = np.ma.masked_where(np.tile(c_nn, (3, 1)).T, A).filled(0)
        u, s, vh = np.linalg.svd(A_masked, full_matrices=True)
        U.append(u)
        S.append(np.linalg.pinv(diagsvd(s, u.shape[0], vh.shape[0])))
        Vh.append(vh)
        A_masked_list.append(A_masked)

    U = np.array(U)
    S = np.array(S)
    Vh = np.array(Vh)
    A_eff = np.array(A_masked_list)

    U_T = np.einsum('...ij->...ji', U)
    Vh_T = np.einsum('...ij->...ji', Vh)

    K1 = np.nansum((U_T * V_in.T[:, None, :]), axis=2)[..., None]
    K2 = np.einsum('...ik,...kj->...ij', S, K1)
    V_k = np.einsum('...ik,...kj->...ij', Vh_T, K2)

    UVW[kk, ...] = np.squeeze(V_k)
    bad_gate_mask = np.sum(~np.isnan(VR_CNSmax.T), axis=1) < MIN_BEAMS_FOR_RETRIEVAL
    UVW[kk, bad_gate_mask, :] = np.nan

    # Same masked pseudoinverse geometry as the retrieval
    A_r_MP = np.einsum('...ik,...kl,...lj->...ij', Vh_T, S, U_T)
    A_r_MP_T = np.einsum('...ij->...ji', A_r_MP)

    cov = np.einsum('nik,nk,nkl->nil', A_r_MP, SIGMA_r ** 2, A_r_MP_T)
    UVWunc[kk, ...] = np.sqrt(np.clip(np.einsum('...ii->...i', cov), 0, None))
    UVWunc[kk, bad_gate_mask, :] = np.nan

    V_r_est = np.einsum('nij,njk->nik', A_eff, V_k)
    ss_e = ((V_r - V_r_est) ** 2).sum(axis=1)
    ss_t = ((V_r - V_r.mean(axis=1)[:, None, :]) ** 2).sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        r2_tmp = np.squeeze(1 - ss_e / ss_t)
    r2_tmp = np.where(np.squeeze(ss_t) == 0, np.nan, r2_tmp)

    R2[kk, :] = r2_tmp
    R2[kk, bad_gate_mask] = np.nan

    A_eff_T = np.einsum('...ij->...ji', A_eff)
    diag_ata = np.einsum('...ii->...i', A_eff_T @ A_eff)
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_sqrt_diag = np.divide(
            1.0,
            np.sqrt(diag_ata),
            out=np.full_like(diag_ata, np.nan, dtype=float),
            where=diag_ata > 0
        )
    Spp = np.apply_along_axis(np.diag, 1, inv_sqrt_diag)

    mask_A = np.tile(mask_V_in.T[..., None], (1, 1, 3))
    Z = np.ma.masked_where(mask_A, A_eff @ Spp)
    CN[kk, :] = np.squeeze(np.array([CN_est(X) for X in Z]))
    CN[kk, bad_gate_mask] = np.nan

    SPEED[kk, :], SPEEDunc[kk, :] = np.vstack(
        [np.fromiter(uvw_2_spd(val, unc).values(), dtype=float)
         for val, unc in zip(UVW[kk, ...], UVWunc[kk, ...])]
    ).T

    DIREC[kk, :], DIRECunc[kk, :] = np.vstack(
        [np.fromiter(uvw_2_dir(val, unc).values(), dtype=float)
         for val, unc in zip(UVW[kk, ...], UVWunc[kk, ...])]
    ).T

    with np.errstate(all='ignore'):
        SNR_tot[kk, :] = nanmedian_where_valid_2d(SNR_CNS)
        BETA_tot[kk, :] = nanmedian_where_valid_2d(BETA_CNS)
        SNR_max[kk, :] = nanmax_where_valid_2d(SNR_CNS_max)
        SNR_min[kk, :] = nanmin_where_valid_2d(SNR_CNS_min)
        SPEC_tot[kk, :] = nanmedian_where_valid_2d(SPEC_CNS)

def wind_output_to_xarray(
    SPEED,
    SPEEDunc,
    DIREC,
    DIRECunc,
    R2,
    CN,
    n_good,
    UVW,
    UVWunc,
    binary_dir,
    SNR_tot,
    SNR_max,
    SNR_min,
    SPEC_tot,
    BETA_tot,
    confDict,
    NN,
    height,
    width,
    height_bnds,
    time_start,
    time_bnds,
):
    speed = np.copy(SPEED)
    errspeed = np.copy(SPEEDunc)
    wdir = np.copy(DIREC)
    errwdir = np.copy(DIRECunc)
    r2 = np.copy(R2)
    cn = np.copy(CN)
    nvrad = np.copy(n_good)
    u = np.copy(UVW[:, :, 0])
    v = np.copy(UVW[:, :, 1])
    w = np.copy(UVW[:, :, 2])
    erru = np.copy(UVWunc[:, :, 0])
    errv = np.copy(UVWunc[:, :, 1])
    errw = np.copy(UVWunc[:, :, 2])
    binary_dir_out = np.copy(binary_dir)

    qretrieval = ~np.isnan(SPEED)

    r2[np.isnan(r2)] = -999.0
    qr2 = r2 >= float(confDict["R2_THRESHOLD"])

    cn[np.isnan(cn)] = 999.0
    qcn = (cn >= 0) & (cn <= float(confDict["CN_THRESHOLD"]))

    nvrad[np.isnan(nvrad)] = -999.0
    qnvrad = nvrad >= int(confDict["N_VRAD_THRESHOLD"])

    qwind = qretrieval & qnvrad & qcn & qr2

    # Keep flags consistent with stored component data
    qu = qwind.copy()
    qv = qwind.copy()
    qw = qwind.copy()
    qspeed = qwind.copy()

    speed[~qspeed] = -999.0
    errspeed[~qspeed] = -999.0
    wdir[~qspeed] = -999.0
    errwdir[~qspeed] = -999.0
    u[~qu] = -999.0
    v[~qv] = -999.0
    w[~qw] = -999.0
    erru[~qu] = -999.0
    errv[~qv] = -999.0
    errw[~qw] = -999.0
    r2[~qr2] = -999.0
    cn[~qcn] = -999.0
    nvrad[~qnvrad] = -999.0
    binary_dir_out[~qwind] = -999

    if not np.any(qretrieval):
        print("WARNING: bad retrieval quality")
        print("all retrieved velocities are NaN -> check nvrad threshold!")

    configuration = ""
    for dd in confDict:
        if dd not in ["PROC_PATH", "NC_L1_PATH", "NC_L2_PATH", "NC_L2_QL_PATH"]:
            configuration += dd + "=" + confDict[dd] + "\n"

    return xr.Dataset(
        {
            "config": (
                [],
                configuration,
                {"standard_name": "configuration_file"},
            ),
            "snr": (
                ["time", "height"],
                np.float32(SNR_tot[:, NN:]),
                {
                    "units": "fraction",
                    "comments": "median signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1",
                    "standard_name": "snr",
                    "long_name": "Signal-to-Noise-Ratio",
                    "_FillValue": -999.0,
                },
            ),
            "snr_max": (
                ["time", "height"],
                np.float32(SNR_max[:, NN:]),
                {
                    "units": "fraction",
                    "comments": "maximal signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1",
                    "standard_name": "snr_max",
                    "long_name": "maximal_Signal-to-Noise-Ratio",
                    "_FillValue": -999.0,
                },
            ),
            "snr_min": (
                ["time", "height"],
                np.float32(SNR_min[:, NN:]),
                {
                    "units": "fraction",
                    "comments": "minimal signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1",
                    "standard_name": "snr_min",
                    "long_name": "minimal_Signal-to-Noise-Ratio",
                    "_FillValue": -999.0,
                },
            ),
            "beta": (
                ["time", "height"],
                np.float32(BETA_tot[:, NN:]),
                {
                    "units": "m-1 sr-1",
                    "comments": "median attenuated backscatter coefficient along VAD cycle.",
                    "standard_name": "beta",
                    "long_name": "attenuated backscatter coefficient",
                    "_FillValue": -999.0,
                },
            ),
            "wspeed": (
                ["time", "height"],
                np.float32(speed[:, NN:]),
                {
                    "units": "m s-1",
                    "comments": "Scalar wind speed (amount of vector)",
                    "standard_name": "wind_speed",
                    "long_name": "Wind Speed",
                    "_FillValue": -999.0,
                },
            ),
            "spec_w": (
                ["time", "height"],
                np.float32(SPEC_tot[:, NN:]),
                {
                    "units": "m s-1",
                    "comments": "spectral width of wind estimate",
                    "standard_name": "spectral_width",
                    "long_name": "spectral width",
                    "_FillValue": -999.0,
                },
            ),
            "qwind": (
                ["time", "height"],
                qwind[:, NN:].astype(np.int8),
                {
                    "comments": "quality flag 0 or 1 for u, v, w, wspeed, wdir and corresponding errors",
                    "long_name": "wind_quality_flag",
                    "_FillValue": np.array(-128).astype(np.int8),
                    "flag_values": np.arange(0, 2).astype(np.int8),
                    "flag_meanings": "quality_bad quality_good",
                },
            ),
            "qu": (
                ["time", "height"],
                qu[:, NN:].astype(np.int8),
                {
                    "comments": "quality flag 0 or 1 for u and corresponding error",
                    "long_name": "quality_flag_u",
                    "_FillValue": np.array(-128).astype(np.int8),
                    "flag_values": np.arange(0, 2).astype(np.int8),
                    "flag_meanings": "quality_bad quality_good",
                },
            ),
            "qv": (
                ["time", "height"],
                qv[:, NN:].astype(np.int8),
                {
                    "comments": "quality flag 0 or 1 for v and corresponding error",
                    "long_name": "quality_flag_v",
                    "_FillValue": np.array(-128).astype(np.int8),
                    "flag_values": np.arange(0, 2).astype(np.int8),
                    "flag_meanings": "quality_bad quality_good",
                },
            ),
            "qw": (
                ["time", "height"],
                qw[:, NN:].astype(np.int8),
                {
                    "comments": "quality flag 0 or 1 for w and corresponding error",
                    "long_name": "quality_flag_w",
                    "_FillValue": np.array(-128).astype(np.int8),
                    "flag_values": np.arange(0, 2).astype(np.int8),
                    "flag_meanings": "quality_bad quality_good",
                },
            ),
            "errwspeed": (
                ["time", "height"],
                np.float32(errspeed[:, NN:]),
                {
                    "units": "m s-1",
                    "standard": "wind_speed_uncertainty",
                    "long_name": "Wind Speed Uncertainty",
                    "_FillValue": -999.0,
                },
            ),
            "u": (
                ["time", "height"],
                np.float32(u[:, NN:]),
                {
                    "units": "m s-1",
                    "comments": '"Eastward indicates" a vector component which is positive when directed eastward (negative westward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component',
                    "standard_name": "eastward_wind",
                    "long_name": "Zonal Wind",
                    "_FillValue": -999.0,
                },
            ),
            "erru": (
                ["time", "height"],
                np.float32(erru[:, NN:]),
                {
                    "units": "m s-1",
                    "standard_name": "eastward_wind_uncertainty",
                    "long_name": "Zonal Wind Uncertainty",
                    "_FillValue": -999.0,
                },
            ),
            "v": (
                ["time", "height"],
                np.float32(v[:, NN:]),
                {
                    "units": "m s-1",
                    "comments": '"Northward indicates" a vector component which is positive when directed northward (negative southward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component',
                    "standard_name": "northward_wind",
                    "long_name": "Meridional Wind",
                    "_FillValue": -999.0,
                },
            ),
            "errv": (
                ["time", "height"],
                np.float32(errv[:, NN:]),
                {
                    "units": "m s-1",
                    "standard_name": "northward_wind_uncertainty",
                    "long_name": "Meridional Wind Uncertainty",
                    "_FillValue": -999.0,
                },
            ),
            "w": (
                ["time", "height"],
                np.float32(w[:, NN:]),
                {
                    "units": "m s-1",
                    "comments": "Vertical wind component, positive when directed upward (negative downward)",
                    "standard_name": "upward_air_velocity",
                    "long_name": "Upward Air Velocity",
                    "_FillValue": -999.0,
                },
            ),
            "errw": (
                ["time", "height"],
                np.float32(errw[:, NN:]),
                {
                    "units": "m s-1",
                    "standard_name": "upward_air_velocity_uncertainty",
                    "long_name": "Upward Air Velocity Uncertainty",
                    "_FillValue": -999.0,
                },
            ),
            "wdir": (
                ["time", "height"],
                np.float32(wdir[:, NN:]),
                {
                    "units": "degree",
                    "comments": "Wind direction",
                    "standard_name": "wind_from_direction",
                    "long_name": "Wind Direction",
                    "_FillValue": -999.0,
                },
            ),
            "errwdir": (
                ["time", "height"],
                np.float32(errwdir[:, NN:]),
                {
                    "units": "degree",
                    "standard_name": "wind_direction_uncertainty",
                    "long_name": "Wind Direction Uncertainty",
                    "_FillValue": -999.0,
                },
            ),
            "r2": (
                ["time", "height"],
                np.float32(r2[:, NN:]),
                {
                    "comments": "coefficient of determination",
                    "long_name": "coefficient of determination",
                    "standard_name": "coefficient_of_determination",
                    "units": "1",
                    "_FillValue": -999.0,
                },
            ),
            "nvrad": (
                ["time", "height"],
                np.float32(nvrad[:, NN:]),
                {
                    "comments": "number of (averaged) radial velocities used for wind calculation",
                    "long_name": "number of radial velocities",
                    "standard_name": "no_radial_velocities",
                    "units": "1",
                    "_FillValue": -999.0,
                },
            ),
            "cn": (
                ["time", "height"],
                np.float32(cn[:, NN:]),
                {
                    "comments": "condition number",
                    "standard_name": "condition_number",
                    "long_name": "Condition Number",
                    "units": "1",
                    "_FillValue": -999.0,
                },
            ),
            "lat": (
                [],
                np.float32(confDict["SYSTEM_LATITUDE"]),
                {
                    "units": "degrees_north",
                    "long_name": "latitude",
                    "standard_name": "latitude",
                    "comments": "latitude of sensor",
                    "_FillValue": -999.0,
                },
            ),
            "lon": (
                [],
                np.float32(confDict["SYSTEM_LONGITUDE"]),
                {
                    "units": "degrees_east",
                    "long_name": "longitude",
                    "standard_name": "longitude",
                    "comments": "longitude of sensor",
                    "_FillValue": -999.0,
                },
            ),
            "zsl": (
                [],
                np.float32(confDict["SYSTEM_ALTITUDE"]),
                {
                    "units": "m",
                    "comments": "Altitude of sensor above mean sea level",
                    "standard_name": "altitude",
                    "_FillValue": -999.0,
                },
            ),
            "time_bnds": (
                ["time", "nv"],
                time_bnds.T.astype(np.float64),
                {"units": "seconds since 1970-01-01 00:00:00 UTC"},
            ),
            "height_bnds": (
                ["height", "nv"],
                np.float32(height_bnds[NN:, :]),
                {"units": "m"},
            ),
            "frequency": (
                [],
                np.float32(299792458 / float(confDict["SYSTEM_WAVELENGTH"])),
                {
                    "units": "Hz",
                    "comments": "lidar operating frequency",
                    "long_name": "instrument_frequency",
                    "_FillValue": -999.0,
                },
            ),
            "vert_res": (
                [],
                np.float32(np.diff(height).mean()),
                {
                    "units": "m",
                    "comments": "Calculated from pulse wdth and beam elevation",
                    "long_name": "Vertical_resolution_measurement",
                    "_FillValue": -999.0,
                },
            ),
            "hor_width": (
                ["height"],
                np.float32(width[NN:]),
                {
                    "units": "m",
                    "comments": "Calculated from beam elevation and height",
                    "standard_name": "horizontal_sample_width",
                    "_FillValue": -999.0,
                },
            ),
            "binary_dir": (
                ["time", "height"],
                binary_dir_out[:, NN:],
                {
                    "long_name": "directional binary mask",
                    "comments": "Bitmask encoding which azimuthal beams contributed to each wind retrieval. Each bit represents a direction.",
                    "units": "1",
                    "_FillValue": -999,
                },
            ),
            'bearing': ([]
            , np.float32(float(confDict.get("AZIMUTH_OFFSET", confDict.get("BEARING", 0.0))))
            , {
                'units': 'degree',
                'long_name': 'bearing/azimuth offset applied to raw instrument azimuth',
                'comments': 'Clockwise offset added to raw azimuth before wind retrieval'
            }),
        },
        coords={
            "height": (
                ["height"],
                np.float32(height[NN:]),
                {
                    "units": "m",
                    "standard_name": "height",
                    "comments": "vertical distance from sensor to centre of range gate",
                    "bounds": "height_bnds",
                },
            ),
            "time": (
                ["time"],
                time_start.astype(np.float64),
                {
                    "units": "seconds since 1970-01-01 00:00:00",
                    "comments": "Timestamp at the end of the averaging interval",
                    "standard_name": "time",
                    "long_name": "Time",
                    "calendar": "gregorian",
                    "bounds": "time_bnds",
                    "_CoordinateAxisType": "Time",
                },
            ),
            "nv": (["nv"], np.arange(0, 2).astype(np.int8)),
        },
    )


def lvl2vad_standard(ds, date_chosen, confDict):
    n_rays = int(confDict["NUMBER_OF_DIRECTIONS"])
    n_gates = int(confDict["NUMBER_OF_GATES"])
    n = ds.prf.data
    M = ds.nsmpl.data
    B = ds.nqv.data

    elevation = 90 - ds.zenith.data
    # azimuth = ds.azi.data[elevation < 89] % 360
    # time_ds = ds.time.data[elevation < 89]
    # dv = ds.dv.data[elevation < 89]
    # snr = ds.intensity.data[elevation < 89] - 1
    # beta = ds.beta.data[elevation < 89]

    # if "delv" in list(ds.keys()):
    #     delv = ds.delv.data[elevation < 89]
    # else:
    #     delv = np.full(dv.shape, -999)
    #     elevation = 90 - ds.zenith.data
    az_offset = float(confDict.get("AZIMUTH_OFFSET", confDict.get("BEARING", 0.0)))
    azimuth = (ds.azi.data + az_offset) % 360
    time_ds = ds.time.data
    dv = ds.dv.data
    snr = ds.intensity.data - 1
    beta = ds.beta.data
    if "delv" in list(ds.keys()):
        delv = ds.delv.data
    else:
        delv = np.full(dv.shape, -999)

    elev_slanted = elevation[elevation < 89]
    elev_med = np.nanmedian(elev_slanted) * np.pi / 180
    height = ds.range.data * np.sin(elev_med)
    width = ds.range.data * 2 * np.cos(elev_med)
    height_bnds = ds.range_bnds.data.copy()
    height_bnds[:, 0] = np.sin(elev_med) * height_bnds[:, 0]
    height_bnds[:, 1] = np.sin(elev_med) * height_bnds[:, 1]

    calc_idx, time_start, time_bnds = _build_time_chunks(time_ds, date_chosen, confDict)

    if n_gates != dv.shape[1]:
        print("Warning: number of gates in config does not match lvl1 data!")
        n_gates = dv.shape[1]
        print("number of gates changed to " + str(n_gates))

    time_valid = [ii for ii, x in enumerate(calc_idx) if len(x[0]) != 0]

    UVW = np.full((len(calc_idx), n_gates, 3), np.nan)
    UVWunc = np.full((len(calc_idx), n_gates, 3), np.nan)
    SPEED = np.full((len(calc_idx), n_gates), np.nan)
    SPEEDunc = np.full((len(calc_idx), n_gates), np.nan)
    DIREC = np.full((len(calc_idx), n_gates), np.nan)
    DIRECunc = np.full((len(calc_idx), n_gates), np.nan)
    R2 = np.full((len(calc_idx), n_gates), np.nan)
    CN = np.full((len(calc_idx), n_gates), np.nan)
    n_good = np.full((len(calc_idx), n_gates), np.nan)
    binary_dir = np.zeros((len(calc_idx), n_gates), dtype=int)
    SNR_tot = np.full((len(calc_idx), n_gates), np.nan)
    SNR_max = np.full((len(calc_idx), n_gates), np.nan)
    SNR_min = np.full((len(calc_idx), n_gates), np.nan)
    SPEC_tot = np.full((len(calc_idx), n_gates), np.nan)
    BETA_tot = np.full((len(calc_idx), n_gates), np.nan)

    for kk in time_valid:
        process_time_chunk(
            kk,
            n_rays,
            confDict,
            calc_idx,
            azimuth,
            dv,
            snr,
            beta,
            delv,
            elevation,
            time_start,
            M,
            n,
            B,
            "halo",
            UVW,
            UVWunc,
            SPEED,
            SPEEDunc,
            DIREC,
            DIRECunc,
            R2,
            CN,
            n_good,
            binary_dir,
            SNR_tot,
            SNR_max,
            SNR_min,
            SPEC_tot,
            BETA_tot,
        )

    NN = int(confDict.get("BLINDEZONE_GATES", 0))

    return wind_output_to_xarray(
        SPEED,
        SPEEDunc,
        DIREC,
        DIRECunc,
        R2,
        CN,
        n_good,
        UVW,
        UVWunc,
        binary_dir,
        SNR_tot,
        SNR_max,
        SNR_min,
        SPEC_tot,
        BETA_tot,
        confDict,
        NN,
        height,
        width,
        height_bnds,
        time_start,
        time_bnds,
    )


def lvl2wcdbs(ds, date_chosen, confDict):
    n_rays = int(confDict["NUMBER_OF_DIRECTIONS"])
    n_gates = int(confDict["NUMBER_OF_GATES"])
    B = (ds.radial_wind_speed.max() - ds.radial_wind_speed.min()).data / 2
    lrg = ds.range_gate_length.data

    time_ds = pd.to_datetime(ds.time.data, unit="s")
    ds["zenith"] = 90 - ds.elevation
    elevation = ds.elevation.data
    az_offset = float(confDict.get("AZIMUTH_OFFSET", confDict.get("BEARING", 0.0)))
    azimuth = (ds.azimuth.data + az_offset) % 360
    snr = 10 ** (ds.cnr.data / 10)
    dv = ds.radial_wind_speed.data
    beta = ds.relative_beta.data if "relative_beta" in list(ds.keys()) else np.ones(dv.shape)

    if "measurement_height" in list(ds.keys()):
        height = ds.measurement_height.data[0]
    else:
        height = (np.sin(ds.elevation * np.pi / 180) * ds.range).data[0]

    width = (np.cos(ds.elevation * np.pi / 180) * ds.range * 2).data
    width = width[elevation < 89][0]

    # fixed
    height_bnds = np.vstack([height - lrg / 2, height + lrg / 2]).T

    delv = ds.doppler_spectrum_width.data

    calc_idx, time_start, time_bnds = _build_time_chunks(time_ds, date_chosen, confDict)

    if n_gates != dv.shape[1]:
        print("Warning: number of gates in config does not match lvl1 data!")
        n_gates = dv.shape[1]
        print("number of gates changed to " + str(n_gates))

    time_valid = [ii for ii, x in enumerate(calc_idx) if len(x[0]) != 0]

    UVW = np.full((len(calc_idx), n_gates, 3), np.nan)
    UVWunc = np.full((len(calc_idx), n_gates, 3), np.nan)
    SPEED = np.full((len(calc_idx), n_gates), np.nan)
    SPEEDunc = np.full((len(calc_idx), n_gates), np.nan)
    DIREC = np.full((len(calc_idx), n_gates), np.nan)
    DIRECunc = np.full((len(calc_idx), n_gates), np.nan)
    R2 = np.full((len(calc_idx), n_gates), np.nan)
    CN = np.full((len(calc_idx), n_gates), np.nan)
    n_good = np.full((len(calc_idx), n_gates), np.nan)
    binary_dir = np.zeros((len(calc_idx), n_gates), dtype=int)
    SNR_tot = np.full((len(calc_idx), n_gates), np.nan)
    SNR_max = np.full((len(calc_idx), n_gates), np.nan)
    SNR_min = np.full((len(calc_idx), n_gates), np.nan)
    SPEC_tot = np.full((len(calc_idx), n_gates), np.nan)
    BETA_tot = np.full((len(calc_idx), n_gates), np.nan)

    for kk in time_valid:
        process_time_chunk(
            kk,
            n_rays,
            confDict,
            calc_idx,
            azimuth,
            dv,
            snr,
            beta,
            delv,
            elevation,
            time_start,
            None,
            None,
            B,
            "windcube",
            UVW,
            UVWunc,
            SPEED,
            SPEEDunc,
            DIREC,
            DIRECunc,
            R2,
            CN,
            n_good,
            binary_dir,
            SNR_tot,
            SNR_max,
            SNR_min,
            SPEC_tot,
            BETA_tot,
        )

    NN = int(confDict.get("BLINDEZONE_GATES", 0))

    return wind_output_to_xarray(
        SPEED,
        SPEEDunc,
        DIREC,
        DIRECunc,
        R2,
        CN,
        n_good,
        UVW,
        UVWunc,
        binary_dir,
        SNR_tot,
        SNR_max,
        SNR_min,
        SPEC_tot,
        BETA_tot,
        confDict,
        NN,
        height,
        width,
        height_bnds,
        time_start,
        time_bnds,
    )
# import numpy as np
# import pandas as pd
# import xarray as xr
# from scipy.linalg import diagsvd

# from hpl2netCDF_client.signal_calc import in_db, CN_est
# from hpl2netCDF_client.wind_calc import set_azi_dir, calc_sigma_single, consensus, build_Amatrix, uvw_2_spd, uvw_2_dir


# def process_time_chunk(
#     kk, n_rays, confDict, calc_idx, azimuth, dv, snr, beta, delv, elevation, time_start,
#     set_azi_dir, consensus, calc_sigma_single, in_db, M, n, B, mode,
#     UVW, UVWunc, SPEED, SPEEDunc, DIREC, DIRECunc, R2, CN, n_good, binary_dir,
#     SNR_tot, SNR_max, SNR_min, SPEC_tot, BETA_tot
# ):
#     indicator, azi_mean = set_azi_dir(n_rays, calc_idx, azimuth, kk, confDict)
#     r_phi = 360 / (n_rays) / 2
#     if not indicator:
#         print('some issue with the data', n_rays, len(azi_mean), time_start[kk])
#         return

#     # Mode-specific slicing
#     if mode == "halo":
#         VR = dv[calc_idx[kk]]
#         SNR = snr[calc_idx[kk]]
#         BETA = beta[calc_idx[kk]]
#         SPEC = delv[calc_idx[kk]]
#         azi = azimuth[calc_idx[kk]]
#         ele = elevation[calc_idx[kk]]
#     else:  # windcube
#         mask = elevation[calc_idx[kk]] < 89
#         VR = dv[calc_idx[kk]][mask]
#         SNR = snr[calc_idx[kk]][mask]
#         BETA = beta[calc_idx[kk]][mask]
#         SPEC = delv[calc_idx[kk]][mask]
#         azi = azimuth[calc_idx[kk]][mask]
#         ele = elevation[calc_idx[kk]][mask]

#     SNR_CNS = np.full((len(azi_mean), VR.shape[1]), np.nan)
#     SNR_CNS_max = np.full((len(azi_mean), VR.shape[1]), np.nan)
#     SNR_CNS_min = np.full((len(azi_mean), VR.shape[1]), np.nan)
#     SPEC_CNS = np.full((len(azi_mean), VR.shape[1]), np.nan)
#     BETA_CNS = np.full((len(azi_mean), VR.shape[1]), np.nan)
#     VR_CNSmax = np.full((len(azi_mean), VR.shape[1]), np.nan)
#     VR_CNSunc = np.full((len(azi_mean), VR.shape[1]), np.nan)
#     SIGMA_CNS = np.full((len(azi_mean), VR.shape[1]), np.nan)
#     ele_cns = np.full((len(azi_mean),), np.nan)

#     for ii, azi_i in enumerate(azi_mean):
#         azi_idx = (np.mod(360 - np.mod(np.mod(azi - azi_i, 360) - r_phi, 360), 360) <= 2 * r_phi)
#         ele_cns[ii] = np.median(ele[azi_idx])
#         VR_CNSmax[ii, :], idx_tmp, VR_CNSunc[ii, :] = consensus(
#             VR[azi_idx], SNR[azi_idx], BETA[azi_idx],
#             int(confDict['CNS_RANGE']),
#             int(confDict['CNS_PERCENTAGE']),
#             int(confDict['SNR_THRESHOLD']),
#             B
#         )
#         if mode == "halo":
#             sigma_tmp = calc_sigma_single(in_db(SNR[azi_idx]), M, n, 2 * B, 1.316)
#             SIGMA_CNS[ii, :] = np.ma.divide(
#                 np.sqrt(np.nansum(np.where(idx_tmp, sigma_tmp ** 2, np.nan), axis=0)),
#                 np.sum(idx_tmp, axis=0)
#             )
#         else:
#             SIGMA_CNS[ii, :] = VR_CNSunc[ii, :]
#         SNR_CNS[ii,:]= np.nanmean( np.where( idx_tmp, SNR[azi_idx], np.nan), axis=0)
#         SNR_CNS_max[ii,:]= np.nanmax( np.where( idx_tmp, SNR[azi_idx], np.nan), axis=0)
#         SNR_CNS_min[ii,:]= np.nanmin( np.where( idx_tmp, SNR[azi_idx], np.nan), axis=0)
#         SPEC_CNS[ii,:]= np.nanmean( np.where( idx_tmp, SPEC[azi_idx], np.nan), axis=0)
#         BETA_CNS[ii,:]= np.nanmean( np.where(idx_tmp, BETA[azi_idx], np.nan), axis=0)

#     n_good_kk = (~np.isnan(VR_CNSmax)).sum(axis=0)
#     n_good[kk, :] = n_good_kk
#     binary_dir[kk, :] = (~np.isnan(VR_CNSmax)).T.dot(1 << np.arange(n_rays-1, -1, -1))

#     V_r = np.ma.masked_where(np.isnan(VR_CNSmax), VR_CNSmax).T[..., None]
#     mask_V_in = (np.isnan(VR_CNSmax)) | (np.tile(n_good_kk, (azi_mean.shape[0], 1)) < 4)
#     V_in = np.ma.masked_where(mask_V_in, VR_CNSmax)
#     A = build_Amatrix(azi_mean, ele_cns)
#     A_r = np.tile(A, (VR_CNSmax.shape[1], 1, 1))
#     A_r_MP = np.tile(np.linalg.pinv(A), (VR_CNSmax.shape[1], 1, 1))
#     A_r_MP_T = np.einsum('...ij->...ji', A_r_MP)
#     SIGMA_r = np.ma.masked_where(np.isnan(VR_CNSmax), SIGMA_CNS).T

#     condi = np.isnan(VR_CNSmax)
#     A = np.round(build_Amatrix(azi_mean, ele_cns), 6)
#     U, S, Vh = [], [], []
#     for c_nn in condi.T:
#         u, s, vh = np.linalg.svd(np.ma.masked_where(np.tile(c_nn, (3, 1)).T, A).filled(0), full_matrices=True)
#         U.append(u)
#         S.append(np.linalg.pinv(diagsvd(s, u.shape[0], vh.shape[0])))
#         Vh.append(vh)
#     U, S, Vh = np.array(U), np.array(S), np.array(Vh)

#     U_T = np.einsum('...ij->...ji', U)
#     Vh_T = np.einsum('...ij->...ji', Vh)
#     K1 = np.nansum((U_T * V_in.T[:, None, :]), axis=2)[..., None]
#     K2 = np.einsum('...ik,...kj->...ij', S, K1)
#     V_k = np.einsum('...ik,...kj->...ij', Vh_T, K2)
#     UVW[kk, ...] = np.squeeze(V_k)
#     UVW[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4, :] = np.squeeze(np.full((3, 1), np.nan))

#     UVWunc[kk, ...] = abs(np.einsum('...ii->...i', np.sqrt(
#         np.einsum('nik,nk,nkl->nil', A_r_MP, SIGMA_r**2, A_r_MP_T).astype(complex)).real))
#     UVWunc[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4, :] = np.squeeze(np.full((3, 1), np.nan))

#     V_r_est = A_r @ V_k
#     ss_e = ((V_r - V_r_est) ** 2).sum(axis=1)
#     ss_t = ((V_r - V_r.mean(axis=1)[:, None, :]) ** 2).sum(axis=1)
#     R2[kk, :] = np.squeeze(1 - ss_e / ss_t)
#     R2[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

#     mask_A = np.tile(mask_V_in.T[..., None], (1, 1, 3))
#     A_r_T = np.einsum('...ij->...ji', A_r)
#     Spp = np.apply_along_axis(np.diag, 1, 1 / np.sqrt(np.einsum('...ii->...i', A_r_T @ A_r)))
#     Z = np.ma.masked_where(mask_A, A_r @ Spp)
#     CN[kk, :] = np.squeeze(np.array([CN_est(X) for X in Z]))
#     CN[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

#     SPEED[kk, :], SPEEDunc[kk, :] = np.vstack(
#         [np.fromiter(uvw_2_spd(val, unc).values(), dtype=float) for val, unc in
#          zip(UVW[kk, ...], UVWunc[kk, ...])]).T
#     DIREC[kk, :], DIRECunc[kk, :] = np.vstack(
#         [np.fromiter(uvw_2_dir(val, unc).values(), dtype=float) for val, unc in
#          zip(UVW[kk, ...], UVWunc[kk, ...])]).T
#     SNR_tot[kk, :], BETA_tot[kk, :], SNR_max[kk, :], SNR_min[kk, :], SPEC_tot[kk, :] = (
#         np.nanmedian(SNR_CNS, axis=0),
#         np.nanmedian(BETA_CNS, axis=0),
#         np.nanmax(SNR_CNS_max, axis=0),
#         np.nanmin(SNR_CNS_min, axis=0),
#         np.nanmedian(SPEC_CNS, axis=0)
#     )


# def wind_output_to_xarray(
#     SPEED, SPEEDunc, DIREC, DIRECunc, R2, CN, n_good, UVW, UVWunc, binary_dir,
#     SNR_tot, SNR_max, SNR_min, SPEC_tot, BETA_tot,
#     confDict, NN, height, width, height_bnds, time_start, time_bnds
# ):
#     speed = np.copy(SPEED)
#     errspeed = np.copy(SPEEDunc)
#     wdir = np.copy(DIREC)
#     errwdir = np.copy(DIRECunc)
#     r2 = np.copy(R2)
#     cn = np.copy(CN)
#     nvrad = np.copy(n_good)
#     u = np.copy(UVW[:, :, 0])
#     v = np.copy(UVW[:, :, 1])
#     w = np.copy(UVW[:, :, 2])
#     erru = np.copy(UVWunc[:, :, 0])
#     errv = np.copy(UVWunc[:, :, 1])
#     errw = np.copy(UVWunc[:, :, 2])

#     qspeed = (~np.isnan(SPEED))
#     r2[np.isnan(R2)] = -999.
#     qr2 = r2 >= float(confDict['R2_THRESHOLD'])
#     cn[np.isnan(CN)] = +999.
#     qcn = (cn >= 0) & (cn <= float(confDict['CN_THRESHOLD']))
#     nvrad[np.isnan(n_good)] = -999
#     qnvrad = nvrad >= int(confDict['N_VRAD_THRESHOLD'])

#     qwind = qspeed & qnvrad & qcn & qr2
#     qu = np.copy(qspeed)
#     qv = np.copy(qspeed)
#     qw = np.copy(qspeed)
#     qspeed = qwind
#     speed[~qspeed] = -999.
#     errspeed[~qspeed] = -999.
#     wdir[~qspeed] = -999.
#     errwdir[~qspeed] = -999.
#     u[~qspeed] = -999.
#     v[~qspeed] = -999.
#     w[~qspeed] = -999.
#     erru[~qspeed] = -999.
#     errv[~qspeed] = -999.
#     errw[~qspeed] = -999.
#     r2[~qr2] = -999.
#     cn[~qcn] = -999.
#     nvrad[~qnvrad] = -999.
#     binary_dir[~qspeed] = -999

#     if np.all(np.isnan(speed)):
#         print('WARNING: bad retrieval quality')
#         print('all retrieved velocities are NaN -> check nvrad threshold!')

#     configuration = ""
#     for dd in confDict:
#         if dd not in ['PROC_PATH', 'NC_L1_PATH', 'NC_L2_PATH', 'NC_L2_QL_PATH']:
#             configuration += dd + '=' + confDict[dd] + '\n'
#     if 'BLINDEZONE_GATES' in confDict:
#         NN = int(confDict['BLINDEZONE_GATES'])
#     else:
#         NN = 0

#     return xr.Dataset({
#         'config': ([]
#                    , configuration
#                    , {'standard_name': 'configuration_file'}
#                    ),
#         'snr': (['time', 'height'], np.float32(SNR_tot[:, NN:]), {'units': 'fraction', 'comments': 'median signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1', 'standard_name': 'snr', 'long_name': 'Signal-to-Noise-Ratio', '_FillValue': -999.}),
#         'snr_max': (['time', 'height'], np.float32(SNR_max[:, NN:]), {'units': 'fraction', 'comments': 'maximal signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1', 'standard_name': 'snr_max', 'long_name': 'maximal_Signal-to-Noise-Ratio', '_FillValue': -999.}),
#         'snr_min': (['time', 'height'], np.float32(SNR_min[:, NN:]), {'units': 'fraction', 'comments': 'minimal signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1', 'standard_name': 'snr_min', 'long_name': 'minimal_Signal-to-Noise-Ratio', '_FillValue': -999.}),
#         'beta': (['time', 'height'], np.float32(BETA_tot[:, NN:]), {'units': 'm-1 sr-1', 'comments': 'median attenuated backscatter coefficient along VAD cycle.', 'standard_name': 'beta', 'long_name': 'attenuated backscatter coefficient', '_FillValue': -999.}),
#         'wspeed': (['time', 'height'], np.float32(speed[:, NN:]), {'units': 'm s-1', 'comments': 'Scalar wind speed (amount of vector)', 'standard_name': 'wind_speed', 'long_name': 'Wind Speed', '_FillValue': -999.}),
#         'spec_w': (['time', 'height'], np.float32(SPEC_tot[:, NN:]), {'units': 'm s-1', 'comments': 'spectral width of wind estimate', 'standard_name': 'spectral_width', 'long_name': 'spectral width', '_FillValue': -999.}),
#         'qwind': (['time', 'height'], qwind[:, NN:].astype(np.int8), {'comments': 'quality flag 0 or 1 for u, v, w, wspeed, wdir and corresponding errors', 'long_name': 'wind_quality_flag', '_FillValue': np.array(-128).astype(np.int8), 'flag_values': np.arange(0, 2).astype(np.int8), 'flag_meanings': 'quality_bad quality_good'}),
#         'qu': (['time', 'height'], qu[:, NN:].astype(np.int8), {'comments': 'quality flag 0 or 1 for u and corresponding error', 'long_name': 'quality_flag_u', '_FillValue': np.array(-128).astype(np.int8), 'flag_values': np.arange(0, 2).astype(np.int8), 'flag_meanings': 'quality_bad quality_good'}),
#         'qv': (['time', 'height'], qv[:, NN:].astype(np.int8), {'comments': 'quality flag 0 or 1 for v and corresponding error', 'long_name': 'quality_flag_v', '_FillValue': np.array(-128).astype(np.int8), 'flag_values': np.arange(0, 2).astype(np.int8), 'flag_meanings': 'quality_bad quality_good'}),
#         'qw': (['time', 'height'], qw[:, NN:].astype(np.int8), {'comments': 'quality flag 0 or 1 for w and corresponding error', 'long_name': 'quality_flag_w', '_FillValue': np.array(-128).astype(np.int8), 'flag_values': np.arange(0, 2).astype(np.int8), 'flag_meanings': 'quality_bad quality_good'}),
#         'errwspeed': (['time', 'height'], np.float32(errspeed[:, NN:]), {'units': 'm s-1', 'standard': 'wind_speed_uncertainty', 'long_name': 'Wind Speed Uncertainty', '_FillValue': -999.}),
#         'u': (['time', 'height'], np.float32(u[:, NN:]), {'units': 'm s-1', 'comments': '"Eastward indicates" a vector component which is positive when directed eastward (negative westward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component', 'standard_name': 'eastward_wind', 'long_name': 'Zonal Wind', '_FillValue': -999.}),
#         'erru': (['time', 'height'], np.float32(erru[:, NN:]), {'units': 'm s-1', 'standard_name': 'eastward_wind_uncertainty', 'long_name': 'Zonal Wind Uncertainty', '_FillValue': -999.}),
#         'v': (['time', 'height'], np.float32(v[:, NN:]), {'units': 'm s-1', 'comments': '"Northward indicates" a vector component which is positive when directed northward (negative southward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component', 'standard_name': 'northward_wind', 'long_name': 'Meridional Wind', '_FillValue': -999.}),
#         'errv': (['time', 'height'], np.float32(errv[:, NN:]), {'units': 'm s-1', 'standard_name': 'northward_wind_uncertainty', 'long_name': 'Meridional Wind Uncertainty', '_FillValue': -999.}),
#         'w': (['time', 'height'], np.float32(w[:, NN:]), {'units': 'm s-1', 'comments': 'Vertical wind component, positive when directed upward (negative downward)', 'standard_name': 'upward_air_velocity', 'long_name': 'Upward Air Velocity', '_FillValue': -999.}),
#         'errw': (['time', 'height'], np.float32(errw[:, NN:]), {'units': 'm s-1', 'standard_name': 'upward_air_velocity_uncertainty', 'long_name': 'Upward Air Velocity Uncertainty', '_FillValue': -999.}),
#         'wdir': (['time', 'height'], np.float32(wdir[:, NN:]), {'units': 'degree', 'comments': 'Wind direction', 'standard_name': 'wind_from_direction', 'long_name': 'Wind Direction', '_FillValue': -999.}),
#         'errwdir': (['time', 'height'], np.float32(errwdir[:, NN:]), {'units': 'degree', 'standard_name': 'wind_direction_uncertainty', 'long_name': 'Wind Direction Uncertainty', '_FillValue': -999.}),
#         'r2': (['time', 'height'], np.float32(r2[:, NN:]), {'comments': 'coefficient of determination', 'long_name': 'coefficient of determination', 'standard_name': 'coefficient_of_determination', 'units': '1', '_FillValue': -999.}),
#         'nvrad': (['time', 'height'], np.float32(nvrad[:, NN:]), {'comments': 'number of (averaged) radial velocities used for wind calculation', 'long_name': 'number of radial velocities', 'standard_name': 'no_radial_velocities', 'units': '1', '_FillValue': -999.}),
#         'cn': (['time', 'height'], np.float32(cn[:, NN:]), {'comments': 'condition number', 'standard_name': 'condition_number', 'long_name': 'Condition Number', 'units': '1', '_FillValue': -999.}),
#         'lat': ([]
#                 , np.float32(confDict['SYSTEM_LATITUDE']), {'units': 'degrees_north', 'long_name': 'latitude', 'standard_name': 'latitude', 'comments': 'latitude of sensor', '_FillValue': -999.}),
#         'lon': ([]
#                 , np.float32(confDict['SYSTEM_LONGITUDE']), {'units': 'degrees_east', 'long_name': 'longitude', 'standard_name': 'longitude', 'comments': 'longitude of sensor', '_FillValue': -999.}),
#         'zsl': ([]
#                 , np.float32(confDict['SYSTEM_ALTITUDE']), {'units': 'm', 'comments': 'Altitude of sensor above mean sea level', 'standard_name': 'altitude', '_FillValue': -999.}),
#         'time_bnds': (['time', 'nv'], time_bnds.T.astype(np.float64), {'units': 'seconds since 1970-01-01 00:00:00 UTC'}),
#         'height_bnds': (['height', 'nv'], np.float32(height_bnds[NN:, :]), {'units': 'm'}),
#         'frequency': ([]
#                       , np.float32(299792458 / float(confDict['SYSTEM_WAVELENGTH'])), {'units': 'Hz', 'comments': 'lidar operating frequency', 'long_name': 'instrument_frequency', '_FillValue': -999.}),
#         'vert_res': ([]
#                      , np.float32(np.diff(height).mean()), {'units': 'm', 'comments': 'Calculated from pulse wdth and beam elevation', 'long_name': 'Vertical_resolution_measurement', '_FillValue': -999.}),
#         'hor_width': (['height'], np.float32(width[NN:]), {'units': 'm', 'comments': 'Calculated from beam elevation and height', 'standard_name': 'horizontal_sample_width', '_FillValue': -999.}),
#         'binary_dir': (['time', 'height'], binary_dir[:, NN:], {'long_name': 'directional binary mask', 'comments': 'Bitmask encoding which azimuthal beams contributed to each wind retrieval. Each bit represents a direction.', 'units': '1', '_FillValue': -999}),
#     }, coords={
#         'height': (['height'], np.float32(height[NN:]), {'units': 'm', 'standard_name': 'height', 'comments': 'vertical distance from sensor to centre of range gate', 'bounds': 'height_bnds'}),
#         'time': (['time'], time_start.astype(np.float64), {'units': 'seconds since 1970-01-01 00:00:00', 'comments': 'Timestamp at the end of the averaging interval', 'standard_name': 'time', 'long_name': 'Time', 'calendar': 'gregorian', 'bounds': 'time_bnds', '_CoordinateAxisType': 'Time'}),
#         'nv': (['nv'], np.arange(0, 2).astype(np.int8))
#     })

# def lvl2vad_standard(ds, date_chosen, confDict):
#     # read lidar parameters
#     n_rays = int(confDict['NUMBER_OF_DIRECTIONS'])
#     # number of gates
#     n_gates = int(confDict['NUMBER_OF_GATES'])
#     # number of pulses used in the data point aquisition
#     n = ds.prf.data
#     # number of points per range gate
#     M = ds.nsmpl.data
#     # half of detector bandwidth in velocity space
#     B = ds.nqv.data

#     # filter Stares within scan
#     elevation = 90 - ds.zenith.data
#     azimuth = ds.azi.data[elevation < 89] % 360
#     time_ds = ds.time.data[elevation < 89]
#     dv = ds.dv.data[elevation < 89]
#     snr = ds.intensity.data[elevation < 89] - 1
#     beta = ds.beta.data[elevation < 89]
#     if 'delv' in list(ds.keys()):
#         delv = ds.delv.data[elevation < 89]
#     else:
#         delv = np.full(dv.shape, -999)
#     height = ds.range.data * np.sin(np.nanmedian(elevation[elevation < 89]) * np.pi / 180)
#     width = ds.range.data * 2 * np.cos(np.nanmedian(elevation[elevation < 89]) * np.pi / 180)
#     height_bnds = ds.range_bnds.data
#     height_bnds[:, 0] = np.sin(np.nanmedian(elevation[elevation < 89]) * np.pi / 180) * (height_bnds[:, 0])
#     height_bnds[:, 1] = np.sin(np.nanmedian(elevation[elevation < 89]) * np.pi / 180) * (height_bnds[:, 1])

#     # Time chunking
#     if 'UTC_OFFSET' in confDict:
#         time_delta = int(confDict['UTC_OFFSET'])
#     else:
#         time_delta = 0

#     time_vec = np.arange(date_chosen - datetime.timedelta(hours=time_delta),
#                          date_chosen + datetime.timedelta(days=1) - datetime.timedelta(hours=time_delta)
#                          + datetime.timedelta(minutes=int(confDict['AVG_MIN'])),
#                          datetime.timedelta(minutes=int(confDict['AVG_MIN'])))
#     calc_idx = [np.where((ii <= time_ds) * (time_ds < iip1))
#                 for ii, iip1 in zip(time_vec[0:-1], time_vec[1::])]
#     time_start = np.array([int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())
#                            if len(t[0]) != 0
#                            else int(pd.to_datetime(time_vec[ii + 1]).replace(tzinfo=datetime.timezone.utc).timestamp())
#                            for ii, t in enumerate(calc_idx)
#                            ])
#     time_bnds = np.array([[int(pd.to_datetime(time_ds[t[0][0]]).replace(tzinfo=datetime.timezone.utc).timestamp()),
#                            int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())]
#                           if len(t[0]) != 0
#                           else [int(pd.to_datetime(time_vec[ii]).replace(tzinfo=datetime.timezone.utc).timestamp()),
#                                 int(pd.to_datetime(time_vec[ii + 1]).replace(tzinfo=datetime.timezone.utc).timestamp())]
#                           for ii, t in enumerate(calc_idx)
#                           ]).T

#     if n_gates != dv.shape[1]:
#         print('Warning: number of gates in config does not match lvl1 data!')
#         n_gates = dv.shape[1]
#         print('number of gates changed to ' + str(n_gates))

#     time_valid = [ii for ii, x in enumerate(calc_idx) if len(x[0]) != 0]

#     # Allocate arrays
#     UVW = np.full((len(calc_idx), n_gates, 3), np.nan)
#     UVWunc = np.full((len(calc_idx), n_gates, 3), np.nan)
#     SPEED = np.full((len(calc_idx), n_gates), np.nan)
#     SPEEDunc = np.full((len(calc_idx), n_gates), np.nan)
#     DIREC = np.full((len(calc_idx), n_gates), np.nan)
#     DIRECunc = np.full((len(calc_idx), n_gates), np.nan)
#     R2 = np.full((len(calc_idx), n_gates), np.nan)
#     CN = np.full((len(calc_idx), n_gates), np.nan)
#     n_good = np.full((len(calc_idx), n_gates), np.nan)
#     binary_dir = np.zeros((len(calc_idx), n_gates), dtype=int)
#     SNR_tot = np.full((len(calc_idx), n_gates), np.nan)
#     SNR_max = np.full((len(calc_idx), n_gates), np.nan)
#     SNR_min = np.full((len(calc_idx), n_gates), np.nan)
#     SPEC_tot = np.full((len(calc_idx), n_gates), np.nan)
#     BETA_tot = np.full((len(calc_idx), n_gates), np.nan)

#     for kk in time_valid:
#         process_time_chunk(
#             kk, n_rays, confDict, calc_idx, azimuth, dv, snr, beta, delv, elevation, time_start,
#             set_azi_dir, consensus, calc_sigma_single, in_db, M, n, B, "halo",
#             UVW, UVWunc, SPEED, SPEEDunc, DIREC, DIRECunc, R2, CN, n_good, binary_dir,
#             SNR_tot, SNR_max, SNR_min, SPEC_tot, BETA_tot
#         )

#     # Quality control and output (shared)
#     speed = np.copy(SPEED)
#     errspeed = np.copy(SPEEDunc)
#     wdir = np.copy(DIREC)
#     errwdir = np.copy(DIRECunc)
#     r2 = np.copy(R2)
#     cn = np.copy(CN)
#     nvrad = np.copy(n_good)
#     u = np.copy(UVW[:, :, 0])
#     v = np.copy(UVW[:, :, 1])
#     w = np.copy(UVW[:, :, 2])
#     erru = np.copy(UVWunc[:, :, 0])
#     errv = np.copy(UVWunc[:, :, 1])
#     errw = np.copy(UVWunc[:, :, 2])

#     qspeed = (~np.isnan(SPEED))
#     r2[np.isnan(R2)] = -999.
#     qr2 = r2 >= float(confDict['R2_THRESHOLD'])
#     cn[np.isnan(CN)] = +999.
#     qcn = (cn >= 0) & (cn <= float(confDict['CN_THRESHOLD']))
#     nvrad[np.isnan(n_good)] = -999
#     qnvrad = nvrad >= int(confDict['N_VRAD_THRESHOLD'])

#     qwind = qspeed & qnvrad & qcn & qr2
#     qu = np.copy(qspeed)
#     qv = np.copy(qspeed)
#     qw = np.copy(qspeed)
#     qspeed = qwind
#     speed[~qspeed] = -999.
#     errspeed[~qspeed] = -999.
#     wdir[~qspeed] = -999.
#     errwdir[~qspeed] = -999.
#     u[~qspeed] = -999.
#     v[~qspeed] = -999.
#     w[~qspeed] = -999.
#     erru[~qspeed] = -999.
#     errv[~qspeed] = -999.
#     errw[~qspeed] = -999.
#     r2[~qr2] = -999.
#     cn[~qcn] = -999.
#     nvrad[~qnvrad] = -999.
#     binary_dir[~qspeed] = -999

#     if np.all(np.isnan(speed)):
#         print('WARNING: bad retrieval quality')
#         print('all retrieved velocities are NaN -> check nvrad threshold!')

#     configuration = ""
#     for dd in confDict:
#         if dd not in ['PROC_PATH', 'NC_L1_PATH', 'NC_L2_PATH', 'NC_L2_QL_PATH']:
#             configuration += dd + '=' + confDict[dd] + '\n'
#     if 'BLINDEZONE_GATES' in confDict:
#         NN = int(confDict['BLINDEZONE_GATES'])
#     else:
#         NN = 0

#     return wind_output_to_xarray(
#         SPEED, SPEEDunc, DIREC, DIRECunc, R2, CN, n_good, UVW, UVWunc, binary_dir,
#         SNR_tot, SNR_max, SNR_min, SPEC_tot, BETA_tot,
#         confDict, NN, height, width, height_bnds, time_start, time_bnds
#     )

#     # return xr.Dataset({'config': ([]
#     #                               , configuration
#     #                               , {'standard_name': 'configuration_file'}
#     #                               )
#     #                         , 'snr': (['time', 'height']
#     #                                     , np.float32(SNR_tot[:, NN:])
#     #                                     , {'units': 'fraction'
#     #                                     , 'comments': 'median signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1'
#     #                                     , 'standard_name' : 'snr'
#     #                                     , 'long_name' : 'Signal-to-Noise-Ratio' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                        , 'snr_max': (['time', 'height']
#     #                                     , np.float32(SNR_max[:, NN:])
#     #                                     , {'units': 'fraction'
#     #                                     , 'comments': 'maximal signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1'
#     #                                     , 'standard_name' : 'snr_max'
#     #                                     , 'long_name' : 'maximal_Signal-to-Noise-Ratio' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                        , 'snr_min': (['time', 'height']
#     #                                     , np.float32(SNR_min[:, NN:])
#     #                                     , {'units': 'fraction'
#     #                                     , 'comments': 'minimal signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1'
#     #                                     , 'standard_name' : 'snr_min'
#     #                                     , 'long_name' : 'minimal_Signal-to-Noise-Ratio' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                        , 'beta': (['time', 'height']
#     #                                     , np.float32(BETA_tot[:, NN:])
#     #                                     , {'units': 'm-1 sr-1'
#     #                                     , 'comments': 'median attenuated backscatter coefficient along VAD cycle.'
#     #                                     , 'standard_name' : 'beta'
#     #                                     , 'long_name' : 'attenuated backscatter coefficient' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                       , 'wspeed': (['time', 'height']
#     #                                    , np.float32(speed[:, NN:])
#     #                                    , {'units': 'm s-1'
#     #                                        , 'comments': 'Scalar wind speed (amount of vector)'
#     #                                        , 'standard_name': 'wind_speed'
#     #                                        , 'long_name': 'Wind Speed'
#     #                                        , '_FillValue': -999.
#     #                                       }
#     #                                    )
#     #                       , 'spec_w': (['time', 'height']
#     #                                     , np.float32(SPEC_tot[:, NN:])
#     #                                     , {'units': 'm s-1'
#     #                                     , 'comments': 'spectral width of wind estimate'
#     #                                     , 'standard_name' : 'spectral_width'
#     #                                     , 'long_name' : 'spectral width' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                       , 'qwind': (['time', 'height']
#     #                                   , qwind[:, NN:].astype(np.int8)
#     #                                   , {'comments': str(
#     #         'quality flag 0 or 1 for u, v, w, wspeed, wdir and corresponding errors,'
#     #         + '(good quality data = WHERE( R2 > ' + confDict['R2_THRESHOLD']
#     #         + 'AND CN < ' + confDict['CN_THRESHOLD']
#     #         + 'AND NVRAD > ' + confDict['N_VRAD_THRESHOLD'] + ')')
#     #                                       , 'long_name': 'wind_quality_flag'
#     #                                       , '_FillValue': np.array(-128).astype(np.int8)
#     #                                       , 'flag_values': np.arange(0, 2).astype(np.int8)
#     #                                       , 'flag_meanings': 'quality_bad quality_good'
#     #                                   }
#     #                                   )
#     #                       , 'qu': (['time', 'height']
#     #                                , qu[:, NN:].astype(np.int8)
#     #                                , {'comments': 'quality flag 0 or 1 for u and corresponding error'
#     #                                    , 'long_name': 'quality_flag_u'
#     #                                    , '_FillValue': np.array(-128).astype(np.int8)
#     #                                    , 'flag_values': np.arange(0, 2).astype(np.int8)
#     #                                    , 'flag_meanings': 'quality_bad quality_good'
#     #                                   }
#     #                                )
#     #                       , 'qv': (['time', 'height']
#     #                                , qv[:, NN:].astype(np.int8)
#     #                                , {'comments': 'quality flag 0 or 1 for v and corresponding error'
#     #                                    , 'long_name': 'quality_flag_v'
#     #                                    , '_FillValue': np.array(-128).astype(np.int8)
#     #                                    , 'flag_values': np.arange(0, 2).astype(np.int8)
#     #                                    , 'flag_meanings': 'quality_bad quality_good'
#     #                                   }
#     #                                )
#     #                       , 'qw': (['time', 'height']
#     #                                , qw[:, NN:].astype(np.int8)
#     #                                , {'comments': 'quality flag 0 or 1 for w and corresponding error'
#     #                                    , 'long_name': 'quality_flag_w'
#     #                                    , '_FillValue': np.array(-128).astype(np.int8)
#     #                                    , 'flag_values': np.arange(0, 2).astype(np.int8)
#     #                                    , 'flag_meanings': 'quality_bad quality_good'
#     #                                   }
#     #                                )
#     #                       , 'errwspeed': (['time', 'height']
#     #                                       , np.float32(errspeed[:, NN:])
#     #                                       , {'units': 'm s-1'
#     #                                           , 'standard': 'wind_speed_uncertainty'
#     #                                           , 'long_name': 'Wind Speed Uncertainty'
#     #                                           , '_FillValue': -999.
#     #                                          }
#     #                                       )
#     #                       , 'u': (['time', 'height']
#     #                               , np.float32(u[:, NN:])
#     #                               , {'units': 'm s-1'
#     #                                   ,
#     #                                  'comments': '"Eastward indicates" a vector component which is positive when directed eastward (negative westward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
#     #                                   , 'standard_name': 'eastward_wind'
#     #                                   , 'long_name': 'Zonal Wind'
#     #                                   , '_FillValue': -999.
#     #                                  }
#     #                               )
#     #                       , 'erru': (['time', 'height']
#     #                                  , np.float32(erru[:, NN:])
#     #                                  , {'units': 'm s-1'
#     #                                      , 'standard_name': 'eastward_wind_uncertainty'
#     #                                      , 'long_name': 'Zonal Wind Uncertainty'
#     #                                      , '_FillValue': -999.
#     #                                     }
#     #                                  )
#     #                       , 'v': (['time', 'height']
#     #                               , np.float32(v[:, NN:])
#     #                               , {'units': 'm s-1'
#     #                                   ,
#     #                                  'comments': '"Northward indicates" a vector component which is positive when directed northward (negative southward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
#     #                                   , 'standard_name': 'northward_wind'
#     #                                   , 'long_name': 'Meridional Wind'
#     #                                   , '_FillValue': -999.
#     #                                  }
#     #                               )
#     #                       , 'errv': (['time', 'height']
#     #                                  , np.float32(errv[:, NN:])
#     #                                  , {'units': 'm s-1'
#     #                                      , 'standard_name': 'northward_wind_uncertainty'
#     #                                      , 'long_name': 'Meridional Wind Uncertainty'
#     #                                      , '_FillValue': -999.
#     #                                     }
#     #                                  )
#     #                       , 'w': (['time', 'height']
#     #                               , np.float32(w[:, NN:])
#     #                               , {'units': 'm s-1'
#     #                                   ,
#     #                                  'comments': 'Vertical wind component, positive when directed upward (negative downward)'
#     #                                   , 'standard_name': 'upward_air_velocity'
#     #                                   , 'long_name': 'Upward Air Velocity'
#     #                                   , '_FillValue': -999.
#     #                                  }
#     #                               )
#     #                       , 'errw': (['time', 'height']
#     #                                  , np.float32(errw[:, NN:])
#     #                                  , {'units': 'm s-1'
#     #                                      , 'standard_name': 'upward_air_velocity_uncertainty'
#     #                                      , 'long_name': 'Upward Air Velocity Uncertainty'
#     #                                      , '_FillValue': -999.
#     #                                     }
#     #                                  )
#     #                       , 'wdir': (['time', 'height']
#     #                                  , np.float32(wdir[:, NN:])
#     #                                  , {'units': 'degree'
#     #                                      , 'comments': 'Wind direction'
#     #                                      , 'standard_name': 'wind_from_direction'
#     #                                      , 'long_name': 'Wind Direction'
#     #                                      , '_FillValue': -999.
#     #                                     }
#     #                                  )
#     #                       , 'errwdir': (['time', 'height']
#     #                                     , np.float32(errwdir[:, NN:])
#     #                                     , {'units': 'degree'
#     #                                         , 'standard_name': 'wind_direction_uncertainty'
#     #                                         , 'long_name': 'Wind Direction Uncertainty'
#     #                                         , '_FillValue': -999.
#     #                                        }
#     #                                     )
#     #                       , 'r2': (['time', 'height']
#     #                                , np.float32(r2[:, NN:])
#     #                                , {
#     #                                    'comments': 'coefficient of determination - provides a measure of how well observed radial velocities are replicated by the model used to determine u,v,w wind components from the measured line of sight radial velocities'
#     #                                    , 'long_name': 'coefficient of determination'
#     #                                    , 'standard_name': 'coefficient_of_determination'
#     #                                    , 'units': '1'
#     #                                    , '_FillValue': -999.
#     #                                }
#     #                                )
#     #                       , 'nvrad': (['time', 'height']
#     #                                   , np.float32(nvrad[:, NN:])
#     #                                   , {'comments': 'number of (averaged) radial velocities used for wind calculation'
#     #                                       , 'long_name': 'number of radial velocities'
#     #                                       , 'standard_name': 'no_radial_velocities'
#     #                                       , 'units': '1'
#     #                                       , '_FillValue': -999.
#     #                                      }
#     #                                   )
#     #                       , 'cn': (['time', 'height']
#     #                                , np.float32(cn[:, NN:])
#     #                                , {
#     #                                    'comments': 'condition number - provides a measure for the degree of collinearity among the Doppler velocity measurements used for the retrieval of the wind variables (u,v,w,speed,direction).'

#     #                                    , 'standard_name': 'condition_number'
#     #                                    , 'long_name': 'Condition Number'
#     #                                    , 'units': '1'
#     #                                    , '_FillValue': -999.
#     #                                }
#     #                                )
#     #                       , 'lat': ([]
#     #                                 , np.float32(confDict['SYSTEM_LATITUDE'])
#     #                                 , {'units': 'degrees_north'
#     #                                     , 'long_name': 'latitude'
#     #                                     , 'standard_name': 'latitude'
#     #                                     , 'comments': 'latitude of sensor'
#     #                                     , '_FillValue': -999.
#     #                                    }
#     #                                 )
#     #                       , 'lon': ([]
#     #                                 , np.float32(confDict['SYSTEM_LONGITUDE'])
#     #                                 , {'units': 'degrees_east'
#     #                                     , 'long_name': 'longitude'
#     #                                     , 'standard_name': 'longitude'
#     #                                     , 'comments': 'longitude of sensor'
#     #                                     , '_FillValue': -999.
#     #                                    }
#     #                                 )
#     #                       , 'zsl': ([]
#     #                                 , np.float32(confDict['SYSTEM_ALTITUDE'])
#     #                                 , {'units': 'm'
#     #                                     , 'comments': 'Altitude of sensor above mean sea level'
#     #                                     , 'standard_name': 'altitude'
#     #                                     , '_FillValue': -999.
#     #                                    }
#     #                                 )
#     #                       , 'time_bnds': (['time', 'nv']
#     #                                       , time_bnds.T.astype(np.float64)
#     #                                       , {'units': 'seconds since 1970-01-01 00:00:00 UTC'
#     #                                          }
#     #                                       )
#     #                       , 'height_bnds': (['height', 'nv']
#     #                                         # ,np.array([(np.arange(0,n_gates)
#     #                                         #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                         #             ,((np.arange(0,n_gates) + 1.)
#     #                                         #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                         #             ]).T
#     #                                         , np.float32(height_bnds[NN:, :])
#     #                                         , {'units': 'm'
#     #                                            }
#     #                                         )
#     #                       , 'frequency': ([]
#     #                                       , np.float32(299792458 / float(confDict['SYSTEM_WAVELENGTH']))
#     #                                       , {'units': 'Hz'
#     #                                           , 'comments': 'lidar operating frequency'
#     #                                           , 'long_name': 'instrument_frequency'
#     #                                           , '_FillValue': -999.
#     #                                          }
#     #                                       )
#     #                       , 'vert_res': ([]
#     #                                      , np.float32(np.diff(height).mean())
#     #                                      , {'units': 'm'
#     #                                          , 'comments': 'Calculated from pulse wdth and beam elevation'
#     #                                          , 'long_name': 'Vertical_resolution_measurement'
#     #                                          , '_FillValue': -999.
#     #                                         }
#     #                                      )
#     #                       , 'hor_width': (['height']
#     #                                       # ,np.array([(np.arange(0,n_gates)
#     #                                       #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                       #             ,((np.arange(0,n_gates) + 1.)
#     #                                       #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                       #             ]).T
#     #                                       , np.float32(width[NN:])
#     #                                       , {'units': 'm'
#     #                                           , 'comments': 'Calculated from beam elevation and height'
#     #                                           , 'standard_name': 'horizontal_sample_width'
#     #                                           , '_FillValue': -999.
#     #                                          }
#     #                                       )
#     #                     , 'binary_dir': (['time', 'height']
#     #                                      , binary_dir[:, NN:]
#     #                                      , { 'long_name': 'directional binary mask'
#     #                                      , 'comments': 'Bitmask encoding which azimuthal beams contributed to each wind retrieval. Each bit represents a direction.'
#     #                                      , 'units': '1'
#     #                                      , '_FillValue': -999.
#     #                                      }
#     #                                      )
#     #                    }
#     #                   , coords={'height': (['height']
#     #                                        # ,((np.arange(0,n_gates)+.5)*int(confDict['RANGE_GATE_LENGTH'])
#     #                                        # *np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                        , np.float32(height[NN:])
#     #                                        , {'units': 'm'
#     #                                            , 'standard_name': 'height'
#     #                                            , 'comments': 'vertical distance from sensor to centre of range gate'
#     #                                            , 'bounds': 'height_bnds'
#     #                                           }
#     #                                        )
#     #         , 'time': (['time']
#     #                    , time_start.astype(np.float64)
#     #                    , {'units': 'seconds since 1970-01-01 00:00:00'
#     #                        , 'comments': 'Timestamp at the end of the averaging interval'
#     #                        , 'standard_name': 'time'
#     #                        , 'long_name': 'Time'
#     #                        , 'calendar': 'gregorian'
#     #                        , 'bounds': 'time_bnds'
#     #                        , '_CoordinateAxisType': 'Time'
#     #                       })
#     #         , 'nv': (['nv'], np.arange(0, 2).astype(np.int8))
#     #                             }
#     #                   )


# def lvl2wcdbs(ds, date_chosen, confDict):
#     n_rays = int(confDict['NUMBER_OF_DIRECTIONS'])
#     # number of gates
#     n_gates = int(confDict['NUMBER_OF_GATES'])
#     # number of pulses used in the data point aquisition
#     B = (ds.radial_wind_speed.max() - ds.radial_wind_speed.min()).data / 2
#     lrg = ds.range_gate_length.data

#     time_ds = pd.to_datetime(ds.time.data, unit='s')
#     ds['zenith'] = 90 - ds.elevation
#     elevation = ds.elevation.data
#     azimuth = ds.azimuth.data
#     snr = 10 ** (ds.cnr.data / 10)
#     dv = ds.radial_wind_speed.data
#     beta = ds.relative_beta.data if 'relative_beta' in list(ds.keys()) else np.ones(dv.shape)
#     if 'measurement_height' in list(ds.keys()):
#         height = ds.measurement_height.data[0]
#     else:
#         height = (np.sin(ds.elevation * np.pi / 180) * ds.range).data[0]
#     width = (np.cos(ds.elevation * np.pi / 180) * ds.range * 2).data
#     width = width[elevation < 89][0]
#     height_bnds = np.vstack([height - lrg / 2, height - lrg / 2]).T
#     delv = ds.doppler_spectrum_width.data

#     # Time chunking
#     if 'UTC_OFFSET' in confDict:
#         time_delta = int(confDict['UTC_OFFSET'])
#     else:
#         time_delta = 0

#     time_vec = np.arange(date_chosen - datetime.timedelta(hours=time_delta),
#                          date_chosen + datetime.timedelta(days=1) - datetime.timedelta(hours=time_delta)
#                          + datetime.timedelta(minutes=int(confDict['AVG_MIN'])),
#                          datetime.timedelta(minutes=int(confDict['AVG_MIN'])))
#     calc_idx = [np.where((ii <= time_ds) * (time_ds < iip1))
#                 for ii, iip1 in zip(time_vec[0:-1], time_vec[1::])]
#     time_start = np.array([int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())
#                            if len(t[0]) != 0
#                            else int(pd.to_datetime(time_vec[ii + 1]).replace(tzinfo=datetime.timezone.utc).timestamp())
#                            for ii, t in enumerate(calc_idx)
#                            ])
#     time_bnds = np.array([[int(pd.to_datetime(time_ds[t[0][0]]).replace(tzinfo=datetime.timezone.utc).timestamp()),
#                            int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())]
#                           if len(t[0]) != 0
#                           else [int(pd.to_datetime(time_vec[ii]).replace(tzinfo=datetime.timezone.utc).timestamp()),
#                                 int(pd.to_datetime(time_vec[ii + 1]).replace(tzinfo=datetime.timezone.utc).timestamp())]
#                           for ii, t in enumerate(calc_idx)
#                           ]).T

#     if n_gates != dv.shape[1]:
#         print('Warning: number of gates in config does not match lvl1 data!')
#         n_gates = dv.shape[1]
#         print('number of gates changed to ' + str(n_gates))

#     time_valid = [ii for ii, x in enumerate(calc_idx) if len(x[0]) != 0]

#     # Allocate arrays
#     UVW = np.full((len(calc_idx), n_gates, 3), np.nan)
#     UVWunc = np.full((len(calc_idx), n_gates, 3), np.nan)
#     SPEED = np.full((len(calc_idx), n_gates), np.nan)
#     SPEEDunc = np.full((len(calc_idx), n_gates), np.nan)
#     DIREC = np.full((len(calc_idx), n_gates), np.nan)
#     DIRECunc = np.full((len(calc_idx), n_gates), np.nan)
#     R2 = np.full((len(calc_idx), n_gates), np.nan)
#     CN = np.full((len(calc_idx), n_gates), np.nan)
#     n_good = np.full((len(calc_idx), n_gates), np.nan)
#     binary_dir = np.zeros((len(calc_idx), n_gates), dtype=int)
#     SNR_tot = np.full((len(calc_idx), n_gates), np.nan)
#     SNR_max = np.full((len(calc_idx), n_gates), np.nan)
#     SNR_min = np.full((len(calc_idx), n_gates), np.nan)
#     SPEC_tot = np.full((len(calc_idx), n_gates), np.nan)
#     BETA_tot = np.full((len(calc_idx), n_gates), np.nan)

#     for kk in time_valid:
#         process_time_chunk(
#             kk, n_rays, confDict, calc_idx, azimuth, dv, snr, beta, delv, elevation, time_start,
#             set_azi_dir, consensus, calc_sigma_single, in_db, None, None, B, "windcube",
#             UVW, UVWunc, SPEED, SPEEDunc, DIREC, DIRECunc, R2, CN, n_good, binary_dir,
#             SNR_tot, SNR_max, SNR_min, SPEC_tot, BETA_tot
#         )

#     # Quality control and output (shared)
#     speed = np.copy(SPEED)
#     errspeed = np.copy(SPEEDunc)
#     wdir = np.copy(DIREC)
#     errwdir = np.copy(DIRECunc)
#     r2 = np.copy(R2)
#     cn = np.copy(CN)
#     nvrad = np.copy(n_good)
#     u = np.copy(UVW[:, :, 0])
#     v = np.copy(UVW[:, :, 1])
#     w = np.copy(UVW[:, :, 2])
#     erru = np.copy(UVWunc[:, :, 0])
#     errv = np.copy(UVWunc[:, :, 1])
#     errw = np.copy(UVWunc[:, :, 2])

#     qspeed = (~np.isnan(SPEED))
#     r2[np.isnan(R2)] = -999.
#     qr2 = r2 >= float(confDict['R2_THRESHOLD'])
#     cn[np.isnan(CN)] = +999.
#     qcn = (cn >= 0) & (cn <= float(confDict['CN_THRESHOLD']))
#     nvrad[np.isnan(n_good)] = -999
#     qnvrad = nvrad >= int(confDict['N_VRAD_THRESHOLD'])

#     qwind = qspeed & qnvrad & qcn & qr2
#     qu = np.copy(qspeed)
#     qv = np.copy(qspeed)
#     qw = np.copy(qspeed)
#     qspeed = qwind
#     speed[~qspeed] = -999.
#     errspeed[~qspeed] = -999.
#     wdir[~qspeed] = -999.
#     errwdir[~qspeed] = -999.
#     u[~qspeed] = -999.
#     v[~qspeed] = -999.
#     w[~qspeed] = -999.
#     erru[~qspeed] = -999.
#     errv[~qspeed] = -999.
#     errw[~qspeed] = -999.
#     r2[~qr2] = -999.
#     cn[~qcn] = -999.
#     nvrad[~qnvrad] = -999.
#     binary_dir[~qspeed] = -999

#     if np.all(np.isnan(speed)):
#         print('WARNING: bad retrieval quality')
#         print('all retrieved velocities are NaN -> check nvrad threshold!')

#     configuration = ""
#     for dd in confDict:
#         if dd not in ['PROC_PATH', 'NC_L1_PATH', 'NC_L2_PATH', 'NC_L2_QL_PATH']:
#             configuration += dd + '=' + confDict[dd] + '\n'
#     if 'BLINDEZONE_GATES' in confDict:
#         NN = int(confDict['BLINDEZONE_GATES'])
#     else:
#         NN = 0

#     return wind_output_to_xarray(
#         SPEED, SPEEDunc, DIREC, DIRECunc, R2, CN, n_good, UVW, UVWunc, binary_dir,
#         SNR_tot, SNR_max, SNR_min, SPEC_tot, BETA_tot,
#         confDict, NN, height, width, height_bnds, time_start, time_bnds
#     )

#     # return xr.Dataset({'config': ([]
#     #                               , configuration
#     #                               , {'standard_name': 'configuration_file'}
#     #                               )
#     #                         , 'snr': (['time', 'height']
#     #                                     , np.float32(SNR_tot[:, NN:])
#     #                                     , {'units': 'fraction'
#     #                                     , 'comments': 'median signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1'
#     #                                     , 'standard_name' : 'snr'
#     #                                     , 'long_name' : 'Signal-to-Noise-Ratio' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                        , 'snr_max': (['time', 'height']
#     #                                     , np.float32(SNR_max[:, NN:])
#     #                                     , {'units': 'fraction'
#     #                                     , 'comments': 'maximal signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1'
#     #                                     , 'standard_name' : 'snr_max'
#     #                                     , 'long_name' : 'maximal_Signal-to-Noise-Ratio' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                        , 'snr_min': (['time', 'height']
#     #                                     , np.float32(SNR_min[:, NN:])
#     #                                     , {'units': 'fraction'
#     #                                     , 'comments': 'minimal signal-to-noise-ration in fraction along VAD cycle, i.e. signal intensity - 1'
#     #                                     , 'standard_name' : 'snr_min'
#     #                                     , 'long_name' : 'minimal_Signal-to-Noise-Ratio' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                        , 'beta': (['time', 'height']
#     #                                     , np.float32(BETA_tot[:, NN:])
#     #                                     , {'units': 'm-1 sr-1'
#     #                                     , 'comments': 'median attenuated backscatter coefficient along VAD cycle.'
#     #                                     , 'standard_name' : 'beta'
#     #                                     , 'long_name' : 'attenuated backscatter coefficient' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                       , 'wspeed': (['time', 'height']
#     #                                    , np.float32(speed[:, NN:])
#     #                                    , {'units': 'm s-1'
#     #                                        , 'comments': 'Scalar wind speed (amount of vector)'
#     #                                        , 'standard_name': 'wind_speed'
#     #                                        , 'long_name': 'Wind Speed'
#     #                                        , '_FillValue': -999.
#     #                                       }
#     #                                    )
#     #                       , 'spec_w': (['time', 'height']
#     #                                     , np.float32(SPEC_tot[:, NN:])
#     #                                     , {'units': 'm s-1'
#     #                                     , 'comments': 'spectral width of wind estimate'
#     #                                     , 'standard_name' : 'spectral_width'
#     #                                     , 'long_name' : 'spectral width' 
#     #                                     ,'_FillValue' : -999.
#     #                                     }
#     #                                     )
#     #                       , 'qwind': (['time', 'height']
#     #                                   , qwind[:, NN:].astype(np.int8)
#     #                                   , {'comments': str(
#     #         'quality flag 0 or 1 for u, v, w, wspeed, wdir and corresponding errors,'
#     #         + '(good quality data = WHERE( R2 > ' + confDict['R2_THRESHOLD']
#     #         + 'AND CN < ' + confDict['CN_THRESHOLD']
#     #         + 'AND NVRAD > ' + confDict['N_VRAD_THRESHOLD'] + ')')
#     #                                       , 'long_name': 'wind_quality_flag'
#     #                                       , '_FillValue': np.array(-128).astype(np.int8)
#     #                                       , 'flag_values': np.arange(0, 2).astype(np.int8)
#     #                                       , 'flag_meanings': 'quality_bad quality_good'
#     #                                   }
#     #                                   )
#     #                       , 'qu': (['time', 'height']
#     #                                , qu[:, NN:].astype(np.int8)
#     #                                , {'comments': 'quality flag 0 or 1 for u and corresponding error'
#     #                                    , 'long_name': 'quality_flag_u'
#     #                                    , '_FillValue': np.array(-128).astype(np.int8)
#     #                                    , 'flag_values': np.arange(0, 2).astype(np.int8)
#     #                                    , 'flag_meanings': 'quality_bad quality_good'
#     #                                   }
#     #                                )
#     #                       , 'qv': (['time', 'height']
#     #                                , qv[:, NN:].astype(np.int8)
#     #                                , {'comments': 'quality flag 0 or 1 for v and corresponding error'
#     #                                    , 'long_name': 'quality_flag_v'
#     #                                    , '_FillValue': np.array(-128).astype(np.int8)
#     #                                    , 'flag_values': np.arange(0, 2).astype(np.int8)
#     #                                    , 'flag_meanings': 'quality_bad quality_good'
#     #                                   }
#     #                                )
#     #                       , 'qw': (['time', 'height']
#     #                                , qw[:, NN:].astype(np.int8)
#     #                                , {'comments': 'quality flag 0 or 1 for w and corresponding error'
#     #                                    , 'long_name': 'quality_flag_w'
#     #                                    , '_FillValue': np.array(-128).astype(np.int8)
#     #                                    , 'flag_values': np.arange(0, 2).astype(np.int8)
#     #                                    , 'flag_meanings': 'quality_bad quality_good'
#     #                                   }
#     #                                )
#     #                       , 'errwspeed': (['time', 'height']
#     #                                       , np.float32(errspeed[:, NN:])
#     #                                       , {'units': 'm s-1'
#     #                                           , 'standard': 'wind_speed_uncertainty'
#     #                                           , 'long_name': 'Wind Speed Uncertainty'
#     #                                           , '_FillValue': -999.
#     #                                          }
#     #                                       )
#     #                       , 'u': (['time', 'height']
#     #                               , np.float32(u[:, NN:])
#     #                               , {'units': 'm s-1'
#     #                                   ,
#     #                                  'comments': '"Eastward indicates" a vector component which is positive when directed eastward (negative westward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
#     #                                   , 'standard_name': 'eastward_wind'
#     #                                   , 'long_name': 'Zonal Wind'
#     #                                   , '_FillValue': -999.
#     #                                  }
#     #                               )
#     #                       , 'erru': (['time', 'height']
#     #                                  , np.float32(erru[:, NN:])
#     #                                  , {'units': 'm s-1'
#     #                                      , 'standard_name': 'eastward_wind_uncertainty'
#     #                                      , 'long_name': 'Zonal Wind Uncertainty'
#     #                                      , '_FillValue': -999.
#     #                                     }
#     #                                  )
#     #                       , 'v': (['time', 'height']
#     #                               , np.float32(v[:, NN:])
#     #                               , {'units': 'm s-1'
#     #                                   ,
#     #                                  'comments': '"Northward indicates" a vector component which is positive when directed northward (negative southward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
#     #                                   , 'standard_name': 'northward_wind'
#     #                                   , 'long_name': 'Meridional Wind'
#     #                                   , '_FillValue': -999.
#     #                                  }
#     #                               )
#     #                       , 'errv': (['time', 'height']
#     #                                  , np.float32(errv[:, NN:])
#     #                                  , {'units': 'm s-1'
#     #                                      , 'standard_name': 'northward_wind_uncertainty'
#     #                                      , 'long_name': 'Meridional Wind Uncertainty'
#     #                                      , '_FillValue': -999.
#     #                                     }
#     #                                  )
#     #                       , 'w': (['time', 'height']
#     #                               , np.float32(w[:, NN:])
#     #                               , {'units': 'm s-1'
#     #                                   ,
#     #                                  'comments': 'Vertical wind component, positive when directed upward (negative downward)'
#     #                                   , 'standard_name': 'upward_air_velocity'
#     #                                   , 'long_name': 'Upward Air Velocity'
#     #                                   , '_FillValue': -999.
#     #                                  }
#     #                               )
#     #                       , 'errw': (['time', 'height']
#     #                                  , np.float32(errw[:, NN:])
#     #                                  , {'units': 'm s-1'
#     #                                      , 'standard_name': 'upward_air_velocity_uncertainty'
#     #                                      , 'long_name': 'Upward Air Velocity Uncertainty'
#     #                                      , '_FillValue': -999.
#     #                                     }
#     #                                  )
#     #                       , 'wdir': (['time', 'height']
#     #                                  , np.float32(wdir[:, NN:])
#     #                                  , {'units': 'degree'
#     #                                      , 'comments': 'Wind direction'
#     #                                      , 'standard_name': 'wind_from_direction'
#     #                                      , 'long_name': 'Wind Direction'
#     #                                      , '_FillValue': -999.
#     #                                     }
#     #                                  )
#     #                       , 'errwdir': (['time', 'height']
#     #                                     , np.float32(errwdir[:, NN:])
#     #                                     , {'units': 'degree'
#     #                                         , 'standard_name': 'wind_direction_uncertainty'
#     #                                         , 'long_name': 'Wind Direction Uncertainty'
#     #                                         , '_FillValue': -999.
#     #                                        }
#     #                                     )
#     #                       , 'r2': (['time', 'height']
#     #                                , np.float32(r2[:, NN:])
#     #                                , {
#     #                                    'comments': 'coefficient of determination - provides a measure of how well observed radial velocities are replicated by the model used to determine u,v,w wind components from the measured line of sight radial velocities'
#     #                                    , 'long_name': 'coefficient of determination'
#     #                                    , 'standard_name': 'coefficient_of_determination'
#     #                                    , 'units': '1'
#     #                                    , '_FillValue': -999.
#     #                                }
#     #                                )
#     #                       , 'nvrad': (['time', 'height']
#     #                                   , np.float32(nvrad[:, NN:])
#     #                                   , {'comments': 'number of (averaged) radial velocities used for wind calculation'
#     #                                       , 'long_name': 'number of radial velocities'
#     #                                       , 'standard_name': 'no_radial_velocities'
#     #                                       , 'units': '1'
#     #                                       , '_FillValue': -999.
#     #                                      }
#     #                                   )
#     #                       , 'cn': (['time', 'height']
#     #                                , np.float32(cn[:, NN:])
#     #                                , {
#     #                                    'comments': 'condition number - provides a measure for the degree of collinearity among the Doppler velocity measurements used for the retrieval of the wind variables (u,v,w,speed,direction).'

#     #                                    , 'standard_name': 'condition_number'
#     #                                    , 'long_name': 'Condition Number'
#     #                                    , 'units': '1'
#     #                                    , '_FillValue': -999.
#     #                                }
#     #                                )
#     #                       , 'lat': ([]
#     #                                 , np.float32(confDict['SYSTEM_LATITUDE'])
#     #                                 , {'units': 'degrees_north'
#     #                                     , 'long_name': 'latitude'
#     #                                     , 'standard_name': 'latitude'
#     #                                     , 'comments': 'latitude of sensor'
#     #                                     , '_FillValue': -999.
#     #                                    }
#     #                                 )
#     #                       , 'lon': ([]
#     #                                 , np.float32(confDict['SYSTEM_LONGITUDE'])
#     #                                 , {'units': 'degrees_east'
#     #                                     , 'long_name': 'longitude'
#     #                                     , 'standard_name': 'longitude'
#     #                                     , 'comments': 'longitude of sensor'
#     #                                     , '_FillValue': -999.
#     #                                    }
#     #                                 )
#     #                       , 'zsl': ([]
#     #                                 , np.float32(confDict['SYSTEM_ALTITUDE'])
#     #                                 , {'units': 'm'
#     #                                     , 'comments': 'Altitude of sensor above mean sea level'
#     #                                     , 'standard_name': 'altitude'
#     #                                     , '_FillValue': -999.
#     #                                    }
#     #                                 )
#     #                       , 'time_bnds': (['time', 'nv']
#     #                                       , time_bnds.T.astype(np.float64)
#     #                                       , {'units': 'seconds since 1970-01-01 00:00:00 UTC'
#     #                                          }
#     #                                       )
#     #                       , 'height_bnds': (['height', 'nv']
#     #                                         # ,np.array([(np.arange(0,n_gates)
#     #                                         #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                         #             ,((np.arange(0,n_gates) + 1.)
#     #                                         #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                         #             ]).T
#     #                                         , np.float32(height_bnds[NN:, :])
#     #                                         , {'units': 'm'
#     #                                            }
#     #                                         )
#     #                       , 'frequency': ([]
#     #                                       , np.float32(299792458 / float(confDict['SYSTEM_WAVELENGTH']))
#     #                                       , {'units': 'Hz'
#     #                                           , 'comments': 'lidar operating frequency'
#     #                                           , 'long_name': 'instrument_frequency'
#     #                                           , '_FillValue': -999.
#     #                                          }
#     #                                       )
#     #                       , 'vert_res': ([]
#     #                                      , np.float32(np.diff(height).mean())
#     #                                      , {'units': 'm'
#     #                                          , 'comments': 'Calculated from pulse wdth and beam elevation'
#     #                                          , 'long_name': 'Vertical_resolution_measurement'
#     #                                          , '_FillValue': -999.
#     #                                         }
#     #                                      )
#     #                       , 'hor_width': (['height']
#     #                                       # ,np.array([(np.arange(0,n_gates)
#     #                                       #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                       #             ,((np.arange(0,n_gates) + 1.)
#     #                                       #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                       #             ]).T
#     #                                       , np.float32(width[NN:])
#     #                                       , {'units': 'm'
#     #                                           , 'comments': 'Calculated from beam elevation and height'
#     #                                           , 'standard_name': 'horizontal_sample_width'
#     #                                           , '_FillValue': -999.
#     #                                          }
#     #                                       )
#     #                     , 'binary_dir': (['time', 'height']
#     #                                      , binary_dir[:, NN:]
#     #                                      , { 'long_name': 'directional binary mask'
#     #                                      , 'comments': 'Bitmask encoding which azimuthal beams contributed to each wind retrieval. Each bit represents a direction.'
#     #                                      , 'units': '1'
#     #                                      , '_FillValue': -999.
#     #                                      }
#     #                                      )
#     #                    }
#     #                   , coords={'height': (['height']
#     #                                        # ,((np.arange(0,n_gates)+.5)*int(confDict['RANGE_GATE_LENGTH'])
#     #                                        # *np.sin(np.nanmedian(elevation)*np.pi/180))
#     #                                        , np.float32(height[NN:])
#     #                                        , {'units': 'm'
#     #                                            , 'standard_name': 'height'
#     #                                            , 'comments': 'vertical distance from sensor to centre of range gate'
#     #                                            , 'bounds': 'height_bnds'
#     #                                           }
#     #                                        )
#     #         , 'time': (['time']
#     #                    , time_start.astype(np.float64)
#     #                    , {'units': 'seconds since 1970-01-01 00:00:00'
#     #                        , 'comments': 'Timestamp at the end of the averaging interval'
#     #                        , 'standard_name': 'time'
#     #                        , 'long_name': 'Time'
#     #                        , 'calendar': 'gregorian'
#     #                        , 'bounds': 'time_bnds'
#     #                        , '_CoordinateAxisType': 'Time'
#     #                       })
#     #         , 'nv': (['nv'], np.arange(0, 2).astype(np.int8))
#     #                             }
#     #                   )
