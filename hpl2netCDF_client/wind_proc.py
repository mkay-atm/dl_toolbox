import datetime

import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import diagsvd

from hpl2netCDF_client.signal_calc import in_db, CN_est
from hpl2netCDF_client.wind_calc import find_num_dir, calc_sigma_single, consensus, build_Amatrix, uvw_2_spd, uvw_2_dir


def lvl2vad_standard(ds_tmp, date_chosen, confDict):
    # read lidar parameters
    n_rays = int(confDict['NUMBER_OF_DIRECTIONS'])
    # number of gates
    n_gates = int(confDict['NUMBER_OF_GATES'])
    # number of pulses used in the data point aquisition
    n = ds_tmp.prf.data
    # number of points per range gate
    M = ds_tmp.nsmpl.data
    # half of detector bandwidth in velocity space
    B = ds_tmp.nqv.data

    # filter Stares within scan
    elevation = 90 - ds_tmp.zenith.data
    azimuth = ds_tmp.azi.data[elevation < 89] % 360
    time_ds = ds_tmp.time.data[elevation < 89]
    dv = ds_tmp.dv.data[elevation < 89]
    snr = ds_tmp.intensity.data[elevation < 89] - 1
    beta = ds_tmp.beta.data[elevation < 89]

    height = ds_tmp.range.data * np.sin(np.nanmedian(elevation[elevation < 89]) * np.pi / 180)
    width = ds_tmp.range.data * 2 * np.cos(np.nanmedian(elevation[elevation < 89]) * np.pi / 180)
    height_bnds = ds_tmp.range_bnds.data
    height_bnds[:, 0] = np.sin(np.nanmedian(elevation[elevation < 89]) * np.pi / 180) * (height_bnds[:, 0])
    height_bnds[:, 1] = np.sin(np.nanmedian(elevation[elevation < 89]) * np.pi / 180) * (height_bnds[:, 1])

    # define time chunks
    ## Look for UTC_OFFSET in config
    if 'UTC_OFFSET' in confDict:
        time_offset = np.timedelta64(int(confDict['UTC_OFFSET']), 'h')
        time_delta = int(confDict['UTC_OFFSET'])
    else:
        time_offset = np.timedelta64(0, 'h')
        time_delta = 0

    time_vec = np.arange(date_chosen - datetime.timedelta(hours=time_delta)
                         , date_chosen + datetime.timedelta(days=1) - datetime.timedelta(hours=time_delta)
                         + datetime.timedelta(minutes=int(confDict['AVG_MIN']))
                         , datetime.timedelta(minutes=int(confDict['AVG_MIN'])))
    calc_idx = [np.where((ii <= time_ds) * (time_ds < iip1))
                for ii, iip1 in zip(time_vec[0:-1], time_vec[1::])]
    # Keeping only calculation indices that are not empty:
    calc_idx = [x for x in calc_idx if len(x[0]) != 0]
    time_start = np.array([int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())
                           if len(t[0]) != 0
                           else int(pd.to_datetime(time_vec[ii + 1]).replace(tzinfo=datetime.timezone.utc).timestamp())
                           for ii, t in enumerate(calc_idx)
                           ])
    time_bnds = np.array([[int(pd.to_datetime(time_ds[t[0][0]]).replace(tzinfo=datetime.timezone.utc).timestamp())
                              ,
                           int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())]
                          if len(t[0]) != 0
                          else [int(pd.to_datetime(time_vec[ii]).replace(tzinfo=datetime.timezone.utc).timestamp())
        , int(pd.to_datetime(time_vec[ii + 1]).replace(tzinfo=datetime.timezone.utc).timestamp())]
                          for ii, t in enumerate(calc_idx)
                          ]).T

    # compare n_gates in lvl1-file and confdict
    if n_gates != dv.shape[1]:
        print('Warning: number of gates in config does not match lvl1 data!')
        n_gates = dv.shape[1]
        print('number of gates changed to ' + str(n_gates))

    # infer number of directions
    # don't forget to check for empty calc_idx
    time_valid = [ii for ii, x in enumerate(calc_idx) if len(x[0]) != 0]

    # UVW = np.where(np.zeros((len(calc_idx),n_gates,3)),np.nan,np.nan)
    UVW = np.full((len(calc_idx), n_gates, 3), np.nan)
    UVWunc = np.full((len(calc_idx), n_gates, 3), np.nan)
    SPEED = np.full((len(calc_idx), n_gates), np.nan)
    SPEEDunc = np.full((len(calc_idx), n_gates), np.nan)
    DIREC = np.full((len(calc_idx), n_gates), np.nan)
    DIRECunc = np.full((len(calc_idx), n_gates), np.nan)
    R2 = np.full((len(calc_idx), n_gates), np.nan)
    CN = np.full((len(calc_idx), n_gates), np.nan)
    n_good = np.full((len(calc_idx), n_gates), np.nan)
    SNR_tot = np.full((len(calc_idx), n_gates), np.nan)
    BETA_tot = np.full((len(calc_idx), n_gates), np.nan)
    SIGMA_tot = np.full((len(calc_idx), n_gates), np.nan)

    for kk in time_valid:
        print('processed ' + str(np.floor(100 * kk / (len(calc_idx) - 1))) + ' %')
        # read lidar parameters
        n_rays = int(confDict['NUMBER_OF_DIRECTIONS'])
        indicator, n_rays, azi_mean, azi_edges = find_num_dir(n_rays, calc_idx, azimuth, kk)
        # azimuth[azimuth>azi_edges[0]]= azimuth[azimuth>azi_edges[0]]-360
        # azi_edges[0]= azi_edges[0]-360
        r_phi = 360 / (n_rays) / 2
        if ~indicator:
            print('some issue with the data', n_rays, len(azi_mean), time_start[kk])
            continue
        else:

            VR = dv[calc_idx[kk]]
            SNR = snr[calc_idx[kk]]
            BETA = beta[calc_idx[kk]]
            azi = azimuth[calc_idx[kk]]
            ele = elevation[calc_idx[kk]]

            VR_CNSmax = np.full((len(azi_mean), n_gates), np.nan)
            VR_CNSunc = np.full((len(azi_mean), n_gates), np.nan)
            # SNR_CNS= np.full((len(azi_mean),n_gates), np.nan)
            BETA_CNS = np.full((len(azi_mean), n_gates), np.nan)
            SIGMA_CNS = np.full((len(azi_mean), n_gates), np.nan)
            # azi_CNS= np.full((len(azi_mean),n_gates), np.nan)
            ele_cns = np.full((len(azi_mean),), np.nan)

            for ii, azi_i in enumerate(azi_mean):
                # azi_idx = (azi>=azi_edges[ii])*(azi<azi_edges[ii+1])
                azi_idx = (np.mod(360 - np.mod(np.mod(azi - azi_i, 360) - r_phi, 360), 360) <= 2 * r_phi)
                ele_cns[ii] = np.median(ele[azi_idx])
                ## calculate consensus average
                VR_CNSmax[ii, :], idx_tmp, VR_CNSunc[ii, :] = consensus(VR[azi_idx]
                                                                        #   , np.ones(SNR[azi_idx].shape)
                                                                        , SNR[azi_idx], BETA[azi_idx]
                                                                        , int(confDict['CNS_RANGE'])
                                                                        , int(confDict['CNS_PERCENTAGE'])
                                                                        , int(confDict['SNR_THRESHOLD'])
                                                                        , B)
                # next line is just experimental and might be useful in the future                                                                          )
                # azi_CNS[ii,:]= np.array([np.nanmean(azi[azi_idx][xi]) for xi in idx_tmp.T])
                # SNR_CNS[ii,:]= np.nanmean( np.where( idx_tmp
                #                                , SNR[azi_idx]
                #                                , np.nan)
                #                          , axis=0)
                SNR_tmp = SNR[azi_idx]
                sigma_tmp = calc_sigma_single(in_db(SNR[azi_idx]), M, n, 2 * B, 1.316)
                # Probably an error in the calculation, but this is what's written in the IDL-code
                # here: MRSE (mean/root/sum/square)
                # I woulf recommend changing it to RMSE (root/mean/square)
                # SIGMA_CNS[ii,:] = np.sqrt(np.nansum( np.where( idx_tmp
                #                                              , sigma_tmp**2
                #                                              , np.nan)
                #                                     , axis=0)
                #                         )/np.sum(idx_tmp, axis=0)
                SIGMA_CNS[ii, :] = np.ma.divide(np.sqrt(np.nansum(np.where(idx_tmp
                                                                           , sigma_tmp ** 2
                                                                           , np.nan)
                                                                  , axis=0))
                                                , np.sum(idx_tmp, axis=0)
                                                )
                ## calculate BETA, with consensus indices
                # BETA_CNS[ii,:]= np.nanmean(np.where(idx_tmp
                #                            ,BETA[azi_idx]
                #                            ,np.nan), axis=0)

            #     # This approach avoids looping over all range gates, but the method is not as stable
            n_good_kk = (~np.isnan(VR_CNSmax)).sum(axis=0)
            # NVRAD[kk, :] = (~np.isnan(VR_CNSmax)).sum(axis=0)
            n_good[kk, :] = n_good_kk
            V_r = np.ma.masked_where((np.isnan(VR_CNSmax))  # & (np.tile(n_good_kk, (azi_mean.shape[0], 1)) < 4)
                                     , VR_CNSmax).T[..., None]
            mask_V_in = (np.isnan(VR_CNSmax)) | (np.tile(n_good_kk, (azi_mean.shape[0], 1)) < 4)
            V_in = np.ma.masked_where(mask_V_in, VR_CNSmax)
            A = build_Amatrix(azi_mean, ele_cns)
            # A[abs(A)<1e-3] = 0
            A_r = np.tile(A, (VR_CNSmax.shape[1], 1, 1))
            A_r_MP = np.tile(np.linalg.pinv(A), (VR_CNSmax.shape[1], 1, 1))
            A_r_MP_T = np.einsum('...ij->...ji', A_r_MP)
            SIGMA_r = np.ma.masked_where(np.isnan(VR_CNSmax), SIGMA_CNS).T

            condi = np.isnan(VR_CNSmax)
            A = np.round(build_Amatrix(azi_mean, ele_cns), 6)
            U, S, Vh = [], [], []
            for c_nn in condi.T:
                u, s, vh = np.linalg.svd(np.ma.masked_where(np.tile(c_nn, (3, 1)).T
                                                            , A).filled(0)
                                         , full_matrices=True)

                U.append(u)
                S.append(np.linalg.pinv(diagsvd(s, u.shape[0], vh.shape[0])))
                Vh.append(vh)
            U, S, Vh = np.array(U), np.array(S), np.array(Vh)

            U_T = np.einsum('...ij->...ji', U)
            Vh_T = np.einsum('...ij->...ji', Vh)
            K1 = np.nansum((U_T * V_in.T[:, None, :]), axis=2)[..., None]
            K2 = np.einsum('...ik,...kj->...ij', S, K1)

            V_k = np.einsum('...ik,...kj->...ij', Vh_T, K2)
            UVW[kk, ...] = np.squeeze(V_k)
            # plausible winds can only be calculated, when the at least three LOS measurements are present
            UVW[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4, :] = np.squeeze(np.full((3, 1), np.nan))

            UVWunc[kk, ...] = abs(np.einsum('...ii->...i', np.sqrt(
                (A_r_MP @ np.apply_along_axis(np.diag, 1, SIGMA_r ** 2) @ A_r_MP_T).astype(complex)).real))
            # plausible winds can only be calculated, when the at least three LOS measurements are present
            UVWunc[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4, :] = np.squeeze(np.full((3, 1), np.nan))

            V_r_est = A_r @ V_k
            ss_e = ((V_r - V_r_est) ** 2).sum(axis=1)
            ss_t = ((V_r - V_r.mean(axis=1)[:, None, :]) ** 2).sum(axis=1)
            R2[kk, :] = np.squeeze(1 - ss_e / ss_t)
            # R2[kk, :] = 1 - (1 - R2[kk, :]) * (np.sum(~np.isnan(VR_CNSmax.T), axis=1)-1)/(np.sum(~np.isnan(VR_CNSmax.T), axis=1)-2)
            # sqe = ((V_r_est-V_r_est.mean(axis=1)[:, None, :])**2).sum(axis = 1)
            # sqt = ((V_r-V_r.mean(axis=1)[:, None, :])**2).sum(axis = 1)
            # R2[kk, :] = np.squeeze(sqe/sqt)
            R2[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

            mask_A = np.tile(mask_V_in.T[..., None], (1, 1, 3))
            # A_r_m = np.ma.masked_where( mask_A, A_r)
            A_r_T = np.einsum('...ij->...ji', A_r)
            Spp = np.apply_along_axis(np.diag, 1, 1 / np.sqrt(np.einsum('...ii->...i', A_r_T @ A_r)))
            Z = np.ma.masked_where(mask_A, A_r @ Spp)
            CN[kk, :] = np.squeeze(np.array([CN_est(X) for X in Z]))
            # CN[kk, :] = np.array([CN_est(X) for X in A_r_m])
            CN[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

            SPEED[kk, :], SPEEDunc[kk, :] = np.vstack(
                [np.fromiter(uvw_2_spd(val, unc).values(), dtype=float) for val, unc in
                 zip(UVW[kk, ...], UVWunc[kk, ...])]).T
            DIREC[kk, :], DIRECunc[kk, :] = np.vstack(
                [np.fromiter(uvw_2_dir(val, unc).values(), dtype=float) for val, unc in
                 zip(UVW[kk, ...], UVWunc[kk, ...])]).T

    ## do quality control
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

    qspeed = (~np.isnan(SPEED))  # *(abs(w)<.3*np.sqrt(np.nanmedian(u)**2+np.nanmedian(v)**2))
    r2[np.isnan(R2)] = -999.
    qr2 = r2 >= float(confDict['R2_THRESHOLD'])
    cn[np.isnan(CN)] = +999.
    qcn = (cn >= 0) & (cn <= float(confDict['CN_THRESHOLD']))
    nvrad[np.isnan(n_good)] = -999
    qnvrad = nvrad >= int(confDict['N_VRAD_THRESHOLD'])

    qwind = qspeed & qnvrad & qcn & qr2
    qu = np.copy(qspeed)
    qv = np.copy(qspeed)
    qw = np.copy(qspeed)
    # qspeed = qspeed & qnvrad
    qspeed = qwind
    speed[~qspeed] = -999.
    errspeed[~qspeed] = -999.
    wdir[~qspeed] = -999.
    errwdir[~qspeed] = -999.
    u[~qspeed] = -999.
    v[~qspeed] = -999.
    w[~qspeed] = -999.
    erru[~qspeed] = -999.
    errv[~qspeed] = -999.
    errw[~qspeed] = -999.
    r2[~qspeed] = -999.
    cn[~qspeed] = -999.
    nvrad[~qspeed] = -999.

    if np.all(np.isnan(speed)):
        print('WARNING: bad retrieval quality')
        print('all retrieved velocities are NaN -> check nvrad threshold!')

    ## add configuration used to create the file
    configuration = """"""
    for dd in confDict:
        if not dd in ['PROC_PATH', 'NC_L1_PATH', 'NC_L2_PATH', 'NC_L2_QL_PATH']:
            configuration += dd + '=' + confDict[dd] + '\n'
    if 'BLINDEZONE_GATES' in confDict:
        NN = int(confDict['BLINDEZONE_GATES'])
    else:
        NN = 0

    ## save processed data to xarray dataset
    return xr.Dataset({'config': ([]
                                  , configuration
                                  , {'standard_name': 'configuration_file'}
                                  )
                          , 'wspeed': (['time', 'height']
                                       , np.float32(speed[:, NN:])
                                       , {'units': 'm s-1'
                                           , 'comments': 'Scalar wind speed (amount of vector)'
                                           , 'standard_name': 'wind_speed'
                                           , 'long_name': 'Wind Speed'
                                           , '_FillValue': -999.
                                          }
                                       )
                          , 'qwind': (['time', 'height']
                                      , qwind[:, NN:].astype(np.int8)
                                      , {'comments': str(
            'quality flag 0 or 1 for u, v, w, wspeed, wdir and corresponding errors,'
            + '(good quality data = WHERE( R2 > ' + confDict['R2_THRESHOLD']
            + 'AND CN < ' + confDict['CN_THRESHOLD']
            + 'AND NVRAD > ' + confDict['N_VRAD_THRESHOLD'] + ')')
                                          , 'long_name': 'wind_quality_flag'
                                          , '_FillValue': np.array(-128).astype(np.int8)
                                          , 'flag_values': np.arange(0, 2).astype(np.int8)
                                          , 'flag_meanings': 'quality_bad quality_good'
                                      }
                                      )
                          , 'qu': (['time', 'height']
                                   , qu[:, NN:].astype(np.int8)
                                   , {'comments': 'quality flag 0 or 1 for u and corresponding error'
                                       , 'long_name': 'quality_flag_u'
                                       , '_FillValue': np.array(-128).astype(np.int8)
                                       , 'flag_values': np.arange(0, 2).astype(np.int8)
                                       , 'flag_meanings': 'quality_bad quality_good'
                                      }
                                   )
                          , 'qv': (['time', 'height']
                                   , qv[:, NN:].astype(np.int8)
                                   , {'comments': 'quality flag 0 or 1 for v and corresponding error'
                                       , 'long_name': 'quality_flag_v'
                                       , '_FillValue': np.array(-128).astype(np.int8)
                                       , 'flag_values': np.arange(0, 2).astype(np.int8)
                                       , 'flag_meanings': 'quality_bad quality_good'
                                      }
                                   )
                          , 'qw': (['time', 'height']
                                   , qw[:, NN:].astype(np.int8)
                                   , {'comments': 'quality flag 0 or 1 for w and corresponding error'
                                       , 'long_name': 'quality_flag_w'
                                       , '_FillValue': np.array(-128).astype(np.int8)
                                       , 'flag_values': np.arange(0, 2).astype(np.int8)
                                       , 'flag_meanings': 'quality_bad quality_good'
                                      }
                                   )
                          , 'errwspeed': (['time', 'height']
                                          , np.float32(errspeed[:, NN:])
                                          , {'units': 'm s-1'
                                              , 'standard': 'wind_speed_uncertainty'
                                              , 'long_name': 'Wind Speed Uncertainty'
                                              , '_FillValue': -999.
                                             }
                                          )
                          , 'u': (['time', 'height']
                                  , np.float32(u[:, NN:])
                                  , {'units': 'm s-1'
                                      ,
                                     'comments': '"Eastward indicates" a vector component which is positive when directed eastward (negative westward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
                                      , 'standard_name': 'eastward_wind'
                                      , 'long_name': 'Zonal Wind'
                                      , '_FillValue': -999.
                                     }
                                  )
                          , 'erru': (['time', 'height']
                                     , np.float32(erru[:, NN:])
                                     , {'units': 'm s-1'
                                         , 'standard_name': 'eastward_wind_uncertainty'
                                         , 'long_name': 'Zonal Wind Uncertainty'
                                         , '_FillValue': -999.
                                        }
                                     )
                          , 'v': (['time', 'height']
                                  , np.float32(v[:, NN:])
                                  , {'units': 'm s-1'
                                      ,
                                     'comments': '"Northward indicates" a vector component which is positive when directed northward (negative southward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
                                      , 'standard_name': 'northward_wind'
                                      , 'long_name': 'Meridional Wind'
                                      , '_FillValue': -999.
                                     }
                                  )
                          , 'errv': (['time', 'height']
                                     , np.float32(errv[:, NN:])
                                     , {'units': 'm s-1'
                                         , 'standard_name': 'northward_wind_uncertainty'
                                         , 'long_name': 'Meridional Wind Uncertainty'
                                         , '_FillValue': -999.
                                        }
                                     )
                          , 'w': (['time', 'height']
                                  , np.float32(w[:, NN:])
                                  , {'units': 'm s-1'
                                      ,
                                     'comments': 'Vertical wind component, positive when directed upward (negative downward)'
                                      , 'standard_name': 'upward_air_velocity'
                                      , 'long_name': 'Upward Air Velocity'
                                      , '_FillValue': -999.
                                     }
                                  )
                          , 'errw': (['time', 'height']
                                     , np.float32(errw[:, NN:])
                                     , {'units': 'm s-1'
                                         , 'standard_name': 'upward_air_velocity_uncertainty'
                                         , 'long_name': 'Upward Air Velocity Uncertainty'
                                         , '_FillValue': -999.
                                        }
                                     )
                          , 'wdir': (['time', 'height']
                                     , np.float32(wdir[:, NN:])
                                     , {'units': 'degree'
                                         , 'comments': 'Wind direction'
                                         , 'standard_name': 'wind_from_direction'
                                         , 'long_name': 'Wind Direction'
                                         , '_FillValue': -999.
                                        }
                                     )
                          , 'errwdir': (['time', 'height']
                                        , np.float32(errwdir[:, NN:])
                                        , {'units': 'degree'
                                            , 'standard_name': 'wind_direction_uncertainty'
                                            , 'long_name': 'Wind Direction Uncertainty'
                                            , '_FillValue': -999.
                                           }
                                        )
                          , 'r2': (['time', 'height']
                                   , np.float32(r2[:, NN:])
                                   , {
                                       'comments': 'coefficient of determination - provides a measure of how well observed radial velocities are replicated by the model used to determine u,v,w wind components from the measured line of sight radial velocities'
                                       , 'long_name': 'coefficient of determination'
                                       , 'standard_name': 'coefficient_of_determination'
                                       , 'units': '1'
                                       , '_FillValue': -999.
                                   }
                                   )
                          , 'nvrad': (['time', 'height']
                                      , np.float32(nvrad[:, NN:])
                                      , {'comments': 'number of (averaged) radial velocities used for wind calculation'
                                          , 'long_name': 'number of radial velocities'
                                          , 'standard_name': 'no_radial_velocities'
                                          , 'units': '1'
                                          , '_FillValue': -999.
                                         }
                                      )
                          , 'cn': (['time', 'height']
                                   , np.float32(cn[:, NN:])
                                   , {
                                       'comments': 'condition number - provides a measure for the degree of collinearity among the Doppler velocity measurements used for the retrieval of the wind variables (u,v,w,speed,direction).'

                                       , 'standard_name': 'condition_number'
                                       , 'long_name': 'Condition Number'
                                       , 'units': '1'
                                       , '_FillValue': -999.
                                   }
                                   )
                          , 'lat': ([]
                                    , np.float32(confDict['SYSTEM_LATITUDE'])
                                    , {'units': 'degrees_north'
                                        , 'long_name': 'latitude'
                                        , 'standard_name': 'latitude'
                                        , 'comments': 'latitude of sensor'
                                        , '_FillValue': -999.
                                       }
                                    )
                          , 'lon': ([]
                                    , np.float32(confDict['SYSTEM_LONGITUDE'])
                                    , {'units': 'degrees_east'
                                        , 'long_name': 'longitude'
                                        , 'standard_name': 'longitude'
                                        , 'comments': 'longitude of sensor'
                                        , '_FillValue': -999.
                                       }
                                    )
                          , 'zsl': ([]
                                    , np.float32(confDict['SYSTEM_ALTITUDE'])
                                    , {'units': 'm'
                                        , 'comments': 'Altitude of sensor above mean sea level'
                                        , 'standard_name': 'altitude'
                                        , '_FillValue': -999.
                                       }
                                    )
                          , 'time_bnds': (['time', 'nv']
                                          , time_bnds.T.astype(np.float64)
                                          , {'units': 'seconds since 1970-01-01 00:00:00 UTC'
                                             }
                                          )
                          , 'height_bnds': (['height', 'nv']
                                            # ,np.array([(np.arange(0,n_gates)
                                            #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                            #             ,((np.arange(0,n_gates) + 1.)
                                            #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                            #             ]).T
                                            , np.float32(height_bnds[NN:, :])
                                            , {'units': 'm'
                                               }
                                            )
                          , 'frequency': ([]
                                          , np.float32(299792458 / float(confDict['SYSTEM_WAVELENGTH']))
                                          , {'units': 'Hz'
                                              , 'comments': 'lidar operating frequency'
                                              , 'long_name': 'instrument_frequency'
                                              , '_FillValue': -999.
                                             }
                                          )
                          , 'vert_res': ([]
                                         , np.float32(np.diff(height).mean())
                                         , {'units': 'm'
                                             , 'comments': 'Calculated from pulse wdth and beam elevation'
                                             , 'long_name': 'Vertical_resolution_measurement'
                                             , '_FillValue': -999.
                                            }
                                         )
                          , 'hor_width': (['height']
                                          # ,np.array([(np.arange(0,n_gates)
                                          #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                          #             ,((np.arange(0,n_gates) + 1.)
                                          #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                          #             ]).T
                                          , np.float32(width[NN:])
                                          , {'units': 'm'
                                              , 'comments': 'Calculated from beam elevation and height'
                                              , 'standard_name': 'horizontal_sample_width'
                                              , '_FillValue': -999.
                                             }
                                          )
                       }
                      , coords={'height': (['height']
                                           # ,((np.arange(0,n_gates)+.5)*int(confDict['RANGE_GATE_LENGTH'])
                                           # *np.sin(np.nanmedian(elevation)*np.pi/180))
                                           , np.float32(height[NN:])
                                           , {'units': 'm'
                                               , 'standard_name': 'height'
                                               , 'comments': 'vertical distance from sensor to centre of range gate'
                                               , 'bounds': 'height_bnds'
                                              }
                                           )
            , 'time': (['time']
                       , time_start.astype(np.float64)
                       , {'units': 'seconds since 1970-01-01 00:00:00'
                           , 'comments': 'Timestamp at the end of the averaging interval'
                           , 'standard_name': 'time'
                           , 'long_name': 'Time'
                           , 'calendar': 'gregorian'
                           , 'bounds': 'time_bnds'
                           , '_CoordinateAxisType': 'Time'
                          })
            , 'nv': (['nv'], np.arange(0, 2).astype(np.int8))
                                }
                      )


def lvl2wcdbs(ds_comb, date_chosen, confDict):
    n_rays = int(confDict['NUMBER_OF_DIRECTIONS'])
    # number of gates
    n_gates = int(confDict['NUMBER_OF_GATES'])
    # number of pulses used in the data point aquisition
    B = (ds_comb.radial_wind_speed.max() - ds_comb.radial_wind_speed.min()).data / 2
    lrg = ds_comb.range_gate_length.data

    time_ds = pd.to_datetime(ds_comb.time.data, unit='s')
    ds_comb['zenith'] = 90 - ds_comb.elevation
    elevation = ds_comb.elevation.data
    azimuth = ds_comb.azimuth.data
    cnr = 10 ** (ds_comb.cnr.data / 10)
    dv = ds_comb.radial_wind_speed.data
    delv = ds_comb.doppler_spectrum_width.data
    if 'relative_beta' in list(ds_comb.keys()):
        beta = ds_comb.relative_beta.data
    else:
        print('no backscatter data in file, set all beta to 1')
        beta = np.ones(dv.shape)
    if 'measurement_height' in list(ds_comb.keys()):
        height = ds_comb.measurement_height.data[0]
    else:
        height = (np.sin(ds_comb.elevation * np.pi / 180) * ds_comb.range).data[0]

    width = (np.cos(ds_comb.elevation * np.pi / 180) * ds_comb.range * 2).data
    width = width[elevation < 89][0]
    height_bnds = np.vstack([height - lrg / 2, height - lrg / 2]).T

    if 'UTC_OFFSET' in confDict:
        time_offset = np.timedelta64(int(confDict['UTC_OFFSET']), 'h')
        time_delta = int(confDict['UTC_OFFSET'])
    else:
        time_offset = np.timedelta64(0, 'h')
        time_delta = 0

    time_vec = np.arange(date_chosen - datetime.timedelta(hours=time_delta)
                         , date_chosen + datetime.timedelta(days=1) - datetime.timedelta(hours=time_delta)
                         + datetime.timedelta(minutes=int(confDict['AVG_MIN']))
                         , datetime.timedelta(minutes=int(confDict['AVG_MIN'])))
    calc_idx = [np.where((ii <= time_ds) * (time_ds < iip1))
                for ii, iip1 in zip(time_vec[0:-1], time_vec[1::])]
    # Keeping only calculation indices that are not empty:
    calc_idx = [x for x in calc_idx if len(x[0]) != 0]
    time_start = np.array([int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())
                           if len(t[0]) != 0
                           else int(pd.to_datetime(time_vec[ii + 1]).replace(tzinfo=datetime.timezone.utc).timestamp())
                           for ii, t in enumerate(calc_idx)
                           ])
    time_bnds = np.array([[int(pd.to_datetime(time_ds[t[0][0]]).replace(tzinfo=datetime.timezone.utc).timestamp())
                              ,
                           int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())]
                          if len(t[0]) != 0
                          else [int(pd.to_datetime(time_vec[ii]).replace(tzinfo=datetime.timezone.utc).timestamp())
        , int(pd.to_datetime(time_vec[ii + 1]).replace(tzinfo=datetime.timezone.utc).timestamp())]
                          for ii, t in enumerate(calc_idx)
                          ]).T
    if n_gates != dv.shape[1]:
        print('Warning: number of gates in config does not match lvl1 data!')
        n_gates = dv.shape[1]
        print('number of gates changed to ' + str(n_gates))
    # infer number of directions
    # don't forget to check for empty calc_idx
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
    CNR_tot = np.full((len(calc_idx), n_gates), np.nan)
    BETA_tot = np.full((len(calc_idx), n_gates), np.nan)
    # SIGMA_tot = np.full((len(calc_idx), n_gates), np.nan)

    # time_ds = time[np.where(ds)]

    for kk in time_valid:
        #print('processed ' + str(np.floor(100 * kk / (len(calc_idx) - 1))) + ' %')
        # read lidar parameters
        n_rays = int(confDict['NUMBER_OF_DIRECTIONS'])
        indicator, n_rays, azi_mean, azi_edges = find_num_dir(n_rays, calc_idx, azimuth, kk)
        # azimuth[azimuth>azi_edges[0]]= azimuth[azimuth>azi_edges[0]]-360
        # azi_edges[0]= azi_edges[0]-360
        r_phi = 360 / (n_rays) / 2
        if ~indicator:
            print('some issue with the data', n_rays, len(azi_mean), time_start[kk])
            continue
        else:
            VR = dv[calc_idx[kk]][elevation[calc_idx[kk]] < 89]
            # only if vertical values exist
            if np.any(elevation[calc_idx[kk]] > 89):
                WR = dv[calc_idx[kk]][elevation[calc_idx[kk]] > 89]
                CNR_WR = cnr[calc_idx[kk]][elevation[calc_idx[kk]] > 89]
                BETA_WR = dv[calc_idx[kk]][elevation[calc_idx[kk]] > 89]
                SPEC_WR = delv[calc_idx[kk]][elevation[calc_idx[kk]] > 89]
                # estimate consensus of vertical velocity data
                # w_cns, idx_tmp, tmp_tmp = consensus( WR, np.ones(WR.shape), np.ones(WR.shape), .1, 100, 0, B)
                # WR_SPEC = np.ma.masked_where(~idx_tmp, SPEC_WR).mean(axis=0).filled(np.nan)
                w_cns, idx_tmp, tmp_tmp = consensus(WR, np.ones(WR.shape), np.ones(WR.shape), 2, 30, 0, B)
                WR_SPEC = tmp_tmp
                WR_CNS = np.ma.masked_where(~idx_tmp, CNR_WR).mean(axis=0).filled(np.nan)
                WR_BETA = np.ma.masked_where(~idx_tmp, BETA_WR).mean(axis=0).filled(np.nan)
            CNR = cnr[calc_idx[kk]][elevation[calc_idx[kk]] < 89]
            BETA = beta[calc_idx[kk]][elevation[calc_idx[kk]] < 89]
            azi = azimuth[calc_idx[kk]][elevation[calc_idx[kk]] < 89]
            ele = elevation[calc_idx[kk]][elevation[calc_idx[kk]] < 89]
            SPEC = delv[calc_idx[kk]][elevation[calc_idx[kk]] < 89]

            VR_CNSmax = np.full((len(azi_mean), n_gates), np.nan)
            VR_CNSunc = np.full((len(azi_mean), n_gates), np.nan)
            CNR_CNS = np.full((len(azi_mean), n_gates), np.nan)
            BETA_CNS = np.full((len(azi_mean), n_gates), np.nan)
            SIGMA_CNS = np.full((len(azi_mean), n_gates), np.nan)
            ele_cns = np.full((len(azi_mean),), np.nan)

            for ii, azi_i in enumerate(azi_mean):
                # azi_idx = (azi>=azi_edges[ii])*(azi<azi_edges[ii+1])
                azi_idx = (np.mod(360 - np.mod(np.mod(azi - azi_i, 360) - r_phi, 360), 360) <= 2 * r_phi)
                ele_cns[ii] = np.median(ele[azi_idx])
                ## calculate consensus average
                VR_CNSmax[ii, :], idx_tmp, VR_CNSunc[ii, :] = consensus(VR[azi_idx]
                                                                        , CNR[azi_idx], BETA[azi_idx]
                                                                        , int(confDict['CNS_RANGE'])
                                                                        , int(confDict['CNS_PERCENTAGE'])
                                                                        , int(confDict['SNR_THRESHOLD'])
                                                                        , B)
                CNR_CNS[ii, :] = np.ma.masked_where(~idx_tmp, CNR[azi_idx]).mean(axis=0).filled(np.nan)
                # SIGMA_CNS[ii, :] = np.ma.masked_where(~idx_tmp, SPEC[azi_idx]).mean(axis=0).filled(np.nan)
                SIGMA_CNS[ii, :] = VR_CNSunc[ii, :]
            if np.any(elevation[calc_idx[kk]] > 89):
                #         Add vertical Stares to azimuth consensus
                VR_CNSmax = np.vstack([VR_CNSmax, w_cns])
                SIGMA_CNS = np.vstack([SIGMA_CNS, WR_SPEC])
                azi_mean = np.hstack([azi_mean, 0])
                ele_cns = np.hstack([ele_cns, 90])
            #         WR_filt = hp.hpl2netCDF_client.filter_by_snr(WR, CNR_WR, -18).filled(np.nan)
            #         SPEC_filt = hp.hpl2netCDF_client.filter_by_snr(WR, CNR_WR, -18).filled(np.nan)
            #         VR_CNSmax = np.vstack([VR_CNSmax, WR_filt])
            #         SIGMA_CNS = np.vstack([SIGMA_CNS, SPEC_filt])
            #         azi_mean = np.hstack([azi_mean, np.zeros(WR_filt.shape[0])])
            #         ele_cns = np.hstack([ele_cns, 90*np.ones(WR_filt.shape[0])])
            #     # This approach avoids looping over all range gates, but the method is not as stable
            n_good_kk = (~np.isnan(VR_CNSmax)).sum(axis=0)
            # NVRAD[kk, :] = (~np.isnan(VR_CNSmax)).sum(axis=0)
            n_good[kk, :] = n_good_kk
            V_r = np.ma.masked_where((np.isnan(VR_CNSmax))  # & (np.tile(n_good_kk, (azi_mean.shape[0], 1)) < 4)
                                     , VR_CNSmax).T[..., None]
            mask_V_in = (np.isnan(VR_CNSmax)) | (np.tile(n_good_kk, (azi_mean.shape[0], 1)) < 4)
            V_in = np.ma.masked_where(mask_V_in, VR_CNSmax)
            A = build_Amatrix(azi_mean, ele_cns)
            # A[abs(A)<1e-3] = 0
            A_r = np.tile(A, (VR_CNSmax.shape[1], 1, 1))
            A_r_MP = np.tile(np.linalg.pinv(A), (VR_CNSmax.shape[1], 1, 1))
            A_r_MP_T = np.einsum('...ij->...ji', A_r_MP)
            SIGMA_r = np.ma.masked_where(np.isnan(VR_CNSmax), SIGMA_CNS).T

            condi = np.isnan(VR_CNSmax)
            A = np.round(build_Amatrix(azi_mean, ele_cns), 6)
            # include stare measurements
            A_stare = np.round(build_Amatrix(np.zeros(((elevation[calc_idx[kk]] > 89).sum(),))
                                             , 90 * np.ones(((elevation[calc_idx[kk]] > 89).sum(),)))
                               , 6)
            U, S, Vh = [], [], []
            for c_nn in condi.T:
                u, s, vh = np.linalg.svd(np.ma.masked_where(np.tile(c_nn, (3, 1)).T
                                                            , A).filled(0)
                                         , full_matrices=True)

                U.append(u)
                S.append(np.linalg.pinv(diagsvd(s, u.shape[0], vh.shape[0])))
                Vh.append(vh)
            U, S, Vh = np.array(U), np.array(S), np.array(Vh)

            U_T = np.einsum('...ij->...ji', U)
            Vh_T = np.einsum('...ij->...ji', Vh)
            K1 = np.nansum((U_T * V_in.T[:, None, :]), axis=2)[..., None]
            K2 = np.einsum('...ik,...kj->...ij', S, K1)

            V_k = np.einsum('...ik,...kj->...ij', Vh_T, K2)
            UVW[kk, ...] = np.squeeze(V_k)
            # plausible winds can only be calculated, when the at least three LOS measurements are present
            UVW[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4, :] = np.squeeze(np.full((3, 1), np.nan))

            UVWunc[kk, ...] = abs(np.einsum('...ii->...i', np.sqrt(
                (A_r_MP @ np.apply_along_axis(np.diag, 1, SIGMA_r ** 2) @ A_r_MP_T).astype(complex)).real))
            # plausible winds can only be calculated, when the at least three LOS measurements are present
            UVWunc[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4, :] = np.squeeze(np.full((3, 1), np.nan))

            V_r_est = A_r @ V_k
            ss_e = ((V_r - V_r_est) ** 2).sum(axis=1)
            ss_t = ((V_r - V_r.mean(axis=1)[:, None, :]) ** 2).sum(axis=1)
            R2[kk, :] = np.squeeze(1 - ss_e / ss_t)
            # R2[kk, :] = 1 - (1 - R2[kk, :]) * (np.sum(~np.isnan(VR_CNSmax.T), axis=1)-1)/(np.sum(~np.isnan(VR_CNSmax.T), axis=1)-2)
            # sqe = ((V_r_est-V_r_est.mean(axis=1)[:, None, :])**2).sum(axis = 1)
            # sqt = ((V_r-V_r.mean(axis=1)[:, None, :])**2).sum(axis = 1)
            # R2[kk, :] = np.squeeze(sqe/sqt)
            R2[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

            mask_A = np.tile(mask_V_in.T[..., None], (1, 1, 3))
            # A_r_m = np.ma.masked_where( mask_A, A_r)
            A_r_T = np.einsum('...ij->...ji', A_r)
            Spp = np.apply_along_axis(np.diag, 1, 1 / np.sqrt(np.einsum('...ii->...i', A_r_T @ A_r)))
            Z = np.ma.masked_where(mask_A, A_r @ Spp)
            CN[kk, :] = np.squeeze(np.array([CN_est(X) for X in Z]))
            # CN[kk, :] = np.array([CN_est(X) for X in A_r_m])
            CN[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

            SPEED[kk, :], SPEEDunc[kk, :] = np.vstack(
                [np.fromiter(uvw_2_spd(val, unc).values(), dtype=float) for val, unc in
                 zip(UVW[kk, ...], UVWunc[kk, ...])]).T
            DIREC[kk, :], DIRECunc[kk, :] = np.vstack(
                [np.fromiter(uvw_2_dir(val, unc).values(), dtype=float) for val, unc in
                 zip(UVW[kk, ...], UVWunc[kk, ...])]).T

    ## do quality control
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

    qspeed = (~np.isnan(SPEED))  # *(abs(w)<.3*np.sqrt(np.nanmedian(u)**2+np.nanmedian(v)**2))
    r2[np.isnan(R2)] = -999.
    qr2 = r2 >= float(confDict['R2_THRESHOLD'])
    cn[np.isnan(CN)] = +999.
    qcn = (cn >= 0) & (cn <= float(confDict['CN_THRESHOLD']))
    nvrad[np.isnan(n_good)] = -999
    qnvrad = nvrad >= int(confDict['N_VRAD_THRESHOLD'])

    qwind = qspeed & qnvrad & qcn & qr2
    qu = np.copy(qspeed)
    qv = np.copy(qspeed)
    qw = np.copy(qspeed)
    # qspeed = qspeed & qnvrad
    qspeed = qwind
    speed[~qspeed] = -999.
    errspeed[~qspeed] = -999.
    wdir[~qspeed] = -999.
    errwdir[~qspeed] = -999.
    u[~qspeed] = -999.
    v[~qspeed] = -999.
    w[~qspeed] = -999.
    erru[~qspeed] = -999.
    errv[~qspeed] = -999.
    errw[~qspeed] = -999.
    r2[~qspeed] = -999.
    cn[~qspeed] = -999.
    nvrad[~qspeed] = -999.

    if np.all(np.isnan(speed)):
        print('WARNING: bad retrieval quality')
        print('all retrieved velocities are NaN -> check nvrad threshold!')
    ## save processed data to netCDF

    ## add configuration used to create the file
    configuration = """"""
    for dd in confDict:
        if not dd in ['PROC_PATH', 'NC_L1_PATH', 'NC_L2_PATH', 'NC_L2_QL_PATH']:
            configuration += dd + '=' + confDict[dd] + '\n'
    if 'BLINDEZONE_GATES' in confDict:
        NN = int(confDict['BLINDEZONE_GATES'])
    else:
        NN = 0

    return xr.Dataset({'config': ([]
                                  , configuration
                                  , {'standard_name': 'configuration_file'}
                                  )
                          , 'wspeed': (['time', 'height']
                                       , np.float32(speed[:, NN:])
                                       , {'units': 'm s-1'
                                           , 'comments': 'Scalar wind speed (amount of vector)'
                                           , 'standard_name': 'wind_speed'
                                           , 'long_name': 'Wind Speed'
                                           , '_FillValue': -999.
                                          }
                                       )
                          , 'qwind': (['time', 'height']
                                      , qwind[:, NN:].astype(np.int8)
                                      , {'comments': str(
            'quality flag 0 or 1 for u, v, w, wspeed, wdir and corresponding errors,'
            + '(good quality data = WHERE( R2 > ' + confDict['R2_THRESHOLD']
            + 'AND CN < ' + confDict['CN_THRESHOLD']
            + 'AND NVRAD > ' + confDict['N_VRAD_THRESHOLD'] + ')')
                                          , 'long_name': 'wind_quality_flag'
                                          , '_FillValue': np.array(-128).astype(np.int8)
                                          , 'flag_values': np.arange(0, 2).astype(np.int8)
                                          , 'flag_meanings': 'quality_bad quality_good'
                                      }
                                      )
                          , 'qu': (['time', 'height']
                                   , qu[:, NN:].astype(np.int8)
                                   , {'comments': 'quality flag 0 or 1 for u and corresponding error'
                                       , 'long_name': 'quality_flag_u'
                                       , '_FillValue': np.array(-128).astype(np.int8)
                                       , 'flag_values': np.arange(0, 2).astype(np.int8)
                                       , 'flag_meanings': 'quality_bad quality_good'
                                      }
                                   )
                          , 'qv': (['time', 'height']
                                   , qv[:, NN:].astype(np.int8)
                                   , {'comments': 'quality flag 0 or 1 for v and corresponding error'
                                       , 'long_name': 'quality_flag_v'
                                       , '_FillValue': np.array(-128).astype(np.int8)
                                       , 'flag_values': np.arange(0, 2).astype(np.int8)
                                       , 'flag_meanings': 'quality_bad quality_good'
                                      }
                                   )
                          , 'qw': (['time', 'height']
                                   , qw[:, NN:].astype(np.int8)
                                   , {'comments': 'quality flag 0 or 1 for w and corresponding error'
                                       , 'long_name': 'quality_flag_w'
                                       , '_FillValue': np.array(-128).astype(np.int8)
                                       , 'flag_values': np.arange(0, 2).astype(np.int8)
                                       , 'flag_meanings': 'quality_bad quality_good'
                                      }
                                   )
                          , 'errwspeed': (['time', 'height']
                                          , np.float32(errspeed[:, NN:])
                                          , {'units': 'm s-1'
                                              , 'standard': 'wind_speed_uncertainty'
                                              , 'long_name': 'Wind Speed Uncertainty'
                                              , '_FillValue': -999.
                                             }
                                          )
                          , 'u': (['time', 'height']
                                  , np.float32(u[:, NN:])
                                  , {'units': 'm s-1'
                                      ,
                                     'comments': '"Eastward indicates" a vector component which is positive when directed eastward (negative westward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
                                      , 'standard_name': 'eastward_wind'
                                      , 'long_name': 'Zonal Wind'
                                      , '_FillValue': -999.
                                     }
                                  )
                          , 'erru': (['time', 'height']
                                     , np.float32(erru[:, NN:])
                                     , {'units': 'm s-1'
                                         , 'standard_name': 'eastward_wind_uncertainty'
                                         , 'long_name': 'Zonal Wind Uncertainty'
                                         , '_FillValue': -999.
                                        }
                                     )
                          , 'v': (['time', 'height']
                                  , np.float32(v[:, NN:])
                                  , {'units': 'm s-1'
                                      ,
                                     'comments': '"Northward indicates" a vector component which is positive when directed northward (negative southward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
                                      , 'standard_name': 'northward_wind'
                                      , 'long_name': 'Meridional Wind'
                                      , '_FillValue': -999.
                                     }
                                  )
                          , 'errv': (['time', 'height']
                                     , np.float32(errv[:, NN:])
                                     , {'units': 'm s-1'
                                         , 'standard_name': 'northward_wind_uncertainty'
                                         , 'long_name': 'Meridional Wind Uncertainty'
                                         , '_FillValue': -999.
                                        }
                                     )
                          , 'w': (['time', 'height']
                                  , np.float32(w[:, NN:])
                                  , {'units': 'm s-1'
                                      ,
                                     'comments': 'Vertical wind component, positive when directed upward (negative downward)'
                                      , 'standard_name': 'upward_air_velocity'
                                      , 'long_name': 'Upward Air Velocity'
                                      , '_FillValue': -999.
                                     }
                                  )
                          , 'errw': (['time', 'height']
                                     , np.float32(errw[:, NN:])
                                     , {'units': 'm s-1'
                                         , 'standard_name': 'upward_air_velocity_uncertainty'
                                         , 'long_name': 'Upward Air Velocity Uncertainty'
                                         , '_FillValue': -999.
                                        }
                                     )
                          , 'wdir': (['time', 'height']
                                     , np.float32(wdir[:, NN:])
                                     , {'units': 'degree'
                                         , 'comments': 'Wind direction'
                                         , 'standard_name': 'wind_from_direction'
                                         , 'long_name': 'Wind Direction'
                                         , '_FillValue': -999.
                                        }
                                     )
                          , 'errwdir': (['time', 'height']
                                        , np.float32(errwdir[:, NN:])
                                        , {'units': 'degree'
                                            , 'standard_name': 'wind_direction_uncertainty'
                                            , 'long_name': 'Wind Direction Uncertainty'
                                            , '_FillValue': -999.
                                           }
                                        )
                          , 'r2': (['time', 'height']
                                   , np.float32(r2[:, NN:])
                                   , {
                                       'comments': 'coefficient of determination - provides a measure of how well observed radial velocities are replicated by the model used to determine u,v,w wind components from the measured line of sight radial velocities'
                                       , 'long_name': 'coefficient of determination'
                                       , 'standard_name': 'coefficient_of_determination'
                                       , 'units': '1'
                                       , '_FillValue': -999.
                                   }
                                   )
                          , 'nvrad': (['time', 'height']
                                      , np.float32(nvrad[:, NN:])
                                      , {'comments': 'number of (averaged) radial velocities used for wind calculation'
                                          , 'long_name': 'number of radial velocities'
                                          , 'standard_name': 'no_radial_velocities'
                                          , 'units': '1'
                                          , '_FillValue': -999.
                                         }
                                      )
                          , 'cn': (['time', 'height']
                                   , np.float32(cn[:, NN:])
                                   , {
                                       'comments': 'condition number - provides a measure for the degree of collinearity among the Doppler velocity measurements used for the retrieval of the wind variables (u,v,w,speed,direction).'

                                       , 'standard_name': 'condition_number'
                                       , 'long_name': 'Condition Number'
                                       , 'units': '1'
                                       , '_FillValue': -999.
                                   }
                                   )
                          , 'lat': ([]
                                    , np.float32(confDict['SYSTEM_LATITUDE'])
                                    , {'units': 'degrees_north'
                                        , 'long_name': 'latitude'
                                        , 'standard_name': 'latitude'
                                        , 'comments': 'latitude of sensor'
                                        , '_FillValue': -999.
                                       }
                                    )
                          , 'lon': ([]
                                    , np.float32(confDict['SYSTEM_LONGITUDE'])
                                    , {'units': 'degrees_east'
                                        , 'long_name': 'longitude'
                                        , 'standard_name': 'longitude'
                                        , 'comments': 'longitude of sensor'
                                        , '_FillValue': -999.
                                       }
                                    )
                          , 'zsl': ([]
                                    , np.float32(confDict['SYSTEM_ALTITUDE'])
                                    , {'units': 'm'
                                        , 'comments': 'Altitude of sensor above mean sea level'
                                        , 'standard_name': 'altitude'
                                        , '_FillValue': -999.
                                       }
                                    )
                          , 'time_bnds': (['time', 'nv']
                                          , time_bnds.T.astype(np.float64)
                                          , {'units': 'seconds since 1970-01-01 00:00:00 UTC'
                                             }
                                          )
                          , 'height_bnds': (['height', 'nv']
                                            # ,np.array([(np.arange(0,n_gates)
                                            #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                            #             ,((np.arange(0,n_gates) + 1.)
                                            #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                            #             ]).T
                                            , np.float32(height_bnds[NN:, :])
                                            , {'units': 'm'
                                               }
                                            )
                          , 'frequency': ([]
                                          , np.float32(299792458 / float(confDict['SYSTEM_WAVELENGTH']))
                                          , {'units': 'Hz'
                                              , 'comments': 'lidar operating frequency'
                                              , 'long_name': 'instrument_frequency'
                                              , '_FillValue': -999.
                                             }
                                          )
                          , 'vert_res': ([]
                                         , np.float32(np.diff(height).mean())
                                         , {'units': 'm'
                                             , 'comments': 'Calculated from pulse wdth and beam elevation'
                                             , 'long_name': 'Vertical_resolution_measurement'
                                             , '_FillValue': -999.
                                            }
                                         )
                          , 'hor_width': (['height']
                                          # ,np.array([(np.arange(0,n_gates)
                                          #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                          #             ,((np.arange(0,n_gates) + 1.)
                                          #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                          #             ]).T
                                          , np.float32(width[NN:])
                                          , {'units': 'm'
                                              , 'comments': 'Calculated from beam elevation and height'
                                              , 'standard_name': 'horizontal_sample_width'
                                              , '_FillValue': -999.
                                             }
                                          )
                       }
                      , coords={'height': (['height']
                                           # ,((np.arange(0,n_gates)+.5)*int(confDict['RANGE_GATE_LENGTH'])
                                           # *np.sin(np.nanmedian(elevation)*np.pi/180))
                                           , np.float32(height[NN:])
                                           , {'units': 'm'
                                               , 'standard_name': 'height'
                                               , 'comments': 'vertical distance from sensor to centre of range gate'
                                               , 'bounds': 'height_bnds'
                                              }
                                           )
            , 'time': (['time']
                       , time_start.astype(np.float64)
                       , {'units': 'seconds since 1970-01-01 00:00:00'
                           , 'comments': 'Timestamp at the end of the averaging interval'
                           , 'standard_name': 'time'
                           , 'long_name': 'Time'
                           , 'calendar': 'gregorian'
                           , 'bounds': 'time_bnds'
                           , '_CoordinateAxisType': 'Time'
                          })
            , 'nv': (['nv'], np.arange(0, 2).astype(np.int8))
                                }
                      )
