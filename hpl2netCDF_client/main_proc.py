import datetime

import numpy as np
import pandas as pd
import xarray as xr

from hpl2netCDF_client.wind_proc import lvl2vad_standard, lvl2wcdbs, lvl2vad_nrt


def process_dataset(ds_tmp, date_chosen, confDict, nrt=False):
    if confDict['SYSTEM'].lower() == 'windcube':
        if nrt:
            raise ValueError('NRT processing not yet implemented for Windcube')
        # if (len(ds_tmp.range.dims) > 1):
        if 'fixed' not in confDict['SCAN_TYPE'].lower():
            if ('dbs' in confDict['SCAN_TYPE'].lower()) or ('vad' in confDict['SCAN_TYPE'].lower()) or (
                    'ppi' in confDict['SCAN_TYPE'].lower()):
                print("processing 'Windcube-dbs/-vad' setting!")
                ds_lvl2 = lvl2wcdbs(ds_tmp, date_chosen, confDict)
            if ('rhi' in confDict['SCAN_TYPE'].lower()):
                print("settings for RHI not yet implemented!")
                # create place holder dataset
                ds_lvl2 = xr.Dataset()
        if ('fixed' in confDict['SCAN_TYPE'].lower()):
            if ('vad' in confDict['SCAN_TYPE'].lower()):
                print("processing 'Windcube-vad-fixed' setting!...for old system version!!")
                ds_lvl2 = lvl2vad_standard(ds_tmp, date_chosen, confDict)
            if ('stare' in confDict['SCAN_TYPE'].lower()):
                print("processing 'WindCube-stare' setting!")
                print("coming soon!")
                # create place holder dataset
                ds_lvl2 = xr.Dataset()
    if confDict['SYSTEM'].lower() == 'halo':
        if (len(ds_tmp.range.dims) > 1) & (confDict['SCAN_TYPE'] == 'DBS'):
            print("processing 'Streamline-dbs' setting!")
            if nrt:
                ds_lvl2 = lvl2vad_nrt(ds_tmp, date_chosen, confDict)
            else:
                ds_lvl2 = lvl2vad_standard(ds_tmp, date_chosen, confDict)
        if ('vad' in confDict['SCAN_TYPE'].lower()) | ('user' in confDict['SCAN_TYPE'].lower()):
            print("processing 'Streamline-VAD' setting!")
            if nrt:
                ds_lvl2 = lvl2vad_nrt(ds_tmp, date_chosen, confDict)
            else:
                ds_lvl2 = lvl2vad_standard(ds_tmp, date_chosen, confDict)
        if ('stare' in confDict['SCAN_TYPE'].lower()):
            print("processing 'Streamline-Stare' setting!")
            print("coming soon!")
            # create place holder dataset
            ds_lvl2 = xr.Dataset()
    ds_lvl2 = add_attributes(ds_lvl2, confDict)
    return ds_lvl2

def add_attributes(ds_lvl2, confDict):
    ds_lvl2.attrs['title'] = confDict['NC_TITLE']
    ds_lvl2.attrs['institution'] = confDict['NC_INSTITUTION']
    ds_lvl2.attrs['site_location'] = confDict['NC_SITE_LOCATION']
    ds_lvl2.attrs['source'] = confDict['NC_SOURCE']
    ds_lvl2.attrs['instrument_type'] = confDict['NC_INSTRUMENT_TYPE']
    ds_lvl2.attrs['instrument_mode'] = confDict['NC_INSTRUMENT_MODE']
    if 'NC_INSTRUMENT_FIRMWARE_VERSION' in confDict:
        ds_lvl2.attrs['instrument_firmware_version'] = confDict['NC_INSTRUMENT_FIRMWARE_VERSION']
    else:
        ds_lvl2.attrs['instrument_firmware_version'] = 'N/A'
    if 'NC_INSTRUMENT_ID' in confDict:
        ds_lvl2.attrs['instrument_id'] = confDict['NC_INSTRUMENT_ID']
    else:
        ds_lvl2.attrs['instrument_id'] = 'N/A'
    ds_lvl2.attrs['instrument_contact'] = confDict['NC_INSTRUMENT_CONTACT']
    # ds_lvl2.attrs['Source']= "HALO Photonics Doppler lidar (production number: " + confDict['SYSTEM_ID'] + ')'
    # ds_lvl2.attrs['history']= confDict['NC_HISTORY']
    ds_lvl2.attrs['conventions'] = confDict['NC_CONVENTIONS']
    ds_lvl2.attrs['processing_date'] = str(pd.to_datetime(datetime.datetime.now())) + ' UTC'
    # ds_lvl2.attrs['author']= confDict['NC_AUTHOR']
    # ds_lvl2.attrs['licence']= confDict['NC_LICENCE']
    ds_lvl2.attrs['data_policy'] = confDict['NC_DATA_POLICY']
    # attributes for operational use of netCDFs, see E-Profile wind profiler netCDF version 1.7
    if 'NC_WIGOS_STATION_ID' in confDict:
        ds_lvl2.attrs['wigos_station_id'] = confDict['NC_WIGOS_STATION_ID']
    else:
        ds_lvl2.attrs['wigos_station_id'] = 'N/A'
    if 'NC_WMO_ID' in confDict:
        ds_lvl2.attrs['wmo_id'] = confDict['NC_WMO_ID']
    else:
        ds_lvl2.attrs['wmo_id'] = 'N/A'
    if 'NC_PI_ID' in confDict:
        ds_lvl2.attrs['principal_investigator'] = confDict['NC_PI_ID']
    else:
        ds_lvl2.attrs['principal_investigator'] = 'N/A'
    if 'NC_INSTRUMENT_SERIAL_NUMBER' in confDict:
        ds_lvl2.attrs['instrument_serial_number'] = confDict['NC_INSTRUMENT_SERIAL_NUMBER']
    else:
        ds_lvl2.attrs['instrument_serial_number'] = ' '
    ds_lvl2.attrs['history'] = confDict['NC_HISTORY'] + ' version ' + confDict['VERSION'] + ' on ' + str(
        pd.to_datetime(datetime.datetime.now())) + ' UTC'
    if 'OPERATIONAL' in confDict:
        if bool(int(confDict['OPERATIONAL'])):
            # Blocking statur, only important for operational use
            ds_lvl2.attrs['references'] = confDict['NC_REFERENCES']  # 'Doppler lidar PPI-based retrieval, see VAD'
            ds_lvl2.attrs['data_blocking_status'] = confDict['NC_DATA_BLOCKING_STATUS']
            # set all uncertainties to NaN-Value
            for item in ['erru', 'errv', 'errw', 'errwspeed', 'errwdir']:
                ds_lvl2[item] = ds_lvl2[item].where(ds_lvl2[item] == -999., other=-999.)
    #             ds_lvl2[item] = ds_lvl2[item].where(np.isnan(ds_lvl2[item]), other=np.nan)
    ds_lvl2.attrs['comments'] = confDict['NC_COMMENTS']
    return ds_lvl2

def write_netcdf(ds, filename, confDict):
    # compress variables
    comp = dict(zlib=True, complevel=9)
    encoding = {var: comp for var in np.hstack([ds.data_vars, ds.coords])}

    # set UTC offset
    if 'UTC_OFFSET' in confDict:
        time_delta = int(confDict['UTC_OFFSET'])
    else:
        time_delta = 0
    # ds.time.attrs['units'] = ('seconds since 1970-01-01 00:00:00', 'seconds since 1970-01-01 00:00:00 {:+03d}'.format(time_delta))[abs(np.sign(time_delta))]
    ds.time.encoding['units'] = ('seconds since 1970-01-01 00:00:00',
                                 'seconds since 1970-01-01 00:00:00 {:+03d}'.format(time_delta)
                                 )[abs(np.sign(time_delta))]

    ds.to_netcdf(filename, unlimited_dims={'time': True}, encoding=encoding)
