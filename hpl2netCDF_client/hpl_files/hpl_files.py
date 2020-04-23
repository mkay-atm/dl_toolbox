#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import re
import datetime

from pathlib import Path

import hpl2netCDF_client as proc

class hpl_files(object):
    name= []
    time= []

    # The class "constructor" - It's actually an initializer 
    def __init__(self, name, time):
        self.name = name
        self.time = time
    
    def make_file_list(date_chosen, confDict, url):
        path = Path(url + '/'
            + date_chosen.strftime('%Y') + '/'
            + date_chosen.strftime('%Y%m') + '/'
            + date_chosen.strftime('%Y%m%d'))
        #confDict= config.gen_confDict()
        if (confDict['SCAN_TYPE'] == 'Stare') | (confDict['SCAN_TYPE'] == 'VAD') | (confDict['SCAN_TYPE'] == 'RHI'):
            scan_type= confDict['SCAN_TYPE']
        else:
            scan_type= 'User'
        mylist= list(path.glob('**/' + scan_type + '*.hpl'))
        if confDict['SCAN_TYPE']=='Stare':
            file_time= [ datetime.datetime.strptime(x.stem,  scan_type
                                                 + "_"
                                                 + confDict['SYSTEM_ID']
                                                 + "_%Y%m%d_%H")
                    for x in mylist]
            # sort files according to time stamp
            file_list = []
            for ii,idx in enumerate(np.argsort(file_time).astype(int)):
                file_list.append(mylist[idx])
            file_time = [ datetime.datetime.strptime(x.stem
                                                     , scan_type
                                                     + "_"
                                                     + confDict['SYSTEM_ID']
                                                     + "_%Y%m%d_%H")
                         for x in file_list]
        elif (confDict['SCAN_TYPE']=='VAD') | (confDict['SCAN_TYPE']=='RHI'):
            file_time= [ datetime.datetime.strptime(x.stem,  scan_type
                                                 + "_"
                                                 + confDict['SYSTEM_ID']
                                                 + "_%Y%m%d_%H%M%S")
                    for x in mylist]
        # sort files according to time stamp
            file_list = []
            for ii,idx in enumerate(np.argsort(file_time).astype(int)):
                file_list.append(mylist[idx])
            file_time = [ datetime.datetime.strptime(x.stem
                                                     , scan_type
                                                     + "_"
                                                     + confDict['SYSTEM_ID']
                                                     + "_%Y%m%d_%H%M%S")
                         for x in file_list]
        else:
            file_time= [ datetime.datetime.strptime(x.stem
                                                 , scan_type
                                                 + x.name[4]
                                                 + "_"
                                                 + confDict['SYSTEM_ID']
                                                 + "_%Y%m%d_%H%M%S")
                    for x in mylist]
        # sort files according to time stamp
            file_list = []
            for ii,idx in enumerate(np.argsort(file_time).astype(int)):
                file_list.append(mylist[idx])
            file_time = [ datetime.datetime.strptime(x.stem
                                                     , scan_type
                                                     + x.name[4]
                                                     + "_"
                                                     + confDict['SYSTEM_ID']
                                                     + "_%Y%m%d_%H%M%S")
                         for x in file_list]
        return hpl_files(file_list, file_time)   

    #def gen_filedict(filename):
    def split_header(string):
        return [x.strip() for x in re.split('[:\=\-]', re.sub('[\n\t]','',string),1)]
    def split_data(string):
        return re.split('\s+', re.sub('\n','',string).strip())
    #switch_str = {True: split_header(line), False: split_data(line)}
    def split_default(string):
        return string
    def switch(case,string):
        return {
            True: hpl_files.split_header(string),
            False: hpl_files.split_data(string)}.get(case, hpl_files.split_default)
  
    def reader_idx(hpl_list,confDict,chunks=False):
        print(hpl_list.time)
        time_file = pd.to_datetime(hpl_list.time)
        time_vec= np.arange(pd.to_datetime(hpl_list.time[0].date()),(hpl_list.time[0]+datetime.timedelta(days = 1))
                            ,pd.to_timedelta(int(confDict['AVG_MIN']), unit = 'm'))
        if chunks == True:
            return [np.where((ii <= time_file)*(time_file < iip1))
                              for ii,iip1 in zip(time_vec[0:-1],time_vec[1::])]
        if chunks == False:
            return np.arange(0,len(hpl_list.time))
    
    def combine_lvl1(hpl_list,confDict,read_idx):
        ds= xr.concat((hpl_files.read_hpl(hpl_list.name[idx],confDict) for ii,idx in enumerate(read_idx))
                          ,dim='time'#, combine='nested'#,compat='identical'
                          ,data_vars='minimal'
                          ,coords='minimal')
        ds.attrs['Title']= confDict['NC_TITLE']
        ds.attrs['Institution']= confDict['NC_INSTITUTION']
        ds.attrs['Contact_person']= confDict['NC_CONTACT_PERSON']
        ds.attrs['Source']= "HALO Photonics Doppler lidar (system_id: " + confDict['SYSTEM_ID']
        ds.attrs['History']= confDict['NC_HISTORY']
        ds.attrs['Conventions']= confDict['NC_CONVENTIONS']
        ds.attrs['Processing_date']= str(pd.to_datetime(datetime.datetime.now())) + ' UTC'
        ds.attrs['Author']= confDict['NC_AUTHOR']
        ds.attrs['Licence']= confDict['NC_LICENCE'] 
        # adjust time variable to double (aka float64)
        ds.time.data.astype(np.float64)
        path= Path(confDict['NC_L1_PATH'] + '/'
                    + (hpl_list.time[0]).strftime("%Y") + '/'
                    + (hpl_list.time[0]).strftime("%Y%m"))
        path.mkdir(parents=True, exist_ok=True)
        path= path / Path(confDict['NC_L1_BASENAME'] + 'v' + confDict['VERSION'] + '_'  + (hpl_list.time[0]).strftime("%Y%m%d")+ '.nc')                                                               
        ds.to_netcdf(path,unlimited_dims={'time':True})
        ds.close()
        return path
    
    def read_hpl(filename, confDict):
        if not filename.exists():
            print("Oops, file doesn't exist!")
        else:
            print('reading file: ' + filename.name)
        with filename.open() as infile:
            #result = {}
            header_info = True
            mheader = {}
            for line in infile:
                if line.startswith("****"):
                    header_info = False
                    ## Adjust header in order to extract data formats more easily
                    ## 1st for 'Data line 1' , i.e. time of beam etc.
                    tmp = [x.split() for x in mheader['Data line 1'].split('  ')]
                    tmp.append(" ".join([tmp[2][2],tmp[2][3]]))
                    tmp.append(" ".join([tmp[2][4],tmp[2][5]]))
                    tmp[0] = " ".join(tmp[0])
                    tmp[1] = " ".join(tmp[1])
                    tmp[2] = " ".join([tmp[2][0],tmp[2][1]])
                    mheader['Data line 1'] = tmp
                    tmp = mheader['Data line 1 (format)'].split(',1x,')
                    tmp.append(tmp[-1])
                    tmp.append(tmp[-1])
                    mheader['Data line 1 (format)'] = tmp
                    ## Adjust header in order to extract data formats more easily
                    ## 2st for 'Data line 2' , i.e. actual data
                    tmp = [x.split() for x in mheader['Data line 2'].split('  ')]
                    tmp[0] = " ".join(tmp[0])
                    tmp[1] = " ".join(tmp[1])
                    tmp[2] = " ".join(tmp[2])
                    tmp[3] = " ".join(tmp[3])
                    mheader['Data line 2'] = tmp
                    tmp = mheader['Data line 2 (format)'].split(',1x,')
                    mheader['Data line 2 (format)'] = tmp
                    ## start counter for time and range gates
                    counter_ii = 0
                    counter_jj = 0
                    continue # stop the loop and continue with the next line

                tmp = hpl_files.switch(header_info,line)

                if header_info == True:
                    try:
                        if tmp[0][0:1] == 'i':
                            tmp_tmp = {'Data line 2 (format)': tmp[0]}
                        else:
                            tmp_tmp = {tmp[0]: tmp[1]}
                    except:
                        if tmp[0][0] == 'f':
                            tmp_tmp = {'Data line 1 (format)': tmp[0]}
                        else:
                            tmp_tmp = {'blank': 'nothing'}
                    mheader.update(tmp_tmp)
                elif (header_info == False) & (counter_ii == 0) & (counter_jj == 0):
#                     if (confDict['SCAN_TYPE'] == 'Stare') | (confDict['SCAN_TYPE'] == 'CSM'):
#                     # this works 99 percent of the time, if no broken files are registered
#                     if (confDict['SCAN_TYPE'] != 'VAD'):
#                         n_o_rays= (len(filename.open().read().splitlines())-17)//(int(mheader['Number of gates'])+1)
#                     else:
#                         n_o_rays= int(mheader['No. of rays in file'])
                    n_o_rays= (len(filename.open().read().splitlines())-17)//(int(mheader['Number of gates'])+1)
                    mbeam = np.recarray((n_o_rays,),
                                        dtype=np.dtype([('time', 'f8')
                                               , ('azimuth', 'f4')
                                               ,('elevation','f4')
                                               ,('pitch','f4')
                                               ,('roll','f4')]))
                    mdata = np.recarray((n_o_rays,int(mheader['Number of gates'])),
                                        dtype=np.dtype([('range gate', 'i2')
                                               , ('velocity', 'f4')
                                               ,('snrp1','f4')
                                               ,('beta','f4')]))
                if (len(tmp) == 5) & (header_info == False):
                    dt=np.dtype([('time', 'f8'), ('azimuth', 'f4'),('elevation','f4'),('pitch','f4'),('roll','f4')])
                    if counter_jj < n_o_rays:
                        mbeam[counter_jj] = np.array(tuple(tmp), dtype=dt)
                    counter_jj = counter_jj+1
                        #mbeam.append(tmp)
                elif (len(tmp) == 4) & (header_info == False):
                    dt=np.dtype([('range gate', 'i2'), ('velocity', 'f4'),('snrp1','f4'),('beta','f4')])
                    mdata[counter_jj-1,counter_ii] = np.array(tuple(tmp), dtype=dt)
                    counter_ii = counter_ii+1
                    if counter_ii == int(confDict['NUMBER_OF_GATES']):
                        counter_ii = 0
        
#         time_tmp= ((pd.to_datetime(datetime.datetime.strptime(mheader['Start time'], '%Y%m%d %H:%M:%S.%f').date())
#                                              +pd.to_timedelta(np.squeeze(mbeam['time']), unit = 'h')).astype(np.int64) / 10**9).values
        time_tmp= pd.to_numeric(pd.to_timedelta(pd.DataFrame(mbeam)['time'], unit = 'h')
                                +pd.to_datetime(datetime.datetime.strptime(mheader['Start time'], '%Y%m%d %H:%M:%S.%f').date())
                               ).values / 10**9
        time_ds= [x+(datetime.timedelta(days=1)).total_seconds()
                    if time_tmp[0]-x>0 else x
                   for x in time_tmp]          
        range_bnds= np.array([mdata['range gate'][0,:] * float(mheader['Range gate length (m)'])
                                                    ,(mdata['range gate'][0,:] + 1.) * float(mheader['Range gate length (m)'])]
                                                    ).T        
        tgint = (2*(range_bnds[0,1]-range_bnds[0,0]) / 3e8).astype('f4')
#         SNR_tmp= np.copy(np.squeeze(mdata['snrp1']))-1
        SNR_tmp= np.copy(mdata['snrp1'])-1
        SNR_tmp[SNR_tmp<=0]= np.nan
        ## calculate SNR in dB
        SNR_dB= 10*np.log10(np.ma.masked_values(SNR_tmp, np.nan)).filled(np.nan)
        ## calculate measurement uncertainty, with consensus indices
        sigma_tmp= proc.hpl2netCDF_client.calc_sigma_single(SNR_dB
                                        ,int(mheader['Gate length (pts)'])
                                        ,int(confDict['PULSES_PER_DIRECTION'])
                                        ,float(mheader['Gate length (pts)'])/tgint/2*float(confDict['SYSTEM_WAVELENGTH'])
                                        ,1.316)
        return xr.Dataset({ 
                          'dv': (['time', 'range']
#                                 , np.squeeze(mdata['velocity'])
                                , mdata['velocity']
                                , {'units': 'm s-1'
                                  ,'long_name' : 'radial velocity of scatterers away from instrument' 
                                  ,'standard_name' : 'doppler_velocity'
                                  ,'comments' : 'A velocity is a vector quantity; the component of the velocity of the scatterers along the line of sight of the instrument where positive implies movement away from the instrument'
                                  ,'_FillValue': -999.
                                  ,'_CoordinateAxes': ['time','range']
                                  }
                                )
                        , 'errdv': (['time', 'range']
                                , sigma_tmp.astype('f4')
                                , {'units': 'm s-1'
                                  ,'long_name' : 'error of Doppler velocity' 
                                  ,'standard_name' : 'doppler_velocity_error'
                                  ,'comments' : 'error of radial velocity calculated from Cramer-Rao lower bound (CRLB)'
                                  ,'_FillValue': -999.
                                  ,'_CoordinateAxes': ['time','range']
                                  }
                                )
                        , 'intensity': (['time', 'range']
#                                         , np.squeeze(mdata['snrp1'])
                                        , mdata['snrp1']                                     
                                        , {'units': '1'
                                           ,'long_name' : 'backscatter intensity: b_int = snr+1, where snr denotes the signal-to-noise-ratio'
                                           ,'standard_name' : 'backscatter_intensity'
                                           ,'comments' : 'backscatter intensity: b_int = snr+1'
                                           ,'_FillValue': -999.
                                           ,'_CoordinateAxes': ['time','range']
                                          }
                                       )
                        , 'beta': (['time', 'range']
#                                    , np.squeeze(mdata['beta'])
                                   , mdata['beta']                               
                                   , {'units': 'm-1 sr-1'
                                     ,'long_name' : 'attenuated backscatter coefficient'
                                     ,'standard_name' : 'volume_attenuated_backwards_scattering_function_in_air'
                                     ,'comments' : 'determined from SNR by means of lidar equation; uncalibrated and uncorrected'  
                                     ,'_FillValue': -999.
                                     ,'_CoordinateAxes': ['time','range']
                                     }
                                   )
                        , 'azi': ('time'
#                                  , np.squeeze(mbeam['azimuth'])
                                 , mbeam['azimuth']                               
                                 , {'units' : 'degree'
                                   ,'long_name' : 'sensor azimuth due reference point'
                                   ,'standard_name' : 'sensor_azimuth_angle'
                                   ,'comments' : 'sensor_azimuth_angle is the horizontal angle between the line of sight from the observation point to the sensor and a reference direction at the observation point, which is often due north. The angle is measured clockwise positive, starting from the reference direction. A comment attribute should be added to a data variable with this standard name to specify the reference direction. A standard name also exists for platform_azimuth_angle, where \"platform\" refers to the vehicle from which observations are made e.g. aeroplane, ship, or satellite. For some viewing geometries the sensor and the platform cannot be assumed to be close enough to neglect the difference in calculated azimuth angle.'
                                   }
                                  )
#                         , 'ele': ('time'
#                                  , mbeam['elevation']                                  
#                                  , {'units' : 'degree'
#                                    ,'long_name' : 'beam direction due elevation'
#                                    ,'standard_name' : 'elevation_angle'
#                                    ,'comments' : 'elevation angle of the beam to local horizone; a value of 90 is directly overhead'
#                                    } 
#                                   )
                        , 'zenith': ('time'
#                                     , 90-np.squeeze(mbeam['elevation'])
                                    , 90-mbeam['elevation']                                 
                                    , {'units' : 'degree'
                                      ,'long_name' : 'beam direction due zenith'
                                      ,'standard_name' : 'zenith_angle'
                                      ,'comments' : 'zenith angle of the beam to the local vertical; a value of zero is directly overhead'
                                      }
                                    )
                        , 'lat': ([]
                                 , np.float32(confDict['SYSTEM_LATITUDE'])
                                 , {'units': 'degrees_north'
                                   ,'long_name': 'latitude'
                                   ,'standard_name': 'latitude'
                                   ,'comments': 'latitude of sensor'
                                   ,'_FillValue': -999.
                                   }
                                 )  
                        , 'lon': ([]
                                 , np.float32(confDict['SYSTEM_LONGITUDE'])
                                 , {'units': 'degrees_east'
                                   ,'long_name': 'longitude'
                                   ,'standard_name': 'longitude'
                                   ,'comments': 'longitude of sensor'
                                   ,'_FillValue': -999.
                                   }
                                 )  
                        , 'zsl': ([]
                                 , np.float32(confDict['SYSTEM_ALTITUDE'])
                                 , {'units': 'm'
                                   ,'comments': 'system altitude above mean sea level'
                                   ,'standard_name': 'altitude'
                                   ,'_FillValue': -999.
                                   }
                                 )
                        , 'wl': ([]
                                , np.float32(confDict['SYSTEM_WAVELENGTH'])
                                , {'units': 'm'
                                  ,'long_name': 'laser center wavelength'
                                  ,'standard_name': 'radiation_wavelength'
                                  ,'_FillValue': -999.
                                  }
                                )
                        , 'pd': ([]
                                , np.float32(confDict['PULS_DURATION'])
                                , {'units': 'seconds'
                                  ,'long_name': 'laser duration'
                                  ,'comments': 'duration of the transmitted pulse pd = 2 dr / c'
                                  ,'_FillValue': -999. 
                                  }
                                )
                        , 'nfft': ([]
                                  , np.float32(confDict['FFT_POINTS'])
                                  , {'units': '1'
                                    ,'long_name': 'number of fft points'
                                    ,'comments': 'according to the manufacturer'
                                    ,'_FillValue': -999.
                                    }
                                   )
                        , 'id': ([]
                                , confDict['SYSTEM_ID']
                                , {'long_name': 'system identification number'}
                                )
                        , 'nrg': ([]
                                , np.float32(mheader['Number of gates'])
                                , {'long_name': 'total number of range gates per ray'
                                  ,'units': '1'
                                  ,'_FillValue': -999.     
                                  }
                                 )
                        , 'lrg': ([]
                                 , np.float32(mheader['Range gate length (m)'])
                                 , {'units' : 'm'
                                   ,'long_name': 'range gate length'
                                   ,'_FillValue': -999.           
                                    }
                                  )
                        , 'nsmpl': ([]
                                   , np.float32(mheader['Gate length (pts)'])
                                   , {'long_name': 'points per range gate'
                                     ,'units': '1'
                                     }
                                     )
                        , 'prf': ([]
                                 , np.float32(confDict['PULS_REPETITION_FREQ'])
                                 , {'units' : 's-1'
                                   ,'long_name': 'pulse repetition frequency'
                                   ,'_FillValue': -999.
                                   }
                                 )
                        , 'npls': ([]
                                , np.float32(confDict['PULSES_PER_DIRECTION'])#[int(mheader['Pulses/ray'])]
                                , {'long_name': 'number of pulses per ray'
                                  ,'units': '1'
                                  ,'_FillValue': -999.
                                  }
                                 )
                        , 'focus': ([]
                                    , np.float32(mheader['Focus range'])
                                    , {'units' : 'm'
                                      ,'long_name': 'telescope focus length'
                                      ,'_FillValue': -999.
                                      }
                                    )
                        , 'resv': ([]
                                  , np.float32(mheader['Resolution (m/s)'])
                                  , {'units' : 'm s-1'
                                    ,'long_name': 'resolution of Doppler velocity'
                                    ,'_FillValue': -999.
                                    }
                                  )
                        , 'nqf': ([], (np.float32(mheader['Gate length (pts)'])/tgint/2).astype('f4')
                                             , {'long_name': 'nyquist frequency'
                                                , 'comments' : 'half of the detector sampling frequency; detector bandwidth'
                                                }
                                     )
                        , 'nqv': ([], (np.float32(mheader['Gate length (pts)'])/tgint/2*np.float32(confDict['SYSTEM_WAVELENGTH'])/2).astype('f4')
                                , {'long_name': 'nyquist velocity'
                                  ,'comments' : 'nq_freq*lambda/2; signal bandwidth'
                                  }
                                 )
                        , 'smplf': ([], np.float32(mheader['Gate length (pts)'])/tgint
                                   , {'long_name': 'sampling frequency'
                                     ,'units': 's-1'
                                     ,'comments' : 'nsmpl / tgint'
                                     }
                                   )
                        , 'resf': ([], (np.float32(mheader['Gate length (pts)'])/tgint/float(confDict['FFT_POINTS'])).astype('f4')
                                   , {'long_name': 'frequency resolution'
                                     ,'units': 's-1'
                                     ,'comments' : 'smplf / nfft'
                                     }
                                   )
                       , 'tgint': ([], tgint
                                   , {'long_name': 'total observation time per range gate'
                                     ,'units': 's'
                                     ,'comments' : 'time window used for time gating the time series of the signal received on the detector: tgint = (2 X) / c, with X = range_bnds[range,1] - range_bnds[range,0]'
                                     }
                                   )
                       , 'range_bnds': (['range','nv']
                                        ,  range_bnds.astype('f4')
                                        , {'units': 'm'
                                          ,'_FillValue' : -999.
                                          }
                                        )
    #                     , 'pitch': ('time', np.squeeze(mbeam['pitch']))
    #                     , 'roll': ('time', np.squeeze(mbeam['roll']))
                          }
                        , coords= {  'time': ( ['time']
                                    , time_ds#.astype(np.float64)
                                    ,{'units': 'seconds since 1970-01-01 00:00:00' 
                                     ,'standard_name': 'Time'
                                     ,'long_name': 'time'
                                     ,'calendar':'gregorian'
                                     ,'_CoordinateAxisType': 'Time'
                                     })
                                     ,'range': (['range']
                                             , ((mdata['range gate'][0,:] + 0.5) * np.float32(mheader['Range gate length (m)'])).astype('f4')
                                             , {'units' : 'm'
                                             ,'long_name': 'line of sight distance towards the center of each range gate'
                                             ,'_FillValue': -999.
                                                }
                                                )
                                   , 'nv': (['nv'],np.arange(0,2).astype(np.int8))
                                  }
                        )

