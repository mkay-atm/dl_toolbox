#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
DWD-Pilotstation software source code file

by Markus Kayser. Non-commercial use only.
'''

import numpy as np
import pandas as pd
import xarray as xr
import datetime

# import packackes used for plotting quicklooks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from matplotlib.ticker import MultipleLocator
from pathlib import Path

# import 
from hpl2netCDF_client.hpl_files.hpl_files import hpl_files
from hpl2netCDF_client.config.config import config
from scipy.linalg import diagsvd

from hpl2netCDF_client.main_proc import process_dataset, write_netcdf
from hpl2netCDF_client.plot_helpers import ql_helper
from hpl2netCDF_client.signal_calc import in_db, CN_est
from hpl2netCDF_client.wind_calc import build_Amatrix, uvw_2_spd, uvw_2_dir, calc_sigma_single, consensus, find_num_dir


### functions used for plotting
def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.
    """
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki, key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1)]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

### function for import of processed files
def import_lvl1(date_chosen, confDict):
    path= Path(confDict['NC_L1_PATH'] + '/'  
                + date_chosen.strftime("%Y") + '/'
                + date_chosen.strftime("%Y%m")
              )

    mylist= list(path.glob('**/' + confDict['NC_L1_BASENAME'] + '*' + date_chosen.strftime("%Y%m%d")+ '*.nc'))
    if len(mylist)>1:
        print('!!!multiple files found!!!, only first is processed!')
    try:
        ds_tmp= xr.open_dataset(mylist[0])
        print('processing lvl1 to lvl2...')
        return ds_tmp
    except IndexError:
        print('empty file list')
        print('unable to continue processing!')
    except FileNotFoundError:
        print('no such file exists: ' + path.name + '... .nc')
        print('unable to continue processing!')
    except:
        print("something went wrong!") 

def import_lvl2(date_chosen, confDict):
    path= Path(confDict['NC_L2_PATH'] + '/'  
                + date_chosen.strftime("%Y") + '/'
                + date_chosen.strftime("%Y%m")
              )

    mylist= list(path.glob('**/' + confDict['NC_L2_BASENAME'] + '*' + date_chosen.strftime("%Y%m%d")+ '*.nc'))
    if len(mylist)>1:
        print('!!!multiple files found!!!, only first is processed!')
    try:
        ds_tmp= xr.open_dataset(mylist[0])
        print('processing lvl2 to quicklooks...')
        return ds_tmp
    except IndexError:
        print('empty file list')
        print('unable to continue processing!')
    except FileNotFoundError:
        print('no such file exists: ' + path.name + '... .nc')
        print('unable to continue processing!')
    except:
        print("something went wrong!")

### the actual processing is done in this class
class hpl2netCDFClient(object):
    def __init__(self, config_dir, cmd, date2proc):
        self.config_dir = config_dir
        self.cmd= cmd
        self.date2proc= date2proc
        
    def display_config_dir(self):
        print('config-file taken from ' + self.config_dir)
        
    def display_configDict(self):
        confDict= config.gen_confDict(url= self.config_dir)
        print(confDict)

            ## do processiong!!



    def dailylvl1(self):
        date_chosen = self.date2proc
        print(date_chosen)
        confDict= config.gen_confDict(url= self.config_dir)
        hpl_list= hpl_files.make_file_list(date_chosen, confDict, url=confDict['PROC_PATH'])
        if not hpl_list.name:
            print('no files found')
        else:
            print('combining files to daily lvl1...')
            print(' ...')
        ## look at the previous and following day for potential files
        # and add to hpl_list
        print('looking at the previous day')
        hpl_listm1 = hpl_files.make_file_list(date_chosen + datetime.timedelta(minutes=-30), confDict, url=confDict['PROC_PATH'])
        print('looking at the following day')
        hpl_listp1 = hpl_files.make_file_list(date_chosen + datetime.timedelta(days=+1, minutes=30), confDict, url=confDict['PROC_PATH'])
        namelist = hpl_list.name
        timelist = hpl_list.time
        #print('check 1')
        if  len(hpl_listm1.time) > 0:
            if date_chosen - hpl_listm1.time[-1] <= datetime.timedelta(minutes=30):
                namelist = [hpl_listm1.name[-1]] + namelist
                timelist = np.array([hpl_listm1.time[-1]] + list(timelist))
                print('adding last file of previous day before')
        #print('check 2')
        if  len(hpl_listp1.time) > 0:
            if hpl_listp1.time[0] - date_chosen  <= datetime.timedelta(days=1, minutes=30):
                namelist = namelist + [hpl_listp1.name[0]]
                timelist = np.array(list(timelist) + [hpl_listp1.time[0]])
                print('adding first file of following day after')
        
        hpl_list = hpl_files(namelist, timelist)
        # print('check 3')
        # read_idx= hpl_files.reader_idx(hpl_list,confDict,chunks=False)
        # print(hpl_list.name)
        nc_name= hpl_files.combine_lvl1(hpl_list, confDict, date_chosen)
        print(nc_name)
        ds_tmp= xr.open_dataset(nc_name)
        print(ds_tmp.info)
        ds_tmp.close()
        
    def lvl2_from_filelist(self, filelist, infile_prefix='XXX_', version_in_filename=False, time_chosen=None):
        """generate file containing wind field time series from list of raw input files

        Args:
            filelist: list of raw input files to be processed
            infile_prefix (optional): prefix of input filenames before date starts. Only number of underscores matter.
                Leave at default value to process files with standard filename convention produced by halo and windcube.
            version_in_filename(optional): include version specified in config to output filename. Defaults to False.
        """

        date_chosen = self.date2proc
        confDict = config.gen_confDict(url=self.config_dir)
        files_hpl = hpl_files.filelist_to_hpl_files(filelist, confDict['SYSTEM'], infile_prefix)
        ds_tmp = hpl_files.combine_lvl1_to_ds(files_hpl, confDict, date_chosen, time_chosen=time_chosen)
        ds_lvl2 = process_dataset(ds_tmp, date_chosen, confDict)
        ds_lvl2.attrs['scan_type'] = confDict['SCAN_TYPE']

        # to output file
        timestamp_out = files_hpl.time[0].strftime("%Y%m%d%H%M")  # set stamp of output file to stamp of first infile
        aux_fn_info = ''
        if version_in_filename:
            aux_fn_info += 'v' + confDict['VERSION'] + '_'
        file_out = Path(confDict['NC_L2_PATH'] + '/'
                        + confDict['NC_L2_BASENAME'] + aux_fn_info + timestamp_out + '.nc')
        print('writing results to {}'.format(file_out))
        write_netcdf(ds_lvl2, file_out, confDict)

    def dailylvl2(self):
        date_chosen = self.date2proc
        confDict= config.gen_confDict(url= self.config_dir)
        
        ds_tmp = import_lvl1(date_chosen, confDict)

        ds_lvl2 = process_dataset(ds_tmp, date_chosen, confDict)

        path= Path(confDict['NC_L2_PATH'] + '/' 
                    + date_chosen.strftime("%Y") + '/'
                    + date_chosen.strftime("%Y%m"))
        path.mkdir(parents=True, exist_ok=True)  
        path= path / Path(confDict['NC_L2_BASENAME'] + 'v' +  confDict['VERSION'] + '_' + date_chosen.strftime("%Y%m%d") + '.nc')

        print(path)
        write_netcdf(ds_lvl2, path, confDict)
        print(path)
        print(ds_lvl2.info)
        ds_lvl2.close()
        ds=xr.open_dataset(path)
        print(ds.info)
        ds.close()

    def lvl2ql(self):
        date_chosen = self.date2proc
        confDict= config.gen_confDict(url= self.config_dir)
        ## Look for UTC_OFFSET in config
        if 'UTC_OFFSET' in confDict:
            time_offset = np.timedelta64(int(confDict['UTC_OFFSET']), 'h') 
            time_delta = int(confDict['UTC_OFFSET'])
        else:
            time_offset = np.timedelta64(0, 'h') 
            time_delta = 0

        ds = import_lvl2(date_chosen, confDict)
            
        fig, axes= plt.subplots(1,1,figsize=(18,12))
        # set figure bachground color, "'None'" means transparent; use 'w' as alternative
        fig.set_facecolor('w')
        ax= axes
        # make spines' linewidth thicker
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

       #load variables and prepare mesh for plotting
        X,Y= np.meshgrid(ds.time.data, ds.height.data)
        U= np.copy(ds.u.data)
        if np.all(np.isnan(U)):
            print('all input is NaN -> check nvrad threshold!')
            print('no quick look  is saved')
        else:
            V= np.copy(ds.v.data)
            WS= np.copy(ds.wspeed.data)
            qwind= np.copy(ds.qwind.data)

            mask= (qwind<1)
            # masked_u = np.ma.masked_where(mask,U)
            # masked_v = np.ma.masked_where(mask,V)
            vel_sq_sum = U**2 + V**2
            qvels = np.sqrt(  vel_sq_sum
                            , out=np.zeros(vel_sq_sum.shape)
                            , where=~np.isnan(vel_sq_sum)) >= 2.5
            qwind = qwind * qvels
            # qwind = qwind * ( np.sqrt(masked_u**2 + masked_v**2) >= 2.5)
            mask= (qwind<1)

            masked_u = np.ma.masked_where(mask,U)
            masked_v = np.ma.masked_where(mask,V)
            masked_WS = np.ma.masked_where(mask,WS)

            # define adjustable and discretized colormap
            # wsmax = max(np.round(masked_WS.max(),-1), 10)
            wsmax = max(np.round(np.nanpercentile(masked_WS.filled(np.nan).flatten(), 95), -1) + 10, 10)
            palette = plt.get_cmap(cmap_discretize(cm.jet,int(wsmax)))
            palette.set_under('white', 1.0)
            # define x-axis values
            # d= pd.to_datetime(ds.time.data[0]).date()
            # dp1=pd.to_datetime(ds.time.data[0]).date()+datetime.timedelta(days=1)
            # dticks= np.arange(d, dp1)
            # print(d, dp1)
            d0 = date_chosen.date()   
            d = datetime.datetime(d0.year, d0.month, d0.day) - datetime.timedelta(hours=time_delta)    
            dp1 =datetime.datetime(d0.year, d0.month, d0.day) + datetime.timedelta(days=1) - datetime.timedelta(hours=time_delta)                                                                     
            print(d, dp1)
            dticks= np.arange(d, dp1, datetime.timedelta(hours=1))

            # plot colored barbs
            clims = [0, wsmax]
            c= ax.barbs(  X.T, Y.T, masked_u, masked_v, masked_WS
                        , clim= clims
                        , pivot='middle'#, flip_barb=True
                        , barb_increments=dict( half=2.5, full=5, flag=25)
                        , sizes=dict(emptybarb=.25, spacing=.1, height=.5, width=.3)
                        , cmap=palette
                        )
            # set x-axis limits
            ax.set_xlim(d,dp1)
            ax.set_aspect('auto')
            # add colorbar and adjust its settings
            cticks = np.linspace(0, wsmax, int(wsmax/5 + 1))
            cbar = fig.colorbar(c, ax=ax, extend='both', pad=0.02,ticks=cticks)
            cbar.set_label( r'$\rm{wind\;speed}\;/\;\rm{m}\,\rm{s}^{-1}$'
                            , rotation=270
                            , fontsize=22
                            , labelpad=30
                            )
            cbar.ax.tick_params(  labelsize=18
                                , length = 0
                                , width = 2
                                , direction= 'in'
                                )
            # set time axis
            # plt.setp(ax, xticks=np.hstack([dticks,dp1+datetime.timedelta(hours=1)]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            # ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))
            # ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, 1)))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=np.mod(range(0-time_delta
                                                                      ,24-time_delta
                                                                      ,6), 24)))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=np.mod(range(0-time_delta
                                                                      ,24-time_delta
                                                                      ,1), 24)))
            # put x-and y-label
            ax.set_xlabel(date_chosen.strftime('%Y-%m-%d') + '\n' + 'time (UTC)', fontsize=22)
            # ax.set_xlabel(d.strftime('%Y-%m-%d') + '\n' + ('time (UTC)', 'time (UTC{:+03d})'.format(time_delta))[abs(np.sign(time_delta))]
            #                             , fontsize=22)
            ax.set_ylabel(r'$\rm{height}\;/\;\rm{m}$', fontsize=22)

            # find maximum height
            ind = np.unravel_index(qwind==1, WS.shape)
            hmax= ds.height.data[np.sum(ind[1],axis=0).astype(bool)][-1]
            # set dynamic ticks
            if hmax>=6000:
                delmajor= 2000
                delnom= 4
            elif (hmax>2000)*(hmax<6000):
                delmajor= 1000
                delnom= 4
            else:
                delmajor= 500
                delnom=5
            ax.yaxis.set_major_locator(MultipleLocator(delmajor))
            ax.yaxis.set_minor_locator(MultipleLocator(delmajor//delnom))
            # set y-axis limits
            ylims_1 = [0, (np.round(hmax,-2)+100)]
            ax.set_ylim(ylims_1)
            # plot smaller than 2.5 as sticks
            U= np.copy(ds.u.data)
            V= np.copy(ds.v.data)
            WS= np.copy(ds.wspeed.data)
            qwind= np.copy(ds.qwind.data)

            mask= (qwind<1)
            # masked_u = np.ma.masked_where(mask,U)
            # masked_v = np.ma.masked_where(mask,V)
            vel_sq_sum = U**2 + V**2
            qvels = np.sqrt( vel_sq_sum
                            , out=999. * np.ones(vel_sq_sum.shape)
                            , where=~np.isnan(vel_sq_sum)
                            ) < 2.5 

            qwind = qwind * qvels
            mask= (qwind<1)

            masked_u = np.ma.masked_where(mask,U)
            masked_v = np.ma.masked_where(mask,V)
            masked_WS = np.ma.masked_where(mask,WS)

            c= ax.barbs(  X.T, Y.T
                        , masked_u
                        , masked_v
                        , masked_WS
                        , clim= [0,wsmax]
                        , rounding=False
                        , pivot='middle'#, flip_barb=True
                        , barb_increments=dict( half=.25, full=5, flag=25)
                        , sizes=dict(emptybarb=.25, spacing=.1, height=0., width=0.)
                        , cmap=palette
                        )   

            # set tick parameters
            ax.tick_params(axis='both', labelsize=18, length = 34, width = 2. ,pad=7.78
                        , which='major', direction= 'in', top=True, right=True)
            ax.tick_params(axis='both', labelsize=18, length = 23, width = 1.
                        , which='minor', direction= 'in', top=True, right=True)
            ## save wind quicklook           
            path= Path(confDict['NC_L2_QL_PATH'] + '/' + date_chosen.strftime('%Y') + '/' + date_chosen.strftime('%Y%m'))
            print('saving wind retrieval quicklook as... \n ... '
                    + str('{}/' + confDict['NC_L2_BASENAME'] + 'ql_' + date_chosen.strftime('%Y%m%d') 
                    + '_' + str(confDict['AVG_MIN']) + 'min' + '.png').format(path)
                 )
            path.mkdir(parents=True,exist_ok=True)
            fig.savefig(str('{}/' + confDict['NC_L2_BASENAME'] + 'ql_' + date_chosen.strftime('%Y%m%d') + '_' + str(confDict['AVG_MIN']) + 'min' + '.png').format(path)
                        ,transparent=False, bbox_inches='tight')

    def bckql(self):
        date_chosen = self.date2proc
        confDict= config.gen_confDict(url= self.config_dir)
        if 'UTC_OFFSET' in confDict:
            time_offset = np.timedelta64(int(confDict['UTC_OFFSET']), 'h') 
            time_delta = int(confDict['UTC_OFFSET'])
        else:
            time_offset = np.timedelta64(0, 'h') 
            time_delta = 0

        ds = import_lvl1(date_chosen, confDict)
        condi = ('relative_beta' not in list(ds.keys())) and ('beta' not in list(ds.keys()))
        beta_max, time_mean, range_vec, elevation, vmin, vmax = ql_helper(ds, confDict)

        fig, axes= plt.subplots(1,1,figsize=(18, 12))
        # set figure bachground color, "'None'" means transparent; use 'w' as alternative
        fig.set_facecolor('w')
        ax= axes
        # make spines' linewidth thicker
        ax.spines['left'].set_linewidth(2)
        ax.spines['right'].set_linewidth(2)
        ax.spines['top'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)

        #load variables and prepare mesh for plotting
        X, Y = np.meshgrid(mdates.date2num(pd.to_datetime(time_mean)), range_vec*np.sin(np.pi/180 * (elevation.mean())))                          
        Z = np.copy(beta_max)
        if np.all(np.isnan(Z)):
            print('all input is NaN -> check nvrad threshold!')
            print('no quick look  is saved')
        else:
            mask= np.isnan(Z)
            masked_Z = np.ma.masked_where(mask, Z)
            # print(list(ds.keys()))
            if condi:
                print(vmin, vmax)
                str_bck = r'$\rm{CNR}\;/\;\rm{dB}$'
                c_temp = ax.pcolormesh(  X.T, Y.T
                                    , masked_Z
                                    , cmap=cm.gnuplot2
                                    , vmin=vmin, vmax=vmax
                                    )
            else:
                str_bck = r'$\rm{attenuated\;backscatter}\;/\;\rm{m}^{-1}\,\rm{sr}^{-1}$'
                c_temp = ax.pcolormesh(  X.T, Y.T
                                    , masked_Z
                                    , cmap=cm.gnuplot2
                                    , norm=mcolors.LogNorm(vmin=vmin, vmax=vmax)
                                    )
            
            cbar = fig.colorbar(c_temp, ax=ax, extend='both', pad=0.01)
            cbar.set_label(str_bck, rotation=270,
                        fontsize=22, labelpad=37)
            # cbar.ax.tick_params(labelsize=27, length = 0, width = 2, dir17ection= 'in')
            cbar.ax.tick_params(which='major', direction= 'out', length = 14, width = 2, labelsize=22)
            cbar.ax.tick_params(which='minor', direction= 'out', length = 8, width = 2, labelsize=22)
            # set x-axis limits
            # define x-axis values
            # d= pd.to_datetime(ds.time.data[0]).date()
            # dp1=pd.to_datetime(ds.time.data[0]).date()+datetime.timedelta(days=1)
            d0 = date_chosen.date()   
            d = datetime.datetime(d0.year, d0.month, d0.day) - datetime.timedelta(hours=time_delta)    
            dp1 =datetime.datetime(d0.year, d0.month, d0.day) + datetime.timedelta(days=1) - datetime.timedelta(hours=time_delta)                                                                     
            print(d, dp1)
            dticks= np.arange(d, dp1, datetime.timedelta(hours=1))
            # dticks= np.arange(d, dp1+datetime.timedelta(hours=2), datetime.timedelta(hours=1))
            ax.set_xlabel(date_chosen.strftime('%Y-%m-%d') + '\n' + 'time (UTC)', fontsize=22)
            # ax.set_xlabel(d.strftime('%Y-%m-%d') + '\n' + ('time (UTC)', 'time (UTC{:+03d})'.format(time_delta))[abs(np.sign(time_delta))]
            #                             , fontsize=22)
            ax.set_ylabel(r'$\rm{height}\;/\;\rm{m}$', fontsize=22)
            ax.set_xlim(d, dp1)
            ax.set_aspect('auto')
            # set time axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            # ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0
            #                                                           ,24
            #                                                           ,6)))
            # ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0
            #                                                           ,24
            #                                                           ,1)))
            ax.xaxis.set_major_locator(mdates.HourLocator(byhour=np.mod(range(0-time_delta
                                                                      ,24-time_delta
                                                                      ,6), 24)))
            ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=np.mod(range(0-time_delta
                                                                      ,24-time_delta
                                                                      ,1), 24)))

            # find maximum height
            hmax= range_vec[-1]*np.sin(np.pi/180 * (elevation.mean()))
            # set dynamic ticks
            if hmax>=6000:
                delmajor= 2000
                delnom= 4
            elif (hmax>2000)*(hmax<6000):
                delmajor= 1000
                delnom= 4
            else:
                delmajor= 500
                delnom=5
            ax.yaxis.set_major_locator(MultipleLocator(delmajor))
            ax.yaxis.set_minor_locator(MultipleLocator(delmajor//delnom))
            # set y-axis limits
            ylims_1 = [0, (np.round(hmax,-2)+100)]
            ax.set_ylim(ylims_1)
            # ax.set_xlim(d, dp1)
            # set tick parameters
            ax.tick_params(axis='both', labelsize=18, length = 34, width = 2. ,pad=7.78
                        , which='major', direction= 'in', top=True, right=True)
            ax.tick_params(axis='both', labelsize=18, length = 23, width = 1.
                        , which='minor', direction= 'in', top=True, right=True)
            ## save wind quicklook
            path= Path(confDict['NC_L2_QL_PATH'] + '/' + date_chosen.strftime('%Y') + '/' + date_chosen.strftime('%Y%m') + '/')
            print('saving backscatter quicklook as... \n ... '
                    + str('{}' + confDict['NC_L2_BASENAME'] + 'bck_ql_' + date_chosen.strftime('%Y%m%d') 
                    + '_cycle_max' + '.png').format(path)
                 )
            path.mkdir(parents=True,exist_ok=True)
            fig.savefig(str('{}/' + confDict['NC_L2_BASENAME'] + 'bck_ql_' + date_chosen.strftime('%Y%m%d') + '_cycle_max' + '.png').format(path)
                        ,transparent=False, bbox_inches='tight')

    def nrtlvl1(self):
        # get configuration
        confDict= config.gen_confDict(url= self.config_dir)
        # timy, wimy, wobbly stuff
        # account for possible optional utc offset (!!!recheck this, I might have the sign wrong!!!)
        if 'UTC_OFFSET' in confDict:
            time_delta = int(confDict['UTC_OFFSET'])
        else:
            time_delta = 0
        time_chosen = self.date2proc + datetime.timedelta(hours=time_delta)
        delt = int(confDict['AVG_MIN'])
        date_chosen = datetime.datetime(time_chosen.year, time_chosen.month, time_chosen.day)
        start, end = time_chosen - datetime.timedelta(minutes=delt+3), time_chosen + datetime.timedelta(minutes=3)
        
        # get daily file list
        hpl_list= hpl_files.make_file_list(date_chosen, confDict, url=confDict['PROC_PATH'])
        if not hpl_list.name:
            print('no files found')
        else:
            print('combining files to daily lvl1...')
            print(' ...')
        # reduce to time window of interest
        namelist = list(map( hpl_list.name.__getitem__
                           , np.where((np.array(hpl_list.time) >= start.replace(tzinfo=None)) & (np.array(hpl_list.time) <= end.replace(tzinfo=None)))[0]
                           )
                           )
        timelist = list(map( hpl_list.time.__getitem__
                           , np.where((np.array(hpl_list.time) >= start.replace(tzinfo=None)) & (np.array(hpl_list.time) <= end.replace(tzinfo=None)))[0]
                           )
                           )
        ## look at the previous and following day for potential files
        # and add to hpl_list
        if (time_chosen.hour < 1):
            print('looking at the previous day')
            hpl_listm1 = hpl_files.make_file_list(date_chosen + datetime.timedelta(minutes=-30), confDict, url=confDict['PROC_PATH'])
            if  len(hpl_listm1.time) > 0:
                if date_chosen - hpl_listm1.time[-1] <= datetime.timedelta(minutes=33):
                    namelist = [hpl_listm1.name[-1]] + namelist
                    timelist = [hpl_listm1.time[-1]] + timelist
                    print('adding last file of previous day before')
        if (time_chosen.hour >= 23):    
            print('looking at the following day')
            hpl_listp1 = hpl_files.make_file_list(date_chosen + datetime.timedelta(days=+1, minutes=30), confDict, url=confDict['PROC_PATH'])
            if  len(hpl_listp1.time) > 0:
                if hpl_listp1.time[0] - date_chosen  <= datetime.timedelta(days=1, minutes=33):
                    namelist = namelist + [hpl_listp1.name[0]]
                    timelist = timelist + [hpl_listp1.time[0]]
                    print('adding first file of following day after')
        # reduce "again" to time window of interest
        # print(timelist)
        namelist = list(map( namelist.__getitem__
                           , np.where((np.array(timelist) >= start.replace(tzinfo=None)) & (np.array(timelist) <= end.replace(tzinfo=None)))[0]
                           )
                           )
        timelist = list(map( timelist.__getitem__
                           , np.where((np.array(timelist) >= start.replace(tzinfo=None)) & (np.array(timelist) <= end.replace(tzinfo=None)))[0]
                           )
                           )
        
        print(namelist, timelist)
        # finalize hpl_list object
        hpl_list = hpl_files(namelist, timelist)
        # read_idx= hpl_files.reader_idx(hpl_list,confDict,chunks=False)
        # combine l1 files to single file
        nc_name= hpl_files.combine_lvl1(hpl_list, confDict, date_chosen, time_chosen)
        print(nc_name)
        ds_tmp= xr.open_dataset(nc_name)
        print(ds_tmp.info)
        ds_tmp.close()      

    def rmlvl1(self):
        confDict= config.gen_confDict(url= self.config_dir)
        if 'UTC_OFFSET' in confDict:
            time_delta = int(confDict['UTC_OFFSET'])
        else:
            time_delta = 0
        time_chosen = self.date2proc + datetime.timedelta(hours=time_delta)
        path= Path(confDict['NC_L1_PATH']) / Path(confDict['NC_L1_BASENAME'] + 'v' + confDict['VERSION'] + '_'  + time_chosen.strftime("%Y%m%d%H%M")+ '.nc')
        print(path)
        try:
            path.unlink()
        except:
            print('no such file exists: ' + path.name + '... .nc')

    def nrtlvl2(self):
        # get configuration
        confDict= config.gen_confDict(url= self.config_dir)
        # timy, wimy, wobbly stuff
        # account for possible optional utc offset (!!!recheck this, I might have the sign wrong!!!)
        if 'UTC_OFFSET' in confDict:
            time_delta = int(confDict['UTC_OFFSET'])
        else:
            time_delta = 0
        time_chosen = self.date2proc + datetime.timedelta(hours=time_delta)
        delt = int(confDict['AVG_MIN'])
        date_chosen = datetime.datetime(time_chosen.year, time_chosen.month, time_chosen.day)
        # read L1 netCDF
        path= Path(confDict['NC_L1_PATH'])

        mylist= list(path.glob('**/' + confDict['NC_L1_BASENAME'] + '*' + time_chosen.strftime("%Y%m%d%H%M")+ '*.nc'))
        print(mylist[0])
        if len(mylist)>1:
            print('!!!multiple files found!!!, only first is processed!')
        try:
            ds_tmp= xr.open_dataset(mylist[0])
        except:
            print('no such file exists: ' + path.name + '... .nc')
        if not ds_tmp:
            print('unable to continue processing!')
        else:
            print('processing lvl1 to lvl2...')

        ## do processiong!!

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
        elevation= 90-ds_tmp.zenith.data
        azimuth= ds_tmp.azi.data[elevation < 89] % 360
        time_ds = ds_tmp.time.data[elevation < 89]
        dv= ds_tmp.dv.data[elevation < 89]
        snr= ds_tmp.intensity.data[elevation < 89]-1
        beta= ds_tmp.beta.data[elevation < 89]

        height= ds_tmp.range.data*np.sin(np.nanmedian(elevation[elevation < 89])*np.pi/180)
        width= ds_tmp.range.data*2*np.cos(np.nanmedian(elevation[elevation < 89])*np.pi/180)
        height_bnds= ds_tmp.range_bnds.data
        height_bnds[:,0]= np.sin(np.nanmedian(elevation[elevation < 89])*np.pi/180)*(height_bnds[:,0])
        height_bnds[:,1]= np.sin(np.nanmedian(elevation[elevation < 89])*np.pi/180)*(height_bnds[:,1])

        # define time chunks
        ## Look for UTC_OFFSET in config
        if 'UTC_OFFSET' in confDict:
            time_delta = int(confDict['UTC_OFFSET'])
        else:
            time_delta = 0
            
        start_dt = (pd.to_datetime(time_chosen - datetime.timedelta(minutes=int(confDict['AVG_MIN']))) - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
        end_dt = (pd.to_datetime(time_chosen) - pd.Timestamp("1970-01-01")) / pd.Timedelta('1s')
        
        if len(time_ds) != 0:
            time_bnds = np.array([[  int(pd.to_datetime(time_ds[0]).replace(tzinfo=datetime.timezone.utc).timestamp())
                                   , int(pd.to_datetime(time_ds[-1]).replace(tzinfo=datetime.timezone.utc).timestamp())]
                                ]).T
        else:
            time_bnds = np.array([[start_dt, end_dt]]).T
                                       
        time_start = time_bnds[-1, :]
        calc_idx = np.array([np.arange(len(time_ds))])

        # compare n_gates in lvl1-file and confdict
        if n_gates != dv.shape[1]:
            print('Warning: number of gates in config does not match lvl1 data!')
            n_gates= dv.shape[1]
            print('number of gates changed to ' + str(n_gates))

        # infer number of directions
            # don't forget to check for empty calc_idx

        # UVW = np.where(np.zeros((len(calc_idx),n_gates,3)),np.nan,np.nan)
        UVW = np.full((1, n_gates, 3), np.nan)
        UVWunc = np.full((1, n_gates, 3), np.nan)
        SPEED = np.full((1, n_gates), np.nan)
        SPEEDunc = np.full((1, n_gates), np.nan)
        DIREC = np.full((1, n_gates), np.nan)
        DIRECunc = np.full((1, n_gates), np.nan)
        R2 = np.full((1, n_gates), np.nan)
        CN = np.full((1, n_gates), np.nan)
        n_good = np.full((1, n_gates), np.nan)
        SNR_tot = np.full((1, n_gates), np.nan)
        BETA_tot = np.full((1, n_gates), np.nan)
        SIGMA_tot = np.full((1, n_gates), np.nan)

        if len(time_ds) != 0:
            print('nrt L2 processing...')

            indicator, n_rays, azi_mean, azi_edges = find_num_dir(n_rays, calc_idx, azimuth, 0)
            # azimuth[azimuth>azi_edges[0]]= azimuth[azimuth>azi_edges[0]]-3
            # azi_edges[0]= azi_edges[0]-360
            r_phi = 360/(n_rays)/2 
            if ~indicator:
                print('some issue with the data', n_rays, len(azi_mean), time_start[0])
            else:

                VR = dv
                SNR = snr
                BETA = beta
                azi = azimuth
                ele = elevation           

                VR_CNSmax = np.full((len(azi_mean),n_gates), np.nan)
                VR_CNSunc = np.full((len(azi_mean),n_gates), np.nan)
                # SNR_CNS= np.full((len(azi_mean),n_gates), np.nan)
                BETA_CNS = np.full((len(azi_mean),n_gates), np.nan)
                SIGMA_CNS = np.full((len(azi_mean),n_gates), np.nan)
                # azi_CNS= np.full((len(azi_mean),n_gates), np.nan)
                ele_cns = np.full((len(azi_mean),), np.nan)

                for ii, azi_i in enumerate(azi_mean):
                    # azi_idx = (azi>=azi_edges[ii])*(azi<azi_edges[ii+1])
                    azi_idx = (np.mod(360-np.mod(np.mod(azi-azi_i, 360)-r_phi, 360), 360)<=2*r_phi)
                    ele_cns[ii] = np.median(ele[azi_idx])
                    ## calculate consensus average
                    VR_CNSmax[ii,:], idx_tmp, VR_CNSunc[ii,:] = consensus(VR[azi_idx]
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
                    SIGMA_CNS[ii,:] = np.ma.divide( np.sqrt( np.nansum( np.where( idx_tmp
                                                                                , sigma_tmp**2
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
                # NVRAD[0, :] = (~np.isnan(VR_CNSmax)).sum(axis=0)
                n_good[0, :] = n_good_kk
                V_r=np.ma.masked_where( (np.isnan(VR_CNSmax)) #& (np.tile(n_good_kk, (azi_mean.shape[0], 1)) < 4)
                                    , VR_CNSmax).T[..., None]
                mask_V_in = (np.isnan(VR_CNSmax)) | (np.tile(n_good_kk, (azi_mean.shape[0], 1)) < 4)
                V_in=np.ma.masked_where( mask_V_in, VR_CNSmax)
                A = build_Amatrix(azi_mean, ele_cns)
                # A[abs(A)<1e-3] = 0
                A_r = np.tile( A,  (VR_CNSmax.shape[1], 1, 1))
                A_r_MP = np.tile( np.linalg.pinv(A), (VR_CNSmax.shape[1],1,1))
                A_r_MP_T = np.einsum('...ij->...ji', A_r_MP)
                SIGMA_r = np.ma.masked_where(np.isnan(VR_CNSmax), SIGMA_CNS).T


                condi = np.isnan(VR_CNSmax)
                A = np.round(build_Amatrix(azi_mean, ele_cns), 6)
                U, S, Vh = [], [], []
                for c_nn in condi.T:
                    u, s, vh = np.linalg.svd( np.ma.masked_where( np.tile(c_nn, (3, 1)).T
                                                                , A).filled(0)
                                            , full_matrices=True)

                    U.append(u)
                    S.append(np.linalg.pinv(diagsvd(s,u.shape[0],vh.shape[0])))
                    Vh.append(vh)
                U, S, Vh = np.array(U), np.array(S), np.array(Vh)


                U_T = np.einsum('...ij->...ji', U)
                Vh_T = np.einsum('...ij->...ji', Vh)
                K1 = np.nansum((U_T * V_in.T[:, None, :]), axis=2)[..., None]
                K2 = np.einsum('...ik,...kj->...ij', S, K1)

                V_k = np.einsum('...ik,...kj->...ij', Vh_T, K2)
                UVW[0, ...] = np.squeeze(V_k)
                # plausible winds can only be calculated, when the at least three LOS measurements are present
                UVW[0, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4 , :] = np.squeeze(np.full((3,1), np.nan))

                UVWunc[0, ...]= abs(np.einsum('...ii->...i', np.sqrt((A_r_MP @ np.apply_along_axis(np.diag, 1, SIGMA_r**2) @ A_r_MP_T).astype(complex)).real))
                # plausible winds can only be calculated, when the at least three LOS measurements are present
                UVWunc[0, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4 , :] = np.squeeze(np.full((3,1), np.nan))

                V_r_est = A_r @ V_k
                ss_e = ((V_r-V_r_est)**2).sum(axis = 1)
                ss_t = ((V_r-V_r.mean(axis=1)[:, None, :])**2).sum(axis = 1)
                R2[0, :] = np.squeeze(1 - ss_e/ss_t)
                # R2[0, :] = 1 - (1 - R2[0, :]) * (np.sum(~np.isnan(VR_CNSmax.T), axis=1)-1)/(np.sum(~np.isnan(VR_CNSmax.T), axis=1)-2)  
                # sqe = ((V_r_est-V_r_est.mean(axis=1)[:, None, :])**2).sum(axis = 1)
                # sqt = ((V_r-V_r.mean(axis=1)[:, None, :])**2).sum(axis = 1)
                # R2[0, :] = np.squeeze(sqe/sqt)
                R2[0, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan


                mask_A = np.tile( mask_V_in.T[..., None], (1, 1, 3))
                # A_r_m = np.ma.masked_where( mask_A, A_r)
                A_r_T = np.einsum('...ij->...ji', A_r)
                Spp =  np.apply_along_axis(np.diag, 1, 1/np.sqrt(np.einsum('...ii->...i', A_r_T @ A_r)))
                Z = np.ma.masked_where( mask_A, A_r @ Spp)
                CN[0, :] =  np.squeeze(np.array([CN_est(X) for X in Z]))
                # CN[0, :] = np.array([CN_est(X) for X in A_r_m])
                CN[0, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

                SPEED[0, :], SPEEDunc[0, :] = np.vstack([np.fromiter(uvw_2_spd(val, unc).values(), dtype=float) for val, unc in zip(UVW[0, ...], UVWunc[0, ...])]).T
                DIREC[0, :], DIRECunc[0, :] = np.vstack([np.fromiter(uvw_2_dir(val, unc).values(), dtype=float) for val, unc in zip(UVW[0, ...], UVWunc[0, ...])]).T

        ## do quality control
        speed = np.copy(SPEED)
        errspeed = np.copy(SPEEDunc)
        wdir = np.copy(DIREC)
        errwdir = np.copy(DIRECunc)
        r2 = np.copy(R2)
        cn = np.copy(CN)
        nvrad = np.copy(n_good)
        u = np.copy(UVW[:,:,0])
        v = np.copy(UVW[:,:,1])
        w = np.copy(UVW[:,:,2])
        erru = np.copy(UVWunc[:,:,0])
        errv = np.copy(UVWunc[:,:,1])
        errw = np.copy(UVWunc[:,:,2])

        qspeed = (~np.isnan(SPEED))#*(abs(w)<.3*np.sqrt(np.nanmedian(u)**2+np.nanmedian(v)**2))
        r2[np.isnan(R2)] = -999.
        qr2 = r2>=float(confDict['R2_THRESHOLD'])
        cn[np.isnan(CN)] = +999.
        qcn = (cn >= 0) & (cn<=float(confDict['CN_THRESHOLD']))
        nvrad[np.isnan(n_good)] = -999
        qnvrad = nvrad>=int(confDict['N_VRAD_THRESHOLD'])

        qwind = qspeed & qnvrad & qcn & qr2
        qu = np.copy(qspeed)
        qv = np.copy(qspeed)
        qw = np.copy(qspeed)
        # qspeed = qspeed & qnvrad
        # qspeed = qwind
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
                configuration += dd + '=' + confDict[dd]+'\n'
        if 'BLINDZONE_GATES' in confDict:
            NN = int(confDict['BLINDZONE_GATES'])
        else:
            NN = 0
        ds_lvl2= xr.Dataset({ 'config': ([]
                                        , configuration
                                        , {'standard_name' : 'configuration_file'}
                                        )
                            , 'wspeed': (['time', 'height']
                                        , np.float32(speed[:, NN:])
                                        , {'units': 'm s-1'
                                        , 'comments': 'Scalar wind speed (amount of vector)'
                                        , 'standard_name' : 'wind_speed'
                                        , 'long_name' : 'Wind Speed' 
                                        ,'_FillValue' : -999.
                                        }
                                        )
                            , 'qwind': (['time', 'height']
                                        , qwind[:, NN:].astype(np.int8)
                                        , {'comments' : str('quality flag 0 or 1 for u, v, w, wspeed, wdir and corresponding errors,'
                                            + '(good quality data = WHERE( R2 > ' + confDict['R2_THRESHOLD'] 
                                            + 'AND CN < ' + confDict['CN_THRESHOLD'] 
                                            + 'AND NVRAD > '+ confDict['N_VRAD_THRESHOLD']  + ')')
                                        ,'long_name': 'wind_quality_flag'
                                        ,'_FillValue': np.array(-128).astype(np.int8)
                                        ,'flag_values': np.arange(0,2).astype(np.int8)
                                        ,'flag_meanings': 'quality_bad quality_good'
                                        }
                                        )
                            , 'qu': (['time', 'height']
                                        , qu[:, NN:].astype(np.int8)
                                        , {'comments' : 'quality flag 0 or 1 for u and corresponding error'
                                        ,'long_name': 'quality_flag_u'
                                        ,'_FillValue': np.array(-128).astype(np.int8)
                                        ,'flag_values': np.arange(0,2).astype(np.int8)
                                        ,'flag_meanings': 'quality_bad quality_good'
                                        }
                                        )
                            , 'qv': (['time', 'height']
                                        , qv[:, NN:].astype(np.int8)
                                        , {'comments' : 'quality flag 0 or 1 for v and corresponding error'
                                        ,'long_name': 'quality_flag_v'
                                        ,'_FillValue': np.array(-128).astype(np.int8)
                                        ,'flag_values': np.arange(0,2).astype(np.int8)
                                        ,'flag_meanings': 'quality_bad quality_good'
                                        }
                                        )
                            , 'qw': (['time', 'height']
                                        , qw[:, NN:].astype(np.int8)
                                        , {'comments' : 'quality flag 0 or 1 for w and corresponding error'
                                        ,'long_name': 'quality_flag_w'
                                        ,'_FillValue': np.array(-128).astype(np.int8)
                                        ,'flag_values': np.arange(0,2).astype(np.int8)
                                        ,'flag_meanings': 'quality_bad quality_good'
                                        }
                                        )                         
                            ,'errwspeed': (['time', 'height']
                                        , np.float32(errspeed[:, NN:])
                                        , {'units': 'm s-1'
                                        , 'standard' : 'wind_speed_uncertainty'
                                        , 'long_name' : 'Wind Speed Uncertainty'
                                        , '_FillValue' : -999.
                                        }
                                        )
                            ,'u': (['time', 'height']
                                        , np.float32(u[:, NN:])
                                        , {'units': 'm s-1'
                                        , 'comments': '"Eastward indicates" a vector component which is positive when directed eastward (negative westward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
                                        , 'standard_name' : 'eastward_wind'
                                        , 'long_name' : 'Zonal Wind'
                                        , '_FillValue' : -999.
                                        }
                                        )
                            ,'erru': (['time', 'height']
                                        , np.float32(erru[:, NN:])
                                        , {'units': 'm s-1'
                                        , 'standard_name' : 'eastward_wind_uncertainty'
                                        , 'long_name' : 'Zonal Wind Uncertainty'
                                        ,'_FillValue' : -999.
                                        }
                                        )
                            ,'v': (['time', 'height']
                                        , np.float32(v[:, NN:])
                                        , {'units': 'm s-1'
                                        , 'comments': '"Northward indicates" a vector component which is positive when directed northward (negative southward). Wind is defined as a two-dimensional (horizontal) air velocity vector, with no vertical component'
                                        , 'standard_name' : 'northward_wind'
                                        , 'long_name' : 'Meridional Wind'
                                        ,'_FillValue' : -999.
                                        }
                                        )
                            ,'errv': (['time', 'height']
                                        , np.float32(errv[:, NN:])
                                        , {'units': 'm s-1'
                                        , 'standard_name' : 'northward_wind_uncertainty'
                                        , 'long_name' : 'Meridional Wind Uncertainty'
                                        , '_FillValue' : -999.
                                        }
                                        )
                            ,'w': (['time', 'height']
                                        , np.float32(w[:, NN:])
                                        , {'units': 'm s-1'
                                        , 'comments': 'Vertical wind component, positive when directed upward (negative downward)'
                                        , 'standard_name' : 'upward_air_velocity'
                                        , 'long_name' : 'Upward Air Velocity'
                                        , '_FillValue' : -999.
                                        }
                                        )
                            ,'errw': (['time', 'height']
                                        , np.float32(errw[:, NN:])
                                        , {'units': 'm s-1'
                                        , 'standard_name' : 'upward_air_velocity_uncertainty'
                                        , 'long_name' : 'Upward Air Velocity Uncertainty'
                                        ,'_FillValue' : -999.
                                        }
                                        )
                            ,'wdir': (['time', 'height']
                                        , np.float32(wdir[:, NN:])
                                        , {'units': 'degree'
                                        , 'comments': 'Wind direction'
                                        , 'standard_name' : 'wind_from_direction'
                                        , 'long_name' : 'Wind Direction'
                                        ,'_FillValue' : -999.
                                        }
                                        )
                            ,'errwdir': (['time', 'height']
                                        , np.float32(errwdir[:, NN:])
                                        , {'units': 'degree'
                                        , 'standard_name' : 'wind_direction_uncertainty'
                                        , 'long_name' : 'Wind Direction Uncertainty'
                                        , '_FillValue' : -999.
                                        }
                                        )
                            ,'r2': (['time', 'height']
                                        , np.float32(r2[:, NN:])
                                        , {'comments' : 'coefficient of determination - provides a measure of how well observed radial velocities are replicated by the model used to determine u,v,w wind components from the measured line of sight radial velocities'
                                            , 'long_name': 'coefficient of determination'
                                            , 'standard_name': 'coefficient_of_determination'
                                            , 'units': '1'
                                            , '_FillValue': -999.
                                        }
                                        )  
                            ,'nvrad': (['time', 'height']
                                        , np.float32(nvrad[:, NN:])
                                        , { 'comments' : 'number of (averaged) radial velocities used for wind calculation'
                                            , 'long_name': 'number of radial velocities'
                                            , 'standard_name': 'no_radial_velocities'
                                            , 'units': '1'
                                            , '_FillValue': -999.
                                        }
                                        )  
                            ,'cn': (['time', 'height']
                                        , np.float32(cn[:, NN:])
                                        , {'comments' : 'condition number - provides a measure for the degree of collinearity among the Doppler velocity measurements used for the retrieval of the wind variables (u,v,w,speed,direction).'

                                           , 'standard_name': 'condition_number'
                                           , 'long_name': 'Condition Number'
                                           , 'units': '1'
                                           , '_FillValue': -999.
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
                                        ,'comments': 'Altitude of sensor above mean sea level'
                                        ,'standard_name': 'altitude'
                                        ,'_FillValue': -999.
                                        }
                                        )
                            ,'time_bnds': (['time','nv']
                                            ,time_bnds.T.astype(np.float64)
                                            ,{'units': 'seconds since 1970-01-01 00:00:00 UTC'                                         
                                            }
                                            )
                            ,'height_bnds': (['height','nv']
                                            # ,np.array([(np.arange(0,n_gates)
                                            #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                            #             ,((np.arange(0,n_gates) + 1.)
                                            #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                            #             ]).T
                                            , np.float32(height_bnds[NN:, :])
                                            ,{'units': 'm'                                         
                                            }
                                            )
                            ,'frequency': ([]
                                , np.float32(299792458 / float(confDict['SYSTEM_WAVELENGTH']))
                                , {  'units': 'Hz'
                                    ,'comments': 'lidar operating frequency'
                                    ,'long_name': 'instrument_frequency'
                                    ,'_FillValue': -999.
                                  }
                                )                
                            ,'vert_res': ([]
                                        , np.float32(np.diff(height).mean())
                                        ,{ 'units': 'm'
                                          ,'comments': 'Calculated from pulse wdth and beam elevation'
                                          ,'long_name': 'Vertical_resolution_measurement'
                                          ,'_FillValue': -999.
                                         }
                                        )                
                            ,'hor_width': (['height']
                                            # ,np.array([(np.arange(0,n_gates)
                                            #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                            #             ,((np.arange(0,n_gates) + 1.)
                                            #             * float(confDict['RANGE_GATE_LENGTH'])*np.sin(np.nanmedian(elevation)*np.pi/180))
                                            #             ]).T
                                        , np.float32(width[NN:])
                                        ,{ 'units': 'm'
                                          ,'comments': 'Calculated from beam elevation and height'
                                          ,'standard_name': 'horizontal_sample_width'
                                          ,'_FillValue': -999.
                                         }
                                        )
                            }
                            , coords= { 'height': (['height']
                                                    # ,((np.arange(0,n_gates)+.5)*int(confDict['RANGE_GATE_LENGTH'])
                                                    # *np.sin(np.nanmedian(elevation)*np.pi/180))
                                                    , np.float32(height[NN:])
                                                    ,{'units': 'm'
                                                    ,'standard_name': 'height'
                                                    ,'comments': 'vertical distance from sensor to centre of range gate'
                                                    ,'bounds': 'height_bnds'
                                                    }
                                                )
                                        , 'time': ( ['time']
                                                , time_start.astype(np.float64)
                                                , {  'units': 'seconds since 1970-01-01 00:00:00'
                                                    ,'comments': 'Timestamp at the end of the averaging interval'
                                                    ,'standard_name': 'time'
                                                    ,'long_name': 'Time'
                                                    ,'calendar':'gregorian'
                                                    ,'bounds': 'time_bnds'
                                                    ,'_CoordinateAxisType': 'Time'
                                            })
                                        ,'nv': (['nv'], np.arange(0,2).astype(np.int8))
                                        }
                        )

        ds_lvl2.attrs['title']= confDict['NC_TITLE']
        ds_lvl2.attrs['institution']= confDict['NC_INSTITUTION']
        ds_lvl2.attrs['site_location']= confDict['NC_SITE_LOCATION']
        ds_lvl2.attrs['source']= confDict['NC_SOURCE']
        ds_lvl2.attrs['instrument_type']= confDict['NC_INSTRUMENT_TYPE']
        ds_lvl2.attrs['instrument_mode']= confDict['NC_INSTRUMENT_MODE']
        if 'NC_INSTRUMENT_FIRMWARE_VERSION' in confDict:
            ds_lvl2.attrs['instrument_firmware_version']= confDict['NC_INSTRUMENT_FIRMWARE_VERSION']
        else:
            ds_lvl2.attrs['instrument_firmware_version']= 'N/A'

        if 'NC_INSTRUMENT_ID' in confDict:
            ds_lvl2.attrs['instrument_id']= confDict['NC_INSTRUMENT_ID']
        else:
            ds_lvl2.attrs['instrument_id']= 'N/A'    
        ds_lvl2.attrs['instrument_contact']= confDict['NC_INSTRUMENT_CONTACT']
        # ds_lvl2.attrs['Source']= "HALO Photonics Doppler lidar (production number: " + confDict['SYSTEM_ID'] + ')'
        # ds_lvl2.attrs['history']= confDict['NC_HISTORY']
        ds_lvl2.attrs['conventions']= confDict['NC_CONVENTIONS']
        ds_lvl2.attrs['processing_date']= str(pd.to_datetime(datetime.datetime.now())) + ' UTC'
        # ds_lvl2.attrs['author']= confDict['NC_AUTHOR']
        # ds_lvl2.attrs['licence']= confDict['NC_LICENCE']
        ds_lvl2.attrs['data_policy']= confDict['NC_DATA_POLICY']

        # attributes for operational use of netCDFs, see E-Profile wind profiler netCDF version 1.7
        if 'NC_WIGOS_STATION_ID' in confDict:
            ds_lvl2.attrs['wigos_station_id']= confDict['NC_WIGOS_STATION_ID']
        else:
            ds_lvl2.attrs['wigos_station_id']= 'N/A'

        if 'NC_WMO_ID' in confDict:
            ds_lvl2.attrs['wmo_id']= confDict['NC_WMO_ID']
        else:
            ds_lvl2.attrs['wmo_id']= 'N/A'

        if 'NC_PI_ID' in confDict:
            ds_lvl2.attrs['principal_investigator']= confDict['NC_PI_ID']
        else:
            ds_lvl2.attrs['principal_investigator']= 'N/A'

        if 'NC_INSTRUMENT_SERIAL_NUMBER' in confDict:
            ds_lvl2.attrs['instrument_serial_number']= confDict['NC_INSTRUMENT_SERIAL_NUMBER']
        else:
            ds_lvl2.attrs['instrument_serial_number']= ' '

        ds_lvl2.attrs['history']= confDict['NC_HISTORY'] + ' version ' + confDict['VERSION'] + ' on ' + str(pd.to_datetime(datetime.datetime.now())) + ' UTC'

        if 'OPERATIONAL' in confDict:
            if bool(int(confDict['OPERATIONAL'])):
                # Blocking statur, only important for operational use
                ds_lvl2.attrs['references']= confDict['NC_REFERENCES'] #'Doppler lidar PPI-based retrieval, see VAD'
                ds_lvl2.attrs['data_blocking_status']= confDict['NC_DATA_BLOCKING_STATUS']
                # set all uncertainties to NaN-Value
                for item in ['erru', 'errv', 'errw', 'errwspeed', 'errwdir']:
                    ds_lvl2[item] = ds_lvl2[item].where(ds_lvl2[item] == -999., other=-999.)
        #             ds_lvl2[item] = ds_lvl2[item].where(np.isnan(ds_lvl2[item]), other=np.nan)
        ds_lvl2.attrs['comments']= confDict['NC_COMMENTS']

        path= Path(confDict['NC_L2_PATH'])
        path.mkdir(parents=True, exist_ok=True)  
        path= path / Path(confDict['NC_L2_BASENAME'] + 'v' +  confDict['VERSION'] + '_' + time_chosen.strftime("%Y%m%d%H%M") + '.nc')

        print(path)
        # compress variables
        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in np.hstack([ds_lvl2.data_vars,ds_lvl2.coords])}

        # ds_lvl2.time.attrs['units'] = ('seconds since 1970-01-01 00:00:00', 'seconds since 1970-01-01 00:00:00 {:+03d}'.format(time_delta))[abs(np.sign(time_delta))]
        ds_lvl2.time.encoding['units'] = ('seconds since 1970-01-01 00:00:00', 'seconds since 1970-01-01 00:00:00 {:+03d}'.format(time_delta))[abs(np.sign(time_delta))]
        # ## add configuration used to create the file
        # configuration = """"""
        # for dd in confDict:
        #     if not dd in ['PROC_PATH', 'NC_L1_PATH', 'NC_L2_PATH', 'NC_L2_QL_PATH']:
        #         configuration += dd + '=' + confDict[dd]+'\n'
        # ds_lvl2.attrs['File_Configuration']= configuration
        ## save file to path
        ds_lvl2.to_netcdf(path, unlimited_dims={'time':True}, encoding=encoding)
        print(path)
        print(ds_lvl2.info)
        ds_lvl2.close()
        ds=xr.open_dataset(path)
        print(ds.info)
        ds.close()
