#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
DWD-Pilotstation software source code file

by Markus Kayser. Non-commercial use only.
'''

import numpy as np
import pandas as pd
import xarray as xr
import re
import datetime
import itertools as it
import operator as op
import warnings

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
from scipy.linalg import diagsvd, svdvals

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


### functions for the retrieval
def build_Amatrix(azimuth_vec,elevation_vec):
    return np.einsum('ij -> ji',
                    np.vstack(
                        [
                        np.sin((np.pi/180)*(azimuth_vec))*np.sin((np.pi/180)*(90-elevation_vec))
                        ,np.cos((np.pi/180)*(azimuth_vec))*np.sin((np.pi/180)*(90-elevation_vec))
                        ,np.cos((np.pi/180)*(90-elevation_vec))
                        ])
                    )
#Note: numpy's lstsq-function uses singular value decomposition already
def VAD_retrieval(azimuth_vec,elevation_vec,Vr):
#     u, s, vh = np.linalg.svd(build_Amatrix(azimuth_vec,elevation_vec), full_matrices=True)
#     A = build_Amatrix(azimuth_vec,elevation_vec)
#     return vh.transpose() @ np.linalg.pinv(diagsvd(s,u.shape[0],vh.shape[0])) @ u.transpose() @ Vr
    return np.linalg.lstsq(build_Amatrix(azimuth_vec,elevation_vec), Vr, rcond=-1)

def uvw_2_spd(uvw,uvw_unc):
    if (np.isfinite(uvw[0]) * np.isfinite(uvw[1])) & (~np.isnan(uvw[0]) * ~np.isnan(uvw[1])):
        speed = np.sqrt((uvw[0])**2.+(uvw[1])**2.)
    else:
        speed = np.nan   
    if speed > 0:
        df_du = uvw[0] * 1/speed
        df_dv = uvw[1] * 1/speed
        error = np.sqrt((df_du*uvw_unc[0])**2 + (df_dv*uvw_unc[1])**2)
    else:
        error = np.nan
    return {'speed': speed, 'error': error} 

def uvw_2_dir(uvw,uvw_unc):
    if (np.isfinite(uvw[0]) * np.isfinite(uvw[1])) & (~np.isnan(uvw[0]) * ~np.isnan(uvw[1])):
        wdir = np.arctan2(uvw[0],uvw[1])*180/np.pi + 180
    else:
        wdir = np.nan    
    if np.isfinite(wdir):
        error = (180/np.pi)*np.sqrt((uvw[0]*uvw_unc[0])**2 + (uvw[1]*uvw_unc[1])**2)/(uvw[0]**2 + uvw[1]**2)       
    else:
        error = np.nan
    return {'wdir': wdir, 'error': error}

def calc_sigma_single(SNR_dB,Mpts,nsmpl,BW,delta_v):
    'calculates the instrument uncertainty: SNR in dB!'
    # SNR_dB = np.ma.masked_values(SNR_dB, np.nan)
    # SNR_dB = np.ma.masked_invalid(SNR_dB)
    SNR= 10**(SNR_dB/10)
    bb = np.sqrt(2.*np.pi)*(delta_v/BW)
    alpha = SNR/bb
    Np = Mpts*nsmpl*SNR
        
    # a1 = (2.*np.sqrt(np.sqrt(np.pi)/alpha)).filled(np.nan)
    # a1 = 2.*np.sqrt( np.divide(np.sqrt(np.pi), alpha
    #                , out=np.full((alpha.shape), np.nan)
    #                , where=alpha!=0)
    #                )
    a1 = 2.*(np.sqrt(np.ma.divide(np.sqrt(np.pi), alpha)))#.filled(np.nan)
    a2 = (1+0.16*alpha)#.filled(np.nan)
    a3 = np.ma.divide(delta_v, np.sqrt(Np))#.filled(np.nan) ##here, Cramer Rao lower bound!
    SNR= SNR#.filled(np.nan)
    sigma = np.ma.masked_where(  SNR_dB > -5
                                , (a1*a2*a3).filled(np.nan)
                                ).filled(a3.filled(np.nan))

    # sigma= np.where(~np.isnan(SNR)
    #                 ,np.where(SNR_dB <= -5., (a1*a2*a3), a3)
    #                 ,np.nan)
    return sigma
    
def log10_inf(x):
    result = np.zeros(x.shape)
    result[x>0] = np.log10(x[x>0])
    result[x<0] = -float('Inf')
    return result
# def in_dB(x):
#     return np.real(10*log10_inf(np.float64(x)))
    
def consensus_mean(Vr,SNR,CNS_range,CNS_percentage,SNR_threshold):
    if SNR_threshold < 0:
        SNR_threshold= 10**(SNR_threshold/10)
    with np.errstate(divide='ignore', invalid='ignore'):
        Vr_X= np.expand_dims(Vr, axis=0)
        AjdM = (abs(np.einsum('ij... -> ji...',Vr_X)-Vr_X)<CNS_range).astype(int)
        SUMlt= np.sum(AjdM, axis=0)
        X= np.sum( np.einsum('il...,lj... -> ij...',AjdM
                 , np.where(  np.sum(SNR>SNR_threshold, axis=0)/SNR.shape[0] >= CNS_percentage/100
                            , np.apply_along_axis(np.diag, 0,(SUMlt/np.sum(SNR>SNR_threshold, axis=0) >= CNS_percentage/100).astype(int))
                            ,0))#[:,:,kk]
                 , axis=0)#[:,kk]
        W= np.where(X>0,X/np.sum(X, axis=0),np.nan)
        mask= np.isnan(W)
        Wm= np.ma.masked_where(mask,W)
        Xm= np.ma.masked_where(mask,Vr)
        OutCNS=Xm*Wm
        MEAN= OutCNS.sum(axis=0).filled(np.nan)
        diff= Vr- MEAN
        mask_m= abs(diff)<3
        Vr_m = np.ma.masked_where(~mask_m,Vr)
        # Vr_m.mean(axis=0).filled(np.nan)
        IDX= mask_m
        UNC= (Vr_m.max(axis=0)-Vr_m.min(axis=0)).filled(np.nan)/2
        return MEAN, IDX, UNC
        
def consensus_median(Vr,SNR,CNS_range,CNS_percentage,SNR_threshold):
    if SNR_threshold < 0:
        SNR_threshold= 10**(SNR_threshold/10)
    with np.errstate(divide='ignore', invalid='ignore'):
        Vr_X= np.expand_dims(Vr, axis=0)
        AjdM = (abs(np.einsum('ij... -> ji...',Vr_X)-Vr_X)<CNS_range).astype(int)
        SUMlt= np.sum(AjdM, axis=0)
        X= np.sum(np.einsum('il...,lj... -> ij...',AjdM
                  ,np.where(np.sum(SNR>SNR_threshold, axis=0)/SNR.shape[0] >= CNS_percentage/100
                            ,np.apply_along_axis(np.diag, 0,(SUMlt/np.sum(SNR>SNR_threshold, axis=0) >= CNS_percentage/100).astype(int))
                            ,0))#[:,:,kk]
                  , axis=0)#[:,kk]
        W= np.where(X>0,X/np.sum(X, axis=0),np.nan)
        mask= np.isnan(W)
        Wm= np.ma.masked_where(mask,W)
        Xm= np.ma.masked_where(mask,Vr)
        OutCNS=Xm*Wm
        MEAN= OutCNS.sum(axis=0).filled(np.nan)

        diff= Vr- MEAN
        diff= np.ma.masked_values(diff, np.nan)
        mask_m= (abs(diff)<3)*(~np.isnan(diff))
        Vr_m= np.ma.masked_where(~mask_m,Vr)
        MEAN= np.ma.median(Vr_m, axis =0).filled(np.nan)
        IDX= ~np.isnan(W)
        UNC= (Vr_m.max(axis=0)-Vr_m.min(axis=0)).filled(np.nan)/2
        return MEAN, IDX, UNC       

###############################################################################################
# functions used to identify single cycles
###############################################################################################
def process(lst,mon):
    # Guard clause against empty lists
    if len(lst) < 1:
        return lst

    # use an object here to work around closure limitations
    state = type('State', (object,), dict(prev=lst[0], n=0))

    def grouper_proc(x):
        if mon==1:
            if x < state.prev:
                state.n += 1
        elif mon==-1:
            if x > state.prev:
                state.n += 1
        state.prev = x
        return state.n

    return { k: list(g) for k, g in it.groupby(lst, grouper_proc) }

def get_cycles(lst,mon):
    ll= 0
    res= {}
    for key, lst in process(lst,int(np.median(np.sign(np.diff(np.array(lst)))))).items():
        # print(key,np.arange(ll,ll+len(lst)),lst)
        id_tmp= np.arange(ll,ll+len(lst))
        ll+=  len(lst)
        res.update( { key:{'indices': list(id_tmp), 'values': lst} } )
    return res
###############################################################################################

def grouper(iterable, n, fillvalue=None):
    '''Collect data into fixed-length chunks or blocks'''
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue=fillvalue)
# def calc_node_degree(Vr,CNS_range):
#     '''takes masked array as input'''
#     f_abs_pairdiff = lambda x,y: op.abs(op.sub(x,y))<CNS_range
#     with np.errstate(invalid='ignore'):
#         return np.array(list(grouper(it.starmap(f_abs_pairdiff,((it.permutations(Vr.filled(np.nan),2)))),Vr.shape[0]-1))).sum(axis=1)

def calc_node_degree(Vr,CNS_range,B, metric='l1norm'):
    '''takes masked array as input'''
    if metric == 'l1norm':
        f_abs_pairdiff = lambda x,y: op.abs(op.sub(x,y))<CNS_range
    if metric == 'l1norm_aa':
        f_abs_pairdiff = lambda x,y: op.sub(B,op.abs(op.sub(op.abs(op.sub(x,y)),B)))<CNS_range
    with np.errstate(invalid='ignore'):
        return np.array(list(grouper(it.starmap(f_abs_pairdiff,((it.permutations(Vr.filled(np.nan),2)))),Vr.shape[0]-1))).sum(axis=1)    

def diff_aa(x,y,c):
    '''calculate aliasing independent differences'''
    return (c-abs(abs(x-y)-c))    
    
# def consensus(Vr,SNR,CNS_range,CNS_percentage,SNR_threshold):
def consensus(Vr,SNR,BETA,CNS_range,CNS_percentage,SNR_threshold,B):
    '''
    consensus(Vr,SNR,CNS_range,CNS_percentage,SNR_threshold)
        Calculate consensus average:
        --> row-wise, calculate the arithmetic mean using only the members of the most likely cluster
            , i.e. the cluster with the maximum number of edges, if the total number of valid clusters is greater than a specified limit.
    
        Parameters
        ----------
        Vr(time,height) : array_like (intended for using numpy array)
            n-dimensional array representing a signal.
        SNR(time,height) : array_like (intended for using numpy array)
            n-dimensional array of the same dimension as x representing the signal to noise ratio.
        CNS_range(scalar) : scalar value, i.e. 0-dimensional
            scalar value giving the radius for neighboring values in Vr.
        CNS_percentage(scalar) : scalar value, i.e. 0-dimensional
            scalar value stating the minimum percentage for the relative number of valid clusters compared to the totaö number of clusters.
        SNR_threshold(scalar) : scalar value, i.e. 0-dimensional
            scalar value giving the lower bounded threshold of the signal to noise threshold.
            
        order : {'Vr', 'SNR', 'CNS_range', 'CNS_percentage', 'SNR_threshold'}
        
        Returns
        -------
        (MEAN, IDX, UNC) --> [numpy array, boolean array, numpy array] 
            MEAN - consensus average of array, ...
            IDX - index of values used for the consensus, ...
            UNC - standard deviation of centered around the consensus average of array, ...
            ...for each row
        
        Dependencies
        ------------
        functions : check_if_db(x), in_mag(snr), filter_by_snr(vr,snr,snr_threshold)
        
        Notes
        -----
        All array inputs must have the same dimensions, namely (time,height).
        If the input SNR is already given in dB, do NOT filter the input SNR in advance for missing values, because filtering will be done during the calculation
        Translation between dB and magnitude can be done with the functions "in_dB(x)" and "in_mag(x)".
        This function implicitely uses machine epsilon (np.float16) for the numerical value of 0, see "filter_by_snr(vr,snr,snr_threshold)".
    '''
    condi_0 = SNR > 0
    if SNR_threshold == 0:
        condi_snr = condi_0
    else:
        condi_snr = (10*np.log10(SNR.astype(complex)).real > SNR_threshold) & (BETA > 0)

    Vr_m = np.ma.masked_where( ~condi_snr, Vr)
    condi_vr = (abs(Vr_m.filled(-999.)) <= B)
    Vr_m = np.ma.masked_where( ~condi_vr, Vr_m)
    ### calculate the number of points within the consensusrange
    ## easy-to-understand way
    # SUMlt= 1 + np.sum(
    #             np.einsum(  'ij...,ik...-> ij...'
    #                         , (abs(np.einsum('ij... -> ji...', Vr_m[None,...]) - Vr_m[None,...]) < CNS_range).filled(False).astype(int)
    #                         , np.apply_along_axis(np.diag, 0, condi_vr).astype(int))
    #             , axis=0) - (condi_vr).astype(int)
    ## performance strong way using iterators
    SUMlt= 1 + calc_node_degree(Vr_m, CNS_range, B, metric='l1norm') 

    Vr_maxim= np.ma.masked_where( ~((100*np.max(SUMlt, axis=0)/CNS_percentage >= condi_vr.sum(axis=0)) & (condi_vr.sum(axis=0) >= Vr.shape[0]/100*60.))
                                    # ~((100*np.max(SUMlt, axis=0)/condi_vr.sum(axis=0) >= CNS_percentage) & (100*condi_vr.sum(axis=0)/Vr.shape[0] > 60.))
                                    , Vr_m[-(np.argmax(np.flipud(SUMlt),axis=0)+1), np.arange(0,SUMlt.shape[1])]
                                    # , Vr_m[np.argmax(SUMlt,axis=0), np.arange(0,SUMlt.shape[1])]
                                )
    mask_m= abs(Vr_m.filled(999.) - Vr_maxim.filled(-999.)) < CNS_range
    Vr_m= np.ma.masked_where(~(mask_m), Vr_m.filled(-999.)) 
    MEAN= Vr_m.sum(axis=0).filled(np.nan)/np.max(SUMlt, axis=0)
    IDX= mask_m
    UNC= np.nanstd(Vr_m-MEAN.T, axis=0)
    UNC[np.isnan(MEAN)]= np.nan

    ### memory and time efficient option
    # SUMlt= 1 + calc_node_degree(Vr_m, CNS_range, B, metric='l1norm') 

    ### this code is more efficient, but less intuitive and accounts for one-time velocity folding
    #SUMlt= 1 + calc_node_degree(Vr_m, CNS_range, B, metric='l1norm_aa')        
    # Vr_maxim= np.ma.masked_where( ~((100*np.max(SUMlt, axis=0)/condi_vr.sum(axis=0) >= CNS_percentage) & (100*condi_vr.sum(axis=0)/Vr.shape[0] > 60.))
    #                                 , Vr_m[-(np.argmax(np.flipud(SUMlt),axis=0)+1), np.arange(0,SUMlt.shape[1])]
    #                                 # , Vr_m[np.argmax(SUMlt,axis=0), np.arange(0,SUMlt.shape[1])]
    #                             )
    # mask_m= diff_aa(Vr_m, V_max, B) < 3
    # Vr_m = np.ma.masked_where((mask_m), Vr).filled(Vr-np.sign(Vr-V_max)*2*B*np.heaviside(abs(Vr-V_max)-B, 1))
    # Vr_m = np.ma.masked_where(~(mask_m), Vr_m)        
    # MEAN= Vr_m.mean(axis=0).filled(np.nan)
    # IDX= mask_m
    # UNC= np.nanstd(Vr_m-MEAN.T, axis=0)
    # UNC[np.isnan(MEAN)]= np.nan  

    return np.round(MEAN, 4), IDX, UNC  
    
    
def check_if_db(x):
    '''
    check_if_db(X)
        Static method for checking if the input is in dB
        Parameters
        ----------
        x : array or scalar
            representing the signal to noise of a lidar signal.

        Returns
        -------
        bool
            stating wether the input "likely" in dB.

        Notes
        -----
        The method is only tested empirically and therefore not absolute.    
    '''
    return np.any(x<-1)   

def filter_by_snr(x,snr,snr_threshold):
    '''
    filter_by_snr(X,SNR,SNR_threshold)
        Masking an n-dimensional array (X) according to a given signal to noise ratio (SNR) and specified threshold (SNR_threshold).
    
        Parameters
        ----------
        x : array_like (intended for using numpy array)
            n-dimensional array representing a signal.
        snr : array_like (intended for using numpy array)
            n-dimensional array of the same dimension as x representing the signal to noise ratio.
        snr_threshold : scalar value, i.e. 0-dimensional
            scalar value giving the lower bounded threshold of the signal to noise threshold.
            
        order : {'x', 'snr', 'snr_threshold'}
        
        Returns
        -------
        masked_array, i.e. [data, mask]
            Masked numpy array to be used in further processing. 
        
        Dependencies
        ------------
        functions : check_if_db(x), in_mag(snr)
        
        Notes
        -----
        If the input SNR is already given in dB, do NOT filter the input SNR in advance for missing values.
        Translation between dB and magnitude can be done with the functions "in_dB(x)" and "in_mag(x)".
        This functions uses machine epsilon (np.float16) for the numerical value of 0.
    '''
    
    if check_if_db(snr)==True:
        print('SNR interpreted as dB')
        print(snr.min(),snr.max())
        snr= in_mag(snr)
    if check_if_db(snr_threshold)==True:
        print('SNR-threshold interpreted as dB')
        snr_threshold= in_mag(snr_threshold)
    snr_threshold+= np.finfo(np.float32).eps
    return np.ma.masked_where(~(snr>snr_threshold), x)

def in_db(x):
    '''
    in_db(X)
        Calculates dB values of a given input (X). The intended input is the signal to noise ratio of a Doppler lidar.
    
        Parameters
        ----------
        x : array_like (intended for using numpy array) OR numerical scalar
            n-dimensional array
        
        Returns
        -------
        X in dB
            
        Dependencies
        ------------
        functions : check_if_db(x)
                 
        Notes
        -----
        If the input X is already given in dB, X is returned without further processing.
        Please, do NOT filter the input in advance for missing values.
        This functions uses machine epsilon (np.float32) for the numerical value of 0.
    '''
    
    if check_if_db(x)==True:
        print('Input already in dB')
        return x
    else: 
        epsilon_val=  np.finfo(np.float32).eps
        if np.ma.size(x)==0:
            print('0-dimensional input!')
        else:
            if np.ma.size(x)>1:
                x[x<=0]= epsilon_val
                return 10*np.log10(np.ma.masked_where((x<= epsilon_val), x)).filled(10*np.log10(epsilon_val))
            else:
                if x<=0:
                    x= epsilon_val
                return 10*np.log10(np.ma.masked_where((x<= epsilon_val), x)).filled(10*np.log10(epsilon_val))   
def in_mag(x):
    '''
    in_mag(X)
        Calculates the magnitude values of a given dB input (X). The intended input is the signal to noise ratio of a Doppler lidar.
    
        Parameters
        ----------
        x : array_like (intended for using numpy array) OR numerical scalar
            n-dimensional array
        
        Returns
        -------
        X in magnitude
            
        Dependencies
        ------------
        functions : check_if_db(x)
                 
        Notes
        -----
        If the input X is already given in magnitde, X is returned without further processing.
        Please, do NOT filter the input in advance for missing values.
        This functions uses machine epsilon (np.float32) for the numerical value of 0.
    '''    
    if check_if_db(x)==False:
        print('Input already in magnitude')
        return x
    else: 
        epsilon_val=  np.finfo(np.float32).eps
        if np.ma.size(x)==0:
            print('0-dimensional input!')
        else:
            if np.ma.size(x)>1:
                res= 10**(x/10)
                res[res<epsilon_val]= epsilon_val
                return res
            else:
                res= 10**(x/10)
                if res<=epsilon_val:
                    res= epsilon_val
                return res

def CN_est(X):
    Fill_Val = 0
    X_f = X.filled(Fill_Val)
    if np.all(X_f == 0):
        return np.inf
    else:
        max_val = svdvals(X_f).max()
        min_val = svdvals(X_f).min()
        if min_val == 0:
            return np.inf
        else:
            return max_val/min_val

def check_num_dir(n_rays,calc_idx,azimuth,idx_valid):
    h, be = np.histogram(np.mod(azimuth[calc_idx[idx_valid]],360), bins=2*n_rays, range=(0, 360))
    counts = np.sum(np.r_[h[-1], h[:-1]].reshape(-1, 2), axis=1) # rotate and sum
    edges = np.r_[np.r_[be[-2], be[:-2]][::2], be[-2]]         # rotate and skip
    kk_idx= counts >= 3
    return kk_idx, np.arange(0,360,360//n_rays), edges

def find_num_dir(n_rays,calc_idx,azimuth,idx_valid):
    if np.all(check_num_dir(n_rays,calc_idx,azimuth,idx_valid)[0]):
        return np.all(check_num_dir(n_rays,calc_idx,azimuth,idx_valid)[0]), n_rays, check_num_dir(n_rays,calc_idx,azimuth,idx_valid)[1], check_num_dir(n_rays,calc_idx,azimuth,idx_valid)[2]
    elif ~np.all(check_num_dir(n_rays,calc_idx,azimuth,idx_valid)[0]):
        if n_rays > 4:
            print('number of directions to high...try' + str(n_rays//2) + '...instead of ' + str(n_rays))
            return find_num_dir(n_rays//2,calc_idx,azimuth,idx_valid)
        elif n_rays < 4:
            print('number of directions to high...try' + str(4) + '...instead' )
            return find_num_dir(4,calc_idx,azimuth,idx_valid)
        else:
            print('not enough valid directions!-->skip non-convergent time windows' )
            return np.all(check_num_dir(n_rays,calc_idx,azimuth,idx_valid)[0]), n_rays, check_num_dir(n_rays,calc_idx,azimuth,idx_valid)[1], check_num_dir(n_rays,calc_idx,azimuth,idx_valid)[2]
###################################################################################################
## helper functions for plotting
def pbdist_alt(x_i,x_ip1,L):
    return ((x_ip1-x_i) - L*np.rint((x_ip1-x_i) / L))

def ql_helper(ds, confDict):
    # define limits for the range of backscatter values
    if confDict['SYSTEM'].lower() == 'halo':
        vmin, vmax= 1e-7, 1e-4
            
    if confDict['SYSTEM'].lower() == 'windcube':
        vmin, vmax= 1e-8, 1e-6
        if ('relative_beta' not in list(ds.keys())) and ('beta' not in list(ds.keys())):
            print('no backscatter data in file, plot CNR instead')
            ds['relative_beta'] = ds.cnr
            vmin, vmax= -40, 10

    # use equal names        
    try:
        ds.elevation
        ds = ds.rename({'azimuth': 'azi', 'relative_beta': 'beta'})
    except:
        ds['elevation'] = 90 - ds.zenith
              
    # check if cycles need to be identified    
    if len(np.unique(ds.azi.round() % 360))<4:
        condi = False
        azi = ds.azi.data
        beta = ds.beta.data
        time = ds.time
        if len(list(ds.range.shape)) < 2:
            range_vec =  ds.range.data
        else:
            range_vec =  ds.range.data[0]
        elevation = ds.elevation.data
    else:
        condi = True
        idt = np.where(ds.elevation.data<89)[0]
        time = ds.time[idt]
        azi = ds.azi.data[idt]
        beta = ds.beta.data[idt]
        if len(list(ds.range.shape)) < 2:
            range_vec =  ds.range.data
        else:
            range_vec =  ds.range.data[idt][0]
        elevation = ds.elevation.data[idt]
    # prepare data for plotting   
    # indentify maxima of each cycle
    if condi:
        #         data = np.mod(azi-azi[0], 360)
#         index = np.where(abs(np.diff(data)) > 93)[0]
#         ## old method
#         # cycles = get_cycles(data, int(np.median(np.sign(np.diff(np.array(data)))))) 
#         ## new method
#         cycles = {}
#         start = 0
#         for ii,ind in enumerate(index):
#             if ind == index[-1]:
#                 cycles.update( { ii: {'indices': np.arange(start, len(azi)), 'values': azi[start:len(azi)]} })
#                 # print('finished cycling!')
#             else:
#                 cycles.update( { ii: {'indices': np.arange(start, ind+1), 'values': azi[start:ind+1]} })
#                 start = ind+1
#         df= pd.DataFrame.from_dict(cycles, orient='index')
#         df['indices'].apply(lambda row: len(row)).median()


#         Z = beta
#         mask= (np.isnan(Z)) | (Z==-999.)
#         masked_Z = np.ma.masked_where(mask, Z)
#         beta_max = np.empty((df.__len__(), beta.shape[1]))
#         time_mean = np.empty((df.__len__()))

#         for ii in range(df.__len__()):
#             time_mean[ii] = time[df['indices'][ii]].mean(dim='time')
#             beta_max[ii] = np.max(Z[df['indices'][ii]], axis=0)
        Z = beta
        mask= (np.isnan(Z)) | (Z==-999.)
        masked_Z = np.ma.masked_where(mask, Z)
        # id_condi = np.round(pbdist_alt(azi[:-1].round() % 360, azi[1:].round() % 360, 180)) < 0
        id_condi = np.round(azi[:-1].round() % 360 - azi[1:].round() % 360) >= 180
        idx = np.where(np.hstack([True, id_condi]))[0]
        beta_max = np.full((sum(id_condi), beta.shape[1]), np.nan)
        time_mean = np.empty((sum(id_condi),))
        for ii, (start, end) in enumerate(zip(idx[:-1], idx[1:])):
            if end-start > 3:
                time_mean[ii] = time[start:end].mean().data
                beta_max[ii] = Z[start:end].max(axis=0)
            else:
                time_mean[ii] = time[start:end].mean().data
                continue

    else:
        Z = beta
        mask= (np.isnan(Z)) | (Z==-999.)
        beta_max = Z
        time_mean =  time.data
        
    return beta_max, time_mean, range_vec, elevation, vmin, vmax
                    
        
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
    @staticmethod
    def lvl2vad_standard(ds_tmp, date_chosen, confDict):
        # read lidar parameters
        n_rays= int(confDict['NUMBER_OF_DIRECTIONS'])
                    # number of gates
        n_gates= int(confDict['NUMBER_OF_GATES'])
                    # number of pulses used in the data point aquisition
        n= ds_tmp.prf.data
                    # number of points per range gate
        M= ds_tmp.nsmpl.data
                    # half of detector bandwidth in velocity space
        B= ds_tmp.nqv.data
                    
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
            time_offset = np.timedelta64(int(confDict['UTC_OFFSET']), 'h') 
            time_delta = int(confDict['UTC_OFFSET'])
        else:
            time_offset = np.timedelta64(0, 'h') 
            time_delta = 0
            
        time_vec= np.arange(date_chosen - datetime.timedelta(hours=time_delta)
                ,date_chosen+datetime.timedelta(days = 1) - datetime.timedelta(hours=time_delta)
                    +datetime.timedelta(minutes= int(confDict['AVG_MIN']))
                        ,datetime.timedelta(minutes= int(confDict['AVG_MIN'])))
        calc_idx= [np.where((ii <= time_ds)*(time_ds < iip1))
                                    for ii,iip1 in zip(time_vec[0:-1],time_vec[1::])]
        time_start= np.array([int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())
                            if len(t[0]) != 0 
                            else int(pd.to_datetime(time_vec[ii+1]).replace(tzinfo=datetime.timezone.utc).timestamp())
                            for ii,t in enumerate(calc_idx)
                            ])
        time_bnds= np.array([[ int(pd.to_datetime(time_ds[t[0][0]]).replace(tzinfo=datetime.timezone.utc).timestamp())
                                ,int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())]
                            if len(t[0]) != 0
                            else [int(pd.to_datetime(time_vec[ii]).replace(tzinfo=datetime.timezone.utc).timestamp())
                                ,int(pd.to_datetime(time_vec[ii+1]).replace(tzinfo=datetime.timezone.utc).timestamp())]
                                for ii,t in enumerate(calc_idx)
                            ]).T

        # compare n_gates in lvl1-file and confdict
        if n_gates != dv.shape[1]:
            print('Warning: number of gates in config does not match lvl1 data!')
            n_gates= dv.shape[1]
            print('number of gates changed to ' + str(n_gates))
            
        # infer number of directions
            # don't forget to check for empty calc_idx
        time_valid= [ii for ii,x in enumerate(calc_idx) if len(x[0]) != 0]        

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
            print('processed ' + str(np.floor(100*kk/(len(calc_idx)-1))) +' %')
            # read lidar parameters
            n_rays= int(confDict['NUMBER_OF_DIRECTIONS'])
            indicator, n_rays, azi_mean, azi_edges= find_num_dir(n_rays,calc_idx,azimuth,kk)
            # azimuth[azimuth>azi_edges[0]]= azimuth[azimuth>azi_edges[0]]-360
            # azi_edges[0]= azi_edges[0]-360
            r_phi = 360/(n_rays)/2 
            if ~indicator:
                print('some issue with the data', n_rays, len(azi_mean), time_start[kk])
                continue
            else:
                
                VR = dv[calc_idx[kk]]
                SNR = snr[calc_idx[kk]]
                BETA = beta[calc_idx[kk]]
                azi = azimuth[calc_idx[kk]]
                ele = elevation[calc_idx[kk]]                
                
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
                    VR_CNSmax[ii,:], idx_tmp, VR_CNSunc[ii,:] = consensus( VR[azi_idx]
                                                                        #   , np.ones(SNR[azi_idx].shape)
                                                                        ,SNR[azi_idx], BETA[azi_idx]
                                                                        ,int(confDict['CNS_RANGE'])
                                                                        ,int(confDict['CNS_PERCENTAGE'])
                                                                        ,int(confDict['SNR_THRESHOLD']) 
                                                                        , B)
                    # next line is just experimental and might be useful in the future                                                                          )
                    # azi_CNS[ii,:]= np.array([np.nanmean(azi[azi_idx][xi]) for xi in idx_tmp.T])
                    # SNR_CNS[ii,:]= np.nanmean( np.where( idx_tmp
                    #                                , SNR[azi_idx]
                    #                                , np.nan)
                    #                          , axis=0)
                    SNR_tmp = SNR[azi_idx]
                    sigma_tmp = calc_sigma_single(in_db(SNR[azi_idx]),M,n,2*B,1.316)
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
                # NVRAD[kk, :] = (~np.isnan(VR_CNSmax)).sum(axis=0)
                n_good[kk, :] = n_good_kk
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
                UVW[kk, ...] = np.squeeze(V_k)
                # plausible winds can only be calculated, when the at least three LOS measurements are present
                UVW[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4 , :] = np.squeeze(np.full((3,1), np.nan))

                UVWunc[kk, ...]= abs(np.einsum('...ii->...i', np.sqrt((A_r_MP @ np.apply_along_axis(np.diag, 1, SIGMA_r**2) @ A_r_MP_T).astype(complex)).real))
                # plausible winds can only be calculated, when the at least three LOS measurements are present
                UVWunc[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4 , :] = np.squeeze(np.full((3,1), np.nan))

                V_r_est = A_r @ V_k
                ss_e = ((V_r-V_r_est)**2).sum(axis = 1)
                ss_t = ((V_r-V_r.mean(axis=1)[:, None, :])**2).sum(axis = 1)
                R2[kk, :] = np.squeeze(1 - ss_e/ss_t)
                # R2[kk, :] = 1 - (1 - R2[kk, :]) * (np.sum(~np.isnan(VR_CNSmax.T), axis=1)-1)/(np.sum(~np.isnan(VR_CNSmax.T), axis=1)-2)  
                # sqe = ((V_r_est-V_r_est.mean(axis=1)[:, None, :])**2).sum(axis = 1)
                # sqt = ((V_r-V_r.mean(axis=1)[:, None, :])**2).sum(axis = 1)
                # R2[kk, :] = np.squeeze(sqe/sqt)
                R2[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan
                

                mask_A = np.tile( mask_V_in.T[..., None], (1, 1, 3))
                # A_r_m = np.ma.masked_where( mask_A, A_r)
                A_r_T = np.einsum('...ij->...ji', A_r)
                Spp =  np.apply_along_axis(np.diag, 1, 1/np.sqrt(np.einsum('...ii->...i', A_r_T @ A_r)))
                Z = np.ma.masked_where( mask_A, A_r @ Spp)
                CN[kk, :] =  np.squeeze(np.array([CN_est(X) for X in Z]))
                # CN[kk, :] = np.array([CN_est(X) for X in A_r_m])
                CN[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

                SPEED[kk, :], SPEEDunc[kk, :] = np.vstack([np.fromiter(uvw_2_spd(val, unc).values(), dtype=float) for val, unc in zip(UVW[kk, ...], UVWunc[kk, ...])]).T
                DIREC[kk, :], DIRECunc[kk, :] = np.vstack([np.fromiter(uvw_2_dir(val, unc).values(), dtype=float) for val, unc in zip(UVW[kk, ...], UVWunc[kk, ...])]).T

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
                configuration += dd + '=' + confDict[dd]+'\n'
        if 'BLINDZONE_GATES' in confDict:
            NN = int(confDict['BLINDZONE_GATES'])
        else:
            NN = 0
        return xr.Dataset({ 'config': ([]
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
    @staticmethod                    
    def lvl2wcdbs(ds_comb, date_chosen, confDict):
        n_rays= int(confDict['NUMBER_OF_DIRECTIONS'])
                    # number of gates
        n_gates= int(confDict['NUMBER_OF_GATES'])
                    # number of pulses used in the data point aquisition
        B = (ds_comb.radial_wind_speed.max()-ds_comb.radial_wind_speed.min()).data/2
        lrg = ds_comb.range_gate_length.data

        time_ds = pd.to_datetime(ds_comb.time.data, unit='s')
        ds_comb['zenith'] = 90 - ds_comb.elevation
        elevation = ds_comb.elevation.data
        azimuth = ds_comb.azimuth.data
        cnr = 10**(ds_comb.cnr.data/10)
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
            height = (np.sin(ds_comb.elevation*np.pi/180)*ds_comb.range).data[0]
            
        width = (np.cos(ds_comb.elevation*np.pi/180)*ds_comb.range*2).data
        width = width[elevation<89][0]
        height_bnds = np.vstack([height-lrg/2, height-lrg/2]).T
        
        if 'UTC_OFFSET' in confDict:
            time_offset = np.timedelta64(int(confDict['UTC_OFFSET']), 'h') 
            time_delta = int(confDict['UTC_OFFSET'])
        else:
            time_offset = np.timedelta64(0, 'h') 
            time_delta = 0

        time_vec= np.arange(date_chosen - datetime.timedelta(hours=time_delta)
                            ,date_chosen+datetime.timedelta(days = 1) - datetime.timedelta(hours=time_delta)
                            + datetime.timedelta(minutes= int(confDict['AVG_MIN']))
                            , datetime.timedelta(minutes= int(confDict['AVG_MIN'])))
        calc_idx= [np.where((ii <= time_ds)*(time_ds < iip1))
                for ii,iip1 in zip(time_vec[0:-1],time_vec[1::])]
        time_start= np.array([int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())
                            if len(t[0]) != 0 
                            else int(pd.to_datetime(time_vec[ii+1]).replace(tzinfo=datetime.timezone.utc).timestamp())
                            for ii,t in enumerate(calc_idx)
                            ])
        time_bnds= np.array([[ int(pd.to_datetime(time_ds[t[0][0]]).replace(tzinfo=datetime.timezone.utc).timestamp())
                                ,int(pd.to_datetime(time_ds[t[0][-1]]).replace(tzinfo=datetime.timezone.utc).timestamp())]
                            if len(t[0]) != 0
                            else [int(pd.to_datetime(time_vec[ii]).replace(tzinfo=datetime.timezone.utc).timestamp())
                                ,int(pd.to_datetime(time_vec[ii+1]).replace(tzinfo=datetime.timezone.utc).timestamp())]
                                for ii,t in enumerate(calc_idx)
                            ]).T    
        if n_gates != dv.shape[1]:
            print('Warning: number of gates in config does not match lvl1 data!')
            n_gates= dv.shape[1]
            print('number of gates changed to ' + str(n_gates))
        # infer number of directions
        # don't forget to check for empty calc_idx
        time_valid= [ii for ii,x in enumerate(calc_idx) if len(x[0]) != 0]

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
            print('processed ' + str(np.floor(100*kk/(len(calc_idx)-1))) +' %')
            # read lidar parameters
            n_rays= int(confDict['NUMBER_OF_DIRECTIONS'])
            indicator, n_rays, azi_mean, azi_edges= find_num_dir(n_rays,calc_idx,azimuth,kk)
            # azimuth[azimuth>azi_edges[0]]= azimuth[azimuth>azi_edges[0]]-360
            # azi_edges[0]= azi_edges[0]-360
            r_phi = 360/(n_rays)/2
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
                    w_cns, idx_tmp, tmp_tmp = consensus( WR, np.ones(WR.shape), np.ones(WR.shape), 2, 30, 0, B)
                    WR_SPEC = tmp_tmp   
                    WR_CNS = np.ma.masked_where(~idx_tmp, CNR_WR).mean(axis=0).filled(np.nan)
                    WR_BETA = np.ma.masked_where(~idx_tmp, BETA_WR).mean(axis=0).filled(np.nan)
                CNR = cnr[calc_idx[kk]][elevation[calc_idx[kk]] < 89]
                BETA = beta[calc_idx[kk]][elevation[calc_idx[kk]] < 89]
                azi = azimuth[calc_idx[kk]][elevation[calc_idx[kk]] < 89]
                ele = elevation[calc_idx[kk]][elevation[calc_idx[kk]] < 89]
                SPEC = delv[calc_idx[kk]][elevation[calc_idx[kk]] < 89]

                VR_CNSmax = np.full((len(azi_mean),n_gates), np.nan)
                VR_CNSunc = np.full((len(azi_mean),n_gates), np.nan)
                CNR_CNS= np.full((len(azi_mean),n_gates), np.nan)
                BETA_CNS = np.full((len(azi_mean),n_gates), np.nan)
                SIGMA_CNS = np.full((len(azi_mean),n_gates), np.nan)
                ele_cns = np.full((len(azi_mean),), np.nan)

                

                for ii, azi_i in enumerate(azi_mean):
                    # azi_idx = (azi>=azi_edges[ii])*(azi<azi_edges[ii+1])
                    azi_idx = (np.mod(360-np.mod(np.mod(azi-azi_i, 360)-r_phi, 360), 360)<=2*r_phi)
                    ele_cns[ii] = np.median(ele[azi_idx])
                    ## calculate consensus average
                    VR_CNSmax[ii,:], idx_tmp, VR_CNSunc[ii,:] = consensus( VR[azi_idx]
                                                                                            ,CNR[azi_idx], BETA[azi_idx]
                                                                                            ,int(confDict['CNS_RANGE'])
                                                                                            ,int(confDict['CNS_PERCENTAGE'])
                                                                                            ,int(confDict['SNR_THRESHOLD'])
                                                                                            , B)
                    CNR_CNS[ii, :] = np.ma.masked_where(~idx_tmp, CNR[azi_idx]).mean(axis=0).filled(np.nan)
                    # SIGMA_CNS[ii, :] = np.ma.masked_where(~idx_tmp, SPEC[azi_idx]).mean(axis=0).filled(np.nan)
                    SIGMA_CNS[ii, :] = VR_CNSunc[ii,:]
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
                # include stare measurements
                A_stare = np.round(build_Amatrix(  np.zeros(((elevation[calc_idx[kk]] > 89).sum(),))
                                                , 90*np.ones(((elevation[calc_idx[kk]] > 89).sum(),)))
                        , 6 )
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
                UVW[kk, ...] = np.squeeze(V_k)
                # plausible winds can only be calculated, when the at least three LOS measurements are present
                UVW[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4 , :] = np.squeeze(np.full((3,1), np.nan))

                UVWunc[kk, ...]= abs(np.einsum('...ii->...i', np.sqrt((A_r_MP @ np.apply_along_axis(np.diag, 1, SIGMA_r**2) @ A_r_MP_T).astype(complex)).real))
                # plausible winds can only be calculated, when the at least three LOS measurements are present
                UVWunc[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4 , :] = np.squeeze(np.full((3,1), np.nan))

                V_r_est = A_r @ V_k
                ss_e = ((V_r-V_r_est)**2).sum(axis = 1)
                ss_t = ((V_r-V_r.mean(axis=1)[:, None, :])**2).sum(axis = 1)
                R2[kk, :] = np.squeeze(1 - ss_e/ss_t)
                # R2[kk, :] = 1 - (1 - R2[kk, :]) * (np.sum(~np.isnan(VR_CNSmax.T), axis=1)-1)/(np.sum(~np.isnan(VR_CNSmax.T), axis=1)-2)  
                # sqe = ((V_r_est-V_r_est.mean(axis=1)[:, None, :])**2).sum(axis = 1)
                # sqt = ((V_r-V_r.mean(axis=1)[:, None, :])**2).sum(axis = 1)
                # R2[kk, :] = np.squeeze(sqe/sqt)
                R2[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan


                mask_A = np.tile( mask_V_in.T[..., None], (1, 1, 3))
                # A_r_m = np.ma.masked_where( mask_A, A_r)
                A_r_T = np.einsum('...ij->...ji', A_r)
                Spp =  np.apply_along_axis(np.diag, 1, 1/np.sqrt(np.einsum('...ii->...i', A_r_T @ A_r)))
                Z = np.ma.masked_where( mask_A, A_r @ Spp)
                CN[kk, :] =  np.squeeze(np.array([CN_est(X) for X in Z]))
                # CN[kk, :] = np.array([CN_est(X) for X in A_r_m])
                CN[kk, np.sum(~np.isnan(VR_CNSmax.T), axis=1) < 4] = np.nan

                SPEED[kk, :], SPEEDunc[kk, :] = np.vstack([np.fromiter(uvw_2_spd(val, unc).values(), dtype=float) for val, unc in zip(UVW[kk, ...], UVWunc[kk, ...])]).T
                DIREC[kk, :], DIRECunc[kk, :] = np.vstack([np.fromiter(uvw_2_dir(val, unc).values(), dtype=float) for val, unc in zip(UVW[kk, ...], UVWunc[kk, ...])]).T


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
                configuration += dd + '=' + confDict[dd]+'\n'
        if 'BLINDZONE_GATES' in confDict:
            NN = int(confDict['BLINDZONE_GATES'])
        else:
            NN = 0
        
        return xr.Dataset({ 'config': ([]
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
        
    def dailylvl2(self):
        date_chosen = self.date2proc
        confDict= config.gen_confDict(url= self.config_dir)
        
        ds_tmp = import_lvl1(date_chosen, confDict)
        
        if confDict['SYSTEM'].lower() == 'windcube':
            
            #if (len(ds_tmp.range.dims) > 1):
            if 'fixed' not in confDict['SCAN_TYPE'].lower():
                if ('dbs' in confDict['SCAN_TYPE'].lower()) or ('vad' in confDict['SCAN_TYPE'].lower()) or ('ppi' in confDict['SCAN_TYPE'].lower()):
                    print("processing 'Windcube-dbs/-vad' setting!")
                    ds_lvl2 = hpl2netCDFClient.lvl2wcdbs(ds_tmp, date_chosen, confDict)
                if ('rhi' in confDict['SCAN_TYPE'].lower()):
                    print("settings for RHI not yet implemented!")
                    # create place holder dataset
                    ds_lvl2 = xr.Dataset()
            if ('fixed' in confDict['SCAN_TYPE'].lower()):
                if ('vad' in confDict['SCAN_TYPE'].lower()):
                    print("processing 'Windcube-vad-fixed' setting!...for old system version!!")
                    ds_lvl2 = hpl2netCDFClient.lvl2vad_standard(ds_tmp, date_chosen, confDict)           
                if ('stare' in confDict['SCAN_TYPE'].lower()):
                    print("processing 'WindCube-stare' setting!")
                    print("coming soon!")
                    # create place holder dataset
                    ds_lvl2 = xr.Dataset()
                
        if confDict['SYSTEM'].lower() == 'halo':
            if (len(ds_tmp.range.dims) > 1) & (confDict['SCAN_TYPE'] == 'DBS'):
                print("processing 'Streamline-dbs' setting!")
                ds_lvl2 = hpl2netCDFClient.lvl2vad_standard(ds_tmp, date_chosen, confDict)
            if ('vad' in confDict['SCAN_TYPE'].lower()) | ('user' in confDict['SCAN_TYPE'].lower()):
                print("processing 'Streamline-VAD' setting!")
                ds_lvl2 = hpl2netCDFClient.lvl2vad_standard(ds_tmp, date_chosen, confDict)
            if ('stare' in confDict['SCAN_TYPE'].lower()):
                print("processing 'Streamline-Stare' setting!")
                print("coming soon!")
                # create place holder dataset
                ds_lvl2 = xr.Dataset()

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
        
        path= Path(confDict['NC_L2_PATH'] + '/' 
                    + date_chosen.strftime("%Y") + '/'
                    + date_chosen.strftime("%Y%m"))
        path.mkdir(parents=True, exist_ok=True)  
        path= path / Path(confDict['NC_L2_BASENAME'] + 'v' +  confDict['VERSION'] + '_' + date_chosen.strftime("%Y%m%d") + '.nc')

        print(path)
        # compress variables
        comp = dict(zlib=True, complevel=9)
        encoding = {var: comp for var in np.hstack([ds_lvl2.data_vars,ds_lvl2.coords])}
        if 'UTC_OFFSET' in confDict:
            time_delta = int(confDict['UTC_OFFSET'])
        else:
            time_delta = 0
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
                    VR_CNSmax[ii,:], idx_tmp, VR_CNSunc[ii,:] = consensus( VR[azi_idx]
                                                                        #   , np.ones(SNR[azi_idx].shape)
                                                                          ,SNR[azi_idx], BETA[azi_idx]
                                                                          ,int(confDict['CNS_RANGE'])
                                                                          ,int(confDict['CNS_PERCENTAGE'])
                                                                          ,int(confDict['SNR_THRESHOLD']) 
                                                                          , B)
                    # next line is just experimental and might be useful in the future                                                                          )
                    # azi_CNS[ii,:]= np.array([np.nanmean(azi[azi_idx][xi]) for xi in idx_tmp.T])
                    # SNR_CNS[ii,:]= np.nanmean( np.where( idx_tmp
                    #                                , SNR[azi_idx]
                    #                                , np.nan)
                    #                          , axis=0)
                    SNR_tmp = SNR[azi_idx]
                    sigma_tmp = calc_sigma_single(in_db(SNR[azi_idx]),M,n,2*B,1.316)
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
