import itertools as it
import operator as op

import numpy as np


def build_Amatrix(azimuth_vec,elevation_vec):
    return np.einsum('ij -> ji',
                    np.vstack(
                        [
                        np.sin((np.pi/180)*(azimuth_vec))*np.sin((np.pi/180)*(90-elevation_vec))
                        ,np.cos((np.pi/180)*(azimuth_vec))*np.sin((np.pi/180)*(90-elevation_vec))
                        ,np.cos((np.pi/180)*(90-elevation_vec))
                        ])
                    )


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


def diff_aa(x,y,c):
    '''calculate aliasing independent differences'''
    return (c-abs(abs(x-y)-c))


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


def grouper(iterable, n, fillvalue=None):
    '''Collect data into fixed-length chunks or blocks'''
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return it.zip_longest(*args, fillvalue=fillvalue)


def calc_node_degree(Vr,CNS_range,B, metric='l1norm'):
    '''takes masked array as input'''
    if metric == 'l1norm':
        f_abs_pairdiff = lambda x,y: op.abs(op.sub(x,y))<CNS_range
    if metric == 'l1norm_aa':
        f_abs_pairdiff = lambda x,y: op.sub(B,op.abs(op.sub(op.abs(op.sub(x,y)),B)))<CNS_range
    with np.errstate(invalid='ignore'):
        return np.array(list(grouper(it.starmap(f_abs_pairdiff,((it.permutations(Vr.filled(np.nan),2)))),Vr.shape[0]-1))).sum(axis=1)


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


# def calc_node_degree(Vr,CNS_range):
#     '''takes masked array as input'''
#     f_abs_pairdiff = lambda x,y: op.abs(op.sub(x,y))<CNS_range
#     with np.errstate(invalid='ignore'):
#         return np.array(list(grouper(it.starmap(f_abs_pairdiff,((it.permutations(Vr.filled(np.nan),2)))),Vr.shape[0]-1))).sum(axis=1)
