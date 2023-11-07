import numpy as np


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
