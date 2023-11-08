import datetime
import os

from hpl2netCDF_client.hpl2netCDF_client import hpl2netCDFClient

if __name__ == '__main__':

    cmd = 'lvl2_from_filelist' #'hpl_l1_l2'  # works: 'hpl_l1_l2'; does not work: 'hpl_l1_l2nrt', 'hpl_l2nrt'

    # Names and metadata of observation sets. inpath and files are just used for lvl2_from_files.
    # Set names are just for display purposes and can be anything.
    # Can comment out obs set (incl conf and date) for not processing it
    obs_sets = {
        'pay_vad_tp': dict(conf='/home/ruf/Desktop/dl_toolbox_test_data/config_ruf/wc_MeteoSwiss_Payerne_VAD_TP.conf',
                           date=datetime.datetime.strptime('2023-02-09', '%Y-%m-%d'),
                           inpath='/home/ruf/Desktop/dl_toolbox_test_data/data_prepared/MeteoSwiss/VAD_selection/2023/202302/20230209/',
                           files=['WLS200s-197_2023-02-09_00-21-34_vad_148_50mTP.nc',
                                  'WLS200s-197_2023-02-09_00-51-33_vad_148_50mTP.nc']),
        'pay_dbs_tp': dict(conf='/home/ruf/Desktop/dl_toolbox_test_data/config_ruf/wc_MeteoSwiss_Payerne_DBS_TP.conf',
                           date=datetime.datetime.strptime('2023-01-01', '%Y-%m-%d'),
                           inpath='/home/ruf/Desktop/dl_toolbox_test_data/data_prepared/MeteoSwiss/Payerne/2023/202301/20230101/',
                           files=['WLS200s-197_2023-01-01_00-00-59_dbs_303_50mTP.nc',
                                  'WLS200s-197_2023-01-01_00-05-11_dbs_216_50mTP.nc',
                                  'WLS200s-197_2023-01-01_00-06-12_dbs_303_50mTP.nc',
                                  'WLS200s-197_2023-01-01_00-07-27_dbs_303_50mTP.nc',
                                  'WLS200s-197_2023-01-01_00-08-42_dbs_303_50mTP.nc',
                                  'WLS200s-197_2023-01-01_00-09-57_dbs_303_50mTP.nc']),
        }

    # Code:
    for obs_name, obs_meta in obs_sets.items():
        print(f'\nProcessing for {obs_name}\n=========================================\n')
        proc_dl = hpl2netCDFClient(obs_meta['conf'], cmd, obs_meta['date'])

        proc_dl.display_config_dir()
        proc_dl.display_configDict()

        if cmd == 'hpl_l1':
            proc_dl.dailylvl1()
        if cmd == 'l1_l2':
            proc_dl.dailylvl2()
        if cmd == 'hpl_l1_l2':
            proc_dl.dailylvl1()
            proc_dl.dailylvl2()
        if cmd == 'l1_l2wql':
            proc_dl.dailylvl2()
            proc_dl.lvl2ql()
            proc_dl.bckql()
        if cmd == 'hpl_l1_l2wql':
            proc_dl.dailylvl1()
            proc_dl.dailylvl2()
            proc_dl.lvl2ql()
            proc_dl.bckql()
        if cmd == 'l1wql':
            proc_dl.bckql()
        if cmd == 'l2wql':
            proc_dl.lvl2ql()
        if cmd == 'hpl_l1_l2nrt':
            proc_dl.nrtlvl1()
            proc_dl.nrtlvl2()
        if cmd == 'hpl_l2nrt':
            proc_dl.nrtlvl1()
            proc_dl.nrtlvl2()
            proc_dl.rmlvl1()
        if cmd == 'lvl2_from_filelist':
            # prepend inpath to filenames
            files = list(map(lambda x: os.path.join(obs_meta['inpath'], x), obs_meta['files']))
            proc_dl.lvl2_from_filelist(files)
