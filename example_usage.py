import datetime

from hpl2netCDF_client.hpl2netCDF_client import hpl2netCDFClient


if __name__ == '__main__':

    # Payerne VAD
    cmd = 'hpl_l1_l2'  # works: 'hpl_l1_l2'; does not work: 'hpl_l1_l2nrt', 'hpl_l2nrt'
    conf = '/home/ruf/Desktop/dl_toolbox_test_data/config_ruf/wc_MeteoSwiss_Payerne_VAD_TP.conf'
    date = datetime.datetime.strptime('2023-02-09', '%Y-%m-%d')

    # Code:
    proc_dl = hpl2netCDFClient(conf, cmd, date)

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
