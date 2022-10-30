#!/usr/bin/env python
'''
DWD-Pilotstation software source code file

by Markus Kayser. Non-commercial use only.
'''
import argparse
from hpl2netCDF_client.hpl2netCDF_client import hpl2netCDFClient
import textwrap
import sys
import datetime
# from version import __version__
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

def try_parsing_date(text):
    for fmt in ('%Y-%m-%dT%H:%M', '%Y-%m-%d'):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('no valid date format found')

def valid_date(s):
    try:
        if s == 'nrt':
            date = datetime.datetime.now(datetime.timezone.utc)
        else:
            date = try_parsing_date(s)
        return date
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)

def main():
    default_path = "C:/Users/mkayser/Documents/Notebooks/wl_testing/wl_44_markus.conf"
    parser = argparse.ArgumentParser(
                                    description='Process Halo photonics Doppler lidar Client',
                                    formatter_class=argparse.RawTextHelpFormatter
                                    )
    # parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)
    parser.add_argument('-u', '--url', dest="path2config",
                        type=str, help="path to configuration file",
                            default=default_path)
    parser.add_argument("-d", "--date", nargs='+'
                        , help="The Date - format YYYY-mm-dd or use nrt for near real time processing using data from the last AVG_MIN minutes prior to calling the client"
                        , required=True, type=valid_date
                        )
    parser.add_argument('-c', '--cmd', required=True, choices=['hpl_l1', 'l1_l2','hpl_l1_l2','l1_l2wql','hpl_l1_l2wql','l1wql','l2wql','hpl_l1_l2nrt','hpl_l2nrt'],
        help=textwrap.dedent(\
        '''Send a command. Supported commands:
        hpl_l1 - combines daily lvl1 files to netCDF
        l1_l2 - processes existing lvl1 netCDF files and combines to lvl2 netCDF
        hpl_l1_l2 - combines hpl to daily and processes directly to lvl2 netCDF
        hpl_l1_l2wql - combines hpl to daily and processes directly to lvl2 netCDF and creates quicklooks
        l1_l2wql - combines hpl to daily and processes directly to lvl2 netCDF and creates quicklooks
        l1wql - takes lvl1 netCDF file and creates backscatter quicklook
        l2wql - takes lvl2 netCDF file and creates backscatter quicklook
        Note that directory paths will be taken from the configuration file!''')
        )
    
    args = parser.parse_args(sys.argv[1:])
        
    proc_dl= hpl2netCDFClient(args.path2config, args.cmd, args.date[0])
    
    proc_dl.display_config_dir()
    proc_dl.display_configDict()
    
    if args.cmd == 'hpl_l1':
        proc_dl.dailylvl1()
    if args.cmd == 'l1_l2':
        proc_dl.dailylvl2()
    if args.cmd == 'hpl_l1_l2':
        proc_dl.dailylvl1()
        proc_dl.dailylvl2()
    if args.cmd == 'l1_l2wql':
        proc_dl.dailylvl2()
        proc_dl.lvl2ql()
        proc_dl.bckql()
    if args.cmd == 'hpl_l1_l2wql':
        proc_dl.dailylvl1()
        proc_dl.dailylvl2()
        proc_dl.lvl2ql()
        proc_dl.bckql()
    if args.cmd == 'l1wql':
        proc_dl.bckql()
    if args.cmd == 'l2wql':
        proc_dl.lvl2ql()
    if args.cmd == 'hpl_l1_l2nrt':
        proc_dl.nrtlvl1()
        proc_dl.nrtlvl2()
    if args.cmd == 'hpl_l2nrt':
        proc_dl.nrtlvl1()
        proc_dl.nrtlvl2()
        proc_dl.rmlvl1()
