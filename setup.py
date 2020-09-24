#!/usr/bin/env python
'''
DWD-Pilotstation software source code file

by Markus Kayser. Non-commercial use only.
'''
from setuptools import setup

with open('hpl2netCDF_client/version.py') as f:
    exec(f.read())


setup(name='hpl2netCDF_client',
      version=__version__,
      description='Halo photonics Doppler lidar to netCDF',
      url='http://lglxs408.dwd.de:8000/webtools/iserver/mol1/dial/',
      author='Markus Kayser',
      license='Non-commercial use only',
      packages=['hpl2netCDF_client','hpl2netCDF_client.hpl_files','hpl2netCDF_client.config'],
      install_requires=[
          'numpy',
          'scipy',
          'xarray',
          'pandas',
          'datetime',
          'argparse',
          'pathlib',
          'netcdf4',
          'matplotlib',
          #'pyqt5==5.9.2',
          ],
      entry_points = {
        'console_scripts': ['hpl2netCDF-client=hpl2netCDF_client.command_line:main'],
      },
      zip_safe=False)
