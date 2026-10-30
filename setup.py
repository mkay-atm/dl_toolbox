#!/usr/bin/env python
'''
DWD-Pilotstation software source code file

by Markus Kayser. Non-commercial use only.
'''
from setuptools import find_packages, setup

with open('hpl2netCDF_client/version.py') as f:
    exec(f.read())

with open('README.md', encoding='utf-8') as f:
    readme = f.read()


setup(name='hpl2netCDF_client',
      version=__version__,
      description='Doppler lidar Level-1 and Level-2 processing toolbox',
      long_description=readme,
      long_description_content_type='text/markdown',
      url='http://lglxs408.dwd.de:8000/webtools/iserver/mol1/dial/',
      author='Markus Kayser',
      license='Non-commercial use only',
      packages=find_packages(),
      python_requires='>=3.10',
      install_requires=[
          'numpy',
          'scipy',
          'xarray >= 0.17.0',
          'pandas',
          'netcdf4',
          'matplotlib',
          #'pyqt5==5.9.2',
          ],
      entry_points = {
        'console_scripts': ['hpl2netCDF-client=hpl2netCDF_client.command_line:main'],
      },
      zip_safe=False)
