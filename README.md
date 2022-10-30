# Documentation - dl_toolbox

##### Table of Contents  
- [Description](#desc)  
- [Usage](#usage) 
    - [Generic example](#generic_example)
    - [Streamline VAD](#streamline_example)
    - [Windcube DBS](#windcube_example)
	- [Other scan types](#other_scans)


<a name="desc"/></a>
## Description
Python software package for standardized processing Doppler wind lidar data. (Note: the client is originally designed for Halo Photonics systems, but the latest version makes it applicaple to windcube lidars. Please, read the corresponding settings sections.)

<a name="usage"/></a>
## Usage
1.	Go to the directory where the package is located and install the client via `pip install`. If installation was not successful, please contact Markus.Kayser@dwd.de
  
2.	After the successful installation the client can be used through parsing the following commands:

	You need to specify a valid date with either `-d` or `-date` and a command to execute with `-c` or `-cmd`.
	The following commands are valid:
		- `hpl_l1` creates a daily netCDF file of lvl1 .hpl-files.
		- `l1_l2` processes a daily lvl1 netCDF file according to the VAD retrieval described in Päschke et al. [2015].
		- `hpl_l1_l2` creates daily lvl1 and lvl2 netCDF files from .hpl-files.
        Note: In case of lvl2 processing you can add `...wql` to the command to create quicklooks.
	You have to specify a custom configuration file in order for the routines to locate .hpl-files and give directories of lvl1 	and lvl2 netCDFs as well as of quicklooks.
    
    The routines will create a data path corresponding to the path specified in the configuration file plus the time 		information, i.e. `yourpath/YYYY/YYYYMM`.
	This is done with the command `-u` or `-url`, followed by the path to the configuration file, including the file name.
    
<a name="generic_example"/></a>
### Geneneric example
The use can start processing with the client by inserting the following command into the terminal.
		`hpl2netCDF-client -u "configuration file" -d "specify date as YYYY-MM-DD" -c "cmd"`	
If an error occurs at this stage, please make sure that the package is installed and that python 3.XX is in the system path.
<a name="streamline_example"/></a>
### Streamline VAD
The user needs to adjust the configuration file in the following way:
1. Make sure that "SCAN_TYPE" refers to the section of the filenames, either "VAD" or "User" plus a number. Otherwise the client is not able to find the desired .hpl-files.
2. Please copy the meta data attributes that can found in the .hpl header into the corresponding entries of the configuration file, i.e. number of gates, range gate length, pulses/ray, etc.
3. Adjust the processing parameters, used for filtering and quality control, to the user needs. These are: "AVG_MIN", "CN_THRESHOLD", "CNS_RANGE", "CNS_PERCENTAGE", "SNR_THRESHOLD", "N_VRAD_THRESHOLD", and "R2_THRESHOLD".
An example file is part of this repository, just have a look at "wl_44_cns_60_snr22.conf". This file is used to process a VAD with 24 directions applying consensus and snr filtering.

<a name="windcube_example"/></a>
### Windcube DBS
With the latest windcube release in October 2022 a new system software was introduced that now enables VAD scan pattern and also an additional range mode, called "TP-mode". This can create issues with the processing client not yet accounted for. Therefore, windcube-users are strongly encouraged to test this toolbox and report issues on github. Helping users to setup processing of windcube DBS data, we recommend the following steps:
1. Look at the individual windcube netCDF files and make note of the scan type, here dbs, the range gate length and wether or not, TP-mode is active. The latter two are found at the end of the file name. Set the SCAN_TYPE entry to "DBS_TP" and the RANGE_GATE_LENGTH to whatever number the filename states, e.g. 50 for 50 m length.
2. Check the individual netCDFs to fill in the remaining meta data, i.e. number of gates, etc. Note, that the current software leaves out a lot of the meta data. So the user has to fill in the gaps.
3. Make sure that the DBS configuration states NUMBER_OF_DIRECTIONS=			4 and that the N_VRAD_THRESHOLD is either 3 or 4. Generally, a threshold greater than the number of directions results in a LV2 file containing only NaN-values.
Note, if your individual filenames do not contain "TP", the SCAN_TYPE entry should be just "DBS". Please look at the example configuration file "wc_233_DBS_cns_60_snr00.conf" and adjust it to your needs.

<a name="other_scans"/></a>
### Other scan types
Even though a LV2-processing for other scan types is not implemented, users can still make use of this toolbox to compile daily LV1 netCDFs. Therefor, they have to follow the previously mentioned steps of identifying configuration parameters from the filenames and from meta data contained in the files. The following table helps the users to make these adjustments and states what products are available.

| System  | SCAN_TYPE  | LV1  | LV2  | Quicklooks  |
| :---  | :---:  | :---:  | :---:  | ---: |
| Streamline  | VAD <br /> UserX  | yes  | yes  | LV1 / LV2 |
| Streamline  | DBS <br /> UserX  | yes  | yes  | LV1 / LV2 |
| Streamline  | Stare  | yes  | no  | LV1 |
| Streamline  | RHI  | yes  | no  | no |
| Windcube  | fixed_VAD <br /> fixed_VAD_TP <br /> VAD <br /> VAD_TP| yes  | yes  | LV1 / LV2 |
| Windcube  | DBS <br /> DBS_TP | yes  | yes  | LV1 / LV2 |
| Windcube  | fixed/Stare <br /> fixed_TP/Stare_TP| yes  | no  | LV1 |
| Windcube  | RHI <br /> RHI_TP | yes  | no  | no |
| Windcube  | PPI <br /> PPI_TP| yes  | no  | no |