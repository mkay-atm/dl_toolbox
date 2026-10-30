# DL Toolbox (`hpl2netCDF_client`)

**Experimental development branch** — Python tools for processing ground-based Doppler wind lidar measurements to NetCDF Level-1 (L1) and Level-2 (L2) products.

The toolbox supports Halo Photonics Stream Line instruments and selected Vaisala/Leosphere WindCube workflows. Its established pathways create daily L1 files, retrieve wind from supported scanning strategies, apply quality control, and generate quicklooks. This branch additionally contains an **experimental vertical-stare turbulence retrieval**.

> **Development status:** This is not a validated operational release. Validate retrieved quantities, especially turbulent kinetic energy dissipation rate (`epsilon`), against independent observations before scientific or operational use.

## Capabilities

- **L1:** daily NetCDF assembly from supported Halo `.hpl` and WindCube `.nc` data.
- **Wind L2:** `u`, `v`, `w`, wind speed and direction for supported VAD/DBS scans, with uncertainty estimates, SNR/backscatter diagnostics, fit quality (`r2`), condition number (`cn`), valid radial-velocity counts (`nvrad`), and quality flags (`qwind`).
- **Quicklooks:** wind profiles and backscatter.
- **Experimental stare L2:** screened vertical Doppler velocity, vertical-velocity variance, noise correction, `epsilon`, associated quality information, optional estimated uncertainties, and three stare quicklooks.

Not all instrument/scan-mode combinations have a complete L2 implementation.

## Install

```bash
git clone <repository-url>
cd <repository-directory>
git switch <development-branch>
python -m venv .venv
# Activate .venv (for example: source .venv/bin/activate on Linux/macOS)
python -m pip install --upgrade pip
python -m pip install -e .
hpl2netCDF-client --help
```

The Python package is named `hpl2netCDF_client`; the installed CLI is `hpl2netCDF-client`. Dependencies are declared in `setup.py` (including NumPy, SciPy, pandas, xarray, netCDF4, and Matplotlib). Platform/Python-version compatibility has not been systematically tested.

## Standard wind-processing workflow

Adapt a supplied configuration file (`wl_template.conf`, `wl_44_cns_60_snr_22.conf`, or `wc_233_DBS_cns_60_snr_00.conf`). Specify instrument metadata, scan geometry, file paths, filename prefixes, averaging, and quality-control thresholds.

```bash
# Convert raw measurements to daily L1 NetCDF
hpl2netCDF-client -u path/to/site.conf -d 2024-07-20 -c hpl_l1

# Retrieve L2 winds from the existing daily L1 file
hpl2netCDF-client -u path/to/site.conf -d 2024-07-20 -c l1_l2

# Combined L1 + L2 + quicklooks
hpl2netCDF-client -u path/to/site.conf -d 2024-07-20 -c hpl_l1_l2wql
```

The input and output directories follow the configuration; daily NetCDF paths conventionally include `YYYY/YYYYMM/`. Use `--help` for additional CLI commands, including near-real-time variants. Those variants are not yet verified.

## Experimental vertical-stare workflow

**Inputs:** (1) a daily **stare L1** NetCDF file containing `dv`, `intensity`, `time`, and `range`; and (2) a daily **wind L2** NetCDF file with horizontal wind speed `wspeed` and compatible time/height coverage. L1 stare observations **alone are not sufficient** for the `epsilon` retrieval.

A supplied real-world example is the AWI Halo Stream Line XR+ stare dataset for **2024-07-20** (instrument S/N 288, 8,873 profiles × 250 range gates). The repository-ready configuration is in [`examples/stare_awi_288.conf`](examples/stare_awi_288.conf), and detailed setup, input checks, outputs, and caveats are documented in **[`docs/stare-processing.md`](docs/stare-processing.md)**.

After preparing the two inputs and editing their directory roots:

```bash
hpl2netCDF-client -u examples/stare_awi_288.conf -d 2024-07-20 -c l1_stare_l2
```

To create the stare product **and its three quicklooks** in one step:

```bash
hpl2netCDF-client -u examples/stare_awi_288.conf -d 2024-07-20 -c l1_stare_l2wql
```

To generate **only quicklooks** from an already processed stare L2 file (no wind L2 input is needed for this step):

```bash
hpl2netCDF-client -u examples/stare_awi_288.conf -d 2024-07-20 -c starewql
```

The equivalent Python API is:

```python
from datetime import datetime
from hpl2netCDF_client.hpl2netCDF_client import hpl2netCDFClient

client = hpl2netCDFClient(
    "examples/stare_awi_288.conf", "l1_stare_l2", datetime(2024, 7, 20)
)
l2_file, quicklook_files = client.dailystarewql()
```

**CLI fixed in this branch:** `starewql` now reads an existing daily stare L2 file and writes quicklooks; `l1_stare_l2wql` executes both processing and plotting. Quicklook names honor `NC_STARE_L2_QL_BASENAME`.

## Testing

```bash
python -m pip install pytest
python -m pytest -q
```

The included tests exercise CLI dispatch, daily filename creation, configuration parsing, wind-speed gap-fill provenance, and generation of all three quicklooks; physical accuracy is not yet fully established. A full-day computational test using **observed** AWI stare L1 and VAD wind L2 from 2024-07-20 is recorded in [the observed-data validation report](docs/observed-validation-20240720.md), with example outputs distributed separately from the repository source. These observational files are not included in the source tree. Scientific validation of epsilon is still ongoing.

## Development and limitations

- **Stare validation:** the uploaded AWI L1 file was inspected, and the retrieval functions ran on a small sample with an *artificial* supporting wind field.
- **Maintenance:** the included regression tests cover the stare command paths and plots; wind/near-real-time modes still need regression tests and scientifically validated reference data.

Please report reproducible problems through GitHub Issues with the instrument, scan mode, config, and a minimal permissible input sample.

## License

This code revision does **not** change the original license texts.
