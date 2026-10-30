# Experimental vertical-stare processing

This guide describes the current implementation in `hpl2netCDF_client/stare_proc.py`, `stare_calc.py`, and `stare_ql.py`. It uses the provided [`stare_awi_288.conf`](../examples/stare_awi_288.conf) and the AWI S/N 288 L1 observations for **20 July 2024**. Retrievals are **experimental** and require independent scientific validation.

## 1. Inputs verified from the supplied AWI L1 file

Original uploaded copy: `awi_rao_dlidST00_l1_any_v04_20240720(1).nc`.

| Item | Value in the file |
| --- | --- |
| Instrument | Halo Photonics Stream Line XR+; serial number 288 |
| Site / institution | Lindenberg / AWI |
| Scan geometry | Stare; zenith = 0°, corresponding to 90° elevation |
| Time coverage | 2024-07-20 00:10:44 to 23:59:58 UTC (approximately) |
| Dimensions | `time=8873`, `range=250` |
| Range-gate centres | 24 to 11,976 m, separated by 48 m |
| Required stare variables | `dv(time,range)`, `intensity(time,range)`, `time`, `range` |
| Also present | `beta`, `delv`, `errdv`, `azi`, `zenith`, and others |
| Sampling | median difference ≈6.22 s; 71 gaps above 10 s, maximum ≈676 s |

The intensity metadata specifies `intensity = SNR + 1`; the implementation constructs a linear SNR-like input as `intensity - 1`. The configured `STARE_SNR_LIMIT=5.0` is used in a noise-normalized screening routine and **must not be interpreted as a threshold of 5 dB**.

## 2. Required files and directory layout

The daily driver appends `YYYY/YYYYMM/` to each configured path and `YYYYMMDD.nc` to each configured basename. Run from the repository root if using the relative directories in the example config:

```text
data/awi_288/
  stare/l1/2024/202407/
    awi_rao_dlidST00_l1_any_v04_20240720.nc
  wind/l2/2024/202407/
    awi_rao_dlidVAD00_l2_wind_v00_20240720.nc
  stare/l2/2024/202407/          # created by processing
  stare/ql/2024/202407/          # created by quicklooks
```

Rename the downloaded L1 copy to remove `(1)` before putting it into `stare/l1/2024/202407/`. The active AWI prefix in the example config matches that filename. The supporting wind filename above has now been **verified** using the supplied AWI VAD wind L2 file for 2024-07-20. It has 48 times, 250 height levels, and `wspeed` and `qwind` variables. Keep the observed input files in your own data directories rather than committing them with the code. **Wind time metadata matters:** `time` marks the end of the wind averaging interval (`time_bnds`), not its midpoint; the current processor uses that end timestamp directly for interpolation. This requires validation, particularly because many averaging intervals last ~10 minutes but occur every ~30 minutes.

**The wind L2 file is mandatory.** It must contain the selected horizontal wind-speed variable (default `wspeed`), with coordinates/dimensions suitable for interpolation to the stare sampling times and heights. The code accepts `height` directly or converts a `range` axis using `WIND_SUPPORT_ELEVATION_DEG` (60° in the supplied configuration).

## 3. Inspect input data before processing

Run in a Python environment with the toolbox dependencies installed (including the netCDF4 backend):

```python
from pathlib import Path
import numpy as np
import xarray as xr

l1 = Path("data/awi_288/stare/l1/2024/202407/awi_rao_dlidST00_l1_any_v04_20240720.nc")
with xr.open_dataset(l1) as ds:
    print(ds.sizes)
    assert {"dv", "intensity", "time", "range"} <= set(ds.variables)
    assert ds["dv"].dims == ("time", "range")
    dt_s = np.diff(ds.time.values) / np.timedelta64(1, "s")
    print("Median interval (s):", np.median(dt_s))
    print("Gaps > 10 s:", np.count_nonzero(dt_s > 10))
    print("Maximum gap (s):", dt_s.max())

wind = Path("data/awi_288/wind/l2/2024/202407/awi_rao_dlidVAD00_l2_wind_v00_20240720.nc")
assert wind.is_file(), "A matching observed wind L2 file is required for epsilon processing"
with xr.open_dataset(wind) as ds:
    assert "wspeed" in ds
    print(ds["wspeed"].dims, ds["wspeed"].sizes)
```

This check is deliberately independent of the epsilon computation. Both real input files were supplied for this test; the source-code ZIP excludes the raw observations, so users must supply their own data paths.

## 4. Run the retrieval

```bash
hpl2netCDF-client -u examples/stare_awi_288.conf -d 2024-07-20 -c l1_stare_l2
```

The processor reads both daily inputs, screens line-of-sight velocity using the intensity-derived signal, computes lagged autocovariances and a Lenschow-style noise correction, and estimates `epsilon` using an O'Connor-style relationship with the supporting horizontal wind speed. The supplied configuration uses 30-minute time bins and lag indices 0–5, fitting indices 1–5 **inclusively** (`xarray.sel(lag=slice(1,5))`).

To produce the Level-2 product **and** three quicklooks (dissipation rate, atmospheric variance, mean vertical velocity):

```bash
hpl2netCDF-client -u examples/stare_awi_288.conf -d 2024-07-20 -c l1_stare_l2wql
```

To generate quicklooks from an **existing** stare L2 file, without rerunning retrieval:

```bash
hpl2netCDF-client -u examples/stare_awi_288.conf -d 2024-07-20 -c starewql
```

The Python API provides the same functionality:

```python
from datetime import datetime
from hpl2netCDF_client.hpl2netCDF_client import hpl2netCDFClient

client = hpl2netCDFClient(
    "examples/stare_awi_288.conf", "l1_stare_l2", datetime(2024, 7, 20)
)
outfile, quicklooks = client.dailystarewql()
print(outfile)
print(quicklooks)
```

Both CLI commands are **implemented in this revision**. `starewql` requires an existing stare L2 file. Quicklooks are written to `NC_STARE_L2_QL_PATH/YYYY/YYYYMM/` and now use `NC_STARE_L2_QL_BASENAME` for their filename prefix, falling back to `NC_STARE_L2_BASENAME` if the quicklook basename is absent.

With the supplied example configuration, the resulting quicklooks are:

```text
awi_rao_dlidST00_l2_stare_ql_epsilon_20240720.png
awi_rao_dlidST00_l2_stare_ql_atmo_var_20240720.png
awi_rao_dlidST00_l2_stare_ql_wmean_20240720.png
```

## 5. Outputs

The daily L2 file uses the configured basename, e.g. `awi_rao_dlidST00_l2_stare_v00_20240720.nc`, and contains time/height grids with:

| Variable | Description |
| --- | --- |
| `epsilon` | Turbulent kinetic energy dissipation rate, m² s⁻³ |
| `qepsilon` | Retrieval flag: 1 = good by implemented finite/threshold checks, 0 = bad |
| `w_mean`, `w_std` | Screened mean vertical velocity; standard deviation of noise-corrected variance |
| `total_variance`, `atmospheric_variance`, `noise_variance` | Lag-covariance-derived variance terms, m² s⁻² |
| `n_samples`, `screen_fraction`, `snr_median` | Availability and screening diagnostics |
| `horizontal_wind_speed_for_epsilon`, `u_fill_flag` | Supporting wind speed and gap-fill provenance |
| Optional: `epsilon_uncertainty`, `atmospheric_variance_uncertainty`, `horizontal_wind_speed_for_epsilon_uncertainty` | Experimental uncertainty estimates when enabled |

`u_fill_flag` indicates 0 = original, unmodified wind speed (when smoothing windows are both one), 1 = smoothed observed value, 2 = value filled by time-domain smoothing or interpolation, 3 = value filled by height-domain smoothing or interpolation, 4 = nearest-neighbor fallback, and 5 = missing. The 2026-10-30 observed-wind validation identified and corrected a provenance-labeling bug: some wind values filled by rolling smoothing were previously assigned 5 (missing), leading to absent epsilon uncertainty estimates. `qepsilon=1` indicates passing the present numerical tests; **it is not proof of physical validity**. Numerical missing values may be stored using a `_FillValue` of `-999`.

## 6. Parameters and validation priorities

| Configuration | Supplied value | Interpretation / check |
| --- | --- | --- |
| `STARE_AVG_MIN` | `30` | Time-bin width in minutes |
| `STARE_SAMPLE_SPACING_SECONDS` | `6.0` | Assumed spacing in epsilon length-scale calculation; observed median ≈6.22 s and numerous gaps. The code estimates a nominal sample count from bin length / spacing, rather than each bin's valid count. |
| `STARE_ELEVATION_DEG` | `90.0` | Correct for the uploaded vertical-stare data |
| `WIND_SUPPORT_ELEVATION_DEG` | `60.0` | Only relevant when supporting wind data use range instead of height; confirm their scan geometry |
| `STARE_BEAM_DIVERGENCE_RAD` | `0.033` | Input to geometric length-scale correction. **Verify physical magnitude and full-angle vs half-angle convention**; the code parameter name says “half angle” yet the formula uses `sin(angle/2)`. |
| `STARE_SNR_LIMIT`, `STARE_N_NEGATIVES` | `5.0`, `37` | Cloudnet-like screening choices; assess retention of valid returns and contamination by noise |
| `STARE_LAGS`, `STARE_FIT_LAGS` | `0,1,2,3,4,5`; `1,5` | Lag covariance and the included fit-lag subset |
| `STARE_U_MIN`, `STARE_U_TIME_SMOOTH`, `STARE_U_HEIGHT_SMOOTH` | `0.1`, `6`, `3` | Input wind threshold and smoothing; inspect interpolation flags |
| `STARE_ADD_EPSILON_UNCERTAINTY` | `1` | Enable propagated estimate, using heuristic support-wind uncertainty assumptions (10%, 20%, 35% by provenance class) |

The code currently does not use `STARE_PRODUCT`/`STARE_TYPE` to dispatch retrievals; they are descriptive config entries. For scientific deployment, compare noise-corrected variance and `epsilon` against reference methods, test sensitivity to time gaps, and check the assumed divergence and uncertainty model.

## Verification status

- **Verified against the uploaded NetCDF metadata:** variable names, shapes, instrument/stare geometry, timestamps, sampling gaps and path-prefix mismatch.
- **Full-day observational computational test:** AWI stare L1 (8,873 profiles) and the matching *observed* VAD wind L2 (48 times, 250 heights) were processed together for 2024-07-20. A 48 × 250 retrieval grid, NetCDF output and three PNG quicklooks were produced. See [the observed-data validation report](observed-validation-20240720.md).
- **Regression tests:** 12 tests cover command dispatch, configuration and path handling, quicklook rendering, and wind fill-flag consistency. Run `python -m pytest -q`.
- **NetCDF backend caveat:** this environment lacks `netCDF4`, so the observed input NetCDF4 files were read with `h5py` into xarray and the resulting Level-2 dataset was written/reopened as an uncompressed SciPy NetCDF3. The standard compressed NetCDF4 CLI write path therefore remains untested here.

**Still unvalidated:** physical accuracy of epsilon, including outliers, time-gap assumptions and beam-divergence convention. A warning about poorly conditioned autocovariance fits appeared in the observational run; inspect the detailed report before using the generated fields scientifically.
