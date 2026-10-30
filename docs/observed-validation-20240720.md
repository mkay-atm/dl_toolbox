# Observed-data software verification: AWI 288, 2024-07-20

**Date tested:** 2026-10-30  
**Status:** completed observational **computational verification**, not independent scientific validation of turbulence dissipation rate.

## Inputs

| Property | Stare Level 1 | VAD wind Level 2 |
| --- | --- | --- |
| Filename | `awi_rao_dlidST00_l1_any_v04_20240720.nc` | `awi_rao_dlidVAD00_l2_wind_v00_20240720.nc` |
| Instrument / site | Halo Stream Line XR+, S/N 288; AWI Lindenberg | Halo Stream Line XR+, S/N 288; AWI Lindenberg |
| Time | 00:10:44–23:59:58 UTC | 00:28:20–23:59:08 UTC |
| Grid | 8,873 time × 250 range | 48 time × 250 height |
| Required data | `dv`, `intensity` | `wspeed`; `qwind` quality flag |
| Vertical coordinates | Range 24–11,976 m | Height 23.18–11,567.93 m |

The median stare sampling interval is about 6.22 s (configured as 6.0 s); the longest observed gap is about 676 s. **Wind temporal-coordinate caveat:** `time` is documented as the **end** of the wind averaging interval, not its midpoint. The supporting `time_bnds` variable shows the first wind interval as 00:00:21–00:28:20 UTC (~28 minutes), but several later intervals span only ~10.3 minutes and recur every ~30 minutes. The current stare processor interpolates wind using the `time` coordinate directly, without re-centering on the interval midpoint. This potential timing offset should be assessed before scientific interpretation. The wind product is approximately half-hourly. Of the wind product's 12,000 time–height cells, **2,469 (20.6%)** have non-fill `wspeed`, matching the cells where `qwind=1`. In this verification, `qwind=0` and `wspeed=-999` were treated as missing.

## Full-day processing results

With `examples/stare_awi_288.conf` (30-minute bins, fitted lags 1–5, enabled uncertainties):

| Metric | Result |
| --- | ---: |
| Output grid | 48 × 250 = 12,000 time–height cells |
| `qepsilon=1` / non-missing `epsilon` | **4,345 (36.2%)** |
| Median `epsilon` among accepted cells | ~2.03 × 10⁻⁵ m² s⁻³ |
| 75th percentile among accepted cells | ~3.84 × 10⁻⁴ m² s⁻³ |
| Maximum accepted `epsilon` | **~106.89 m² s⁻³** (unphysical for most intended atmospheric conditions; investigate) |
| Accepted `epsilon` above 0.1 m² s⁻³ | **269** |
| Accepted `epsilon` above 1 m² s⁻³ | **140** |
| Accepted `epsilon` with fewer than 10 valid stare samples | **54** |
| Accepted `epsilon` with fewer than 30 valid stare samples | **563** |

**Caution:** `qepsilon` currently applies numerical availability/variance criteria, not an independently calibrated turbulence-quality filter. High reported epsilon values and high absolute vertical velocity estimates (up to ~39 m/s) require investigation. A `RankWarning: Polyfit may be poorly conditioned` occurred during the lagged covariance fit, indicating that some lag fits may be unreliable.

## Bug identified and corrected

Previously, the supporting-wind filling routine marked some samples created by rolling smoothing as `u_fill_flag=5` (missing) even though a finite wind speed was used for epsilon. Consequently, **1,572 accepted epsilon cells had no epsilon uncertainty estimate**.

The revised `stare_calc.py` assigns provenance flags consistently across time smoothing, height smoothing and interpolation. Reprocessing the *same observed inputs* produced:

| Test | Before fix | After fix |
| --- | ---: | ---: |
| Accepted epsilon cells | 4,345 | 4,345 |
| Accepted epsilon cells incorrectly flagged as missing wind | 1,572 | **0** |
| Accepted epsilon cells missing `epsilon_uncertainty` | 1,572 | **0** |
| Distribution of `u_fill_flag` among accepted cells | 1: 2,078; 2: 475; 3: 220; 5: 1,572 | 1: 2,078; 2: 1,614; 3: 653; 5: **0** |

The epsilon values and numerical quality flags are unchanged by this correction; **reported uncertainty values change** where the wind provenance was previously misidentified. Flags 2/3 now cover gaps filled by rolling smoothing and later interpolation, with the code's existing relative wind uncertainty assumptions (20% for those classes). These assumptions remain heuristic.

## Execution and backend limitations

- **Done:** Full day of observed stare L1 and observed wind L2 were read via `h5py`, with `_FillValue` values masked and VAD `qwind` applied, then passed into `process_stare_l2_from_ds`.
- **Done:** The output dataset was written to and reopened from **uncompressed NetCDF3** using the SciPy engine.
- **Done:** All three quicklooks (`epsilon`, atmospheric variance, mean vertical velocity) were generated from that NetCDF output.
- **Done:** 12 automated regression tests passed.
- **Not tested here:** direct NetCDF4 input via `xr.open_dataset`, the standard daily CLI's compressed NetCDF4 write path, or scientific agreement with a reference turbulence estimate. The `netCDF4` module is not installed in this environment.

The source ZIP does **not** include the original measurement inputs. A separate `validation-results/` folder in the distribution ZIP stores the derived uncompressed example L2 and three generated PNGs; these are *experimental diagnostics*, not certified Level-2 science products.

## Next validation priorities

1. Examine the upper-tail epsilon retrievals (>0.1 m² s⁻³) together with raw radial velocities, screening fraction, atmospheric variance, covariance fit stability and supporting wind data. Do not simply clip without a scientific rationale.
2. Evaluate minimum valid samples and the effect of 6.0 s assumed vs ~6.22 s observed median cadence and long gaps on the chosen length scale. Examine support-wind `time_bnds` and whether time-end versus interval-midpoint alignment biases the interpolation.
3. Confirm the physical beam divergence magnitude and angular convention (parameter named `beam_divergence_half_angle_rad` but the formula uses `sin(angle/2)`).
4. Evaluate whether additional wind support QC/provenance constraints and uncertainty assumptions are needed.
5. Verify the normal `l1_stare_l2wql` CLI in an environment with a working NetCDF4 backend, then compare against independent turbulence observations.
