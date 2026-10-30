# Verification and scientific validation checklist

**Status: experimental.** Passing software tests is not evidence that epsilon is physically accurate.

## Regression checks

```bash
python -m compileall -q hpl2netCDF_client
python -m pytest -q
hpl2netCDF-client --help
```

These cover imports/syntax, test command dispatch, output path conventions, wind-speed gap-fill provenance, and plotting on synthetic data. An installed package is required to run the CLI; `python -m pip install -e .` requires normal internet/package access for dependencies.

## Before making a scientific L2 release

- [x] Obtain a measured VAD wind L2 product (`wspeed`) for the same day, site, and instrument as the AWI stare L1 product.
- [x] Compute the full 2024-07-20 retrieval with observed wind, write/read a NetCDF3 result, and generate all three PNG quicklooks with the processing API.
- [ ] Confirm the **normal** `l1_stare_l2wql` CLI + compressed NetCDF4 write path in an environment with `netCDF4`.
- [ ] Compare epsilon with an independent turbulence estimate and inspect stability under alternative time/lag/screening settings.
- [ ] Check `STARE_SAMPLE_SPACING_SECONDS=6.0` against the observed ~6.22 s median and investigate data gaps (max ~676 s in the supplied L1).
- [ ] Confirm whether `STARE_BEAM_DIVERGENCE_RAD=0.033` is a full angle or a half angle. The current formula computes `2*z*sin(config_angle/2)` despite the internal parameter name referring to a half angle. No formula was changed.
- [ ] Confirm the support-wind elevation and height reference (`WIND_SUPPORT_ELEVATION_DEG`).
- [ ] Assess the wind `time_bnds` intervals: wind `time` is the **interval end**, and subsequent windows cover ~10.3 min despite ~30 min temporal spacing. The current code interpolates using the endpoint timestamps, not interval midpoints.
- [ ] Review the assumed support-wind uncertainty factors in `estimate_u_uncertainty_from_fill_flag` and the meaning of `qepsilon` (numerical criterion, not independent physical quality assurance).
- [ ] Add small, permissible *observed* regression fixtures and actual compressed NetCDF end-to-end tests.
- [ ] Resolve licensing notices before public redistribution or JOSS submission.

### Test data distribution

On 2026-10-30 both observed inputs were processed for the full day using the science functions and in-memory HDF5 input decoding. An **uncompressed SciPy NetCDF3 output** was successfully written/reopened, and all three quicklooks were saved. The normal compressed NetCDF4 path is still untested because the `netCDF4` package cannot be installed in this environment. Detailed run results and warnings are in [observed-validation-20240720.md](observed-validation-20240720.md).

The original AWI input NetCDF files are intentionally **not committed or bundled** in the source repository. The separate optional validation-output directory contains derived *experimental*, not science-validated, files.
