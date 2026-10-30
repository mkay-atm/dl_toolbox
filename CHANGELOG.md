# Development changelog

## Observed wind L2 integration and provenance fix — 2026-10-30

- Process a full observed AWI stare day (2024-07-20, instrument 288) with its matching observed VAD wind L2: 48 × 250 output grid, 4,345 non-missing epsilon retrievals, three quicklooks.
- Fix `build_u_for_epsilon_xr` incorrectly labeling some finite rolling-smoothed support wind speeds as missing (flag 5). Time- and height-smoothed gap fills now receive flags 2 and 3, respectively; this restores uncertainty estimates for affected retrieved cells.
- Add two regression tests for smoothing provenance and the unsmoothed original-value flag (12 tests passing).
- Add observed-data validation report; document suspicious epsilon outliers and a polyfit rank warning. Physics and numeric `qepsilon` criteria remain unchanged.
- Verified observed processing and uncompressed NetCDF3 write/read using an HDF5 input adapter; normal compressed NetCDF4 CLI output is still untested in this environment.

## Experimental stare documentation and CLI update — 2026-10-30

- Implement `hpl2netCDFClient.stareql()` for quicklook generation from an **existing** stare L2 NetCDF.
- Route `l1_stare_l2wql` to `dailystarewql()` and retain `starewql` for plots only.
- Apply `NC_STARE_L2_QL_BASENAME` in quicklook filenames (fallback: `NC_STARE_L2_BASENAME`).
- Share daily L2 filename construction between processing and plotting; avoid creating input directories just by constructing filenames.
- Require explicit `-u/--url` on CLI rather than relying on a developer-specific Windows path.
- Improve CLI command descriptions and remove duplicate stare processing import.
- Correct two Python string-literal escape warnings in legacy HPL header parsing (regular-expression semantics unchanged).
- Add AWI instrument 288 example and stare processing instructions; expand README.
- Remove redundant standard-library runtime dependencies from setup metadata, include the Markdown README in package metadata, and use package discovery.
- Add synthetic quicklook and command-dispatch regression tests.

**Not changed:** epsilon retrieval physics, the O'Connor length-scale formula, assumed sampling interval, beam-divergence convention, default screening, uncertainty model, the wind processing algorithms, or existing licensing notices.

**Outstanding:** verify geometrical beam divergence and sampling assumptions, validate epsilon scientifically against independent observations, verify production compressed NetCDF4 CLI output, extend regression coverage to wind and NRT modes, and resolve BSD vs non-commercial notices before public release.
