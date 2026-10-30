# Observed-input computation — 2024-07-20

Derived **experimental** output from the AWI Halo Stream Line XR+ instrument 288 stare L1 file and matching VAD wind L2 file, with the example `examples/stare_awi_288.conf` processing settings.

- `stare_observed_wind_20240720_UNCOMPRESSED.nc`: 48 × 250 time–height L2 result in **uncompressed NetCDF3 format**, written using the SciPy backend. This is not the normal compressed NetCDF4 production file.
- `quicklooks/`: three PNG products: epsilon, atmospheric variance, mean vertical velocity.
- The input AWI measurement files are **not** bundled.

**Scientific caution:** These outputs verify code execution with measured support wind, but are **not scientifically validated** epsilon observations. The run includes extreme positive epsilon values, including many marked qepsilon=1, and a poorly conditioned polynomial fit warning. See `../dl_toolbox-experimental/docs/observed-validation-20240720.md` for details.

To reproduce with the *production* CLI after placing both original inputs under the data roots configured in the example and installing required packages including `netCDF4`:

```bash
hpl2netCDF-client -u examples/stare_awi_288.conf -d 2024-07-20 -c l1_stare_l2wql
```

The production command and its compressed NetCDF4 output path were not verified in the packaging environment, which lacks `netCDF4`.
