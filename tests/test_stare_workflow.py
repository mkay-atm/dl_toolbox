"""Regression checks for the experimental stare CLI and quicklooks.

These tests use small, synthetic Level-2 grids; they are not scientific validation.
"""
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hpl2netCDF_client.command_line import main
from hpl2netCDF_client.config.config import config
from hpl2netCDF_client.hpl2netCDF_client import hpl2netCDFClient
from hpl2netCDF_client.stare_proc import (
    build_stare_l1_filename,
    build_wind_l2_filename,
    build_stare_l2_filename,
)
from hpl2netCDF_client.stare_ql import write_stare_l2_quicklooks

DATE = datetime(2024, 7, 20)
EXAMPLE_CONF = Path(__file__).resolve().parents[1] / "examples" / "stare_awi_288.conf"


def test_example_config_is_parseable_and_uses_awi_288():
    cfg = config.gen_confDict(url=str(EXAMPLE_CONF))
    assert cfg["NC_STARE_L1_BASENAME"] == "awi_rao_dlidST00_l1_any_v04_"
    assert cfg["STARE_AVG_MIN"] == "30"
    assert cfg["NC_STARE_L2_QL_BASENAME"] == "awi_rao_dlidST00_l2_stare_ql_"


def test_daily_filename_builders_do_not_create_input_directories(tmp_path):
    cfg = {
        "NC_STARE_L1_PATH": str(tmp_path / "stare_input"),
        "NC_STARE_L1_BASENAME": "stare_",
        "NC_WIND_L2_PATH": str(tmp_path / "wind_input"),
        "NC_WIND_L2_BASENAME": "wind_",
        "NC_STARE_L2_PATH": str(tmp_path / "output"),
        "NC_STARE_L2_BASENAME": "l2_",
    }
    assert build_stare_l1_filename(cfg, DATE).endswith("stare_20240720.nc")
    assert build_wind_l2_filename(cfg, DATE).endswith("wind_20240720.nc")
    assert not (tmp_path / "stare_input").exists()
    assert not (tmp_path / "wind_input").exists()
    assert build_stare_l2_filename(cfg, DATE, create_dir=False).endswith("l2_20240720.nc")
    assert not (tmp_path / "output").exists()
    build_stare_l2_filename(cfg, DATE)
    assert (tmp_path / "output" / "2024" / "202407").is_dir()


@pytest.fixture
def synthetic_stare_l2(tmp_path):
    times = pd.date_range("2024-07-20 00:15", periods=4, freq="30min")
    heights = [48.0, 96.0, 144.0, 192.0]
    shape = (len(times), len(heights))
    ds = xr.Dataset({
        "epsilon": (("time", "height"), np.full(shape, 1e-4, dtype="f4")),
        "atmospheric_variance": (("time", "height"), np.full(shape, 0.04, dtype="f4")),
        "w_mean": (("time", "height"), np.linspace(-0.2, 0.2, np.prod(shape)).reshape(shape).astype("f4")),
    }, coords={"time": times, "height": heights})
    l2_file = tmp_path / "input_stare_l2.nc"
    ds.to_netcdf(l2_file, engine="scipy")
    return l2_file


def test_stare_quicklooks_use_configured_basename(tmp_path, synthetic_stare_l2):
    cfg = {
        "NC_STARE_L2_QL_PATH": str(tmp_path / "ql"),
        "NC_STARE_L2_QL_BASENAME": "stare_preview_",
        "STARE_QL_MAX_HEIGHT": "3000",
    }
    output = write_stare_l2_quicklooks(synthetic_stare_l2, DATE, cfg)
    assert set(output) == {"epsilon", "atmospheric_variance", "wmean"}
    for name, filename in output.items():
        assert filename is not None
        assert Path(filename).is_file()
        assert Path(filename).stat().st_size > 1000
        assert Path(filename).name.startswith("stare_preview_")
        assert filename.endswith("20240720.png")


def test_stareql_missing_l2_does_not_create_output_dir(tmp_path):
    cfg = tmp_path / "config.conf"
    cfg.write_text(f"NC_STARE_L2_PATH = {tmp_path / 'output'}\nNC_STARE_L2_BASENAME = stare_\n")
    client = hpl2netCDFClient(str(cfg), "starewql", DATE)
    with pytest.raises(FileNotFoundError, match="Missing stare L2"):
        client.stareql()
    assert not (tmp_path / "output").exists()


def test_stareql_reuses_existing_l2(tmp_path, synthetic_stare_l2):
    cfg = tmp_path / "config.conf"
    root = tmp_path / "l2"
    expected = root / "2024" / "202407" / "stare_20240720.nc"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(synthetic_stare_l2.read_bytes())
    cfg.write_text(
        f"NC_STARE_L2_PATH = {root}\nNC_STARE_L2_BASENAME = stare_\n"
        f"NC_STARE_L2_QL_PATH = {tmp_path / 'ql'}\n"
        "NC_STARE_L2_QL_BASENAME = trial_\n"
    )
    client = hpl2netCDFClient(str(cfg), "starewql", DATE)
    result = client.stareql()
    assert len(result) == 3
    assert all(Path(v).is_file() for v in result.values())


@pytest.mark.parametrize("cmd,expected", [
    ("l1_stare_l2", "dailystare"),
    ("l1_stare_l2wql", "dailystarewql"),
    ("starewql", "stareql"),
])
def test_stare_command_dispatch(cmd, expected):
    with patch("hpl2netCDF_client.command_line.hpl2netCDFClient") as client_class:
        with patch.object(sys, "argv", ["hpl2netCDF-client", "-u", "site.conf", "-d", "2024-07-20", "-c", cmd]):
            main()
        client = client_class.return_value
        getattr(client, expected).assert_called_once_with()
        for other in {"dailystare", "dailystarewql", "stareql"} - {expected}:
            getattr(client, other).assert_not_called()


def test_cli_requires_explicit_configuration():
    with patch.object(sys, "argv", ["hpl2netCDF-client", "-d", "2024-07-20", "-c", "starewql"]):
        with pytest.raises(SystemExit) as ex:
            main()
    assert ex.value.code == 2


def test_quicklook_prefix_fallback_preserves_legacy_naming(tmp_path):
    from hpl2netCDF_client.stare_ql import build_stare_ql_filename
    cfg = {
        "NC_STARE_L2_QL_PATH": str(tmp_path / "ql"),
        "NC_STARE_L2_BASENAME": "stare_",
    }
    assert Path(build_stare_ql_filename(cfg, DATE, "epsilon")).name == "stare_epsilon_ql_20240720.png"


def test_wind_smoothing_provenance_flags_and_uncertainty():
    """Missing raw samples filled by rolling windows must not be marked missing."""
    import numpy as np
    import pandas as pd
    import xarray as xr

    from hpl2netCDF_client.stare_calc import (
        build_u_for_epsilon_xr, estimate_u_uncertainty_from_fill_flag,
    )

    winds = xr.DataArray(
        np.array([[4.0, np.nan], [np.nan, np.nan], [5.0, 6.0]]),
        dims=("time", "height"),
        coords={"time": pd.date_range("2024-07-20", periods=3, freq="30min"),
                "height": [100., 200.]},
    )
    u, flags = build_u_for_epsilon_xr(
        winds, time_smooth=3, height_smooth=2, min_u=0.1,
    )
    assert bool(u.notnull().all())
    assert not bool(((flags == 5) & u.notnull()).any())
    assert int(flags.sel(time="2024-07-20T00:30:00", height=100)) == 2
    assert int(flags.sel(time="2024-07-20T00:00:00", height=200)) == 3
    uncertainty = estimate_u_uncertainty_from_fill_flag(u, flags)
    assert bool(uncertainty.notnull().all())


def test_wind_no_smoothing_retains_original_flag():
    import numpy as np
    import pandas as pd
    import xarray as xr
    from hpl2netCDF_client.stare_calc import build_u_for_epsilon_xr
    winds = xr.DataArray(np.array([[4.0, np.nan], [5.0, np.nan]]),
       dims=("time", "height"),
       coords={"time": pd.date_range("2024-07-20", periods=2, freq="30min"),
               "height": [100., 200.]})
    _, flags = build_u_for_epsilon_xr(
        winds, time_smooth=1, height_smooth=1,
        interp_time=False, interp_height=False, nearest_fallback=False,
    )
    assert (flags.sel(height=100.) == 0).all().item()
    assert (flags.sel(height=200.) == 5).all().item()
