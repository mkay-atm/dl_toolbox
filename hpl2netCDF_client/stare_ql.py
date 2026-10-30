from __future__ import annotations

import os
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.ticker import MultipleLocator

from hpl2netCDF_client.config.config import config

def _parse_str(confDict, key, default):
    return str(confDict.get(key, default))


def _parse_float(confDict, key, default):
    return float(confDict.get(key, default))

def _dated_subdir(base_path: str, date_chosen: dt.datetime) -> str:
    year = pd.Timestamp(date_chosen).strftime("%Y")
    yearmonth = pd.Timestamp(date_chosen).strftime("%Y%m")
    outdir = os.path.join(base_path, year, yearmonth)
    os.makedirs(outdir, exist_ok=True)
    return outdir


def _infer_instrument_prefix(confDict):
    system = _parse_str(confDict, "SYSTEM", "dl")
    scan_type = _parse_str(confDict, "SCAN_TYPE", "stare")
    sn = _parse_str(confDict, "SYSTEM_ID", confDict.get("SYSTEM_SN", "unknown"))
    return f"{system}_{scan_type}_{sn}"


def build_stare_ql_filename(confDict, date_chosen, field_name):
    out_base = _parse_str(confDict, "NC_STARE_L2_QL_PATH", confDict.get("NC_L2_QL_PATH", "."))
    basename = confDict.get("NC_STARE_L2_QL_BASENAME")
    if basename is None:
        # Preserve legacy quicklook filenames when the dedicated prefix is absent.
        basename = confDict.get("NC_STARE_L2_BASENAME", _infer_instrument_prefix(confDict) + "_l2_stare_")
        field_suffix = f"{field_name}_ql_"
    else:
        field_suffix = f"{field_name}_"

    outdir = _dated_subdir(out_base, date_chosen)
    fname = f"{basename}{field_suffix}{pd.Timestamp(date_chosen).strftime('%Y%m%d')}.png"
    return os.path.join(outdir, fname)


def _setup_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    ax.tick_params(axis="both", which="major", direction="in", top=True, right=True, length=10)
    ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True, length=5)


def _prepare_time_height(ds, max_height=None):
    ds_plot = ds
    if max_height is not None and "height" in ds_plot.coords:
        ds_plot = ds_plot.sel(height=ds_plot.height <= max_height)
    return ds_plot


def _plot_panel(ax, da, title, cmap="viridis", vmin=None, vmax=None, logscale=False, cbar_label=None):
    plot_da = da.copy()

    if logscale:
        plot_da = xr.where(plot_da > 0, np.log10(plot_da), np.nan)

    im = plot_da.T.plot(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=True,
        cbar_kwargs={"label": cbar_label},
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("height [m]")
    _setup_time_axis(ax)
    return im


def _setup_dltoolbox_axes_style(ax):
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)


def _setup_dltoolbox_time_axis(ax, date_chosen, time_delta):
    d0 = date_chosen.date()
    d = dt.datetime(d0.year, d0.month, d0.day) - dt.timedelta(hours=time_delta)
    dp1 = dt.datetime(d0.year, d0.month, d0.day) + dt.timedelta(days=1) - dt.timedelta(hours=time_delta)

    ax.set_xlim(d, dp1)
    ax.set_aspect('auto')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax.xaxis.set_major_locator(
        mdates.HourLocator(byhour=np.mod(range(0 - time_delta, 24 - time_delta, 6), 24))
    )
    ax.xaxis.set_minor_locator(
        mdates.HourLocator(byhour=np.mod(range(0 - time_delta, 24 - time_delta, 1), 24))
    )

    ax.tick_params(axis='both', labelsize=18, length=34, width=2., pad=7.78,
                   which='major', direction='in', top=True, right=True)
    ax.tick_params(axis='both', labelsize=18, length=23, width=1.,
                   which='minor', direction='in', top=True, right=True)


def _setup_dltoolbox_height_axis(ax, height, confDict):
    hmax_cfg = confDict.get('STARE_QL_MAX_HEIGHT', None)
    hmax_raw = float(hmax_cfg) if hmax_cfg is not None else float(np.nanmax(height))
    hmax = np.round(hmax_raw, -2) + 100

    if hmax >= 6000:
        delmajor = 2000
        delnom = 4
    elif (hmax > 2000) * (hmax < 6000):
        delmajor = 1000
        delnom = 4
    else:
        delmajor = 500
        delnom = 5

    ax.yaxis.set_major_locator(MultipleLocator(delmajor))
    ax.yaxis.set_minor_locator(MultipleLocator(delmajor // delnom))
    ax.set_ylim([0, hmax])


def _plot_stare_single_panel(
    date_chosen,
    confDict,
    time_mean,
    height,
    Z,
    *,
    title,
    cbar_label,
    outfile,
    cmap,
    vmin,
    vmax,
    logscale=True,
):
    if 'UTC_OFFSET' in confDict:
        time_delta = int(confDict['UTC_OFFSET'])
    else:
        time_delta = 0

    fig, axes = plt.subplots(1, 1, figsize=(18, 12))
    fig.set_facecolor('w')
    ax = axes

    _setup_dltoolbox_axes_style(ax)

    X, Y = np.meshgrid(mdates.date2num(pd.to_datetime(time_mean)), height)

    if np.all(np.isnan(Z)):
        print('all input is NaN')
        print('no quicklook is saved')
        plt.close(fig)
        return None

    if logscale:
        mask = np.isnan(Z) | (Z <= 0)
        masked_Z = np.ma.masked_where(mask, Z)
        c_temp = ax.pcolormesh(
            X.T, Y.T, masked_Z,
            cmap=cmap,
            norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        )
    else:
        mask = np.isnan(Z)
        masked_Z = np.ma.masked_where(mask, Z)
        c_temp = ax.pcolormesh(
            X.T, Y.T, masked_Z,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    cbar = fig.colorbar(c_temp, ax=ax, extend='both', pad=0.01)
    cbar.set_label(cbar_label, rotation=270, fontsize=22, labelpad=37)
    cbar.ax.tick_params(which='major', direction='out', length=14, width=2, labelsize=22)
    cbar.ax.tick_params(which='minor', direction='out', length=8, width=2, labelsize=22)

    ax.set_title(title, fontsize=22)
    ax.set_xlabel(date_chosen.strftime('%Y-%m-%d') + '\n' + 'time (UTC)', fontsize=22)
    ax.set_ylabel(r'$\rm{height}\;/\;\rm{m}$', fontsize=22)

    _setup_dltoolbox_time_axis(ax, date_chosen, time_delta)
    _setup_dltoolbox_height_axis(ax, height, confDict)

    if outfile is None:
        plt.close(fig)
        return None
    print('saving stare quicklook as... \n ... ' + str(outfile))
    fig.savefig(str(outfile), transparent=False, bbox_inches='tight')
    plt.close(fig)
    return str(outfile)


# def stare_epsilon_ql(self):
#     date_chosen = self.date2proc
#     confDict = config.gen_confDict(url=self.config_dir)

#     if 'UTC_OFFSET' in confDict:
#         time_delta = int(confDict['UTC_OFFSET'])
#     else:
#         time_delta = 0

#     # read stare level-2 product
#     ds = xr.open_dataset(self.l2_stare_filename())
#     try:
#         time_mean = pd.to_datetime(ds.time.data)
#         height = ds.height.data
#         Z = np.array(ds.epsilon.data, dtype=float)

#         vmin = float(confDict.get('STARE_QL_EPS_MIN', 1e-5))
#         vmax = float(confDict.get('STARE_QL_EPS_MAX', 1e-2))

#         fig, axes = plt.subplots(1, 1, figsize=(18, 12))
#         fig.set_facecolor('w')
#         ax = axes

#         ax.spines['left'].set_linewidth(2)
#         ax.spines['right'].set_linewidth(2)
#         ax.spines['top'].set_linewidth(2)
#         ax.spines['bottom'].set_linewidth(2)

#         X, Y = np.meshgrid(mdates.date2num(time_mean), height)

#         if np.all(np.isnan(Z)):
#             print('all epsilon input is NaN')
#             print('no quicklook is saved')
#             plt.close(fig)
#             return

#         mask = np.isnan(Z) | (Z <= 0)
#         masked_Z = np.ma.masked_where(mask, Z)

#         c_temp = ax.pcolormesh(
#             X.T,
#             Y.T,
#             masked_Z,
#             cmap=plt.cm.viridis,
#             norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
#         )

#         cbar = fig.colorbar(c_temp, ax=ax, extend='both', pad=0.01)
#         cbar.set_label(
#             r'$\rm{TKE\;dissipation\;rate}\;/\;\rm{m}^{2}\,\rm{s}^{-3}$',
#             rotation=270,
#             fontsize=22,
#             labelpad=37,
#         )
#         cbar.ax.tick_params(which='major', direction='out', length=14, width=2, labelsize=22)
#         cbar.ax.tick_params(which='minor', direction='out', length=8, width=2, labelsize=22)

#         d0 = date_chosen.date()
#         d = dt.datetime(d0.year, d0.month, d0.day) - dt.timedelta(hours=time_delta)
#         dp1 = dt.datetime(d0.year, d0.month, d0.day) + dt.timedelta(days=1) - dt.timedelta(hours=time_delta)

#         ax.set_xlabel(date_chosen.strftime('%Y-%m-%d') + '\n' + 'time (UTC)', fontsize=22)
#         ax.set_ylabel(r'$\rm{height}\;/\;\rm{m}$', fontsize=22)
#         ax.set_xlim(d, dp1)
#         ax.set_aspect('auto')

#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#         ax.xaxis.set_major_locator(
#             mdates.HourLocator(byhour=np.mod(range(0 - time_delta, 24 - time_delta, 6), 24))
#         )
#         ax.xaxis.set_minor_locator(
#             mdates.HourLocator(byhour=np.mod(range(0 - time_delta, 24 - time_delta, 1), 24))
#         )

#         hmax = float(confDict.get('STARE_QL_MAX_HEIGHT', height[-1]))
#         if hmax >= 6000:
#             delmajor = 2000
#             delnom = 4
#         elif (hmax > 2000) * (hmax < 6000):
#             delmajor = 1000
#             delnom = 4
#         else:
#             delmajor = 500
#             delnom = 5

#         ax.yaxis.set_major_locator(MultipleLocator(delmajor))
#         ax.yaxis.set_minor_locator(MultipleLocator(delmajor // delnom))
#         ax.set_ylim([0, hmax])

#         ax.tick_params(axis='both', labelsize=18, length=34, width=2., pad=7.78,
#                        which='major', direction='in', top=True, right=True)
#         ax.tick_params(axis='both', labelsize=18, length=23, width=1.,
#                        which='minor', direction='in', top=True, right=True)

#         path = Path(confDict['NC_STARE_L2_QL_PATH'] + '/' + date_chosen.strftime('%Y') + '/' + date_chosen.strftime('%Y%m') + '/')
#         path.mkdir(parents=True, exist_ok=True)

#         outfile = str('{}/' + confDict['NC_STARE_L2_BASENAME'] + 'epsilon_ql_' + date_chosen.strftime('%Y%m%d') + '.png').format(path)
#         print('saving epsilon quicklook as... \n ... ' + outfile)
#         fig.savefig(outfile, transparent=False, bbox_inches='tight')
#         plt.close(fig)
#     finally:
#         ds.close()

# def stare_variance_ql(self):
#     date_chosen = self.date2proc
#     confDict = config.gen_confDict(url=self.config_dir)

#     if 'UTC_OFFSET' in confDict:
#         time_delta = int(confDict['UTC_OFFSET'])
#     else:
#         time_delta = 0

#     ds = xr.open_dataset(self.l2_stare_filename())
#     try:
#         time_mean = pd.to_datetime(ds.time.data)
#         height = ds.height.data
#         Z = np.array(ds.atmospheric_variance.data, dtype=float)

#         vmin = float(confDict.get('STARE_QL_VAR_MIN', 1e-4))
#         vmax = float(confDict.get('STARE_QL_VAR_MAX', 1e0))

#         fig, axes = plt.subplots(1, 1, figsize=(18, 12))
#         fig.set_facecolor('w')
#         ax = axes

#         ax.spines['left'].set_linewidth(2)
#         ax.spines['right'].set_linewidth(2)
#         ax.spines['top'].set_linewidth(2)
#         ax.spines['bottom'].set_linewidth(2)

#         X, Y = np.meshgrid(mdates.date2num(time_mean), height)

#         if np.all(np.isnan(Z)):
#             print('all atmospheric variance input is NaN')
#             print('no quicklook is saved')
#             plt.close(fig)
#             return

#         mask = np.isnan(Z) | (Z <= 0)
#         masked_Z = np.ma.masked_where(mask, Z)

#         c_temp = ax.pcolormesh(
#             X.T,
#             Y.T,
#             masked_Z,
#             cmap=plt.cm.plasma,
#             norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
#         )

#         cbar = fig.colorbar(c_temp, ax=ax, extend='both', pad=0.01)
#         cbar.set_label(
#             r'$\rm{atmospheric\;variance}\;/\;\rm{m}^{2}\,\rm{s}^{-2}$',
#             rotation=270,
#             fontsize=22,
#             labelpad=37,
#         )
#         cbar.ax.tick_params(which='major', direction='out', length=14, width=2, labelsize=22)
#         cbar.ax.tick_params(which='minor', direction='out', length=8, width=2, labelsize=22)

#         d0 = date_chosen.date()
#         d = dt.datetime(d0.year, d0.month, d0.day) - dt.datetime.timedelta(hours=time_delta)
#         dp1 = dt.datetime(d0.year, d0.month, d0.day) + dt.datetime.timedelta(days=1) - dt.datetime.timedelta(hours=time_delta)

#         ax.set_xlabel(date_chosen.strftime('%Y-%m-%d') + '\n' + 'time (UTC)', fontsize=22)
#         ax.set_ylabel(r'$\rm{height}\;/\;\rm{m}$', fontsize=22)
#         ax.set_xlim(d, dp1)
#         ax.set_aspect('auto')

#         ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
#         ax.xaxis.set_major_locator(
#             mdates.HourLocator(byhour=np.mod(range(0 - time_delta, 24 - time_delta, 6), 24))
#         )
#         ax.xaxis.set_minor_locator(
#             mdates.HourLocator(byhour=np.mod(range(0 - time_delta, 24 - time_delta, 1), 24))
#         )

#         hmax = float(confDict.get('STARE_QL_MAX_HEIGHT', height[-1]))
#         if hmax >= 6000:
#             delmajor = 2000
#             delnom = 4
#         elif (hmax > 2000) * (hmax < 6000):
#             delmajor = 1000
#             delnom = 4
#         else:
#             delmajor = 500
#             delnom = 5

#         ax.yaxis.set_major_locator(MultipleLocator(delmajor))
#         ax.yaxis.set_minor_locator(MultipleLocator(delmajor // delnom))
#         ax.set_ylim([0, hmax])

#         ax.tick_params(axis='both', labelsize=18, length=34, width=2., pad=7.78,
#                        which='major', direction='in', top=True, right=True)
#         ax.tick_params(axis='both', labelsize=18, length=23, width=1.,
#                        which='minor', direction='in', top=True, right=True)

#         path = Path(confDict['NC_STARE_L2_QL_PATH'] + '/' + date_chosen.strftime('%Y') + '/' + date_chosen.strftime('%Y%m') + '/')
#         path.mkdir(parents=True, exist_ok=True)

#         outfile = str('{}/' + confDict['NC_STARE_L2_BASENAME'] + 'atmo_var_ql_' + date_chosen.strftime('%Y%m%d') + '.png').format(path)
#         print('saving atmospheric variance quicklook as... \n ... ' + outfile)
#         fig.savefig(outfile, transparent=False, bbox_inches='tight')
#         plt.close(fig)
#     finally:
#         ds.close()
# def quicklook_stare_epsilon(ds, date_chosen, confDict, savefig=True):
#     """
#     Two-panel quicklook:
#       - top: mean vertical velocity
#       - bottom: log10(epsilon)
#     """
#     max_height = _parse_float(confDict, "STARE_QL_MAX_HEIGHT", 3000.0)
#     wmean_abs = _parse_float(confDict, "STARE_QL_WMEAN_ABS", 2.0)
#     eps_min = _parse_float(confDict, "STARE_QL_EPS_MIN", 1e-5)
#     eps_max = _parse_float(confDict, "STARE_QL_EPS_MAX", 1e-2)

#     ds_plot = _prepare_time_height(ds, max_height=max_height)

#     fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)

#     _plot_panel(
#         axes[0],
#         ds_plot["w_mean"],
#         title="Mean vertical velocity",
#         cmap="RdBu_r",
#         vmin=-wmean_abs,
#         vmax=wmean_abs,
#         logscale=False,
#         cbar_label="w_mean [m s$^{-1}$]",
#     )

#     _plot_panel(
#         axes[1],
#         ds_plot["epsilon"],
#         title="TKE dissipation rate",
#         cmap="viridis",
#         vmin=np.log10(eps_min),
#         vmax=np.log10(eps_max),
#         logscale=True,
#         cbar_label="log10(epsilon [m$^2$ s$^{-3}$])",
#     )

#     fig.suptitle(f"Stare quicklook: epsilon {pd.Timestamp(date_chosen).strftime('%Y-%m-%d')}")

#     outfile = None
#     if savefig:
#         outfile = build_stare_ql_filename(confDict, date_chosen, "epsilon")
#         fig.savefig(outfile, dpi=150, bbox_inches="tight")
#         plt.close(fig)

#     return outfile

def quicklook_stare_epsilon(ds, date_chosen, confDict, savefig=True):
    time_mean = pd.to_datetime(ds.time.data)
    height = ds.height.data
    Z = np.array(ds.epsilon.data, dtype=float)

    vmin = float(confDict.get('STARE_QL_EPS_MIN', 1e-5))
    vmax = float(confDict.get('STARE_QL_EPS_MAX', 1e-2))

    outfile = build_stare_ql_filename(confDict, date_chosen, "epsilon")

    if not savefig:
        outfile = None

    return _plot_stare_single_panel(
        date_chosen,
        confDict,
        time_mean,
        height,
        Z,
        title=r'$\rm{TKE\;dissipation\;rate}$',
        cbar_label=r'$\rm{TKE\;dissipation\;rate}\;/\;\rm{m}^{2}\,\rm{s}^{-3}$',
        outfile=outfile,
        cmap=cm.CMRmap,
        vmin=vmin,
        vmax=vmax,
        logscale=True,
    )


# def quicklook_stare_atmospheric_variance(ds, date_chosen, confDict, savefig=True):
#     """
#     Two-panel quicklook:
#       - top: mean vertical velocity
#       - bottom: log10(atmospheric variance)
#     """
#     max_height = _parse_float(confDict, "STARE_QL_MAX_HEIGHT", 3000.0)
#     wmean_abs = _parse_float(confDict, "STARE_QL_WMEAN_ABS", 2.0)
#     var_min = _parse_float(confDict, "STARE_QL_VAR_MIN", 1e-4)
#     var_max = _parse_float(confDict, "STARE_QL_VAR_MAX", 1e0)

#     ds_plot = _prepare_time_height(ds, max_height=max_height)

#     fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)

#     _plot_panel(
#         axes[0],
#         ds_plot["w_mean"],
#         title="Mean vertical velocity",
#         cmap="RdBu_r",
#         vmin=-wmean_abs,
#         vmax=wmean_abs,
#         logscale=False,
#         cbar_label="w_mean [m s$^{-1}$]",
#     )

#     _plot_panel(
#         axes[1],
#         ds_plot["atmospheric_variance"],
#         title="Atmospheric variance of vertical velocity",
#         cmap="plasma",
#         vmin=np.log10(var_min),
#         vmax=np.log10(var_max),
#         logscale=True,
#         cbar_label="log10(var [m$^2$ s$^{-2}$])",
#     )

#     fig.suptitle(f"Stare quicklook: atmospheric variance {pd.Timestamp(date_chosen).strftime('%Y-%m-%d')}")

#     outfile = None
#     if savefig:
#         outfile = build_stare_ql_filename(confDict, date_chosen, "atmospheric_variance")
#         fig.savefig(outfile, dpi=150, bbox_inches="tight")
#         plt.close(fig)

#     return outfile

def quicklook_stare_atmospheric_variance(ds, date_chosen, confDict, savefig=True):
    time_mean = pd.to_datetime(ds.time.data)
    height = ds.height.data
    Z = np.array(ds.atmospheric_variance.data, dtype=float)

    vmin = float(confDict.get('STARE_QL_VAR_MIN', 1e-4))
    vmax = float(confDict.get('STARE_QL_VAR_MAX', 1e0))

    outfile = build_stare_ql_filename(confDict, date_chosen, "atmo_var")

    if not savefig:
        outfile = None

    return _plot_stare_single_panel(
        date_chosen,
        confDict,
        time_mean,
        height,
        Z,
        title=r'$\rm{atmospheric\;variance}$',
        cbar_label=r'$\rm{atmospheric\;variance}\;/\;\rm{m}^{2}\,\rm{s}^{-2}$',
        outfile=outfile,
        cmap=cm.Spectral_r,
        vmin=vmin,
        vmax=vmax,
        logscale=True,
    )

def quicklook_stare_wmean(ds, date_chosen, confDict, savefig=True):
    time_mean = pd.to_datetime(ds.time.data)
    height = ds.height.data
    Z = np.array(ds.w_mean.data, dtype=float)

    wmax = float(confDict.get('STARE_QL_WMEAN_ABS', 2.0))

    outfile = build_stare_ql_filename(confDict, date_chosen, "wmean")

    if not savefig:
        outfile = None

    return _plot_stare_single_panel(
        date_chosen,
        confDict,
        time_mean,
        height,
        Z,
        title=r'$\rm{mean\;vertical\;wind\;speed}$',
        cbar_label=r'$\rm{mean\;vertical\;wind\;speed}\;/\;\rm{m}\,\rm{s}^{-1}$',
        outfile=outfile,
        cmap=cm.seismic,
        vmin=-wmax,
        vmax=wmax,
        logscale=False,
    )

def write_stare_l2_quicklooks(l2_stare_file, date_chosen, confDict):
    """
    Read a stare Level-2 file and write three quicklooks.
    """
    ds = xr.open_dataset(l2_stare_file)
    try:
        out_eps = quicklook_stare_epsilon(ds, date_chosen, confDict, savefig=True)
        out_var = quicklook_stare_atmospheric_variance(ds, date_chosen, confDict, savefig=True)
        out_wmean = quicklook_stare_wmean(ds, date_chosen, confDict, savefig=True)
    finally:
        ds.close()

    return {"epsilon": out_eps, "atmospheric_variance": out_var, "wmean": out_wmean}