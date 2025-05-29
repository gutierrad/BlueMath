import os.path as op
from typing import Dict, List, Optional

import cartopy
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from bluemath_tk.tcs.tracks import historic_track_preprocessing
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from .common import custom_cmap

cmap_wa = custom_cmap(15, "YlOrRd", 0.15, 0.9, "YlGnBu_r", 0, 0.85)


def get_storm_color(categ: int) -> str:
    """
    Get the color associated with a storm category.

    Parameters
    ----------
    categ : int
        Storm category (0-6).

    Returns
    -------
    str
        Color name associated with the category.
    """

    dcs: Dict[int, str] = {
        0: "lime",
        1: "yellow",
        2: "orange",
        3: "red",
        4: "purple",
        5: "black",
        6: "gray",
    }

    return dcs[categ]


def fix_extent(ax: Axes, extent: List[float]) -> None:
    """
    Adjusts the y-axis limits of a matplotlib Axes object based on the given extent.

    Parameters
    ----------
    ax : Axes
        The Axes object to adjust.
    extent : List[float]
        The extent of the data in the form [min_longitude, max_longitude, min_latitude, max_latitude].
    """

    mlon = np.mean(extent[:2])
    mlat = np.mean(extent[2:])
    xtrm_data = np.array(
        [[extent[0], mlat], [mlon, extent[2]], [extent[1], mlat], [mlon, extent[3]]]
    )
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    xtrm = proj_to_data.transform(xtrm_data)
    ax.set_ylim(xtrm[:, 1].min(), xtrm[:, 1].max())  # only latitudes


def axplot_cartopy_robin(
    ax: Axes,
    resolution: str = "l",
    cfill: str = "silver",
    cocean: str = "azure",
    mode_simple: bool = False,
) -> Axes:
    """
    Plot a Cartopy Robinson projection on the given axes.

    Parameters
    ----------
    ax : Axes
        The axes on which to plot the map.
    resolution : str, optional
        The resolution of the map features, by default 'l'.
    cfill : str, optional
        The color to fill the land area, by default 'silver'.
    cocean : str, optional
        The color to fill the ocean area, by default 'azure'.
    mode_simple : bool, optional
        Whether to use a simplified mode, by default False.

    Returns
    -------
    Axes
        The modified axes object.
    """

    ax.set_global()
    ax.add_feature(cartopy.feature.LAND, facecolor=cfill, zorder=20)
    ax.add_feature(cartopy.feature.OCEAN, facecolor=cocean)
    ax.gridlines()
    extent = (-180, 180, -60, 58)
    fix_extent(ax, extent)

    return ax


def axplot_shy_swath_cartopy(
    fig: Figure,
    ax: Axes,
    xds_bulk: xr.Dataset,
    storm: xr.Dataset,
    var: str = "hswath",
    vmin: float = 0,
    vmax: float = 10,
    cmap: str = "hot",
    label: str = "Hs (m)",
    name: str = "",
    storm_year: int = 2000,
    colorbar: bool = True,
) -> AxesImage:
    """
    Plot a swath of sea surface height (Hs) using Cartopy.

    Parameters
    ----------
    fig : Figure
        The figure object to plot on.
    ax : Axes
        The axes object to plot on.
    xds_bulk : xr.Dataset
        The dataset containing the bulk parameters.
    storm : xr.Dataset
        The dataset containing the storm information.
    var : str, optional
        The variable to plot, by default 'hswath'.
    vmin : float, optional
        The minimum value for the colorbar, by default 0.
    vmax : float, optional
        The maximum value for the colorbar, by default 10.
    cmap : str, optional
        The colormap to use, by default 'hot'.
    label : str, optional
        The label for the colorbar, by default 'Hs (m)'.
    name : str, optional
        The name of the storm, by default ''.
    storm_year : int, optional
        The year of the storm, by default 2000.
    colorbar : bool, optional
        Whether to show the colorbar, by default True.

    Returns
    -------
    AxesImage
        The image object representing the plotted data.
    """

    df_storm = historic_track_preprocessing(storm, center="WMO")
    lo_tc, la_tc, categ = df_storm.longitude, df_storm.latitude, df_storm.category

    xdlon, xdlat, xdmaxhs = (
        xds_bulk.lon.values,
        xds_bulk.lat.values,
        xds_bulk[var].values,
    )
    XX = np.reshape(xdlon, (np.unique(xdlat).size, np.unique(xdlon).size))
    YY = np.reshape(xdlat, (np.unique(xdlat).size, np.unique(xdlon).size))
    ZZ = np.reshape(xdmaxhs, (np.unique(xdlat).size, np.unique(xdlon).size))

    # plot
    im = ax.pcolor(
        XX, YY, ZZ, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree()
    )
    if colorbar:
        fig.colorbar(im, ax=ax, label=label)
    ax.plot(
        storm.lon.values,
        storm.lat.values,
        "-",
        color="k",
        linewidth=4,
        transform=ccrs.PlateCarree(),
    )
    for j in list(range(1, np.shape(lo_tc)[0])):
        color = get_storm_color(categ[j - 1])
        ax.plot(
            [lo_tc[j - 1], lo_tc[j]],
            [la_tc[j - 1], la_tc[j]],
            "-",
            color=color,
            linewidth=3,
            transform=ccrs.PlateCarree(),
        )
        ax.plot(
            lo_tc[j],
            la_tc[j],
            ".",
            color=color,
            linewidth=1,
            transform=ccrs.PlateCarree(),
        )

    return im


def axplot_st_cartopy(
    fig: Figure,
    ax: Axes,
    storm: xr.Dataset,
    name: str = "",
    storm_year: int = 2000,
    colorbar: bool = True,
) -> None:
    """
    Plot the storm track on a Cartopy map.

    Parameters
    ----------
    fig : Figure
        The figure object to plot on.
    ax : Axes
        The axes object to plot on.
    storm : xr.Dataset
        The storm track data.
    name : str, optional
        The name of the storm, by default ''.
    storm_year : int, optional
        The year of the storm, by default 2000.
    colorbar : bool, optional
        Whether to include a colorbar, by default True.
    """

    df_storm = historic_track_preprocessing(storm, center="WMO")
    lo_tc, la_tc, categ = df_storm.longitude, df_storm.latitude, df_storm.category

    for j in list(range(1, np.shape(lo_tc)[0])):
        color = get_storm_color(categ[j - 1])
        ax.plot(
            [lo_tc[j - 1], lo_tc[j]],
            [la_tc[j - 1], la_tc[j]],
            "-",
            color=color,
            linewidth=3,
            transform=ccrs.PlateCarree(),
        )
        ax.plot(
            lo_tc[j],
            la_tc[j],
            ".",
            color=color,
            linewidth=1,
            transform=ccrs.PlateCarree(),
        )


def plot_storm_track(
    storm: xr.Dataset, area: List[float], itc: str, path_proj: str
) -> None:
    """
    Plots the track of a storm on a map.

    Parameters
    ----------
    storm : xr.Dataset
        The storm data.
    area : List[float]
        The area to be plotted [lon_min, lon_max, lat_min, lat_max].
    itc : str
        The identifier of the storm.
    path_proj : str
        The path where to save the plot.
    """

    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide(central_longitude=200))
    ax = axplot_cartopy_robin(
        ax, resolution="l", cfill="silver", cocean="azure", mode_simple=False
    )
    axplot_st_cartopy(fig, ax, storm)

    ax.set_extent([area[0] - 2, area[1] + 10, area[2] - 10, area[3] + 10])

    fig.savefig(op.join(path_proj, "tc_{0}_track.png".format(itc)), dpi=300)


def plot_swath_shytcwaves(
    storm: xr.Dataset,
    xds_bulk: xr.Dataset,
    area: List[float],
    itc: str,
    path_proj: str,
    var: str = "hswath",
    vmin: float = 0,
    vmax: Optional[float] = None,
    cmap: str = cmap_wa,
) -> None:
    """
    Plots the swath of SHyTCWaves data for a given storm.

    Parameters
    ----------
    storm : xr.Dataset
        The storm data.
    xds_bulk : xr.Dataset
        The SHyTCWaves data as an xarray Dataset.
    area : List[float]
        The area of interest as a list [lon_min, lon_max, lat_min, lat_max].
    itc : str
        The identifier of the storm.
    path_proj : str
        The path to save the plot.
    var : str, optional
        The variable to plot, by default 'hswath'.
    vmin : float, optional
        The minimum value for the colorbar, by default 0.
    vmax : Optional[float], optional
        The maximum value for the colorbar, by default None.
    cmap : str, optional
        The colormap to use for the plot, by default cmap_wa.
    """

    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide(central_longitude=200))
    ax = axplot_cartopy_robin(
        ax, resolution="l", cfill="silver", cocean="azure", mode_simple=False
    )
    _im = axplot_shy_swath_cartopy(
        fig,
        ax,
        xds_bulk,
        storm,
        var=var,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        label="Hs (m)",
        name="",
        storm_year=2000,
        colorbar=True,
    )

    ax.set_extent([area[0] - 2, area[1] + 10, area[2] - 10, area[3] + 10])

    fig.savefig(op.join(path_proj, "tc_{0}_swath.png".format(itc)), dpi=300)


def plot_hs_tp_point(
    xds_bulk: xr.Dataset, area: List[float], lon: float, lat: float
) -> None:
    """
    Plot significant wave height (Hs) and peak period (Tp) at a specific point.

    Parameters
    ----------
    xds_bulk : xr.Dataset
        The dataset containing wave parameters.
    area : List[float]
        The area of interest [lon_min, lon_max, lat_min, lat_max].
    lon : float
        Longitude of the point to plot.
    lat : float
        Latitude of the point to plot.
    """

    xds_point = xds_bulk.isel(
        point=np.argmin(
            np.sqrt((xds_bulk.lon.values - lon) ** 2 + (xds_bulk.lat.values - lat) ** 2)
        )
    )

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 5, figure=fig)

    ax = fig.add_subplot(gs[0:2, 0:2], projection=ccrs.Mollweide(central_longitude=200))
    ax = axplot_cartopy_robin(
        ax, resolution="l", cfill="silver", cocean="azure", mode_simple=False
    )
    ax.scatter(
        xds_bulk.lon.values,
        xds_bulk.lat.values,
        c=xds_bulk.hswath.values,
        cmap=cmap_wa,
        transform=ccrs.PlateCarree(),
    )
    ax.set_extent([area[0] - 2, area[1] + 2, area[2] - 2, area[3] + 2])

    ax.scatter(
        lon,
        lat,
        c="yellow",
        s=200,
        marker="*",
        ec="green",
        label="Selected point",
        zorder=100,
        transform=ccrs.PlateCarree(),
    )
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.legend()

    ax = fig.add_subplot(gs[0, 2:5])
    ax.plot(xds_point.time.values, xds_point.hsbmu.values, color="royalblue", lw=2)
    ax.grid(":", color="lightgrey")
    ax.set_title("Hs at selected point", fontsize=14)
    ax.set_ylabel("Hs (m)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax = fig.add_subplot(gs[1, 2:5])
    ax.plot(xds_point.time.values, xds_point.tpbmu.values, color="firebrick", lw=2)
    ax.grid(":", color="lightgrey")
    ax.set_title("Tp at selected point", fontsize=14)
    ax.set_ylabel("Tp (s)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.show()
