import os.path as op

import cartopy
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from bluemath_tk.tcs.tracks import historic_track_preprocessing

from .common import custom_cmap

cmap_wa = custom_cmap(15, "YlOrRd", 0.15, 0.9, "YlGnBu_r", 0, 0.85)


def get_storm_color(categ):
    dcs = {
        0: "lime",
        1: "yellow",
        2: "orange",
        3: "red",
        4: "purple",
        5: "black",
        6: "gray",
    }

    return dcs[categ]


def fix_extent(ax, extent):
    """
    Adjusts the y-axis limits of a matplotlib Axes object based on the given extent.

    Parameters:
    - ax (matplotlib.axes.Axes): The Axes object to adjust.
    - extent (list): The extent of the data in the form [min_longitude, max_longitude, min_latitude, max_latitude].
    """

    mlon = np.mean(extent[:2])
    mlat = np.mean(extent[2:])
    xtrm_data = np.array(
        [[extent[0], mlat], [mlon, extent[2]], [extent[1], mlat], [mlon, extent[3]]]
    )
    proj_to_data = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    xtrm = proj_to_data.transform(xtrm_data)
    #     ax.set_xlim(xtrm[:,0].min(), xtrm[:,0].max())
    ax.set_ylim(xtrm[:, 1].min(), xtrm[:, 1].max())  # only latitudes


def axplot_cartopy_robin(
    ax, resolution="l", cfill="silver", cocean="azure", mode_simple=False
):
    """
    Plot a Cartopy Robinson projection on the given axes.

    Parameters:
    - ax (cartopy.mpl.geoaxes.GeoAxesSubplot): The axes on which to plot the map.
    - resolution (str, optional): The resolution of the map features. Defaults to 'l'.
    - cfill (str, optional): The color to fill the land area. Defaults to 'silver'.
    - cocean (str, optional): The color to fill the ocean area. Defaults to 'azure'.
    - mode_simple (bool, optional): Whether to use a simplified mode. Defaults to False.

    Returns:
    - ax (cartopy.mpl.geoaxes.GeoAxesSubplot): The modified axes object.
    """

    ax.set_global()
    ax.add_feature(cartopy.feature.LAND, facecolor=cfill, zorder=20)
    ax.add_feature(cartopy.feature.OCEAN, facecolor=cocean)
    ax.gridlines()
    extent = (-180, 180, -60, 58)
    fix_extent(ax, extent)

    return ax


def axplot_shy_swath_cartopy(
    fig,
    ax,
    xds_bulk,
    storm,
    var="hswath",
    vmin=0,
    vmax=10,
    cmap="hot",
    label="Hs (m)",
    name="",
    storm_year=2000,
    colorbar=True,
):
    """
    Plot a swath of sea surface height (Hs) using Cartopy.

    Parameters:
    - fig (matplotlib.figure.Figure): The figure object to plot on.
    - ax (matplotlib.axes.Axes): The axes object to plot on.
    - xds_bulk (xarray.Dataset): The dataset containing the bulk parameters.
    - storm (xarray.Dataset): The dataset containing the storm information.
    - var (str): The variable to plot (default: 'hswath').
    - vmin (float): The minimum value for the colorbar (default: 0).
    - vmax (float): The maximum value for the colorbar (default: 10).
    - cmap (str): The colormap to use (default: 'hot').
    - label (str): The label for the colorbar (default: 'Hs (m)').
    - name (str): The name of the storm (default: '').
    - storm_year (int): The year of the storm (default: 2000).
    - colorbar (bool): Whether to show the colorbar (default: True).

    Returns:
    - im (matplotlib.image.AxesImage): The image object representing the plotted data.
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


def axplot_st_cartopy(fig, ax, storm, name="", storm_year=2000, colorbar=True):
    """
    Plot the storm track on a Cartopy map.

    Parameters:
    - fig: The figure object to plot on.
    - ax: The axes object to plot on.
    - storm: The storm track data.
    - name: The name of the storm.
    - storm_year: The year of the storm.
    - colorbar: Whether to include a colorbar.

    Returns:
    None
    """

    df_storm = historic_track_preprocessing(storm, center="WMO")
    lo_tc, la_tc, categ = df_storm.longitude, df_storm.latitude, df_storm.category

    # plot
    #    ax.plot(storm.lon.values, storm.lat.values, '-', color='k', linewidth=4, transform=ccrs.PlateCarree());
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


def plot_storm_track(storm, area, itc, path_proj):
    """
    Plots the track of a storm on a map.

    Parameters:
    - storm: The storm data.
    - xds_bulk: The bulk parameters data.
    - area: The area to be plotted.
    - itc: The identifier of the storm.

    Returns:
    None
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
    storm, xds_bulk, area, itc, path_proj, var="hswath", vmin=0, vmax=None, cmap=cmap_wa
):
    """
    Plots the swath of SHyTCWaves data for a given storm.

    Parameters:
    - storm: The storm name or identifier.
    - xds_bulk: The SHyTCWaves data as an xarray Dataset.
    - area: The area of interest as a list [lon_min, lon_max, lat_min, lat_max].
    - itc: The identifier of the storm.
    - path_proj: The path to save the plot.
    - var: The variable to plot (default is 'hswath').
    - vmin: The minimum value for the colorbar (default is 0).
    - vmax: The maximum value for the colorbar (default is None).
    - cmap: The colormap to use for the plot (default is cmap_wa).

    Returns:
    None
    """
    fig = plt.figure(figsize=(30, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mollweide(central_longitude=200))
    ax = axplot_cartopy_robin(
        ax, resolution="l", cfill="silver", cocean="azure", mode_simple=False
    )
    im = axplot_shy_swath_cartopy(
        fig,
        ax,
        xds_bulk,
        storm,
        vmin=vmin,
        vmax=vmax,
        var=var,
        cmap=cmap,
        colorbar=False,
    )
    ax.set_extent([area[0] - 2, area[1] + 10, area[2] - 10, area[3] + 10])

    ax.set_title(var, fontsize=14)
    plt.colorbar(im).set_label(var)

    fig.savefig(op.join(path_proj, "tc_{0}_bulk.png".format(itc)), dpi=300)


def plot_hs_tp_point(xds_bulk, area, lon, lat):
    """
    Plots the Hs and Tp values at a selected point on a map.

    Parameters:
    - xds_bulk (xarray.Dataset): Dataset containing bulk parameters.
    - lon (float): Longitude of the selected point.
    - lat (float): Latitude of the selected point.
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
