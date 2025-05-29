from math import sqrt
from typing import List, Tuple

import cmocean
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import interpolate


def GetDivisors(x: int) -> List[int]:
    """
    Calculate all divisors of a given number.

    Parameters
    ----------
    x : int
        The number to find divisors for.

    Returns
    -------
    List[int]
        A list of all divisors of x (excluding x itself).
    """

    l_div = []
    i = 1
    while i < x:
        if x % i == 0:
            l_div.append(i)
        i = i + 1

    return l_div


def GetBestRowsCols(n: int) -> Tuple[int, int]:
    """
    Calculate the optimal number of rows and columns for a grid layout.

    This function tries to create a square-like grid layout by finding the best
    divisors of n. If n is a perfect square, it returns equal rows and columns.
    Otherwise, it finds the closest rectangular layout.

    Parameters
    ----------
    n : int
        The total number of elements to arrange in a grid.

    Returns
    -------
    Tuple[int, int]
        A tuple containing (number_of_rows, number_of_columns).
    """

    sqrt_n = sqrt(n)
    if sqrt_n.is_integer():
        n_r = int(sqrt_n)
        n_c = int(sqrt_n)
    else:
        l_div = GetDivisors(n)
        n_c = l_div[int(len(l_div) / 2)]
        n_r = int(n / n_c)

    return n_r, n_c


def calc_quiver(
    X: np.ndarray, Y: np.ndarray, var: np.ndarray, vdir: np.ndarray, size: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate variables and prepare data for quiver plot visualization.

    This function interpolates the input variables onto a new mesh grid and calculates
    the vector components for quiver plotting. The direction convention is clockwise
    from geographic north.

    Parameters
    ----------
    X : np.ndarray
        X-coordinates mesh grid array.
    Y : np.ndarray
        Y-coordinates mesh grid array.
    var : np.ndarray
        Variable magnitude array.
    vdir : np.ndarray
        Variable direction array in degrees (clockwise from North).
    size : int, optional
        Size of the quiver mesh, by default 30.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
            - x_q: X-coordinates for quiver plot
            - y_q: Y-coordinates for quiver plot
            - var_q: Interpolated variable magnitude
            - u: X-component of the vector
            - v: Y-component of the vector
    """

    step = (X[-1] - X[0]) / size

    size_x = round((X[-1] - X[0]) / step)
    size_y = round((Y[-1] - Y[0]) / step)

    # var and dir interpolators
    vdir_f = vdir.copy()
    vdir_f[np.isnan(vdir_f)] = 0
    vdir_f_y = np.cos(np.deg2rad(vdir_f))
    vdir_f_x = np.sin(np.deg2rad(vdir_f))
    f_dir_x = interpolate.interp2d(X, Y, vdir_f_x, kind="linear")
    f_dir_y = interpolate.interp2d(X, Y, vdir_f_y, kind="linear")

    var_f = var.copy()
    var_f[np.isnan(var_f)] = 0
    f_var = interpolate.interp2d(X, Y, var_f, kind="linear")

    # generate quiver mesh
    x_q = np.linspace(X[0], X[-1], num=size_x)
    y_q = np.linspace(Y[0], Y[-1], num=size_y)

    # interpolate data to quiver mesh
    var_q = f_var(x_q, y_q)
    vdir_q_x = f_dir_x(x_q, y_q)
    vdir_q_y = f_dir_y(x_q, y_q)
    vdir_q = np.rad2deg(np.arctan(vdir_q_x / vdir_q_y))

    # sign correction
    vdir_q[(vdir_q_x > 0) & (vdir_q_y < 0)] = (
        vdir_q[(vdir_q_x > 0) & (vdir_q_y < 0)] + 180
    )
    vdir_q[(vdir_q_x < 0) & (vdir_q_y < 0)] = (
        vdir_q[(vdir_q_x < 0) & (vdir_q_y < 0)] + 180
    )

    # u and v dir components
    u = np.sin(np.deg2rad(vdir_q))
    v = np.cos(np.deg2rad(vdir_q))

    return x_q, y_q, var_q, u, v


def custom_cmap(
    numcolors: int,
    map1: str,
    m1ini: float,
    m1end: float,
    map2: str,
    m2ini: float,
    m2end: float,
) -> colors.ListedColormap:
    """
    Generate a custom colormap by combining two existing colormaps.

    This function creates a custom colormap by combining two existing colormaps
    with specified ranges. Useful for creating smooth transitions between different
    color schemes.

    Parameters
    ----------
    numcolors : int
        Number of colors in the colormap (100 for continuous, 15 for discretization).
    map1 : str
        Name of the first colormap (e.g., 'YlOrRd').
    m1ini : float
        Start point for the first colormap range (0-1).
    m1end : float
        End point for the first colormap range (0-1).
    map2 : str
        Name of the second colormap (e.g., 'YlGnBu_r').
    m2ini : float
        Start point for the second colormap range (0-1).
    m2end : float
        End point for the second colormap range (0-1).

    Returns
    -------
    colors.ListedColormap
        A new custom colormap combining the two input colormaps.
    """

    # color maps
    cmap1 = plt.get_cmap(map1, numcolors)
    cmap2 = plt.get_cmap(map2, numcolors)

    # custom color ranges
    cmap1v = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap1.name, a=m1ini, b=m1end),
        cmap1(np.linspace(m1ini, m1end, 100)),
    )

    cmap2v = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap2.name, a=m2ini, b=m2end),
        cmap2(np.linspace(m2ini, m2end, 100)),
    )

    top = cm.get_cmap(cmap1v, 128)
    bottom = cm.get_cmap(cmap2v, 128)

    newcolors = np.vstack((bottom(np.linspace(0, 1, 128)), top(np.linspace(0, 1, 128))))
    newcmp = colors.ListedColormap(newcolors, name="OrangeBlue")

    return newcmp


def bathy_cmap(bottom_lim: int, top_lim: int) -> colors.ListedColormap:
    """
    Generate a custom colormap specifically designed for bathymetry plots.

    This function creates a specialized colormap combining the 'turbid' colormap
    from cmocean with a reversed 'YlGnBu' colormap, optimized for bathymetry visualization.

    Parameters
    ----------
    bottom_lim : int
        Number of colors for the bottom part of the colormap.
    top_lim : int
        Number of colors for the top part of the colormap.

    Returns
    -------
    colors.ListedColormap
        A custom colormap suitable for bathymetry visualization.
    """

    # colormaps
    colors1 = cmocean.cm.turbid
    colors2 = "YlGnBu_r"

    top = cm.get_cmap(colors1, top_lim)
    bottom = cm.get_cmap(colors2, bottom_lim)

    newcolors = np.vstack(
        (bottom(np.linspace(0, 0.8, bottom_lim)), top(np.linspace(0.1, 1, top_lim)))
    )

    return colors.ListedColormap(newcolors)
