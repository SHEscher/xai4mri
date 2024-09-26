"""
Functions to visualize MRIs.

    Author: Simon M. Hofmann
    Years: 2023-2024
"""

# %% Import
from __future__ import annotations

from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..dataloader.prune_image import find_brain_edges

# %% Set global vars  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Slice planes
PLANES = [
    "sagittal/longitudinal",
    "transverse/superior/horizontal",
    "coronal/posterior/frontal",
]

# Color maps
CMAPS = ["afmhot", "jet", "seismic", "bwr", "cool", "Greys"]


# %% Plotting functions << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class _Axis(Enum):
    SAGITTAL = 0
    TRANSVERSE = 1
    CORONAL = 2


def plot_slice(
    volume: np.ndarray,
    axis: int | _Axis,
    idx_slice: int,
    edges: bool = False,
    c_range: str | None = None,
    **kwargs,
) -> plt.AxesImage:
    """
    Plot one slice of a 3D volume.

    :param volume: 3D volume (MRI).
    :param axis: The volume axis to plot from.
    :param idx_slice: The slice index to plot.
    :param edges: Draw edges around the brain if set to `True`.
    :param c_range: Color range:
                    "full": Full possible spectrum (0-255 or 0-1),
                    "single": Take min/max of given volume.
                    `None`: Default: do not change color range.
    :param kwargs: Additional keyword arguments for `plt.imshow()`.
                   And, `crosshairs: bool = True`, `ticks: bool = False`.
    """
    im, _edges = None, None  # init
    if edges:
        n_dims = 4
        _edges = find_brain_edges(volume if volume.shape[-1] > n_dims else volume[..., -1])
        # works for transparent RGB images (x,y,z,4) and volumes (x,y,z)

    # Set color range
    if c_range == "full":  # takes full possible spectrum
        i_max = 255 if np.max(volume) > 1 else 1.0
        i_min = 0 if np.min(volume) >= 0 else -1.0
    elif c_range == "single":  # takes min/max of given brain
        i_max = np.max(volume)
        i_min = np.min(volume)
    else:  # c_range=None
        if c_range is not None:
            msg = "c_range must be 'full', 'single' or None."
            raise ValueError(msg)
        i_max, i_min = None, None

    # Get kwargs (which are not for imshow)
    crosshairs = kwargs.pop("crosshairs", False)
    ticks = kwargs.pop("ticks", False)

    axis = _Axis(axis)  # this also checks if the axis is valid
    if axis is _Axis.SAGITTAL:  # sagittal
        im = plt.imshow(volume[idx_slice, :, :], vmin=i_min, vmax=i_max, **kwargs)
        if edges:
            plt.hlines(_edges[2] - 1, 2, volume.shape[1] - 2, colors="darkgrey", alpha=0.3)  # == max edges
            plt.hlines(_edges[3] + 1, 2, volume.shape[1] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(_edges[4] - 1, 2, volume.shape[0] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(_edges[5] + 1, 2, volume.shape[0] - 2, colors="darkgrey", alpha=0.3)

    elif axis == _Axis.TRANSVERSE:  # transverse / superior
        im = plt.imshow(
            np.rot90(volume[:, idx_slice, :], axes=(0, 1)),
            vmin=i_min,
            vmax=i_max,
            **kwargs,
        )
        if edges:
            plt.hlines(
                volume.shape[0] - _edges[5] - 1,
                2,
                volume.shape[0] - 2,
                colors="darkgrey",
                alpha=0.3,
            )
            plt.hlines(
                volume.shape[0] - _edges[4] + 1,
                2,
                volume.shape[0] - 2,
                colors="darkgrey",
                alpha=0.3,
            )
            plt.vlines(_edges[0] - 1, 2, volume.shape[1] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(_edges[1] + 1, 2, volume.shape[1] - 2, colors="darkgrey", alpha=0.3)

    elif axis == _Axis.CORONAL:  # coronal / posterior
        im = plt.imshow(
            np.rot90(volume[:, :, idx_slice], axes=(1, 0)),
            vmin=i_min,
            vmax=i_max,
            **kwargs,
        )
        if edges:
            plt.hlines(_edges[2] - 1, 2, volume.shape[0] - 2, colors="darkgrey", alpha=0.3)
            plt.hlines(_edges[3] + 1, 2, volume.shape[0] - 2, colors="darkgrey", alpha=0.3)
            plt.vlines(
                volume.shape[1] - _edges[1] - 1,
                2,
                volume.shape[1] - 2,
                colors="darkgrey",
                alpha=0.3,
            )
            plt.vlines(
                volume.shape[1] - _edges[0] + 1,
                2,
                volume.shape[1] - 2,
                colors="darkgrey",
                alpha=0.3,
            )

    # Add mid-cross ('crosshairs')
    if crosshairs:
        plt.hlines(int(volume.shape[axis.value] / 2), 2, len(volume) - 2, colors="red", alpha=0.3)
        plt.vlines(int(volume.shape[axis.value] / 2), 2, len(volume) - 2, colors="red", alpha=0.3)

    if not ticks:
        plt.axis("off")

    return im


def plot_mid_slice(
    volume: np.ndarray,
    axis: int | _Axis | None = None,
    fig_name: str | None = None,
    edges: bool = True,
    c_range: str | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot 2D-slices of a given 3D-volume.

    If no axis is given, plot for each axis its middle slice.

    :param volume: 3D volume (MRI).
    :param axis: `None`: Slices from all three axes are plotted.
                 To plot a specific axis only, provide `int` or `_Axis`.
    :param fig_name: Name of the figure.
    :param edges:  Draw edges around the brain if set to `True`.
    :param c_range: Color range:
                    "full": Full possible spectrum (0-255 or 0-1),
                    "single": Take min/max of given volume.
                    `None`: Default: do not change color range.
    :param kwargs: Additional keyword arguments for `plt.imshow()`.
                   And, `crosshairs: bool = True`, `ticks: bool = False`.
                   Also, `cbar: bool = False` and `cbar_range: tuple[int, int] = None` for a color bar,
                   `suptitle: str = None` for a title,
                   and `slice_idx: int | tuple[int, int, int] = None` for specific slice indices to plot,
                   if not provided, the middle slice(s) are plotted.
    :return: The figure object of the plot.
    """
    # Get color bar kwargs (if any)
    cbar = kwargs.pop("cbar", False)
    cbar_range = kwargs.pop("cbar_range") if ("cbar_range" in kwargs and cbar) else None
    # only if cbar is active
    suptitle = kwargs.pop("suptitle", None)
    slice_idx = kwargs.pop("slice_idx", None)  # in case mid or other slice is given

    # Set (mid-)slice index
    sl = int(np.round(volume.shape[0] / 2)) if slice_idx is None else slice_idx
    max_n_axes = 3
    if slice_idx is None:
        sl_str = "mid"
    elif isinstance(sl, (list, tuple)):
        if len(sl) != max_n_axes:
            msg = "slice_idx must be tuple or list of length 3, is None or single int."
            raise ValueError(msg)
        sl_str = str(sl)
    else:
        sl_str = str(int(sl))

    # Begin plotting
    if axis is None:
        _fs = {"size": 10}  # define font size

        fig = plt.figure(num=f"{fig_name if fig_name else ''} volume {sl_str}-slice", figsize=(12, 4))
        if isinstance(suptitle, str) and suptitle:
            fig.suptitle(suptitle, fontsize=14)

        # Planes
        ims = []
        axs = []
        sls = [sl] * 3 if isinstance(sl, int) else sl  # is tuple pf length 3 or int
        for ip, _plane in enumerate(PLANES):
            axs.append(fig.add_subplot(1, 3, ip + 1))
            ims.append(
                plot_slice(
                    volume,
                    axis=ip,
                    idx_slice=sls[ip],
                    edges=edges,
                    c_range=c_range,
                    **kwargs,
                )
            )
            plt.title(_plane, fontdict=_fs)

            divider = make_axes_locatable(axs[ip])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cax.axis("off")
            if cbar and ip == len(PLANES) - 1:
                cax_bar = fig.colorbar(ims[-1], ax=cax, fraction=0.048, pad=0.04)  # shrink=.8, aspect=50)
                if cbar_range:
                    cax_bar.set_ticks(ticks=np.linspace(start=0, stop=1, num=7), minor=True)
                    cax_bar.ax.set_yticklabels(
                        labels=[
                            f"{tick:.2g}"
                            for tick in np.linspace(cbar_range[0], cbar_range[1], len(cax_bar.get_ticks()))
                        ]
                    )

    else:  # If specific axis to plot
        axis = _Axis(axis)
        axis_name = axis.name.lower()

        fig = plt.figure(f"{fig_name if fig_name else ''} {axis_name} {sl_str}-slice")
        im = plot_slice(volume, axis.value, idx_slice=sl, edges=edges, c_range=c_range, **kwargs)
        if cbar:
            cax_bar = fig.colorbar(im, fraction=0.048, pad=0.04)  # shrink=0.8, aspect=50)
            if cbar_range:
                cax_bar.set_ticks(ticks=np.linspace(start=0, stop=1, num=7), minor=True)
                cax_bar.ax.set_yticklabels(
                    labels=[
                        f"{tick:.2g}" for tick in np.linspace(cbar_range[0], cbar_range[1], len(cax_bar.get_ticks()))
                    ]
                )

        plt.tight_layout()

    return fig


def slice_through(
    volume: np.ndarray,
    every: int = 2,
    axis: int | _Axis = 0,
    fig_name: str | None = None,
    edges: bool = True,
    c_range: str | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot several slices of a given 3D-volume in the form of a grid.

    For fancy slicing, check: https://docs.pyvista.org/examples/01-filter/slicing.html.

    :param volume: 3D volume (MRI).
    :param every: Plot every n-th slice.
                  That is, for `every=1`, all slices are plotted.
                  And, for `every=2`, every second slice is plotted, and so on.
    :param axis: The volume axis to plot from.
    :param fig_name: Name of the figure.
    :param edges: Draw edges around the brain if set to `True`.
    :param c_range: Color range:
                    "full": Full possible spectrum (0-255 or 0-1),
                    "single": Take min/max of given volume.
                    `None`: Default: do not change color range.
    :param kwargs: Additional keyword arguments for `plt.imshow()`.
                   And, `crosshairs: bool = True`, `ticks: bool = False`.
    :return: The figure object of the plot.
    """
    axis = _Axis(axis)

    n_row_sq_grid = 5
    n_slices = volume.shape[axis.value] // every
    n_figs = np.round(n_slices // n_row_sq_grid**2)

    axis_name = axis.name.lower()

    fig_n = 1
    sub_n = 1
    fig = None  # init
    for scl in range(n_slices):
        if scl % (n_row_sq_grid**2) == 0:
            fig = plt.figure(
                num=f"{fig_name if fig_name else ''} {axis_name} slice-through {fig_n}|{n_figs}",
                figsize=(10, 10),
            )

        plt.subplot(n_row_sq_grid, n_row_sq_grid, sub_n)

        plot_slice(
            volume=volume,
            axis=axis,
            idx_slice=scl + (every - 1),
            edges=edges,
            c_range=c_range,
            **kwargs,
        )

        plt.tight_layout()

        sub_n += 1

        if ((sub_n - 1) % (n_row_sq_grid**2) == 0) or (scl + 1 == n_slices):
            fig_n += 1
            sub_n = 1

    return fig


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
