"""
Functions to visualize analyzer (XAI) heatmaps.

!!! tip "Quick Start"
    Particularly, the `plot_heatmap` function is used to visualize the relevance maps of the `LRP` analyzer from
    `xai4mri.model.interpreter`.

    ```python+
    from xai4mri.model.interpreter import analyze_model
    from xai4mri.visualizer import plot_heatmap

    # Analyze model
    analyzer_obj = analyze_model(model=model, ipt=mri_image, ...)

    # Visualize heatmap / relevance map
    analyzer_fig = plot_heatmap(ipt=mri_image, analyser_obj=analyzer_obj, ...)
    ```
---
    Author: Simon M. Hofmann
    Years: 2023-2024
"""

# %% Import
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from ..utils import cprint, normalize
from .visualize_mri import CMAPS, plot_mid_slice, slice_through

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def gregoire_black_fire_red(analyser_obj: np.ndarray) -> np.ndarray:
    """
    Apply a color scheme to the analyzer object.

    :param analyser_obj: XAI analyzer object (e.g., `LRP` relevance map).
    :return: Colorized relevance map.
    """
    x = analyser_obj.copy()
    x /= np.max(np.abs(x))

    hrp = np.clip(x - 0.00, a_min=0, a_max=0.25) / 0.25  # all pos. values(+) above 0 get red, above .25 full red(=1.)
    hgp = np.clip(x - 0.25, a_min=0, a_max=0.25) / 0.25  # all above .25 get green, above .50 full green
    hbp = np.clip(x - 0.50, a_min=0, a_max=0.50) / 0.50  # all above .50 get blue until full blue at 1. (mix 2 white)

    hbn = np.clip(-x - 0.00, a_min=0, a_max=0.25) / 0.25  # all neg. values(-) below 0 get blue ...
    hgn = np.clip(-x - 0.25, a_min=0, a_max=0.25) / 0.25  # ... green ....
    hrn = np.clip(-x - 0.50, a_min=0, a_max=0.50) / 0.50  # ... red ... mixes to white (1.,1.,1.)

    return np.concatenate(
        [(hrp + hrn)[..., None], (hgp + hgn)[..., None], (hbp + hbn)[..., None]],
        axis=x.ndim,
    )


custom_maps = {
    "black-fire-red": gregoire_black_fire_red,
    # more custom color maps can be added here
}


def _create_cmap(color_fct: callable, res: int = 4999) -> LinearSegmentedColormap:
    """
    Create a color map for a given color function.

    Create the color map, such as `gregoire_black_fire_red`, which can be used for
    color-bars and other purposes.

    The function creates a color-dict in the following form, and
    feeds it to `matplotlib.colors.LinearSegmentedColormap`:

    ```python
    cdict_gregoire_black_fire_red = {
        "red": [[0.0, 1.0, 1.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.625, 1.0, 1.0], [1.0, 1.0, 1.0]],
        "green": [
            [0.0, 1.0, 1.0],
            [0.25, 1.0, 1.0],
            [0.375, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.625, 0.0, 0.0],
            [0.75, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        "blue": [[0.0, 1.0, 1.0], [0.375, 1.0, 1.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0], [1.0, 1.0, 1.0]],
    }
    ```

    :param color_fct: Color function.
    :param res: Resolution of the color map.
    :return: LinearSegmentedColormap.
    """
    # Prep resolution (res):
    min_res: int = 10  # minimum resolution
    if not float(res).is_integer():
        msg = "'res' must be a positive natural number."
        raise ValueError(msg)
    if res < min_res:
        cprint(
            string=f"res={res} is too small to create a detailed cmap. res was set to 999, instead.",
            col="y",
        )
        res = 999
    if res % 2 == 0:
        res += 1
        print("res was incremented by 1 to zero center the cmap.")

    linear_space = np.linspace(-1, 1, res)
    linear_space_norm = normalize(linear_space, lower_bound=0.0, upper_bound=1.0)

    colored_linear_space = color_fct(linear_space)
    red = colored_linear_space[:, 0]
    green = colored_linear_space[:, 1]
    blue = colored_linear_space[:, 2]

    cdict = {
        "red": [[linear_space_norm[i_], col_, col_] for i_, col_ in enumerate(red)],
        "green": [[linear_space_norm[i_], col_, col_] for i_, col_ in enumerate(green)],
        "blue": [[linear_space_norm[i_], col_, col_] for i_, col_ in enumerate(blue)],
    }

    return LinearSegmentedColormap(name=color_fct.__name__, segmentdata=cdict)


def _colorize_matplotlib(analyser_obj: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Colorize XAI-based analyzer / relevance object with `matplotlib` color maps.

    :param analyser_obj: XAI-based analyzer (relevance) object.
    :param cmap_name: Name of the color map.
    :return: Colorized relevance object.
    """
    # fetch color mapping function by string
    cmap = cm.__dict__[cmap_name]

    # bring data to [-1 1]
    analyser_obj /= np.max(np.abs(analyser_obj))

    # push data to [0 1] to avoid automatic color map normalization
    analyser_obj = (analyser_obj + 1) / 2

    sh = analyser_obj.shape

    return cmap(analyser_obj.flatten())[:, 0:3].reshape([*sh, 3])


def list_supported_cmaps():
    """Return a list of supported color maps for heatmap plotting."""
    print(*(list(custom_maps.keys()) + CMAPS), sep="\n")
    return list(custom_maps.keys()) + CMAPS


def _symmetric_clip(analyser_obj: np.ndarray, percentile: float = 1 - 1e-2, min_sym_clip: bool = True) -> np.ndarray:
    """
    Clip XAI-based analyzer (relevance) object symmetrically around zero.

    :param analyser_obj: XAI-based analyzer object, e.g., `LRP` relevance map.
    :param percentile: Percentile to clip.
                       Default: keep out very small intensity values at the border of their distribution.
                       For `percentile=1`, there is no change.
    :param min_sym_clip: Minimum for symmetric clipping.
                         `True`: find the `min(abs(analyser_obj.min), analyser_obj.max)` to clip symmetrically.
    :return: Symmetrically clipped analyzer (relevance) object.
    """
    if percentile > 1 or percentile < 0.5:  # noqa: PLR2004
        msg = "percentile must be in range (.5, 1)!"
        raise ValueError(msg)

    if not (analyser_obj.min() < 0.0 < analyser_obj.max()) and min_sym_clip:
        cprint(
            string="Relevance object has only values larger OR smaller than 0., thus 'min_sym_clip' is switched off!",
            col="y",
        )
        min_sym_clip = False

    # Get cut-off values for lower and upper percentile
    if min_sym_clip:
        # min_clip example: [-7, 10] => clip(-7, 7) | [-10, 7] => clip(-7, 7)
        max_min_q = min(abs(analyser_obj.min()), analyser_obj.max())  # > 0
        min_min_q = -max_min_q  # < 0

    if percentile < 1:
        max_q = -np.percentile(a=-analyser_obj, q=1 - percentile)
        min_q = np.percentile(a=analyser_obj, q=1 - percentile)

        # Clip-off at max-abs percentile value
        max_q = max(abs(min_q), abs(max_q))  # > 0
        # Does opposite of min_clip, example: [-7, 10] => clip(-10, 10) | [-10, 7] => clip(-10, 10)
        if min_sym_clip:
            # However, when both options are active, 'percentile' is prioritized
            max_q = min(max_q, max_min_q)
        min_q = -max_q  # < 0

        return np.clip(a=analyser_obj, a_min=min_q, a_max=max_q)

    if percentile == 1.0 and min_sym_clip:
        return np.clip(a=analyser_obj, a_min=min_min_q, a_max=max_min_q)

    return analyser_obj


def _apply_colormap(
    analyser_obj: np.ndarray,
    input_image: np.ndarray,
    cmap_name: str = "black-fire_red",
    c_intensifier: float = 1.0,
    clip_q: float = 1e-2,
    min_sym_clip: bool = False,
    gamma: float = 0.2,
    gblur: float = 0.0,
    true_scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray]:
    """
    Merge the relevance tensor (analyzer object) with the model input image to receive a heatmap over the input.

    :param analyser_obj: XAI-based analyzer (relevance) map/tensor.
    :param input_image: Model input image.
    :param cmap_name: Name of the color-map (`cmap`) to be applied.
    :param c_intensifier: Color intensifier.
                          In range `[1, ...[`, increase the color strength by multiplying + clipping [DEPRECATED]
    :param clip_q: Clips off given percentile of relevance symmetrically around zero.
                   Range: `[0, .5]`
    :param min_sym_clip: Minimum for symmetric clipping.
                         `True`: find the `min(abs(analyser_obj.min), analyser_obj.max)` to clip symmetrically.
    :param gamma: The smaller the `gamma` (`< 1.`) the brighter the resulting image.
                  For `gamma > 1.`, the image gets darker.
    :param gblur: Apply Gaussian blur [NOT IMPLEMENTED YET].
    :param true_scale: `True`: return min/max value of the analyzer object (after clipping).
                       This is for true color scaling in, e.g., a color bar `cbar`.
    :return: Heatmap merged with input.
    """
    # Prep input image
    img = input_image.copy()
    a = analyser_obj.copy()

    # Check whether image has RGB(A) channels
    n_with_alpha = 4
    if img.shape[-1] <= n_with_alpha:
        img = np.mean(img, axis=-1)  # removes rgb channels

    # Following creates a grayscale image (for the MRI case, no difference)
    img = np.concatenate([img[..., None]] * 3, axis=-1)  # (X,Y,Z, [r,g,b]), where r=g=b (i.e., grayscale)

    # normalize image (0, 1)
    if img.min() < 0.0:  # for img range (-1, 1)
        img += np.abs(img.min())
    img /= np.max(np.abs(img))

    # Symmetrically clip the relevance values around zero for better visualization
    if clip_q < 0.0 or clip_q > 0.5:  # noqa: PLR2004
        msg = "clip_q must be in range (0, .5)!"
        raise ValueError(msg)
    a = _symmetric_clip(analyser_obj=a, percentile=1 - clip_q, min_sym_clip=min_sym_clip)
    r_max = np.abs(a).max()  # symmetric: r_min = -r_max

    # # Normalize relevance tensor
    a /= np.max(np.abs(a))
    # norm to [-1, 1] for real numbers, or [0, 1] for R+, where zero remains zero

    # # Apply chosen cmap
    if cmap_name in custom_maps:
        r_rgb = custom_maps[cmap_name](a)
    elif cmap_name in CMAPS:
        r_rgb = _colorize_matplotlib(a, cmap_name)
    else:
        raise Exception(
            f"You have managed to smuggle in the unsupported colormap {cmap_name} into method "
            f"_apply_colormap. Supported mappings are:\n\t{list_supported_cmaps()}"
        )

    # Increase col-intensity
    if c_intensifier != 1.0:
        if c_intensifier < 1.0:
            msg = "c_intensifier must be 1 (i.e. no change) OR greater (intensify color)!"
            raise ValueError(msg)
        r_rgb *= c_intensifier
        r_rgb = r_rgb.clip(0.0, 1.0)

    # Merge input image with heatmap via inverse alpha channels
    alpha = np.abs(a[..., None])  # as alpha channel, use (absolute) relevance map amplitude.
    alpha = np.concatenate([alpha] * 3, axis=-1) ** gamma  # (X,Y,Z, 3)
    heat_img = (1 - alpha) * img + alpha * r_rgb

    # Apply Gaussian blur
    if gblur > 0:  # there is a bug in opencv, which causes an error with this command
        msg = "'gblur' currently not activated, keep 'gblur=0' for now!"
        raise NotImplementedError(msg)
        # TODO: hm = cv2.GaussianBlur(HM, (gblur, gblur), 0)  # noqa: FIX002

    if true_scale:
        return heat_img, r_rgb, r_max
    return heat_img, r_rgb


# TODO: solve issue with ipt image when it has negative values as in statsmaps  # noqa: FIX002
def plot_heatmap(
    ipt: np.ndarray,
    analyser_obj: np.ndarray,
    cmap_name: str = "black-fire-red",
    mode: str = "triplet",
    fig_name: str = "Heatmap",
    **kwargs,
) -> plt.Figure:
    """
    Plot an XAI-based analyzer object over the model input in the form of a heatmap.

    !!! example "How to use"
        ```python
        from xai4mri.model.interpreter import analyze_model
        from xai4mri.visualizer import plot_heatmap

        # Analyze model
        analyzer_obj = analyze_model(model=model, ipt=mri_image, ...)

        # Visualize heatmap / relevance map
        analyzer_fig = plot_heatmap(ipt=mri_image, analyser_obj=analyzer_obj, ...)
        ```

    :param ipt: Model input image.
    :param analyser_obj: Analyzer object (relevance map) that is computed by the model interpreter (e.g., `LRP`).
                         Both the input image and the analyzer object must have the same shape.
    :param cmap_name: Name of color-map (`cmap`) to be applied.
    :param mode: "triplet": Plot three slices of different axes.
                 "all": Plot all slices (w/ brain OR w/o brain → set: `plot_empty=True` in `kwargs`)
    :param fig_name: name of figure
    :param kwargs: Additional kwargs:
                   "c_intensifier", "clip_q", "min_sym_clip", "true_scale", "plot_empty", "axis", "every", "crosshair",
                    "gamma".
                    And, `kwargs` for `plot_mid_slice()` and `slice_through()` from `xai4mri.visualizer.visualize_mri`.
    :return: `plt.Figure` object of the heatmap plot.
    """
    a = analyser_obj.copy().squeeze()
    mode = mode.lower()
    if mode not in {"triplet", "all"}:
        msg = "mode must be 'triplet', or 'all'!"
        raise ValueError(msg)

    # Extract kwargs
    cintensifier = kwargs.pop("c_intensifier", 1.0)
    clipq = kwargs.pop("clip_q", 1e-2)
    min_sym_clip = kwargs.pop("min_sym_clip", True)
    true_scale = kwargs.pop("true_scale", False)
    plot_empty = kwargs.pop("plot_empty", False)
    axis = kwargs.pop("axis", 0)
    every = kwargs.pop("every", 2)
    crosshairs = kwargs.pop("crosshair", False)
    gamma = kwargs.pop("gamma", 0.2)

    # Render image
    colored_a = _apply_colormap(
        analyser_obj=a,
        input_image=ipt,
        cmap_name=cmap_name,
        c_intensifier=cintensifier,
        clip_q=clipq,
        min_sym_clip=min_sym_clip,
        gamma=gamma,
        true_scale=true_scale,
    )
    heatmap = colored_a[0]

    cbar_range = (-1, 1) if not true_scale else (-colored_a[2], colored_a[2])
    if mode == "triplet":
        fig = plot_mid_slice(
            volume=heatmap,
            fig_name=fig_name,
            cmap=_create_cmap(gregoire_black_fire_red),
            c_range="full",
            cbar=True,
            cbar_range=cbar_range,
            edges=False,
            crosshairs=crosshairs,
            **kwargs,
        )

    else:  # mode == "all"
        if not plot_empty:
            # Remove planes with no information
            heatmap = heatmap.compress(
                ~np.all(
                    heatmap == 0,
                    axis=tuple(ax for ax in range(heatmap.ndim) if ax != axis),
                ),
                axis=axis,
            )

        fig = slice_through(
            volume=heatmap,
            every=every,
            axis=axis,
            fig_name=fig_name,
            edges=False,
            cmap=_create_cmap(gregoire_black_fire_red),
            c_range="full",
            crosshairs=crosshairs,
            **kwargs,
        )
    return fig


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
