"""
Pruning MRIs has the goal to remove background around brains / heads, i.e., reduce the size of a 3D image.

One key objective is to find the smallest box in the whole dataset, which can surround each brain / head in it.
That is, the space of the 'biggest' brain.

    Author: Simon M. Hofmann
    Years: 2023-2024
"""

# %% Import
from __future__ import annotations

from typing import Sequence  # noqa: UP035

import nibabel as nib
import numpy as np

from ..utils import cprint
from .transformation import BG_VALUE, GLOBAL_ORIENTATION_SPACE

# %% Global vars << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class _PruneConfig:
    """Config class for MRI pruning."""

    def __init__(self):
        """Initialize class."""
        self._largest_brain_max_axes = np.array([156, 166, 204])  # asserting 'LIA' orientation space

    @property
    def largest_brain_max_axes(self) -> np.ndarray:
        """Get max-axes values for the theoretically largest brain in the present data."""
        return self._largest_brain_max_axes

    @property
    def orientation_space(self) -> str:
        """Get the orientation space."""
        return GLOBAL_ORIENTATION_SPACE

    @staticmethod
    def __error_msg() -> str:
        """Provide an error message for issues with 'largest_brain_max_axes'."""
        return "largest_brain_max_axes must be a sequence of positive ints with length 3 OR an int."

    @largest_brain_max_axes.setter
    def largest_brain_max_axes(self, value: Sequence[int] | np.ndarray[int] | int):
        """Set max-axes values for the largest brain in the present data."""
        if isinstance(value, (Sequence, np.ndarray)):
            n_dims = 3
            if len(value) != n_dims:
                raise ValueError(self.__error_msg())
            # Has the correct length
            value = np.array(value)
            # Alle elements must be integers
            if not all(e.is_integer() for e in value):
                raise ValueError(self.__error_msg())
            # Check if elements are all positive
            if np.any(value < 0):
                raise ValueError(self.__error_msg())

        elif isinstance(value, (int, float, np.integer, np.floating)):
            if value % 1 != 0:
                raise ValueError(self.__error_msg())
            value = np.array([value, value, value])
        else:
            raise TypeError(self.__error_msg())

        # Minimum values for a brain in 1 mm resolution
        min_value: int = 130
        if np.any(value < min_value):
            msg = (
                "For 'largest_brain_max_axes' a 1 mm isotropic resolution is assumed.\n"
                "During pruning this will be adapted to the resolution of a given image volume.\n"
                f"The given values {value} seem to be too small for enclosing a whole brain in 1 mm resolution.\n"
                f"Check the biggest brain in your dataset and set the values accordingly.\n"
                f"For this you can use `xai4mri.dataloader.prune_image.get_brain_axes_length()`."
            )
            raise ValueError(msg)

        self._largest_brain_max_axes = value.astype(int)


PruneConfig = _PruneConfig()  # this is provided to the user to change the default value of 'largest_brain_max_axes'


def reverse_pruning(
    original_mri: np.ndarray | nib.Nifti1Image,
    pruned_mri: np.ndarray,
    pruned_stats_map: np.ndarray | None = None,
) -> np.ndarray | nib.Nifti1Image:
    """
    Reverse the pruning of an MRI or its corresponding statistical map.

    If a statistical map is given, both the original MRI and the pruned MRI are necessary to find the edges of
    the cut-off during pruning.
    If no statistical map is given, only the original MRI and the pruned MRI are required.
    Note, in this case `reverse_pruning()` is applied to a processed and pruned version of the original MRI.

    Make sure that the original MRI and the pruned MRI have the same orientation.

    :param original_mri: Original (i.e., non-pruned) MRI.
    :param pruned_mri: Pruned MRI.
    :param pruned_stats_map: [Optional] pruned statistical map.
    :return: MRI with original size (if original_mri is given as Nifti1Image, returns Nifti1Image).
    """
    # Check whether original_mri is Nifti1Image
    is_nifti = isinstance(original_mri, nib.Nifti1Image)

    # Define which volume to use for reverse pruning
    volume_to_reverse_pruning = pruned_mri if pruned_stats_map is None else pruned_stats_map

    # Initialize the MRI to fill
    volume_to_fill = np.zeros(shape=original_mri.shape)
    volume_to_fill[...] = BG_VALUE  # set background

    # Find the edges of the brain (slice format)
    original_mri_edge_slices = find_brain_edges(x3d=original_mri.get_fdata() if is_nifti else original_mri, sl=True)
    pruned_mri_edge_slices = find_brain_edges(x3d=pruned_mri, sl=True)

    # Use the edges to place the brain data at the right spot
    volume_to_fill[original_mri_edge_slices] = volume_to_reverse_pruning[pruned_mri_edge_slices]

    if is_nifti:
        return nib.Nifti1Image(
            dataobj=volume_to_fill,
            affine=original_mri.affine,
            header=original_mri.header if pruned_stats_map is None else None,
            extra=original_mri.extra if pruned_stats_map is None else None,
            dtype=original_mri.get_data_dtype() if pruned_stats_map is None else None,
        )

    return volume_to_fill


def prune_mri(
    x3d: np.ndarray,
    make_cube: bool = False,
    max_axis: int | Sequence[int] | np.ndarray[int] | None = None,
    padding: int = 0,
) -> np.ndarray | None:
    """
    Prune given 3D MRI to (smaller) volume with side-length(s) == `max_axis` [`int` OR 3D tuple].

    If `max_axis` is `None`, find the smallest volume, which covers the brain (i.e., remove zero-padding).
    Works very fast.
    [Implementation with `np.pad` is possible, too].

    Compare to: `nilearn.image.crop_img()` for NIfTI's:
        * This crops exactly along the brain only
        * which is the same as: `mri[find_brain_edges(mri, sl=True)]`
        * but it is slower

    :param x3d: 3D MRI
    :param max_axis: Either side-length [int] of a pruned cube; Or
                     pruned side-length for each axis [3D-sequence: [int, int, int]].
    :param make_cube: True: pruned MRI will be a cube; False: each axis will be pruned to `max_axis`
    :param padding: Number of zero-padding layers that should remain around the brain [int >= 0]
    :return: pruned brain image or `None` if `x3d` is not a numpy array
    """
    # Check argument:
    if max_axis is not None:
        if make_cube:
            if not isinstance(max_axis, (int, np.int_)):
                msg = "If the target volume suppose to be a cube, 'max_axis' must of type int!"
                raise TypeError(msg)
        else:
            msg = "If the target volume suppose to be no cube, 'max_axis' must be a 3D-shaped tuple of integers!"
            if not isinstance(max_axis, (Sequence, np.ndarray)) or not all(
                isinstance(e, (int, np.int_)) for e in max_axis
            ):
                raise TypeError(msg)
            n_dims = 3
            if len(max_axis) != n_dims:
                raise ValueError(msg)

    if not isinstance(padding, (int, np.int_)) or padding < 0:
        msg = "'padding' must be an integer >= 0!"
        raise ValueError(msg)

    if isinstance(x3d, np.ndarray):
        # Cut out
        x3d_minimal = x3d[find_brain_edges(x3d, sl=True)]

        # Prune to smaller volume
        if max_axis is None:
            # find the longest axis for cubing [int] OR take the shape of the minimal volume [3D-tuple]
            max_axis = np.max(x3d_minimal.shape) if make_cube else np.array(x3d_minimal.shape)

        # Add padding at the borders (if requested) & make max_axis a 3D shape-tuple/list
        max_axis = [max_axis + padding] * 3 if make_cube else np.array(max_axis) + padding

        # Initialize an empty 3D target volume
        x3d_small_vol = np.zeros(max_axis, dtype=x3d.dtype)
        if x3d.min() != 0.0:
            x3d_small_vol[x3d_small_vol == 0] = x3d.min()  # in case background is e.g. -1

        x3d_small_vol, _ = _place_small_in_middle_of_big(big=x3d_small_vol, small=x3d_minimal)  # _ = cut

    else:
        cprint(string="'x3d' is not a numpy array!", col="r")
        x3d_small_vol = None

    return x3d_small_vol


def _place_small_in_middle_of_big(big: np.ndarray, small: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Place the small 3D array in the middle of a big 3D array.

    If any axis of the small array is larger than the big one, this axis will be cut symmetrically.

    :param big: larger empty 3D array with the correct data type, here x3d_small_vol
    :param small: smaller 3D array, here x3d_minimal
    :return: big array with the small array in the middle, and bool if the small array was cut
    """
    diff_set = np.subtract(big.shape, small.shape) // 2  # take half of the rest
    cutter_of_small = [0 if x > 0 else (-1) * x for x in diff_set]
    offset_in_big = [0 if x < 0 else x for x in diff_set]
    c1, c2, c3 = cutter_of_small
    a1, a2, a3 = offset_in_big
    cut = any(np.array(cutter_of_small) > 0)
    if cut:
        msg = "Brain tissue was pruned as defined maximal axis was smaller than the brain size."
        cprint(string=msg, col="r")  # could also raise an error

    # cut small objects symmetrically if 'small' is larger than 'big'
    small_cut = small[c1 : small.shape[0] - c1, c2 : small.shape[1] - c2, c3 : small.shape[2] - c3]
    big[
        a1 : a1 + small_cut.shape[0],
        a2 : a2 + small_cut.shape[1],
        a3 : a3 + small_cut.shape[2],
    ] = small_cut

    return big, cut


def find_brain_edges(x3d: np.ndarray, sl: bool = False) -> tuple[slice, slice, slice] | tuple[int, ...]:
    """
    Find the on- & the offset of brain (or head) voxels for each plane.

    This will find the tangential edges of the brain in the given 3D volume.

    ```text
                        /      3D-volume    /
                       +-------------------+
                       |   +_____edge____+ |
                       |   |    *****    | |
                       |   |  **     **  | |
                       |   | *  ** **  * | |
                       |   |*    ***    *| |
                       | Y |*    ***    *| |
                       |   |*    ***    *| |  Z
                       |   |*   ** **   *| | /
                       |   | * **   ** * | |/  /
                       |   |  **     **  | |  /
                       |   |    *****    |/| /
                       |   +––––– X –––––+ |/
                       +-------------------+
    ```

    :param x3d: 3D data.
    :param sl: Whether to return `slice`'s.
               Instead, provide coordinates (if set to `False`, default).
    :return: Tuple with six values of slices or coordinates, two values (lower, upper) per dimension / axis.
    """
    # Slice through image until first appearing brain-voxels are detected (i.e., no background)
    # Find 'lower' planes (i.e., low, left, back, respectively)
    il, jl, kl = 0, 0, 0  # initialize
    while np.all(x3d[il, :, :] == BG_VALUE):  # sagittal slide
        il += 1
    while np.all(x3d[:, jl, :] == BG_VALUE):  # transverse slide
        jl += 1
    while np.all(x3d[:, :, kl] == BG_VALUE):  # coronal/posterior/frontal
        kl += 1

    # Now, find 'upper' planes (i.e., upper, right, front, respectively)
    iu, ju, ku = np.array(x3d.shape) - 1
    while np.all(x3d[iu, :, :] == BG_VALUE):  # sagittal/longitudinal
        iu -= 1
    while np.all(x3d[:, ju, :] == BG_VALUE):  # transverse/inferior/horizontal
        ju -= 1
    while np.all(x3d[:, :, ku] == BG_VALUE):  # coronal/posterior/frontal
        ku -= 1

    if sl:  # return slices
        return slice(il, iu + 1), slice(jl, ju + 1), slice(kl, ku + 1)
    # else return coordinates
    return il, iu, jl, ju, kl, ku


def get_brain_axes_length(x3d: np.ndarray) -> Sequence[int]:
    """
    Get the length of each brain axis (x,y,z) in voxels.

    This will find the tangential edges of the brain in the given 3D volume and measure their lengths.

    ```text
                        /      3D-volume    /
                       +-------------------+
                       |   +_____edge____+ |
                       |   |    *****    | |
                       |   |  **     **  | |
                       |   | *  ** **  * | |
                       |   |*    ***    *| |
                       | Y |*    ***    *| |
                       |   |*    ***    *| |  Z
                       |   |*   ** **   *| | /
                       |   | * **   ** * | |/  /
                       |   |  **     **  | |  /
                       |   |    *****    |/| /
                       |   +––––– X –––––+ |/
                       +-------------------+
    ```

    :param x3d: 3D volume holding a brain / mask.
    :return: The brain axes lengths.
    """
    il, iu, jl, ju, kl, ku = find_brain_edges(x3d)
    return [iu + 1 - il, ju + 1 - jl, ku + 1 - kl]


def get_global_max_axes(nifti_img: nib.Nifti1Image, per_axis: bool) -> int | Sequence[int]:
    """
    Get the global max axis-length(s) for the given brain.

    The global lengths are the maximum axis-length for all brain axes.
    It is globally defined for all brains in the dataset.
    The value can be set in the `PruneConfig` class (`PruneConfig.largest_brain_max_axes`).
    These values are used for pruning brain images.

    :param nifti_img: NIfTI image.
    :param per_axis: True: return max axis-length for each axis (for `prune_mode='max'`);
                     False: return max axis-length for all axes (for `prune_mode='cube'`).
    :return: Global max axis-length(s).
    """
    if nib.orientations.aff2axcodes(nifti_img.affine) != tuple(GLOBAL_ORIENTATION_SPACE):
        msg = (
            f"The orientation of the given NIfTI must match the GLOBAL_ORIENTATION_SPACE: "
            f"'{GLOBAL_ORIENTATION_SPACE}'!"
        )
        raise ValueError(msg)

    # PruneConfig.largest_brain_max_axes is defined for 1 mm isotropic resolution
    resolution = np.round(nifti_img.header["pixdim"][1:4], decimals=3)  # image resolution per axis
    # Adapt the global max axes to the resolution of the given image
    global_max_axis = np.round(PruneConfig.largest_brain_max_axes // resolution).astype(int)
    if not per_axis:
        global_max_axis = int(global_max_axis.max())
    return global_max_axis


def permute_array(xd: np.ndarray) -> np.ndarray:
    """
    Swap all entries (e.g., voxels) in the given x-dimensional array (e.g., 3D-MRI).

    :param xd: x-dimensional array
    :return: permuted array
    """
    flat_xd = xd.flatten()
    np.random.shuffle(flat_xd)  # noqa: NPY002
    return flat_xd.reshape(xd.shape)


def permute_nifti(nifti_img: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    Swap all entries (e.g., voxels) in the given NIfTI image.

    :param nifti_img: NIfTI image
    :return: permuted NIfTI image (i.e., a noise image)
    """
    xd = nifti_img.get_fdata()
    flat_xd = xd.flatten()
    np.random.shuffle(flat_xd)  # noqa: NPY002
    xd = flat_xd.reshape(xd.shape)
    return nib.Nifti1Image(dataobj=xd, affine=nifti_img.affine, header=nifti_img.header)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
