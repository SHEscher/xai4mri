"""
Functions to transform MRIs.

    Author: Simon M. Hofmann
    Years: 2023-2024
"""

# %% Import
from __future__ import annotations

from pathlib import Path
from shutil import copyfile
from typing import Any

import nibabel as nib
import numpy as np

from ..utils import normalize

# %% Global image orientation (according to nibabel) << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# Note: We use FreeSurfer output as reference space, which is according to nibabel: ('L', 'I', 'A'),
# whereas the canonical standard of nibabel itself is: ('R', 'A', 'S') [RAS+]
# For more, see: https://nipy.org/nibabel/coordinate_systems.html
GLOBAL_ORIENTATION_SPACE: str = "LIA"  # Note that for ANTsPy this is vice versa, namely: "RSP"
CLIP_PERCENTILE: float = 99.99
BG_VALUE: int = 0  # assert background in volumes (x3d) equals 0

# %% Global vars << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# TODO: Add options to manipulate MRIs, for data augmentation.  # noqa: FIX002
#  This can include: 'rotation', 'translation', 'noise', 'contrast', 'flip' (biologically implausible)
pass

# %% NiBabel based re-orientation functions < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >


def get_orientation_transform(
    affine: np.ndarray,
    reference_space: str = GLOBAL_ORIENTATION_SPACE,
) -> nib.Nifti1Image:
    """
    Get the orientation transform from a given affine matrix to the reference space.

    The resulting orientation transform (`orient_trans`) can be used to reorient an MRI to a reference space:

        nibabel_image_file.as_reoriented(orient_trans)

    :param affine: affine matrix
    :param reference_space: reference space
    :return: orientation transform
    """
    return nib.orientations.ornt_transform(
        start_ornt=nib.orientations.io_orientation(affine),
        end_ornt=nib.orientations.axcodes2ornt(reference_space),
    )


def file_to_ref_orientation(
    image_file: nib.Nifti1Image, reference_space: str = GLOBAL_ORIENTATION_SPACE
) -> nib.Nifti1Image:
    """
    Take a Nibabel NIfTI-file (not array) and return it reoriented to the (global) reference orientation space.

    :param image_file: NIfTI image file.
    :param reference_space: Reference orientation space.
                            For example, 'LIA' (default) or 'RSP' (as in `ANTsPy`)
    :return: reoriented NIfTI image.
    """
    orient_trans = get_orientation_transform(affine=image_file.affine, reference_space=reference_space)
    return image_file.as_reoriented(orient_trans)


def mri_to_ref_orientation(
    image: np.ndarray,
    affine: np.ndarray,
    reference_space: str = GLOBAL_ORIENTATION_SPACE,
) -> np.ndarray:
    """
    Reorient an MRI array to a reference orientation space.

    Take an MRI array plus its corresponding affine matrix
    and return the MRI reoriented to the (global) reference space.

    :param image: MRI array.
    :param affine: Corresponding affine matrix of the MRI array.
    :param reference_space: Reference orientation space.
                            For example, 'LIA' (default) or 'RSP' (as in `ANTsPy`)
    :return: reoriented MRI array.
    """
    # Create orientation transform object first
    orient_trans = get_orientation_transform(affine=affine, reference_space=reference_space)
    return nib.apply_orientation(image, orient_trans)


# %% ANTspy based warping function  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def save_ants_warpers(tx: dict[str, Any], folder_path: str | Path, image_name: str) -> None:
    """
    Save warper files from `ANTsPy`'s `tx` object to the given `folder_path`.

    :param tx: `ANTsPy` transformation object (via `ants.registration()`).
    :param folder_path: folder path to save warper files
    :param image_name: Image name that will be used as prefix for the warper files.
    """
    if "fwdtransforms" not in list(tx.keys()) or "invtransforms" not in list(tx.keys()):
        msg = "tx object misses forward and/or inverse transformation files."
        raise ValueError(msg)

    # Check whether linear transformation ("Rigid") or non-linear ("SyN")
    linear = tx["fwdtransforms"] == tx["invtransforms"]

    # # Set paths
    # for forward warper
    save_path_name_fwd = str(Path(folder_path, f"{image_name}1Warp.nii.gz"))
    # for inverse warper
    save_path_name_inv = str(Path(folder_path, f"{image_name}1InverseWarp.nii.gz"))
    # # Save also linear transformation .mat file
    save_path_name_mat = str(Path(folder_path, f"{image_name}0GenericAffine.mat"))

    # # Copy warper files from the temporary tx folder file to new location
    if linear:
        copyfile(tx["fwdtransforms"][0], save_path_name_mat)
    else:
        copyfile(tx["fwdtransforms"][0], save_path_name_fwd)
        copyfile(tx["invtransforms"][1], save_path_name_inv)
        copyfile(tx["invtransforms"][0], save_path_name_mat)  # == ['fwdtransforms'][1]


# %% Masking & Transformations << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply volume mask to data.

    Data and the corresponding mask must have the same shape.

    :param data: Data to be masked.
    :param mask: Mask to be applied.
    :return: Masked data.
    """
    if data.shape != mask.shape:
        # check orientation/shape
        msg = f"Data shape {data.shape} does not match brain mask shape {mask.shape}."
        raise ValueError(msg)
    return data * mask


def determine_min_max_clip(data: np.ndarray, at_percentile: float = CLIP_PERCENTILE) -> tuple[float, float]:
    """
    Determine the min and max clip value for the given data.

    This clips the data at the given percentile.
    That is, all values below the min clip value are set to the min clip value,
    and all values above the max clip value are set to the max clip value.

    :param data: data as numpy array
    :param at_percentile: percentile of the upperbound
    :return: min and max clipping values
    """
    bg = float(BG_VALUE)  # asserting background value at zero
    d = data[data != bg]  # filter zeros / background, leave informative signal
    n_dec = len(str(at_percentile)) - 1
    max_clip_val = np.nanpercentile(d, q=at_percentile).round(decimals=n_dec)
    min_clip_val = np.minimum(
        bg,  # min_clip_val must be negative or zero
        np.nanpercentile(d, q=100 - at_percentile).round(decimals=n_dec),
    )
    return min_clip_val, max_clip_val


def clip_data(data: np.ndarray, at_percentile: float = CLIP_PERCENTILE) -> np.ndarray:
    """
    Clip provided data at a certain intensity percentile as the clipping threshold.

    :param data: Data to be clipped.
    :param at_percentile: Percentile of the upper bound.
    :return: Clipped data.
    """
    min_clip_val, max_clip_val = determine_min_max_clip(data=data, at_percentile=at_percentile)
    return np.clip(a=data, a_min=min_clip_val, a_max=max_clip_val)


def _check_norm(data: np.ndarray, compress: bool) -> bool:
    """Check if data can be normalized."""
    if data.min() < 0:
        msg = (
            f"Data will not be normalised between 0-255, since its minimum {data.min()} < 0!\n"
            "For statistical maps that include negative values, set 'norm' to False.\n"
            "Otherwise check your data for invalid values smaller than 0.\n"
            f"{'' if compress else f'Also, try to set compress to True, to clip the data at {CLIP_PERCENTILE}%.'}"
        )
        raise ValueError(msg)
    return True


def compress_and_norm(
    data: np.ndarray,
    clip_min: float | int | None,
    clip_max: float | int | None,
    norm: bool | None,
    global_norm_min: float | int | None,
    global_norm_max: float | int | None,
) -> np.ndarray:
    """
    Clip, normalize, and/or compress data.

    :param data: Data to be processed.
    :param clip_min: Minimum clip value.
    :param clip_max: Maximum clip value.
    :param norm: Whether to normalize data.
    :param global_norm_min: Global minimum value for normalization (usually if data is part of bigger dataset).
    :param global_norm_max: Global maximum value for normalization (usually if data is part of bigger dataset).
    :return: Processed data.
    """
    # Check args
    if type(clip_min) is not type(clip_max):
        msg = f"clip_min [{type(clip_min)}] and clip_max [{type(clip_max)}] must be of same type."
        raise TypeError(msg)

    compress = False  # init

    # Clip
    if clip_min is not None and clip_max is not None:
        data = np.clip(a=data, a_min=clip_min, a_max=clip_max)
        compress = True

    # Normalize
    norm = (global_norm_min is not None and global_norm_max is not None) or norm

    if norm and _check_norm(data=data, compress=compress):
        data = normalize(
            array=data,
            lower_bound=0,
            upper_bound=255,
            global_min=global_norm_min,
            global_max=global_norm_max,
        )
    # Compress
    if compress and norm:
        data = np.round(data).astype(np.uint8)
    return data


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
