"""
Register MRIs to a new space and create corresponding brain masks.

    Authors: Simon M. Hofmann | Hannah S. Heinrichs
    Years: 2023-2024
"""

# %% Import
from __future__ import annotations

from pathlib import Path

import ants
import nibabel as nib
import numpy as np
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_img

from ..utils import cprint, normalize
from .prune_image import get_global_max_axes, prune_mri
from .transformation import BG_VALUE, file_to_ref_orientation, save_ants_warpers

# %% Set paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

pass


# %% Registration functions  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_mni_template(
    low_res: bool = True,
    reorient: bool = True,
    prune_mode: str | None = "max",
    norm: tuple[int | float, int | float] = (0, 1),
    mask: bool = False,
    original_template: bool = True,
    as_nii: bool = False,
) -> np.ndarray | nib.Nifti1Image:
    """
    Get the MNI template.

    :param low_res: True: 2 mm; False: 1 mm isotropic resolution.
    :param reorient: Whether to reorient the template to the project orientation space.
    :param prune_mode: If not use `None`; image pruning reduces zero-padding around the brain:
                       "cube": all axes have the same length; "max": maximally prune all axes independently
                       Pruning asserts that MRI background to be zero.
    :param norm: Whether to normalize image values between 0-1.
    :param mask: Whether to return a binary mask only.
    :param original_template: With `v.0.8.1` `nilearn` reshaped its MNI template (91,109,91) → (99,117,95).
                              If toggled `True`: this functions uses the previous template.
    :param as_nii: Whether to return template as NIfTI image, else it will be a numpy array.
    :return: MNI template image
    """
    # MNI152 since 2009 (nonlinear version)
    if norm[0] != BG_VALUE:
        msg = "Function works only for zero-background (i.e., min-value = 0)!"
        raise ValueError(msg)
    if isinstance(prune_mode, str):
        prune_mode = prune_mode.lower()
        if prune_mode not in {"cube", "max"}:
            msg = "prune_mode must be 'cube' OR 'max' OR None"
            raise ValueError(msg)

    if low_res:
        # Nilearn has 2mm resolution
        mni_temp = load_mni152_template(resolution=2)
        if original_template:
            # With v.0.8.1, nilearn:
            #  1) reshaped (91, 109, 91) -> (99, 117, 95) &
            #  2) changed the affine to np.array([[2., 0., 0., -98.],
            #                                     [0., 2., 0., -134.],  # noqa: ERA001
            #                                     [0., 0., 2., -72.],  # noqa: ERA001
            #                                     [0., 0., 0., 1.]])
            #  3) rescaled (0-8339) -> (0-255) the MNI template
            #  https://github.com/nilearn/nilearn/blob/d91545d9dd0f74ca884cc91dca751f8224f67d99/doc/changes/0.8.1.rst#enhancements
            mni_temp = resample_img(
                img=mni_temp,
                target_affine=np.array([
                    [-2.0, 0.0, 0.0, 90.0],
                    [0.0, 2.0, 0.0, -126.0],
                    [0.0, 0.0, 2.0, -72.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]),
                target_shape=(91, 109, 91),
            )

            # Remove very small values from interpolation
            mni_temp = nib.Nifti1Image(
                dataobj=np.round(mni_temp.get_fdata(), decimals=3),
                affine=mni_temp.affine,
                header=mni_temp.header,
            )

    else:
        # ANTs template has 1 mm resolution
        from ants import get_ants_data  # , image_read,

        mni_path = get_ants_data("mni")
        mni_temp = nib.load(mni_path)  # ants.image_read(mni_path)

    # Re-orient to global/project orientation space
    if reorient:
        mni_temp = file_to_ref_orientation(image_file=mni_temp)

    if as_nii:
        if isinstance(prune_mode, str) or mask:
            cprint(string="No pruning or masking is done for MNI templates that are returned as NIfTI!", col="r")
        return mni_temp

    if isinstance(prune_mode, str):
        prune_mode = prune_mode.lower()
        global_max = get_global_max_axes(nifti_img=mni_temp, per_axis=prune_mode == "max")
        mni_temp = prune_mri(x3d=mni_temp.get_fdata(), make_cube=prune_mode == "cube", max_axis=global_max)
    else:
        mni_temp = mni_temp.get_fdata()

    # Normalize
    mni_temp = normalize(array=mni_temp, lower_bound=norm[0], upper_bound=norm[1])

    # Create a brain mask version
    if mask:
        mni_temp[mni_temp > BG_VALUE] = 1

    return mni_temp


def register_to_mni(
    moving_mri: nib.nifti1.Nifti1Image,
    resolution: int,
    type_of_transform: str,
    save_path_mni: str | Path | None = None,
    verbose: bool = False,
) -> nib.Nifti1Image:
    """
    Register a NIfTI image to the MNI template.

    Note, if there are issues with file handling,
    functions of `antspyx`, which is used for registration (`ants.registration`),
    cannot handle `*.mgz` | `*.mgh` files,
    use `xai4mri.dataloader.mri_dataloader.mgz2nifti()` for conversion.

    :param moving_mri: Input image to be registered to MNI space.
    :param resolution: Either 1 or 2 mm isotropic resolution.
    :param type_of_transform: Either linear registration: 'Rigid' OR non-linear warping: 'SyN'.
    :param save_path_mni: Path to registered MRI.
                          If provided, save or fetch MRI in MNI space.
    :param verbose: Show output of the registration process.
    :return: MRI in MNI space (NIfTI)
    """
    if save_path_mni and Path(save_path_mni).is_file():
        # Load file as a nibabel object
        return nib.load(save_path_mni)  # ants.to_nibabel(mni_ants)

    if resolution not in {1, 2}:
        msg = "resolution must be either 1 or 2"
        raise ValueError(msg)

    # Create the file with registration to mni
    mni_template = get_mni_template(
        low_res=resolution == 2,  # noqa: PLR2004
        reorient=False,
        prune_mode=None,
        original_template=True,
        as_nii=True,
    )  # 1mm, shape: (182, 218, 182) | 2mm, shape: (91, 109, 91)
    mni_tx = ants.registration(
        fixed=ants.from_nibabel(mni_template),  # reference MRI in MNI space
        moving=ants.from_nibabel(moving_mri),
        type_of_transform=type_of_transform,  # linear: 'Rigid' vs non-linear warping: 'SyN'
        verbose=verbose,
    )

    # Save to the given path of the cache folder
    if save_path_mni is not None:
        save_path_mni = Path(save_path_mni)
        save_path_mni.parent.mkdir(parents=True, exist_ok=True)
        mni_tx["warpedmovout"].to_file(save_path_mni)  # save image in MNI space

        # Save transformation files to cache folder, too
        save_path_transformers = save_path_mni.parent / "transforms2mni"
        save_path_transformers.mkdir(parents=True, exist_ok=True)
        save_ants_warpers(tx=mni_tx, folder_path=str(save_path_transformers), image_name="transform")

    mni_nib = mni_tx["warpedmovout"].to_nibabel()

    if verbose:
        cprint(
            string=f"Given image with original shape {moving_mri.shape} is now in {resolution}mm MNI152 space with "
            f"shape {mni_nib.shape}.",
            col="y",
        )

    return mni_nib


def is_mni(img: nib.nifti1.Nifti1Image | np.ndarray) -> bool:
    """
    Check if the given image is in any MNI space.

    :param img: MR image.
    :return: True if the image is in MNI space, else False
    """
    img_shape = tuple(sorted(img.shape))

    return img_shape in {
        (91, 91, 109),  # sorted(get_mni_template(low_res=True, as_nii=True).shape)
        (95, 99, 117),  # sorted(load_mni152_template(resolution=2).shape)
        (182, 182, 218),  # sorted(get_mni_template(low_res=False, as_nii=True).shape)
        (181, 181, 217),
        (189, 197, 233),  # sorted(load_mni152_template(resolution=1).shape)
        (260, 260, 311),  # sorted(HCP MNI format)
    }


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
