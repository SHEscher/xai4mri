"""
Backend functions to load MRI and target data of different datasets.

This is more or less the backend of the data loading process in `xai4mri.dataloader.datasets`.

    Authors: Simon M. Hofmann | Hannah S. Heinrichs
    Years: 2023-2024
"""

# %% Import
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import (
    _load_obj,
    _save_obj,
    ask_true_false,
    check_storage_size,
    cprint,
)
from .mri_registration import is_mni, register_to_mni
from .prune_image import get_global_max_axes, prune_mri
from .transformation import (
    CLIP_PERCENTILE,
    apply_mask,
    compress_and_norm,
    determine_min_max_clip,
    file_to_ref_orientation,
)

# %% Global vars < o >><< o >>><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Paths
CACHE_DIR = "~/.xai4mri/cache"

# Constants
MAXIMUM_SIZE_BYTES: int = 15 * 10**9  # 15 GB


# %% Loader functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _check_regis(regis_mni: int | None) -> bool:
    """Check if the registration argument 'regis_mni' is valid."""
    if regis_mni is None:
        return False
    msg = "regis_mni must be 1 or 2 [mm], OR None / False."
    if isinstance(regis_mni, bool):
        if regis_mni:
            raise ValueError(msg)
        return False
    if isinstance(regis_mni, int) and regis_mni in {1, 2}:
        return True
    raise ValueError(msg)


def get_metadata_path(
    project_id: str,
    mri_seq: str,
    regis_mni: int | None,
    path_brain_mask: str | None,
    norm: bool,
    prune_mode: str | None,
    path_to_dataset: str | Path | None,
) -> Path:
    """
    Get the path to the metadata table of a project's dataset.

    :param project_id: project ID
    :param mri_seq: MRI sequence (e.g., 't1w')
    :param regis_mni: set when data was transformed to MNI space (1 or 2 mm) or None
    :param path_brain_mask: if used, path to the applied brain mask, else None
    :param norm: if images were normalized
    :param prune_mode: if not used: None; else pruning mode: "cube" or "max".
    :param path_to_dataset: Optional path to folder containing project data (if not in globally set `cache_dir`)
    :return: path to the metadata table of the project dataset
    """
    mri_set_name = get_mri_set_name(
        project_id=project_id,
        mri_seq=mri_seq,
        regis_mni=regis_mni,
        brain_masked=isinstance(path_brain_mask, str),
        norm=norm,
        prune_mode=prune_mode,
    )

    return Path(
        get_mri_set_path(
            mri_set_name=mri_set_name + "_metadata",
            path_to_folder=path_to_dataset,
        ).replace(".npy", ".csv")
    )


def _load_data_as_full_array(
    sid_list: list[str] | np.ndarray[str],
    project_id: str,
    mri_path_constructor: Callable[[str], str],
    mri_seq: str,
    regis_mni: int | None = None,
    norm: bool = True,
    path_brain_mask: str | None = None,
    compress: bool = True,
    prune_mode: str | None = "max",
    path_to_dataset: str | Path | None = None,
    cache_files: bool = True,
    save_after_processing: bool = True,
    cache_dir: str | Path = CACHE_DIR,
    **save_kwargs,
) -> tuple[np.ndarray[Any, np.dtype[np.uint8 | np.float32]], np.ndarray[Any, np.dtype[str]]] | None:
    """
    Load MRI data as a full numpy array and process them if necessary.

    :param sid_list: list of subject ID's
    :param project_id: project ID
    :param mri_path_constructor: function to construct path to MRI for the given project
    :param mri_seq: MRI sequence (e.g., 't1')
    :param regis_mni: transform to mni space (1 or 2 mm)
    :param norm: normalize image to [0, 255]
    :param path_brain_mask: path to brain mask, None if brain mask not available/required
    :param compress: clip, and if norm=True, convert image to lower level data type (uint8)
    :param prune_mode: if not use None; image pruning reduces zero-padding around the brain:
                       "cube": all axes have the same length; "max": maximally prune all axes independently
                       Pruning asserts that MRI background to be zero.
    :param path_to_dataset: Optional path to data folder (if not in cache_dir)
    :param cache_files: cache interim files in 1 or 2 mm MNI space if requested
    :param save_after_processing: save processed MRI data as a large numpy array (N_sids, x,y,z, channel=1)
    :param cache_dir: directory to cache files
    :param save_kwargs: keyword arguments for _save_mri_set() & get_mri_set_path()
    :return: data array (MRIs) of shape [n_subjects, x,y,z, channel=1], &
             ordered list of corresponding subject ID's
    """
    # Specify the name and location of the MRI set
    mri_set_name = get_mri_set_name(
        project_id=project_id,
        mri_seq=mri_seq,
        regis_mni=regis_mni,
        brain_masked=isinstance(path_brain_mask, str),
        norm=norm,
        prune_mode=prune_mode,
    )

    # Extract kwargs
    mmap_mode = save_kwargs.pop("mmap_mode", None)
    force = save_kwargs.pop("force", False)

    # Create the path to the data
    if not isinstance(path_to_dataset, (str, Path)):
        path_to_dataset = cache_dir
    mri_set_path = Path(
        get_mri_set_path(mri_set_name=mri_set_name + "_set", path_to_folder=path_to_dataset, **save_kwargs)
    )
    metadata_path = Path(str(mri_set_path).replace("_set.npy", "_metadata.csv"))

    # Process the MRI set
    if len(list(mri_set_path.parent.glob(f"{mri_set_name}_*"))) > 1:  # must be *set.npy, *sids.npy, & *metadata.csv
        # If the MRI set is already there, load it
        cprint(string=f"MRI set exists. Loading '{mri_set_name}' into workspace ...", col="g")
        volume_data = _load_obj(name=mri_set_name + "_set", folder=mri_set_path.parent, mmap_mode=mmap_mode)
        sid_list = _load_obj(
            name=mri_set_name + "_sids",
            folder=mri_set_path.parent,
            functimer=False,
            mmap_mode=mmap_mode,
        )
        return volume_data, sid_list

    try:
        if metadata_path.exists() and not save_after_processing:
            # Load from metadata
            input_sid_list = np.array(sid_list)
            volume_data, sid_list = load_files_from_metadata(sid_list=sid_list, path_to_metadata=metadata_path)
            if set(sid_list) != set(input_sid_list):
                cprint(
                    string=f"Some requested files were not found in the metadata table '{metadata_path}'.\n"
                    f"Will reprocess the data including the metadata table ...",
                    col="y",
                )
                sid_list = input_sid_list
            else:
                return volume_data, sid_list
    except FileNotFoundError:
        cprint(
            string=f"Metadata table '{metadata_path}' contains information on files that were not found on disk.\n"
            f"Will reprocess the data including the metadata table ...",
            col="y",
        )

    # Load and save
    if not force and not ask_true_false(question="Prepare dataset (may take a while)? ", col="b"):
        msg = "Aborting preparation of dataset ..."
        raise KeyboardInterrupt(msg)

    volume_data, sid_list = _process_mri_set(
        mri_set_name=mri_set_name,
        project_id=project_id,
        mri_path_constructor=mri_path_constructor,
        sid_list=sid_list,
        prune_mode=prune_mode,
        norm=norm,
        compress=compress,
        path_brain_mask=path_brain_mask,
        dtype=np.float32,
        regis_mni=regis_mni,
        cache_files=cache_files,
        cache_dir=cache_dir,
    )

    if save_after_processing:
        # Save MRI data
        _save_mri_set(data=volume_data, set_path=mri_set_path, **save_kwargs)
        # Save corresponding subject ID's
        _save_mri_set(
            data=sid_list,
            set_path=str(mri_set_path).replace("_set", "_sids"),
            verbose=False,
            functimer=False,
            **save_kwargs,
        )
        cprint(string=f"Saved {project_id.upper()} data set: {mri_set_path}", col="g")

    return volume_data, sid_list


def _load_data_as_file_paths(
    sid_list: list[str] | np.ndarray[str],
    project_id: str,
    mri_path_constructor: Callable[[str], str],
    mri_seq: str,
    regis_mni: int | None = None,
    norm: bool = True,
    path_brain_mask: str | None = None,
    prune_mode: str | None = "max",
    path_to_dataset: str | Path | None = None,
    cache_dir: str | Path = CACHE_DIR,
    **save_kwargs,
) -> tuple[np.ndarray[Any, np.dtype[str | Path]], np.ndarray[Any, np.dtype[str]]]:
    """
    Load MRI data as file paths and process them if necessary.

    :param sid_list: list of subject ID's
    :param project_id: project ID
    :param mri_path_constructor: function to construct path to MRI for the given project
    :param mri_seq: MRI sequence (e.g., 't1')
    :param regis_mni: transform to mni space (1 or 2 mm)
    :param norm: normalize image to [0, 255]
    :param path_brain_mask: path to brain mask, None if brain mask not available/required
    :param prune_mode: if not use None; image pruning reduces zero-padding around the brain:
                       "cube": all axes have the same length; "max": maximally prune all axes independently
                       Pruning asserts that MRI background to be zero:
    :param path_to_dataset: Optional path to data folder (if not in cache_dir)
    :param cache_files: cache interim files in 1 or 2 mm MNI space if requested
    :param cache_dir: directory to cache files
    :param save_kwargs: keyword arguments for _save_mri_set() & get_mri_set_path()
    :return: data array (MRIs) of shape [n_subjects, x,y,z, channel=1], &
             ordered list of corresponding subject ID's
    """
    # Specify the name and location of the MRI set
    mri_set_name = get_mri_set_name(
        project_id=project_id,
        mri_seq=mri_seq,
        regis_mni=regis_mni,
        brain_masked=isinstance(path_brain_mask, str),
        norm=norm,
        prune_mode=prune_mode,
    )

    # First pop unnecessary kwargs for _load_data_as_file_paths()
    _ = save_kwargs.pop("mmap_mode", None)
    _ = save_kwargs.pop("cache_files", None)
    _ = save_kwargs.pop("save_after_processing", None)
    _ = save_kwargs.pop("compress", None)
    # Extract further kwargs
    force = save_kwargs.pop("force", False)
    exist_check = save_kwargs.pop("exist_check", True)

    # Create the path to the data
    if not isinstance(path_to_dataset, (str, Path)):
        path_to_dataset = cache_dir
    metadata_path = Path(
        get_mri_set_path(
            mri_set_name=mri_set_name + "_metadata",
            path_to_folder=path_to_dataset,
            **save_kwargs,
        ).replace(".npy", ".csv")
    )

    # Process the MRI set
    try:
        if metadata_path.exists():
            # Load from metadata
            input_sid_list = np.array(sid_list)
            volume_data, sid_list = load_file_paths_from_metadata(
                sid_list=sid_list, path_to_metadata=metadata_path, exist_check=exist_check
            )
            if set(sid_list) != set(input_sid_list):
                cprint(
                    string=f"Some requested files were not found in the metadata table '{metadata_path}'.\n"
                    f"Will reprocess the data including the metadata table ...",
                    col="y",
                )
                sid_list = input_sid_list
            else:
                return volume_data, sid_list
    except FileNotFoundError:
        cprint(
            string=f"Metadata table '{metadata_path}' contains information on files that were not found on disk.\n"
            f"Will reprocess the data including the metadata table ...",
            col="y",
        )

    # Load and save
    if not force and not ask_true_false(question="Prepare dataset (may take a while)? ", col="b"):
        msg = "Aborting preparation of dataset ..."
        raise KeyboardInterrupt(msg)

    volume_data, sid_list = _process_mri_set_as_individual_files(
        mri_set_name=mri_set_name,
        project_id=project_id,
        mri_path_constructor=mri_path_constructor,
        sid_list=sid_list,
        prune_mode=prune_mode,
        path_brain_mask=path_brain_mask,
        dtype=np.float32,
        regis_mni=regis_mni,
        cache_dir=cache_dir,
    )

    return volume_data, sid_list


def _prepare_metadata(path_to_metadata: str | Path) -> tuple[pd.DataFrame, str]:
    """
    Prepare metadata table.

    :param path_to_metadata: path to the metadata table
    ;return: metadata table, column name for information on processed MRIs
    """
    metadata_table = pd.read_csv(path_to_metadata, index_col="sid", dtype={"sid": str})
    metadata_table.index = metadata_table.index.astype(str)
    processed_col = [c for c in metadata_table.columns if not ("min" in c or "max" in c or "source" in c)][-1]
    return metadata_table, processed_col


def load_files_from_metadata(
    sid_list: list[str] | np.ndarray[str], path_to_metadata: str | Path
) -> tuple[np.ndarray[Any, np.dtype[np.uint8 | np.float32]], np.ndarray[Any, np.dtype[str]]]:
    """
    Load MRI data from a project's metadata table.

    :param sid_list: List of subject ID's.
    :param path_to_metadata: Path to the metadata table of a project dataset.
    :return: Array of MRI files with the shape `[n_subjects, x, y, z, 1]`,
             and an ordered list of corresponding subject ID's.
    """
    # Load metadata
    metadata_table, processed_col = _prepare_metadata(path_to_metadata=path_to_metadata)
    compress = len([c for c in metadata_table.columns if "clip" in c]) > 0
    norm = "-n-" in Path(path_to_metadata).name
    metadata_table = metadata_table.dropna()  # this cleans up the metadata table, when it has not been fully processed

    # Check if all SIDs in metadata table
    sids_not_in_metadata = set(sid_list).difference(metadata_table.index)
    if len(sids_not_in_metadata) > 0:
        cprint(
            string=f"Metadata table does not contain all given SIDs, missing: {sids_not_in_metadata}.\n"
            f"Return only SIDs data which are also in the metadata table.",
            col="y",
        )
        sid_list = [sid for sid in sid_list if sid not in sids_not_in_metadata]

    # Load data
    all_mri = None
    for sid_idx, sid in tqdm(
        enumerate(sid_list),
        total=len(sid_list),
        desc=f"Loading MRI data for {Path(path_to_metadata).name.split('_metadata.csv')[0]}",
    ):
        path_to_mri = Path(metadata_table.loc[sid, processed_col])
        single_data4d = _load_obj(name=path_to_mri.name, folder=path_to_mri.parent, functimer=False)
        single_data4d = np.expand_dims(single_data4d, axis=0)

        if all_mri is None:
            all_mri = np.empty(
                shape=(
                    len(sid_list),
                    single_data4d.shape[1],
                    single_data4d.shape[2],
                    single_data4d.shape[3],
                ),
                dtype=np.float32,
            )  # init data set (to be filled)
        all_mri[sid_idx, :, :, :] = single_data4d

    # Clip, normalize and, or compress all images
    clip_min = np.nanmean(metadata_table[f"{processed_col}_min_clip"]) if compress else None
    clip_max = np.nanmean(metadata_table[f"{processed_col}_max_clip"]) if compress else None
    all_mri = compress_and_norm(data=all_mri, clip_min=clip_min, clip_max=clip_max, norm=norm)

    # expand to empty dimension (batch_size, x, y, z, channel=1)
    all_mri = np.expand_dims(all_mri, axis=4)

    return all_mri, np.array(sid_list)


def load_file_paths_from_metadata(
    sid_list: list[str] | np.ndarray[str], path_to_metadata: str | Path, exist_check: bool = True
) -> tuple[np.ndarray[Any, np.dtype[str | Path]], np.ndarray[Any, np.dtype[str]]]:
    """
    Load file paths to MRI data from a project's metadata table.

    :param sid_list: List of subject ID's.
    :param path_to_metadata: Path to the metadata table of a project dataset.
    :param exist_check: Check if image files exist.
    :return: Array of MRI file paths and ordered list of corresponding subject ID's.
    """
    # Load metadata
    metadata_table, processed_col = _prepare_metadata(path_to_metadata=path_to_metadata)
    metadata_table = metadata_table.dropna()  # clean up the metadata table, when it has not been fully processed

    # Check if all SIDs in metadata table
    sids_not_in_metadata = set(sid_list).difference(metadata_table.index)
    if len(sids_not_in_metadata) > 0:
        cprint(
            string=f"Metadata table does not contain all requested SIDs, missing: {sids_not_in_metadata}.\n"
            f"Return only SIDs data which are also in the metadata table.",
            col="y",
        )
        sid_list = [sid for sid in sid_list if sid not in sids_not_in_metadata]

    # Load data
    all_mri_paths = metadata_table.loc[sid_list, processed_col].to_numpy()

    if exist_check and not all(Path(p).is_file() for p in all_mri_paths):
        msg = f"Not all requested files exist in: '{path_to_metadata}' (column: ['{processed_col}'])!"
        raise FileNotFoundError(msg)

    return all_mri_paths, np.array(sid_list)


def mgz2nifti(nib_mgh: nib.freesurfer.mghformat.MGHImage) -> nib.nifti1.Nifti1Image:
    """
    Convert Freesurfer's MGH-NMR [`*.mgh` | `*.mgz`] file to NIfTI [`*.nii`].

    :param nib_mgh: `nibabel` `MGHImage` object
    :return: `nibabel` `Nifti1Image` object
    """
    return nib.Nifti1Image(dataobj=nib_mgh.get_fdata(caching="unchanged"), affine=nib_mgh.affine)


def get_nifti(mri_path: str | Path, reorient: bool) -> nib.nifti1.Nifti1Image:
    """
    Get NIfTI image from its file path.

    This works for both NIfTI [`*.nii` | `*.nii.gz`] and MGH [`*.mgh` | `*.mgz`] files.

    :param mri_path: path to an MRI file
    :param reorient: reorient the image to the global project orientation space
    :return: nibabel Nifti1Image object
    """
    nifti_img = nib.load(mri_path)
    # Define input space
    if isinstance(nifti_img, nib.freesurfer.mghformat.MGHImage):
        nifti_img = mgz2nifti(nifti_img)

    if reorient:
        nifti_img = file_to_ref_orientation(image_file=nifti_img)

    return nifti_img


def process_single_mri(
    mri_path: str | Path,
    dtype: type = np.float32,
    prune_mode: str | None = "max",
    path_brain_mask: str | Path | None = None,
    regis_mni: int | None = None,
    path_cached_mni: str | Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Load an individual MRI of an individual subject as a numpy array.

    :param mri_path: path to the original NIfTI MRI file
    :param dtype: data type of returned MRI (default: `np.float32`)
    :param prune_mode: if not use: `None`; image pruning reduces zero-padding around the brain:
                       "cube": all axes have the same length; "max": maximally prune all axes independently
    :param path_brain_mask: path to the brain mask; if no mask should be applied use `None`
    :param regis_mni: transform MRI to MNI space in 1 or 2 mm resolution [int], or `None` for no registration
    :param verbose: be verbose about the process or not
    :param path_cached_mni: if a path is provided, save interim file in MNI space to this cache path
    :return: 4D numpy array (MRI) of shape `[(empty), x, y, z]`
    """
    nifti_img = get_nifti(mri_path=mri_path, reorient=False)  # reorient after registration below

    # Resample to MNI space with 1 or 2 mm resolution
    if _check_regis(regis_mni):
        nifti_img = register_to_mni(
            moving_mri=nifti_img,
            resolution=regis_mni,
            save_path_mni=path_cached_mni,
            type_of_transform="Rigid" if is_mni(img=nifti_img) else "SyN",
            verbose=verbose,
        )

    # Reorient image to global project orientation space
    nifti_img = file_to_ref_orientation(image_file=nifti_img)

    # Get image as numpy array
    data3d = nifti_img.get_fdata(dtype=dtype, caching="unchanged")

    # Prune image, i.e. minimize zero-padding
    if isinstance(prune_mode, str):
        # Set max-axes lengths for pruning
        global_max = get_global_max_axes(nifti_img=nifti_img, per_axis=prune_mode.lower() == "max")
        try:
            data3d = prune_mri(x3d=data3d, make_cube=prune_mode.lower() == "cube", max_axis=global_max)
        except IndexError as e:
            cprint(string=f"\nCould not prune MRI: {mri_path}\n", col="r")
            raise e

    if isinstance(path_brain_mask, (str, Path)):
        bm = nib.load(path_brain_mask).get_fdata(caching="unchanged")  # could pass dtype=np.uint8
        data3d = apply_mask(data=data3d, mask=bm)

    return np.expand_dims(data3d, axis=0)  # now 4d: (*dims, 1)


def _create_cache_files_path(
    f_path: str | Path,
    project_id: str,
    sid: str,
    suffix: str,
    folder_name: str | None = "",
    ext: str | None = None,
    cache_dir: str | Path = CACHE_DIR,
) -> str:
    """
    Create the file path (cache) for processed MRI files.

    :param f_path: file path
    :param project_id: project ID
    :param sid: subject ID
    :param suffix: suffix for file name
    :param folder_name: folder_name of cache files
    :param ext: extensions (should be .pkl, .npy, .nii, (optional: + .*.gz for zip-files [not for npy]))
    :return: path to the cache file
    """
    # Construct the path to the cached file
    cache_file_dir = Path(cache_dir, project_id, str(sid), folder_name)
    cache_file_name = ".".join(Path(f_path).name.split(".")[:-1])  # remove extension
    if isinstance(ext, str):
        ext = ext[1:] if ext.startswith(".") else ext
    else:  # ext is None
        ext = ".".join(Path(f_path).name.split(".")[1:])
    cache_file_name += f"_{suffix}.{ext}" if suffix else f".{ext}"

    return str(cache_file_dir / cache_file_name)


def _process_mri_set(
    mri_set_name: str,
    project_id: str,
    mri_path_constructor: Callable[[str | Path], str],
    sid_list: list[str],
    dtype: type = np.float32,
    prune_mode: str | None = "max",
    norm: bool = True,
    path_brain_mask: str | Path | None = None,
    compress: bool = True,
    regis_mni: int | None = None,
    cache_files: bool = True,
    cache_dir: str | Path = CACHE_DIR,
) -> tuple[np.ndarray[Any, np.dtype[np.uint8 | np.float32]], np.ndarray[Any, np.dtype[Any]]]:
    """
    Construct a numpy array with images (3D) of all subjects (→4D) and additional empty dimension (→5D).

    Ideally, this function is run only once, and all processed files are saved in one (big) data object,
    which _load_data_as_full_array() retrieves from memory (at location mri_set_path).

    :param mri_set_name: name of the MRI dataset
    :param project_id: Project ID
    :param mri_path_constructor: function to construct single mri paths using arguments sid
    :param sid_list: list of subject ID's
    :param dtype: data type
    :param prune_mode: if not use None; image pruning reduces zero-padding around the brain:
                       "cube": all axes have the same length; "max": maximally prune all axes independently
    :param norm: normalize image to [0, 255]
    :param path_brain_mask: path to brain mask, None if brain mask not available/required
    :param compress: clip, and if norm=True, convert image to lower level data type (uint8)
    :param regis_mni: transform to mni space (1 or 2 mm)
    :param cache_files: cache interim files in MNI space if requested
    :param cache_dir: directory to cache files
    :return: data array (MRIs) of shape [n_subjects, x,y,z, channel=1], and
             ordered list of corresponding subject ID's
    """
    # Check arguments
    if isinstance(prune_mode, str):
        prune_mode = prune_mode.lower()
        if prune_mode not in {"cube", "max"}:
            msg = "prune_mode must be 'cube' OR 'max' OR None"
            raise ValueError(msg)

    # Full procedure
    invalid_paths = []  # collect unavailable NIfTIs in list
    sids_with_mri = []
    all_mri = None

    # Prepare suffix for cache files
    path_suffix_register = f"mni{regis_mni}mm" if (_check_regis(regis_mni) and cache_files) else ""
    path_suffix_numpy = [path_suffix_register] if path_suffix_register else []
    if isinstance(prune_mode, str):
        path_suffix_numpy.append(f"{prune_mode[0]}-pruned")  # "c-pruned" (cube) OR "m-pruned" (max)
    if isinstance(path_brain_mask, str):
        path_suffix_numpy.append("brain_masked")
    path_suffix_numpy = "_".join(path_suffix_numpy)  # == "" if no suffix so far
    path_suffix_numpy = path_suffix_numpy if path_suffix_numpy else "cached_numpy"

    # Prepare metadata table
    source_mri_col = "source_mri"
    path_to_metadata = Path(cache_dir, f"{mri_set_name}_metadata.csv")
    if path_to_metadata.exists():
        metadata_table = pd.read_csv(path_to_metadata, index_col="sid", dtype={"sid": str})
    else:
        metadata_columns = [source_mri_col]
        metadata_columns += [path_suffix_register] if path_suffix_register else []
        metadata_columns += [
            path_suffix_numpy,
            f"{path_suffix_numpy}_min_intensity",  # of processed images
            f"{path_suffix_numpy}_max_intensity",
            f"{path_suffix_numpy}_min_clip",  # of processed images
            f"{path_suffix_numpy}_max_clip",  # this is then the max intensity value for all processed images
        ]
        metadata_table = pd.DataFrame(index=sid_list, columns=metadata_columns)
        metadata_table.index.name = "sid"

    cprint(string=f"Loading {len(sid_list)} samples of the MRI set for {project_id} ...", col="b")
    for sid_idx, sid in tqdm(enumerate(sid_list), total=len(sid_list)):
        # Load single 3D files to data
        source_mri_path = mri_path_constructor(str(sid))
        if not Path(source_mri_path).is_file():
            invalid_paths.append(source_mri_path)
            if all_mri is not None:
                all_mri = all_mri[:-1, ...]  # remove empty slot
            continue
        metadata_table.loc[sid, source_mri_col] = source_mri_path

        # Get the full path to cache files
        path_cache_registered = (
            _create_cache_files_path(
                f_path=source_mri_path,
                project_id=project_id,
                sid=sid,
                suffix=path_suffix_register,
                ext="nii.gz",
                cache_dir=cache_dir,
            )
            if cache_files
            else None
        )
        if cache_files and path_suffix_register:
            metadata_table.loc[sid, path_suffix_register] = path_cache_registered

        path_cache_preprocessed = _create_cache_files_path(
            f_path=source_mri_path,
            project_id=project_id,
            sid=sid,
            suffix="" if path_suffix_numpy == "cached_numpy" else path_suffix_numpy,
            ext="npz",  # numpy arrays will be zipped in cache → .npz
            cache_dir=cache_dir,
        )
        if cache_files:
            metadata_table.loc[sid, path_suffix_numpy] = path_cache_preprocessed
        path_cache_preprocessed = Path(path_cache_preprocessed)

        #  Check whether processed MRI (as numpy) already there:
        if path_cache_preprocessed.is_file() and not (
            cache_files and _check_regis(regis_mni) and not Path(path_cache_registered).is_file()
        ):
            # Numpy array found and a MNI-registered image does not have to be cached (anymore)
            single_data4d = _load_obj(
                name=path_cache_preprocessed.name,
                folder=path_cache_preprocessed.parent,
                functimer=False,
            )
            single_data4d = np.expand_dims(single_data4d, axis=0)

        else:
            # Process MRI of current participant
            single_data4d = process_single_mri(
                mri_path=source_mri_path,
                dtype=dtype,
                prune_mode=prune_mode,
                path_brain_mask=path_brain_mask,
                regis_mni=regis_mni,
                path_cached_mni=path_cache_registered,
            )

            if cache_files:
                # Save data object accordingly
                _save_obj(
                    obj=single_data4d.squeeze(),
                    name=Path(path_cache_preprocessed).name,
                    folder=Path(path_cache_preprocessed).parent,
                    save_as="npy",
                    as_zip=True,
                    functimer=False,
                )
                # Note, this data is not compressed nor normed yet
        # TODO: parallelize [all points also valid for _process_mri_set_as_individual_files()] # noqa: FIX002
        # TODO: for non-registered (non-pruned) data, consider creating symlinks or linking directly to the source data # noqa: E501, FIX002
        # TODO: for only-pruned data, consider saving prune-axes in metadata table & prune while loading # noqa: FIX002

        # Fill intensity & clip values
        metadata_table.loc[sid, f"{path_suffix_numpy}_min_intensity"] = single_data4d.min()
        metadata_table.loc[sid, f"{path_suffix_numpy}_max_intensity"] = single_data4d.max()
        min_clip, max_clip = determine_min_max_clip(data=single_data4d, at_percentile=CLIP_PERCENTILE)
        metadata_table.loc[sid, f"{path_suffix_numpy}_min_clip"] = min_clip
        metadata_table.loc[sid, f"{path_suffix_numpy}_max_clip"] = max_clip

        # Save current state of metadata table
        path_to_metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata_table.to_csv(path_to_metadata)

        # Gather data ([empty,x,y,z]) in 4d array
        sids_with_mri.append(sid)
        if all_mri is None:
            all_mri = np.empty(
                shape=(
                    len(sid_list) - len(invalid_paths),
                    single_data4d.shape[1],
                    single_data4d.shape[2],
                    single_data4d.shape[3],
                ),
                dtype=dtype,
            )  # init data set (to be filled)
        all_mri[sid_idx - len(invalid_paths), :, :, :] = single_data4d

    # Clip, normalize and, or compress all images
    clip_min = np.nanmean(metadata_table[f"{path_suffix_numpy}_min_clip"]) if compress else None
    clip_max = np.nanmean(metadata_table[f"{path_suffix_numpy}_max_clip"]) if compress else None
    all_mri = compress_and_norm(
        # global_norm_min|max are not necessary, since all volumes are used at once
        data=all_mri,
        clip_min=clip_min,
        clip_max=clip_max,
        norm=norm,
        global_norm_min=None,
        global_norm_max=None,
    )

    # Expand to empty dimension (batch_size, x, y, z, channel=1)
    all_mri = np.expand_dims(all_mri, axis=4)

    if len(invalid_paths) > 0:
        cprint(string="Participants with non-existent data:\n" + "\n".join(f"\t{i}" for i in invalid_paths), col="y")

    return all_mri, np.array(sids_with_mri)


def _process_mri_set_as_individual_files(
    mri_set_name: str,
    project_id: str,
    mri_path_constructor: Callable[[str | Path], str],
    sid_list: list[str],
    dtype: type = np.float32,
    prune_mode: str | None = "max",
    path_brain_mask: str | None = None,
    regis_mni: int | None = None,
    cache_dir: str | Path = CACHE_DIR,
) -> tuple[np.ndarray[Any, np.dtype[str | Path]], np.ndarray[Any, np.dtype[str]]]:
    """
    Construct a numpy array with paths to volumes of all subjects (sid).

    Ideally, this function is run only once, and all processed files are saved as individual data objects,
    which _load_data_as_file_paths() retrieves from memory (at location mri_set_path).

    :param mri_set_name: name of the MRI dataset
    :param project_id: Project ID
    :param mri_path_constructor: function to construct single mri paths using arguments sid
    :param sid_list: list of subject ID's
    :param dtype: data type
    :param prune_mode: if not use None; image pruning reduces zero-padding around the brain:
                       "cube": all axes have the same length; "max": maximally prune all axes independently
    :param path_brain_mask: path to brain mask, None if brain mask not available/required
    :param regis_mni: transform to MNI space (1 or 2 mm [int]), or None for no registration
    :param cache_dir: directory to cache files
    :return: data array (MRIs) of shape [n_subjects, x,y,z, channel=1], &
             ordered list of corresponding subject ID's
    """
    # Check arguments
    if isinstance(prune_mode, str):
        prune_mode = prune_mode.lower()
        if prune_mode not in {"cube", "max"}:
            msg = "prune_mode must be 'cube' OR 'max' OR None"
            raise ValueError(msg)

    # Full procedure
    # TODO: parallelize # noqa: FIX002
    invalid_paths = []  # collect unavailable NIfTIs in list
    sids_with_mri = []
    all_mri_paths = None

    # Prepare suffix for cache files
    path_suffix_register = f"mni{regis_mni}mm" if _check_regis(regis_mni) else ""
    path_suffix_numpy = [path_suffix_register] if path_suffix_register else []
    if isinstance(prune_mode, str):
        path_suffix_numpy.append(f"{prune_mode[0]}-pruned")  # "c-pruned" (cube) OR "m-pruned" (max)
    if isinstance(path_brain_mask, str):
        path_suffix_numpy.append("brain_masked")
    path_suffix_numpy = "_".join(path_suffix_numpy)  # == "" if no suffix so far
    path_suffix_numpy = path_suffix_numpy if path_suffix_numpy else "cached_numpy"

    # Prepare metadata table
    source_mri_col = "source_mri"
    path_to_metadata = Path(cache_dir, f"{mri_set_name}_metadata.csv")
    if path_to_metadata.exists():
        metadata_table = pd.read_csv(path_to_metadata, index_col="sid", dtype={"sid": str})
    else:
        metadata_columns = [source_mri_col]
        metadata_columns += [path_suffix_register] if path_suffix_register else []
        metadata_columns += [
            path_suffix_numpy,
            f"{path_suffix_numpy}_min_intensity",  # of processed images
            f"{path_suffix_numpy}_max_intensity",
            f"{path_suffix_numpy}_min_clip",  # of processed images
            f"{path_suffix_numpy}_max_clip",  # this is then the max intensity value for all processed images
        ]
        metadata_table = pd.DataFrame(index=np.array(str(sid) for sid in sid_list), columns=metadata_columns)
        metadata_table.index.name = "sid"
    fill_min_max_cols = [c for c in metadata_table.columns if f"{path_suffix_numpy}_m" in c]

    cprint(string=f"Loading {len(sid_list)} samples of the MRI set for {project_id} ...", col="b")
    for sid_idx, sid in tqdm(enumerate(sid_list), total=len(sid_list)):
        # Load single 3D files to data
        source_mri_path = mri_path_constructor(str(sid))
        if not Path(source_mri_path).is_file():
            invalid_paths.append(source_mri_path)
            if all_mri_paths is not None:
                all_mri_paths = all_mri_paths[:-1, ...]  # remove empty slot
            continue
        metadata_table.loc[sid, source_mri_col] = source_mri_path

        # Get the full path to cache files
        path_cache_registered = _create_cache_files_path(
            f_path=source_mri_path,
            project_id=project_id,
            sid=sid,
            suffix=path_suffix_register,
            ext="nii.gz",
            cache_dir=cache_dir,
        )
        if path_suffix_register:
            metadata_table.loc[sid, path_suffix_register] = path_cache_registered

        path_cache_preprocessed = _create_cache_files_path(
            f_path=source_mri_path,
            project_id=project_id,
            sid=sid,
            suffix="" if path_suffix_numpy == "cached_numpy" else path_suffix_numpy,
            ext="npz",  # numpy arrays will be zipped in cache → *.npz
            cache_dir=cache_dir,
        )
        metadata_table.loc[sid, path_suffix_numpy] = path_cache_preprocessed
        path_cache_preprocessed = Path(path_cache_preprocessed)

        # Check whether processed MRI (as numpy) already there:
        if path_cache_preprocessed.is_file() and not (
            _check_regis(regis_mni) and not Path(path_cache_registered).is_file()
        ):
            # Only load and extract values from volume if data is missing
            if not metadata_table.loc[sid, fill_min_max_cols].isna().any():
                continue
            # Numpy array found and a MNI-registered image does not have to be cached (anymore)
            single_data4d = _load_obj(
                name=path_cache_preprocessed.name,
                folder=path_cache_preprocessed.parent,
                functimer=False,
            )
        else:
            # Process MRI of current participant
            single_data4d = process_single_mri(
                mri_path=source_mri_path,
                dtype=dtype,
                prune_mode=prune_mode,
                path_brain_mask=path_brain_mask,
                regis_mni=regis_mni,
                path_cached_mni=path_cache_registered,
            )
            # Save data object accordingly
            _save_obj(
                obj=single_data4d.squeeze(),
                name=Path(path_cache_preprocessed).name,
                folder=Path(path_cache_preprocessed).parent,
                save_as="npy",
                as_zip=True,
                functimer=False,
            )
            # Note, this data is not compressed nor normed yet.

        # Fill intensity & clip values
        metadata_table.loc[sid, f"{path_suffix_numpy}_min_intensity"] = single_data4d.min()
        metadata_table.loc[sid, f"{path_suffix_numpy}_max_intensity"] = single_data4d.max()
        min_clip, max_clip = determine_min_max_clip(data=single_data4d, at_percentile=CLIP_PERCENTILE)
        metadata_table.loc[sid, f"{path_suffix_numpy}_min_clip"] = min_clip
        metadata_table.loc[sid, f"{path_suffix_numpy}_max_clip"] = max_clip

        # Save current state of metadata table
        path_to_metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata_table.to_csv(path_to_metadata)

        # Gather data ([empty,x,y,z]) in 4d array
        sids_with_mri.append(sid)
        if all_mri_paths is None:
            all_mri_paths = np.empty(shape=(len(sid_list) - len(invalid_paths),), dtype=np.object_)  # init data set
        all_mri_paths[sid_idx - len(invalid_paths)] = path_cache_preprocessed

    if len(invalid_paths) > 0:
        cprint(string="Participants with non-existent data:\n" + "\n".join(f"\t{i}" for i in invalid_paths), col="y")

    return all_mri_paths, np.array(sids_with_mri)


def get_mri_set_name(
    project_id: str,
    mri_seq: str,
    regis_mni: int | None,
    brain_masked: bool,
    norm: bool,
    prune_mode: str | None,
) -> str:
    """
    Construct a name for the MRI set which is/will be saved as `*.pkl` object.

    The full name describes different pre-processing steps.

    :param project_id: name of the project containing the data set, e.g., lemon, hcp, or other projects
    :param mri_seq: MRI sequence
    :param regis_mni: registered to MNI space in 1 or 2 mm resolution [int], or None for no registration
    :param brain_masked: brain mask has been applied
    :param norm: if data is normalized
    :param prune_mode: if data is pruned: None OR "cube" OR "max"
    :return: final name of MRI set
    """
    return (
        f"{project_id}_"
        f"{mri_seq}"
        f'{f"-mni{regis_mni}mm" if _check_regis(regis_mni) else ""}'
        f'{"-bm" if brain_masked else ""}'
        f'{"-n" if norm else ""}'
        f'{"-p" + f"{prune_mode[0]}" if isinstance(prune_mode, str) else ""}'
    )


def get_mri_set_path(
    mri_set_name: str,
    path_to_folder: str | Path | None = None,
    as_npy: bool = True,
    as_zip: bool = False,
) -> str:
    """
    Get the absolute path to the MRI set.

    :param mri_set_name: Name of MRI set (constructed by `get_mri_set_name()`).
    :param path_to_folder: The path where the MRI set is supposed to be located.
    :param as_npy: True: Save as a numpy (`*.npy`) else as a pickle (`*.pkl`) object
    :param as_zip: zipped file (`*.pkl.gz`; `*.npz`)
    :return: absolute path to the MRI set
    """
    suffix = (".npz" if as_zip else ".npy") if as_npy else ".pkl.gz" if as_zip else ".pkl"
    if path_to_folder is None:  # Default: look in cache dir
        mri_set_path = Path(CACHE_DIR, mri_set_name).with_suffix(suffix)
    elif isinstance(path_to_folder, (str, Path)):
        if not Path(path_to_folder).is_dir():
            cprint(string=f"Note: the folder '{path_to_folder}' does not exist.", col="y")
        mri_set_path = Path(path_to_folder, mri_set_name).with_suffix(suffix)
    else:
        raise ValueError("path_to_folder must be path to folder [str|Path] or None.")
    return str(mri_set_path)


def _save_mri_set(
    data: np.ndarray,
    set_path: str | Path,
    max_size: int = MAXIMUM_SIZE_BYTES,
    as_npy: bool = True,
    as_zip: bool = False,
    verbose: bool = True,
    **kwargs,
) -> str:
    """
    Save the MRI dataset externally as a pickle-object.

    Check if the size is not so large to save time when loading the data later.

    Timing test (%timeit; with 999 MRIs, total shape (999, 98, 98, 98, 1)) provided loading times (in s):
         1. .npy    : 1.44 s (3.8 GB)
         2. .pkl    : 4.33 s (3.8 GB)
         3. .npz    : 11.8 s (0.85 GB) → if memory size is an issue numpy-zip should be chosen
         4. .pkl.gz : 15.5 s (0.85 GB)

    Note, very large MRI datasets (n > 3,000) should be saved and loaded as individual files
    (see load_mode='file_paths').

    :param data: numpy array with 3D-brain images of all subjects (5D) (n_subjects, x, y, z, 1)
    :param set_path: path to the data set, name including specification of preprocessing
    :param max_size: maximum size in bytes of data set that is allowed to be saved
    :param as_npy: is saved as a numpy array (otherwise pickle)
    :param as_zip: True: zip the pickle
    :param verbose: verbose or not
    :return: output path, i.e., location of saved data set
    """
    if check_storage_size(data, verbose=verbose) > max_size:
        cprint(
            string="Object is too large and cannot be saved.\n"
            "\tCompress file, or change MAXIMUM_SIZE_BYTES, or use load_mode=='file_paths'.",
            col="y",
        )
        set_path = None
    else:
        set_path = str(set_path)
        if not (
            set_path.endswith(".pkl")
            or set_path.endswith(".pkl.gz")
            or set_path.endswith(".npy")
            or set_path.endswith(".npz")
        ):
            ext = ".npy" if as_npy else ".pkl"
            set_path += ext

        if verbose:
            cprint(string=f"\nSaving the MRI set to {set_path} ...", col="y")
        _save_obj(
            obj=data,
            name=Path(set_path).name,
            folder=Path(set_path).parent,
            as_zip=as_zip,
            **kwargs,
        )

    return set_path


def _get_unpruned_mri(
    sid: str,
    project_id: str,
    mri_path_constructor: Callable[[str], str],
    regis_mni: int | None,
    cache_dir: str | Path,
) -> nib.Nifti1Image:
    """Get the processed MRI data for a given subject ID, before pruning."""
    mri_path = mri_path_constructor(sid=sid)

    if _check_regis(regis_mni):
        mri_path = _create_cache_files_path(
            f_path=mri_path,
            project_id=project_id,
            sid=sid,
            suffix=f"mni{regis_mni}mm",
            ext="nii.gz",
            cache_dir=cache_dir,
        )

    return get_nifti(mri_path=mri_path, reorient=True)


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
