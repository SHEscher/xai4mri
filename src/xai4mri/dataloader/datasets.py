"""
Use `BaseDataSet` for creating a project-specific dataset class.

    Authors: Simon M. Hofmann | Hannah S. Heinrichs
    Years: 2021-2023
"""

# %% Import
from __future__ import annotations  # to make package compatible with Python 3.9

import random
import shutil
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tensorflow import keras

from ..utils import (
    _load_obj_from_abs_path,
    bytes_to_rep_string,
    chop_microseconds,
    compute_array_size,
    cprint,
)
from .mri_dataloader import (
    CACHE_DIR,
    _get_unpruned_mri,
    _load_data_as_file_paths,
    _load_data_as_full_array,
    compress_and_norm,
    get_metadata_path,
    get_nifti,
)

if TYPE_CHECKING:
    import nibabel as nib


# %% Dataset classes  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _extract_values_from_metadata(table: pd.DataFrame, compress: bool, norm: bool) -> tuple:
    """
    Extract values from metadata table.

    :param table: metadata table
    :param compress: compress data
    :param norm: normalize data
    :return: clip_min, clip_max, global_norm_min, global_norm_max, norm
    """
    # Extract arguments from metadata table
    min_clip_col = [c for c in table.columns if "min_clip" in c]
    max_clip_col = [c for c in table.columns if "max_clip" in c]
    clip_min = np.nanmean(table[min_clip_col]) if compress else None
    clip_max = np.nanmean(table[max_clip_col]) if compress else None
    min_intensity_col = [c for c in table.columns if "min_intensity" in c]
    negative_values = np.nanmin(table[min_intensity_col]) < 0
    global_norm_min = None
    global_norm_max = None
    if negative_values:
        # For images with negative values (e.g., statistical maps) no normalization in 0-255 is done
        norm = False
    if norm and compress:
        # When data is clipped, use the clipped values as global min/max for the normalization
        global_norm_min = clip_min
        global_norm_max = clip_max
    if norm and not compress:
        max_intensity_col = [c for c in table.columns if "max_intensity" in c]
        global_norm_min = np.nanmin(table[min_intensity_col])
        global_norm_max = np.nanmax(table[max_intensity_col])

    return clip_min, clip_max, global_norm_min, global_norm_max, norm


class BaseDataSet(ABC):
    """
    Base class for an MRI dataset.

    In the case of a research project with multiple MRI sequences, each sequence must have its own dataset class that
    inherits from the BaseDataSet class.
    """

    def __init__(
        self,
        study_table_or_path: pd.DataFrame | str | Path,
        project_id: str,
        mri_sequence: str,
        register_to_mni: int | None = None,
        cache_dir: str | Path = CACHE_DIR,
        load_mode: str = "full_array",
        **kwargs,
    ):
        """
        Initialize BaseDataSet.

        !!! example "Usage"
            ```python
            # Create a study-specific dataset class
            class MyStudyData(BaseDataSet):
                def __init__(self):
                    super().__init__(
                        study_table_or_path="PATH/TO/STUDY_TABLE.csv",  # one column must be 'sid' (subject ID)
                        project_id="MyProjectID",
                        mri_sequence="t1w",  # this is of descriptive nature, for projects with multiple MRI sequences
                        load_mode="full_array",  # or 'file_paths' for very large datasets
                    )

                # Define mri_path_constructor
                def mri_path_constructor(sid: str) -> str | Path:
                    return f"/path/to/mri/{sid}.nii.gz"


            # Instantiate the dataset class
            my_study_data = MyStudyData()
            ```

        :param study_table_or_path: The study table, OR the absolute or relative path to the table [`*.csv` | `*.tsv`].
                                    The table must have 'sid' as the index column, containing subject IDs.
        :param project_id: The project ID.
        :param mri_sequence: MRI sequence ('t1_mni_1mm', 't2', 'dwi', or similar).
                             This is of a descriptive nature for projects with multiple MRI sequences,
                             hence multiple offsprings of `BaseDataSet`.
        :param register_to_mni: Register MRIs to the MNI space (1 mm, 2 mm)
                                using [`ANTs`](https://antspyx.readthedocs.io/en/latest/),
                                OR not [`None`].
        :param cache_dir: Path to the cache directory, where intermediate and processed data is stored.
        :param load_mode: Load mode for the dataset:
                          'file_paths': Load the MRI data from file paths (recommended for very large datasets).
                          'full_array': Load the MRI data as a full array (default).
        :param kwargs: Additional keyword arguments for MRI processing.
                       Find details to `kwargs` in docs of
                       `xai4mri.dataloader.mri_dataloader._load_data_as_full_array()`
                       or `_load_data_as_file_paths()`.
        """
        self._study_table = None  # init
        self._study_table_path: str | Path | None = None  # init

        self.study_table = study_table_or_path
        self.project_id = project_id
        self.mri_sequence = mri_sequence
        self.cache_dir = cache_dir
        self.load_mode = load_mode

        self._sid_list = None  # init
        self._split_dict: dict[str, np.ndarray] | None = None  # init

        # Following variables should ideally remain untouched after dataset class is instantiated.
        # They get passed via self.get_data to .mri_dataloader._load_data_as_full_array() (find kwargs details there)
        self._regis_mni: int | None = register_to_mni
        self._norm: bool = kwargs.pop("norm", True)
        self._path_brain_mask: str | None = kwargs.pop("path_brain_mask", None)
        self._compress: bool = kwargs.pop("compress", True)
        self._prune_mode: str | None = kwargs.pop("prune_mode", "max")  # None: no pruning, OR "max" or "cube"
        self._path_to_dataset: str | None = kwargs.pop("path_to_dataset", None)  # if set not in cache dir
        self._cache_files: bool = kwargs.pop("cache_files", True)
        self._save_after_processing: bool = kwargs.pop("save_after_processing", self.load_mode != "file_paths")
        # **save_kwargs # as_npy, as_zip in get_mri_set_path() &

        # Run check
        if not kwargs.pop("ignore_checks", False):
            self._check_mri_path_constructor()

        # Print unknown kwargs
        if kwargs:
            cprint(
                f"At init of the DataSet class, unknown kwargs were passed: {kwargs}, which will be ignored!", col="y"
            )

    def _check_mri_path_constructor(self):
        """Check if the MRI path constructor is implemented correctly."""
        # Pick a random subject ID
        sid = random.choice(self.sid_list)  # noqa: S311
        path_to_mri = Path(self.mri_path_constructor(sid=sid))
        if not path_to_mri.exists():
            msg = (
                f"Checking mri_path_constructor(sid), with random sid '{sid}'. '{path_to_mri}' does not exist!\n"
                f"Please check your mri_path_constructor() implementation."
            )
            raise FileNotFoundError(msg)

    @staticmethod
    @abstractmethod
    def mri_path_constructor(sid: str) -> str | Path:
        """
        Construct the path to the original MRI file of a subject given its ID (sid).

        Define this function in the dataset class which inherits from BaseDataSet.

        :param sid: subject ID
        :return: path to the original MRI file of the subject
        """
        pass

    @property
    def study_table_path(self) -> str | None:
        """
        Return the path to the study table if it has been provided.

        Can be provided as `study_table_or_path` at initialization, or it can be set later.

        :return: path to the study table
        :rtype: str | None
        """
        return self._study_table_path

    @study_table_path.setter
    def study_table_path(self, study_table_path: str | Path) -> None:
        """Set the path to the study table after running checks on the given path."""
        # Check input
        if not isinstance(study_table_path, (str, Path)):
            msg = f"study_table_path must be a str or Path, but is of type {type(study_table_path)}"
            raise TypeError(msg)

        # Check if the path suffix is valid
        if not str(study_table_path).endswith((".csv", ".tsv")):
            msg = f"Study table '{study_table_path}' must be either *.csv or *.tsv!"
            raise ValueError(msg)

        study_table_path = Path(study_table_path).resolve()  # turn it into an absolute path
        if not study_table_path.exists():
            msg = f"Study table '{study_table_path}' does not exist!"
            raise FileNotFoundError(msg)

        # Set the path to the study table
        self._study_table_path = str(study_table_path)

        # Load study table
        if self.study_table_path.endswith(".csv"):
            self.study_table = pd.read_csv(self.study_table_path, index_col="sid", dtype={"sid": str})
        elif self.study_table_path.endswith(".tsv"):
            self.study_table = pd.read_csv(self.study_table_path, sep="\t", index_col="sid", dtype={"sid": str})

    @property
    def study_table(self) -> pd.DataFrame:
        """
        Get the study table.

        Ideally, each BaseDataSet has its own study table, except if for all participants of a research
        project all MRI sequences are available.
        In this case, the study table can be the same for all MRI sequences and derivatives.

        :return: study table
        :rtype: pd.DataFrame
        """
        return self._study_table

    @study_table.setter
    def study_table(self, study_table_or_path: pd.DataFrame | str | Path) -> None:
        """Set study table and run checks."""
        if isinstance(study_table_or_path, (str, Path)):
            # Run setter including checks for the path and load the study table from the file
            self.study_table_path = study_table_or_path
            return

        if not isinstance(study_table_or_path, pd.DataFrame):
            msg = (
                f"study_table must be a valid table [pandas.DataFrame], or a path to a table file [str|Path], "
                f"but is of type {type(study_table_or_path)}"
            )
            raise TypeError(msg)

        study_table = study_table_or_path  # it is a DataFrame
        if study_table.index.name != "sid":
            msg = "The study table must have 'sid' (subject IDs) as its index column!"
            raise ValueError(msg)

        if study_table.index.dtype not in {str, np.dtype("object")}:
            msg = "The index 'sid' (subject IDs) of the study table must be of dtype str (i.e., dtype('O') in pandas)!"
            raise TypeError(msg)

        if not study_table.index.is_unique:
            msg = "The study table must have unique subject IDs!"
            raise ValueError(msg)

        self._study_table = study_table

    @property
    def load_mode(self) -> str:
        """
        Return the load mode for the dataset.

        The load mode can be either 'file_paths' or 'full_array'.

        - 'file_paths': Load the MRI data from file paths.
                        That is, individual files are stored separately.
        - 'full_array': Load the MRI data as a full array. Data is saved as a single large file.
        """
        return self._load_mode

    @load_mode.setter
    def load_mode(self, load_mode: str) -> None:
        """Set the load mode."""
        valid_load_modes = {"file_paths", "full_array"}
        load_mode = load_mode.lower()
        if load_mode not in valid_load_modes:
            msg = f"load_mode must be in: {valid_load_modes}."
            raise ValueError(msg)
        self._load_mode = load_mode
        if load_mode == "file_paths":
            # For file_paths, all individual files are saved after processing,
            # and if self._save_after_processing is True, it will reprocess the data during self.get_data().
            # Hence, only pass save_after_processing=True to self.get_data() if reprocessing is desired.
            self._save_after_processing = False

    @property
    def sid_list(self) -> np.ndarray[str]:
        """
        Return the list of subject IDs.

        :return: list of subject IDs
        :rtype: np.ndarray[str]
        """
        if self._sid_list is None:
            self._sid_list = self.study_table.index.astype(str).to_numpy()
        return self._sid_list

    @sid_list.setter
    def sid_list(self, sid_list: list[str] | np.ndarray[str]) -> None:
        """Set the list of subject IDs and run checks."""
        if not isinstance(sid_list, (list, np.ndarray, tuple)):
            msg = f"sid_list must be a list or np.ndarray, but is of type {type(sid_list)}"
            raise TypeError(msg)
        if not all(isinstance(sid, str) for sid in sid_list):
            msg = "sid_list must only contain strings!"
            raise TypeError(msg)
        if not all(sid in self.study_table.index for sid in sid_list):
            # sid_list must be a subset of the study table (index)
            msg = "sid_list must only contain subject IDs that are in the study table!"
            raise ValueError(msg)
        self._sid_list = np.array(sid_list)

    def get_data(self, **kwargs) -> tuple[np.ndarray, np.ndarray[str]]:
        """
        Load dataset into workspace.

        :param kwargs: Additional keyword arguments for MRI processing.
                       Find details to `kwargs` in docs of
                       `xai4mri.dataloader.mri_dataloader._load_data_as_full_array()`
                        or `_load_data_as_file_paths()`.
                        Note, these `kwargs` should only be used
                        if the deviation from the `__init__`-`kwargs` is intended.
        :return: Processed MRI data: either 5D data array of shape `[n_subjects, x,y,z, channel=1]`
                           for `self.load_mode='full_array'`,
                           or 1D array (paths to processed MRIs) for `self.load_mode='file_paths'`,
                 and ordered list of corresponding subject ID's
        """
        # Unpack kwargs and update class attributes accordingly
        self._norm = kwargs.pop("norm", self._norm)
        self._path_brain_mask = kwargs.pop("path_brain_mask", self._path_brain_mask)
        self._compress = kwargs.pop("compress", self._compress)
        self._regis_mni = kwargs.pop("regis_mni", self._regis_mni)
        self._prune_mode = kwargs.pop("prune_mode", self._prune_mode)
        self._path_to_dataset = kwargs.pop("path_to_dataset", self._path_to_dataset)
        self._cache_files = kwargs.pop("cache_files", self._cache_files)
        self._save_after_processing = kwargs.pop("save_after_processing", self._save_after_processing)

        # Load data
        load_volume_data = _load_data_as_file_paths if self.load_mode == "file_paths" else _load_data_as_full_array

        volume_data, sid_list = load_volume_data(
            sid_list=self.sid_list,
            project_id=self.project_id,
            mri_path_constructor=self.mri_path_constructor,
            mri_seq=self.mri_sequence,
            cache_dir=self.cache_dir,
            norm=self._norm,
            path_brain_mask=self._path_brain_mask,
            compress=self._compress,
            regis_mni=self._regis_mni,
            prune_mode=self._prune_mode,
            path_to_dataset=self._path_to_dataset,
            cache_files=self._cache_files,
            save_after_processing=self._save_after_processing,
            **kwargs,  # == **save_kwargs remain, see _load_data_as_full_array() for details
        )

        if self.load_mode == "file_paths" and self._save_after_processing:
            # Reset to default for load_mode == "file_paths"
            self._save_after_processing = False

        sids_without_data = set(self.sid_list) - set(sid_list)
        if sids_without_data:
            msg = (
                f"No MRI data found for following subjects (ID's): {sids_without_data}.\n"
                f"Please locate data or remove SIDs from study table to proceed!"
            )
            raise ValueError(msg)

        return volume_data, sid_list

    def get_metadata_table(self):
        """Get the metadata table for the MRIs of the project dataset."""
        path_to_metadata = get_metadata_path(
            project_id=self.project_id,
            mri_seq=self.mri_sequence,
            regis_mni=self._regis_mni,
            path_brain_mask=self._path_brain_mask,
            norm=self._norm,
            prune_mode=self._prune_mode,
            path_to_dataset=self._path_to_dataset
            if isinstance(self._path_to_dataset, (str, Path))
            else self.cache_dir,
        )

        if Path(path_to_metadata).is_file():
            return pd.read_csv(path_to_metadata, index_col="sid", dtype={"sid": str})
        msg = f"Metadata table not found at '{path_to_metadata}'.\nRun data processing to create metadata table."
        raise FileNotFoundError(msg)

    def get_size_of_prospective_mri_set(
        self,
        estimate_with_n: int = 3,
        estimate_processing_time: bool = True,
        **process_mri_kwargs,
    ) -> None:
        """
        Estimate the prospective storage sized, which is necessary to save the pre-processed project data.

        Additionally, estimate the time needed to process the entire dataset.

        :param estimate_with_n: use n samples to approximate the size of the whole processed dataset.
                                If approx_with >= N, the entire dataset will be taken.
        :param estimate_processing_time: estimate the time needed to process the entire dataset.
        :param process_mri_kwargs: `kwargs` for `process_single_mri()`, e.g., `prune_mode`.
                                   These should overlap with or be equal to the `kwargs` for `self.get_data()`.
                                   Note, these `kwargs` should only be used
                                   if the deviation from the `__init__`-`kwargs` is intended.
        """
        max_n = 10
        if estimate_with_n > max_n:
            cprint(
                string=f"estimate_with_n is set down to {max_n}. "
                f"Too many samples make the calculation unnecessarily slow.",
                col="y",
            )
        estimate_with_n = np.clip(estimate_with_n, a_min=1, a_max=max_n)

        # Prep temporary sid_list
        full_sid_list = self.study_table.index.to_list()
        np.random.shuffle(full_sid_list)  # noqa: NPY002

        def print_estimated_size(size_in_bytes: int, cached_single_files: bool) -> None:
            """Print the estimated size of the pre-processed data."""
            cprint(
                string=f"\nEstimated size of all pre-processed {self.project_id.upper()} "
                f"data{' (mri-set + cached single files)' if cached_single_files else ''}: "
                f"{bytes_to_rep_string(size_in_bytes)}",
                col="b",
            )

        # Extract kwargs
        cache_files = process_mri_kwargs.pop("cache_files", self._cache_files)  # bool
        if not (cache_files or estimate_processing_time or process_mri_kwargs.get("register_to_mni", self._regis_mni)):
            # This is the fastest way to estimate the size of the processed data.
            # However, it only works if single files are not cached and no processing time is to be estimated
            from .prune_image import PruneConfig

            # Get shape
            temp_mri = get_nifti(self.mri_path_constructor(sid=full_sid_list[0]), reorient=True)
            resolution = np.round(temp_mri.header["pixdim"][1:4], decimals=3)  # image resolution per axis
            if process_mri_kwargs.get("prune_mode", self._prune_mode) is None:
                mri_shape = temp_mri.shape
            else:
                # If self._prune_mode == "max" or "cube"
                mri_shape = np.round(PruneConfig.largest_brain_max_axes // resolution).astype(int)  # "max"
                if process_mri_kwargs.get("prune_mode", self._prune_mode) == "cube":
                    mri_shape = (int(mri_shape.max()),) * 3

            # Get dtype
            dtype = (
                np.uint8
                if process_mri_kwargs.get("compress", self._compress) and process_mri_kwargs.get("norm", self._norm)
                else temp_mri.get_fdata().dtype
            )
            # Compute size
            total_bytes = compute_array_size(shape=(len(full_sid_list), *mri_shape), dtype=dtype, verbose=False)
            print_estimated_size(size_in_bytes=total_bytes, cached_single_files=cache_files)

        else:
            # Prepare the path to the temporary cache directory
            cache_dir_parent = Path(process_mri_kwargs.pop("cache_dir", self.cache_dir))
            cache_dir_existed = cache_dir_parent.exists()  # init
            if not cache_dir_existed:
                cache_dir_parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=cache_dir_parent) as temp_cache_dir:
                try:
                    start_time = perf_counter()
                    # Prepare and load temp data
                    load_volume_data = (
                        _load_data_as_file_paths if self.load_mode == "file_paths" else _load_data_as_full_array
                    )
                    _, _ = load_volume_data(  # _, _ == volume_data, sid_list
                        sid_list=full_sid_list[:estimate_with_n],
                        project_id=self.project_id,
                        mri_path_constructor=self.mri_path_constructor,
                        mri_seq=self.mri_sequence,
                        cache_dir=temp_cache_dir,
                        norm=process_mri_kwargs.pop("norm", self._norm),
                        path_brain_mask=process_mri_kwargs.pop("path_brain_mask", self._path_brain_mask),
                        compress=process_mri_kwargs.pop("compress", self._compress),
                        regis_mni=process_mri_kwargs.pop("regis_mni", self._regis_mni),
                        prune_mode=process_mri_kwargs.pop("prune_mode", self._prune_mode),
                        path_to_dataset=None,  # must be temp cache dir
                        cache_files=cache_files,
                        save_after_processing=process_mri_kwargs.pop(
                            "save_after_processing", self._save_after_processing
                        ),
                        force=True,  # force processing (ignores ask_true_false in _load_data_as_full_array())
                        **process_mri_kwargs,  # <- dtype, verbose, path_cached_mni
                    )

                    # Take average time (in seconds) per sample and multiply with total population size
                    total_time = perf_counter() - start_time  # in seconds
                    time_per_sample = total_time / estimate_with_n
                    total_time = time_per_sample * len(full_sid_list)  # time for all
                    total_time = timedelta(seconds=total_time)  # convert seconds into timedelta object

                    # Take the average size per sample and multiply with the total population size
                    total_bytes = sum(f.stat().st_size for f in Path(temp_cache_dir).glob("**/*") if f.is_file())
                    total_bytes *= len(full_sid_list) / estimate_with_n

                    data_suffix = ""
                    if self.load_mode == "full_array" and cache_files:
                        data_suffix = "(mri-set + cached single files)"
                    elif self.load_mode == "file_paths" and cache_files:
                        data_suffix = " (cached single files)"
                    cprint(
                        string=f"\nEstimated size of all pre-processed {self.project_id.upper()} "
                        f"data{data_suffix}: {bytes_to_rep_string(total_bytes)}",
                        col="b",
                    )
                    cprint(
                        string=f"Estimated time to process all data: {chop_microseconds(total_time)} [hh:mm:ss]\n",
                        col="b",
                    )
                except Exception as e:  # noqa: BLE001
                    # Catch any Exception here, such that temp data will be deleted afterward.
                    cprint(str(e), col="r")

                if not cache_dir_existed:
                    sleep(0.5)
                    shutil.rmtree(cache_dir_parent)

    def create_data_split(
        self,
        target: str,
        batch_size: int = 1,
        split_ratio: tuple[float, float, float] | None = (0.8, 0.1, 0.1),
        split_dict: dict[str, str] | None = None,
        **get_data_kwargs,
    ) -> tuple[dict[str, np.ndarray], _DataSetGenerator, _DataSetGenerator, _DataSetGenerator]:
        """
        Create data split with a training, validation, and test set.

        The data subsets are provided as generator objects, and can be used for model training and evaluation.

        !!! example "Usage"
            ```python
            # Create a data split for model training and evaluation
            split_dict, train_gen, val_gen, test_gen = mydata.create_data_split(target="age")

            # Train a model
            model.fit(train_gen, validation_data=val_gen, ...)
            ```

        :param target: Prediction target.
                       `target` must match a column in the study table.
        :param batch_size: Batch size in the returned data generators per data split.
                           MRIs are arther large files; hence, it is recommended to keep batches rather small.
        :param split_ratio: Ratio of the data split (train, validation, test).
                            Must add up to 1.
        :param split_dict: Dictionary with 'train', 'validation', & 'test' as keys, and subject IDs as values.
                           If a `split_dict` is provided, it overrules `split_ratio`.
                           Providing `split_dict` is useful when specific subject data shall be used in a split.
        :return: split_dict, and the data generators for the training, validation, and test set
        """
        # Load subject data
        volume_data, sid_list = self.get_data(mmap_mode="r", **get_data_kwargs)  # 'r' does not load data to RAM
        sid_list = np.array(sid_list)

        if split_dict is None:
            if not (split_ratio is not None and round(sum(split_ratio), 3) == 1):
                msg = "Either split_dict or split_ratio must be provided. split_ratio must sum to 1."
                raise ValueError(msg)

            # Create split indices
            indices = list(range(len(sid_list)))
            random.shuffle(indices)
            n_train = round(len(sid_list) * split_ratio[0])
            train_indices = indices[:n_train]
            n_val = round(len(sid_list) * split_ratio[1])
            val_indices = indices[n_train : n_train + n_val]
            n_test = len(sid_list) - n_train - n_val
            test_indices = indices[-n_test:]

            split_dict = {
                "train": sid_list[train_indices],
                "validation": sid_list[val_indices],
                "test": sid_list[test_indices],
            }

        else:
            all_sids = [item for sublist in split_dict.values() for item in sublist]
            if not set(all_sids).issubset(set(sid_list)):
                msg = "All SID's in split_dict must be part of the dataset!"
                raise ValueError(msg)
            if len(set(all_sids)) != len(all_sids):
                msg = "SID's must only appear once in the split!"
                raise ValueError(msg)
            if split_dict.keys() != {"train", "validation", "test"}:
                msg = "split_dict must have keys 'train', 'validation', 'test'!"
                raise ValueError(msg)

            # Create split indices
            train_indices = [list(sid_list).index(sid) for sid in split_dict["train"]]
            val_indices = [list(sid_list).index(sid) for sid in split_dict["validation"]]
            test_indices = [list(sid_list).index(sid) for sid in split_dict["test"]]

        # Prepare target (y) data
        if target not in self.study_table.columns:
            msg = "target variable must be in study table!"
            raise ValueError(msg)
        ydata = self.study_table[target]

        # Save split dict
        self._split_dict = split_dict

        # Define preprocessor
        if self.load_mode == "full_array":
            # Data is preprocessed already in the full array mode
            preprocessor = None
        else:
            # Set arguments for compress_and_norm() function
            metadata_table = self.get_metadata_table()

            clip_min, clip_max, global_norm_min, global_norm_max, self._norm = _extract_values_from_metadata(
                table=metadata_table,
                compress=self._compress,
                norm=self._norm,
            )

            # Set preprocessor
            preprocessor = partial(
                compress_and_norm,
                clip_min=clip_min,
                clip_max=clip_max,
                norm=self._norm,
                global_norm_min=global_norm_min,
                global_norm_max=global_norm_max,
            )

        return (
            split_dict,
            _DataSetGeneratorFactory.create_generator(
                name="train",  # training set
                x_data=volume_data[train_indices],
                y_data=ydata.loc[sid_list[train_indices]].to_numpy(),
                batch_size=batch_size,
                data_indices=train_indices,
                preprocess=preprocessor,
            ),
            _DataSetGeneratorFactory.create_generator(
                name="validation",  # validation set
                x_data=volume_data[val_indices],
                y_data=ydata.loc[sid_list[val_indices]].to_numpy(),
                batch_size=batch_size,
                data_indices=val_indices,
                preprocess=preprocessor,
            ),
            _DataSetGeneratorFactory.create_generator(
                name="test",  # test set
                x_data=volume_data[test_indices],
                y_data=ydata.loc[sid_list[test_indices]].to_numpy(),
                batch_size=1,
                data_indices=test_indices,
                preprocess=preprocessor,
            ),
        )

    @property
    def current_split_dict(self) -> dict[str, np.ndarray[str]] | None:
        """
        Return the current split dictionary.

        The split dictionary is created by calling `create_data_split()`, and has the following structure:

            {'train': np.ndarray[str], 'validation': np.ndarray[str], 'test': np.ndarray[str]}

        :return: split dictionary
        """
        return self._split_dict

    def save_split_dict(
        self, split_dict: dict[str, np.ndarray[str]] | None = None, save_path: str | Path | None = None
    ) -> str:
        """
        Save a split dictionary to a file.

        If no split dictionary is given, the `self.current_split_dict` is saved.
        `self.current_split_dict` is set after calling `self.create_data_split()`.

        :param split_dict: data split dictionary: {'train': ['sub-42', ...], 'validation': [...], 'test': [...]}
        :param save_path: path to file
        :return: the path to the file
        """
        # Use the default path if no path is given
        if save_path is None:
            save_path = Path(
                self.cache_dir,
                "data_splits",
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.project_id}_{self.mri_sequence}_"
                f"split_dict.npy",
            )
        else:
            save_path = Path(save_path)

        # Use current data split dict if no split dict is given
        split_dict = self._split_dict if split_dict is None else split_dict

        if split_dict is None:
            raise ValueError("No split dictionary provided!")

        # Create directory if it does not exist
        save_path.parent.mkdir(exist_ok=True, parents=True)

        # Save split dictionary
        np.save(save_path, split_dict, allow_pickle=True)
        print(f"Saved split dictionary to {save_path}.")

        return str(save_path)

    @staticmethod
    def load_split_dict(split_dict_path: str | Path) -> dict[str, np.ndarray[str]]:
        """
        Load the split dictionary from the given file path.

        :param split_dict_path: path to split dictionary file
        :return: split dictionary
        """
        # Add .npy-extension if it is not provided
        split_dict_path = Path(split_dict_path).with_suffix(".npy")
        split_dict = np.load(split_dict_path, allow_pickle=True).item()
        if not isinstance(split_dict, dict):
            msg = f"split_dict should be a dict, but is of type {type(split_dict)}"
            raise TypeError(msg)
        return split_dict

    def get_unpruned_mri(self, sid: str) -> nib.Nifti1Image:
        """
        Get the processed MRI data for a given subject ID, in the state before the image is pruned.

        :param sid: subject ID
        :return: Non-pruned MRI data as a NIfTI image
        """
        if isinstance(self._regis_mni, int) and not self._cache_files:
            msg = "MNI NIfTI files were not cached during processing!"
            raise ValueError(msg)

        return _get_unpruned_mri(
            sid=sid,
            project_id=self.project_id,
            mri_path_constructor=self.mri_path_constructor,
            regis_mni=self._regis_mni,
            cache_dir=self.cache_dir,
        )


class _DataSetGenerator(keras.utils.Sequence, ABC):
    """
    Abstract class for data set generators.

    !!! example "Usage"
        ```python
        # Generators
        training_generator = _DataSetGenerator(partition["train"], labels, **params)
        validation_generator = _DataSetGenerator(partition["validation"], labels, **params)

        # Train model on dataset
        model.fit(training_generator, validation_data=validation_generator, **kwargs)
        ```
    """

    @abstractmethod
    def __init__(
        self,
        name: str,
        x_data: np.ndarray,
        y_data: np.ndarray,
        batch_size: int,
        data_indices: list[int],
        preprocess: callable | None = None,
    ):
        """
        Initialize _DataSetGenerator.

        :param name: name of the data set
        :param x_data: input data to the model
        :param y_data: target data for the model to predict
        :param batch_size: batch size
        :param data_indices: indices of the whole data set, which are present in this `_DataSetGenerator`
        """
        self.name = name
        self.batch_size = batch_size
        self.x = x_data
        self.y = y_data
        self._data_indices = data_indices
        self._current_indices = None
        self.preprocess = preprocess

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        pass  # also required by keras.utils.Sequence

    def __repr__(self):
        """Return string representation of the object."""
        return f"{self.__class__.__name__}('{self.name}', n_samples={self.n_samples}, batch_size={self.batch_size})"

    def __str__(self):
        """Return string representation of the object."""
        return self.__repr__()

    def __len__(self):
        """Denote the number of batches per epoch."""
        return int(np.ceil(len(self.y) / self.batch_size))

    @property
    def n_samples(self):
        """Return number of samples in the data set."""
        return len(self.y)

    @property
    def data_indices(self):
        """Return indices of the whole data set."""
        return self._data_indices

    @data_indices.setter
    def data_indices(self, data_indices: np.ndarray[int] | list[int] | tuple[int]) -> None:
        """Set indices of the whole data set."""
        if not isinstance(data_indices, (np.ndarray, list, tuple)):
            msg = f"data_indices must be a np.ndarray, list, or tuple, but is of type {type(data_indices)}"
            raise TypeError(msg)
        # Check if all elements are integers
        if not all(isinstance(i, (int, np.int64)) for i in data_indices):
            msg = "data_indices must only contain integers!"
            raise TypeError(msg)
        self._data_indices = np.array(data_indices)

    @property
    def current_indices(self):
        """Return indices of the current batch."""
        return self._current_indices

    @property
    def preprocess(self) -> callable[..., np.ndarray[np.float32, np.int_]]:
        """Preprocess input data."""
        return self._preprocess

    @preprocess.setter
    def preprocess(self, preprocess: callable | None) -> None:
        """Set preprocess function."""
        if preprocess is None:
            self._preprocess = lambda x: x
            return

        if not callable(preprocess):
            msg = "preprocess must be a callable function!"
            raise TypeError(msg)
        self._preprocess = preprocess


class _DataSetGeneratorFromFullArray(_DataSetGenerator):
    """
    Generates data for keras models.

    This gets the x_data and y_data as full arrays, holding the actual data.
    The generator will directly slice the data into batches.
    This is useful when the data is small enough to fit into memory.
    """

    def __init__(
        self,
        name: str | None,
        x_data: np.ndarray,
        y_data: np.ndarray | list | tuple,
        batch_size: int,
        data_indices: np.ndarray[int] | list[int] | tuple[int],
        preprocess: callable | None = None,
    ):
        """
        Initialize _DataSetGeneratorFromFullArray.

        :param x_data: model input data, holding the actual data
        :param y_data: model target data
        :param batch_size: batch size
        :param data_indices: indices of the whole data set, which are present in this _DataSetGenerator
        """
        self.name = name
        self.batch_size = batch_size
        self.x = x_data
        self.y = y_data
        self._data_indices = data_indices
        self._current_indices = None
        self.preprocess = preprocess

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        self._current_indices = self._data_indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        return batch_x, batch_y


class _DataSetGeneratorFromFilePaths(_DataSetGenerator):
    """
    Generates data for keras models.

    This gets the x_data (and y_data) as arrays, holding the paths to the individual data files.
    The generator will first load subsets of the data and then provide them as batches.
    This is useful when the data is too large to fit in memory.
    """

    def __init__(
        self,
        name: str | None,
        x_data: np.ndarray[str | Path] | list[str | Path] | tuple[str | Path],
        y_data: np.ndarray | list | tuple,
        batch_size: int,
        data_indices: np.ndarray[int] | list[int] | tuple[int],
        preprocess: callable | None = None,
    ):
        """
        Initialize _DataSetGeneratorFromFilePaths.

        :param x_data: model input data, holding the paths to the individual data files
        :param y_data: model target data
        :param batch_size: batch size
        :param data_indices: indices of the whole data set, which are present in this _DataSetGenerator
        """
        self.name = name
        self.batch_size = batch_size
        self.x = x_data
        self.y = y_data
        self._data_indices = data_indices
        self._current_indices = None
        self.is_nifti = Path(x_data[0]).name.endswith((".nii.gz", ".nii", ".mgz", ".mgh"))
        if self.is_nifti:
            self.single_file_loader = partial(get_nifti, reorient=True)
        elif Path(x_data[0]).name.endswith((".pkl", ".pkl.gz", ".npy", ".npz")):
            self.single_file_loader = partial(_load_obj_from_abs_path, functimer=False)
        else:
            msg = "File format not supported! For instance, see x_data[0]."
            raise ValueError(msg)
        self.preprocess = preprocess

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate one batch of data."""
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]
        # Load x data from file paths
        batch_x_arr = np.array([self.single_file_loader(x) for x in batch_x])
        if self.is_nifti:
            # Get data arrays from NIfTI images
            batch_x_arr = np.array([x.get_fdata() for x in batch_x_arr])
        batch_x_arr = self.preprocess(batch_x_arr)  # preprocess input data

        # Expand to empty dimension (batch_size, x, y, z, channel=1)
        batch_x_arr = np.expand_dims(batch_x_arr, axis=-1)  # add channel dimension

        batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
        self._current_indices = self._data_indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        return batch_x_arr, batch_y

    @property
    def x(self) -> np.ndarray[str | Path]:
        """Return x_data."""
        return self._x

    @x.setter
    def x(self, x_data: np.ndarray[str | Path] | list[str | Path] | tuple[str | Path]) -> None:
        """Set x_data and run checks."""
        if not isinstance(x_data, (np.ndarray, list, tuple)):
            msg = f"x_data must be a np.ndarray, list, or tuple, but is of type {type(x_data)}"
            raise TypeError(msg)
        if not all(isinstance(x, (str, Path)) for x in x_data):
            msg = "x_data must only contain strings or Path objects!"
            raise TypeError(msg)
        self._x = np.array(x_data)

    @property
    def single_file_loader(self) -> callable:
        """Return single_file_loader."""
        return self._single_file_loader

    @single_file_loader.setter
    def single_file_loader(self, single_file_loader: callable) -> None:
        """Set single_file_loader."""
        if not callable(single_file_loader):
            msg = "single_file_loader must be a callable function!"
            raise TypeError(msg)
        self._single_file_loader = single_file_loader


class _DataSetGeneratorFactory:
    """Factory class for creating data set generators."""

    @staticmethod
    def create_generator(
        name: str | None,
        x_data: np.ndarray | list | tuple,
        y_data: np.ndarray | list | tuple,
        batch_size: int,
        data_indices: np.ndarray[int] | list[int] | tuple[int],
        preprocess: callable[..., np.ndarray[np.float32, np.int_]] | None,
    ) -> _DataSetGenerator:
        """
        Create a data set generator.

        example "Usage"
            ```python
            generator = _DataSetGeneratorFactory.create_generator(name, x_data, y_data, batch_size, data_indices)
            ```

        For model training, too large datasets cannot be loaded in working memory as a whole block.
        Here generators come handy.

        :param name: Name of the data set.
        :param x_data: Model input data.
        :param y_data: Model target data.
        :param batch_size: Batch size.
        :param data_indices: Indices of the whole data set, which are present in this `_DataSetGenerator`.
        :param preprocess: Preprocess function for input data, which will be applied when loading the data.
        :return: Data set generator.
        """
        dataset_generator = (
            _DataSetGeneratorFromFilePaths if isinstance(x_data[0], (str, Path)) else _DataSetGeneratorFromFullArray
        )
        return dataset_generator(
            name=name,
            x_data=x_data,
            y_data=y_data,
            batch_size=batch_size,
            data_indices=data_indices,
            preprocess=preprocess,
        )


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
