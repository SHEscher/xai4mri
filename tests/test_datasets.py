"""Running the `test_datasets` script."""

# %% Import
from __future__ import annotations

import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from xai4mri.dataloader import PruneConfig
from xai4mri.dataloader.datasets import (
    BaseDataSet,
    _DataSetGenerator,
    _DataSetGeneratorFromFilePaths,
    _extract_values_from_metadata,
)

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

PROJECT_ROOT = Path(__file__).parent.parent
TEST_CACHE_DIR = Path(PROJECT_ROOT, "data", "test_cache")
PATH_TO_TEST_DATA = Path(PROJECT_ROOT, "data/demo/LIFE")
TEST_NUMERIC_SIDS: bool = False  # TODO: pass in fixture # noqa: FIX002
REGISTER_MNI = None  # 1, 2, or None  # TODO: pass in fixture # noqa: FIX002

# %% Set fixtures < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@pytest.fixture(scope="module")
def dataset():
    """Create a temporary dataset."""
    project_id = "TestImplementationNr" if TEST_NUMERIC_SIDS else "TestImplementation"
    sid_table_name = "sid_tab_nr.tsv" if TEST_NUMERIC_SIDS else "sid_tab.csv"

    PruneConfig.largest_brain_max_axes = np.array([160, 165, 198])

    # Build temporary dataset
    class MyImplementationTestData(BaseDataSet):
        """FA dataset class."""

        def __init__(self):
            """Init FA dataset."""
            super().__init__(
                study_table_or_path=PATH_TO_TEST_DATA / sid_table_name,
                project_id=project_id,
                mri_sequence="t1",
                register_to_mni=REGISTER_MNI,
                cache_dir=TEST_CACHE_DIR,
                cache_files=False,
                # MRI processing kwargs
                prune_mode="max",
                no_arg="to-be-ignored",  # invalid argument that should be ignored
            )

        @staticmethod
        def mri_path_constructor(sid: str) -> str | Path:
            """
            Construct the MRI path for LIFE t1 demo data.

            :param sid: subject ID
            :return: absolute file path
            """
            return Path(PROJECT_ROOT, "data/demo/LIFE", sid, "mri", "brain.finalsurfs.mgz")

    yield MyImplementationTestData()

    # Teardown
    # TODO: check if whole set should be deleted # noqa: FIX002
    # Path(PROJECT_ROOT, "data", project_id).unlink()  # noqa: ERA001
    for file in TEST_CACHE_DIR.glob(f"{project_id}_*"):
        file.unlink()

    # Remove split_dict dir
    data_split_dir = Path(TEST_CACHE_DIR, "data_splits")
    if data_split_dir.exists():
        for file in data_split_dir.glob(f"*_{project_id}_*_split_dict.npy"):
            file.unlink()
        # Check if dir is empty
        if not list(data_split_dir.iterdir()):
            # If yes, delete dir
            data_split_dir.rmdir()

    # Check if dir is empty
    if not list(TEST_CACHE_DIR.iterdir()):
        # If yes, delete dir
        TEST_CACHE_DIR.rmdir()


@pytest.fixture(scope="module")
def dataset_from_paths():
    """Create a temporary dataset."""
    project_id = "TestImplementationNrFromFilePaths" if TEST_NUMERIC_SIDS else "TestImplementationFromFilePaths"
    sid_table_name = "sid_tab_nr.tsv" if TEST_NUMERIC_SIDS else "sid_tab.csv"

    PruneConfig.largest_brain_max_axes = np.array([160, 165, 198])

    # Build temporary dataset
    class MyImplementationTestDataFromFilePaths(BaseDataSet):
        """FA dataset class."""

        def __init__(self):
            """Init FA dataset."""
            super().__init__(
                study_table_or_path=PATH_TO_TEST_DATA / sid_table_name,
                project_id=project_id,
                mri_sequence="t1",
                register_to_mni=REGISTER_MNI,
                cache_dir=TEST_CACHE_DIR,
                load_mode="file_paths",
                cache_files=False,
                # MRI processing kwargs
                prune_mode="max",
                save_after_processing=True,  # should be overwritten
            )

        @staticmethod
        def mri_path_constructor(sid: str) -> str | Path:
            """
            Construct the MRI path for LIFE t1 demo data.

            :param sid: subject ID
            :return: absolute file path
            """
            return Path(PROJECT_ROOT, "data/demo/LIFE", sid, "mri", "T1.mgz")

    yield MyImplementationTestDataFromFilePaths()

    # Teardown
    # Path(PROJECT_ROOT, "data", project_id).unlink()  # noqa: ERA001
    for file in TEST_CACHE_DIR.glob(f"{project_id}_*"):
        file.unlink()

    # Delete all files and folders in TEST_CACHE_DIR using shutil
    if Path(TEST_CACHE_DIR, project_id).exists():
        shutil.rmtree(TEST_CACHE_DIR / project_id)

    # Remove split_dict dir
    data_split_dir = Path(TEST_CACHE_DIR, "data_splits")
    if data_split_dir.exists():
        for file in data_split_dir.glob(f"*_{project_id}_*_split_dict.npy"):
            file.unlink()
        # Check if dir is empty
        if not list(data_split_dir.iterdir()):
            # If yes, delete dir
            data_split_dir.rmdir()

    # Check if dir is empty
    if not list(TEST_CACHE_DIR.iterdir()):
        # If yes, delete dir
        TEST_CACHE_DIR.rmdir()


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_extract_values_from_metadata_table(dataset):
    """Test the _extract_values_from_metadata function."""
    with pytest.raises(FileNotFoundError, match=r"Metadata table not found at '"):
        _ = dataset.get_metadata_table()

    # Generate metadata table first
    _ = dataset.get_data(force=True)

    metadata_table = dataset.get_metadata_table()
    min_clip_col = [c for c in metadata_table.columns if "min_clip" in c]
    max_clip_col = [c for c in metadata_table.columns if "max_clip" in c]
    min_intens_col = [c for c in metadata_table.columns if "min_intensity" in c]
    max_intens_col = [c for c in metadata_table.columns if "max_intensity" in c]

    # Test no norm no compress
    clip_min, clip_max, global_norm_min, global_norm_max, norm = _extract_values_from_metadata(
        metadata_table, compress=False, norm=False
    )
    assert not norm
    assert clip_min is None
    assert clip_max is None
    assert global_norm_min is None
    assert global_norm_max is None

    # Test norm & compress
    clip_min, clip_max, global_norm_min, global_norm_max, norm = _extract_values_from_metadata(
        metadata_table, compress=True, norm=True
    )
    assert norm
    assert clip_min == np.nanmean(metadata_table[min_clip_col])
    assert clip_max == np.nanmean(metadata_table[max_clip_col])
    assert global_norm_min == clip_min
    assert global_norm_max == clip_max

    # Test norm & not compress
    clip_min, clip_max, global_norm_min, global_norm_max, norm = _extract_values_from_metadata(
        metadata_table, compress=False, norm=True
    )
    assert norm
    assert clip_min is None
    assert clip_max is None
    assert global_norm_min == np.nanmin(metadata_table[min_intens_col])
    assert global_norm_max == np.nanmax(metadata_table[max_intens_col])

    # Test negative value case (compress & norm -> no norm)
    metadata_table[min_intens_col] = metadata_table[min_intens_col] - 1
    clip_min, clip_max, global_norm_min, global_norm_max, norm = _extract_values_from_metadata(
        metadata_table, compress=True, norm=True
    )
    assert not norm
    assert clip_min == np.nanmean(metadata_table[min_clip_col])
    assert clip_max == np.nanmean(metadata_table[max_clip_col])
    assert global_norm_min is None
    assert global_norm_max is None


def test_datasetgenerator(dataset):
    """Test the DataSetGenerator class."""
    split_dict, _, _, test_data_gen = dataset.create_data_split(target="fake_age", force=True)

    # Test __repr__ and __str__
    assert "test" in test_data_gen.__repr__()
    assert "n_samples=" in test_data_gen.__repr__()
    assert "batch_size=" in test_data_gen.__repr__()
    assert "_DataSetGenerator" in f"{test_data_gen}"  # __str__

    assert len(test_data_gen) == len(split_dict["test"])
    assert len(test_data_gen.data_indices) == len(split_dict["test"])
    assert test_data_gen.n_samples == len(split_dict["test"])

    for batch_x, bath_y in test_data_gen:
        assert isinstance(batch_x, np.ndarray)
        assert isinstance(bath_y, np.ndarray)
        assert batch_x.shape[0] == bath_y.shape[0]

    assert set(test_data_gen.current_indices).issubset(test_data_gen.data_indices)

    # Test data_indices setter
    with pytest.raises(TypeError, match=r"data_indices must be a np.ndarray, list, or tuple, but is of type "):
        test_data_gen.data_indices = "no indices"

    with pytest.raises(TypeError, match=r"data_indices must only contain integers!"):
        test_data_gen.data_indices = (1, 4, "no index")

    test_data_gen.data_indices = np.arange(len(test_data_gen), dtype=int)  # valid
    assert test_data_gen.data_indices.shape[0] == len(test_data_gen)

    # Test preprocess setter
    with pytest.raises(TypeError, match=r"preprocess must be a callable function!"):
        test_data_gen.preprocess = "no preprocess"


def test_datasetgenerator_from_filepaths(dataset_from_paths):
    """Test the _DataSetGeneratorFromFilePaths class."""
    split_dict, _, _, test_data_gen = dataset_from_paths.create_data_split(target="fake_age", force=True)

    with pytest.raises(TypeError, match=r"x_data must be a np.ndarray, list, or tuple, but is of type"):
        test_data_gen.x = "invalid data"

    with pytest.raises(TypeError, match=r"x_data must only contain strings or Path objects!"):
        test_data_gen.x = ("path_1", Path("path_2"), 44)

    with pytest.raises(TypeError, match=r"single_file_loader must be a callable function!"):
        test_data_gen.single_file_loader = "invalid function"

    # Check NifTi file loader
    temp_test_gen = _DataSetGeneratorFromFilePaths(
        name="TempTest",
        x_data=[
            Path(PATH_TO_TEST_DATA, "0D6F638973", "mri", "T1.mgz"),
            Path(PATH_TO_TEST_DATA, "0CF200BEF5", "mri", "T1.mgz"),
        ],
        y_data=[1, 2],
        batch_size=1,
        data_indices=[0, 1],
    )
    assert isinstance(temp_test_gen[0][0], np.ndarray)

    # Test file format checker
    with pytest.raises(ValueError, match=r"File format not supported! For instance, see x_data"):
        _ = _DataSetGeneratorFromFilePaths(
            name="TempTestInvalid",
            x_data=["path_1.invalid", "path_2.invalid"],
            y_data=[1, 2],
            batch_size=1,
            data_indices=[0, 1],
        )


def test_mri_path_constructor(dataset):
    for sid in dataset.sid_list:
        assert Path(dataset.mri_path_constructor(sid=sid)).exists()


def test_study_table(dataset):
    assert isinstance(dataset.study_table, pd.DataFrame)
    assert dataset.study_table.index.name == "sid"


def test_sid_list(dataset):
    assert len(dataset.sid_list) > 0
    assert isinstance(dataset.sid_list, np.ndarray)
    assert all(isinstance(sid, str) for sid in dataset.sid_list)

    # Test setting sid_list (list -> np.ndarray)
    n_sids = 2
    former_sid_list = dataset.sid_list
    dataset.sid_list = dataset.study_table.index.to_list()[:n_sids]
    assert isinstance(dataset.sid_list, np.ndarray)
    assert len(dataset.sid_list) == n_sids
    # Reset (since dataset is used across tests)
    dataset.sid_list = former_sid_list


def test_load_mode(dataset_from_paths):
    """Check if the load_mode is set correctly and processed accurately."""
    assert dataset_from_paths.load_mode == "file_paths"

    split_dict, _, _, test_data_gen = dataset_from_paths.create_data_split(target="fake_age", force=True)

    assert len(test_data_gen) == len(split_dict["test"])
    assert len(test_data_gen.data_indices) == len(split_dict["test"])
    assert test_data_gen.n_samples == len(split_dict["test"])

    for batch_x, bath_y in test_data_gen:
        assert isinstance(batch_x, np.ndarray)
        assert isinstance(bath_y, np.ndarray)
        assert batch_x.shape[0] == bath_y.shape[0]

    assert set(test_data_gen.current_indices).issubset(test_data_gen.data_indices)

    # Set wrong load_mode
    with pytest.raises(ValueError, match=r"load_mode must be in: "):
        dataset_from_paths.load_mode = "wrong_mode"


def test_get_data(dataset):
    mriset, sid_list = dataset.get_data(force=True)

    assert np.all(sid_list == dataset.sid_list)
    assert isinstance(mriset, np.ndarray)
    assert mriset.shape[0] == len(sid_list)

    # Subjects without data
    previous_study_table = dataset.study_table.copy()
    dataset.study_table.rename(index={dataset.study_table.index[2]: "no-data-sub"}, inplace=True)  # noqa: PD002
    dataset._sid_list = None
    with pytest.raises(ValueError, match=r"No MRI data found for following subjects*"):
        _ = dataset.get_data(force=True)
    dataset.study_table = previous_study_table  # reset
    dataset._sid_list = None  # reset


def test_get_size_of_prospective_mri_set(dataset, capsys):
    dataset.get_size_of_prospective_mri_set(estimate_with_n=2)
    captured = capsys.readouterr()
    assert "Estimated size of all pre-processed" in captured.out
    assert "MB" in captured.out
    assert "Estimated time" in captured.out
    assert "[hh:mm:ss]" in captured.out

    max_n = 10
    dataset.get_size_of_prospective_mri_set(estimate_with_n=max_n + 1)
    captured = capsys.readouterr()
    assert f"estimate_with_n is set down to {max_n}." in captured.out

    # Test exception
    dataset.get_size_of_prospective_mri_set(estimate_with_n=1, regis_mni="invalid")
    captured = capsys.readouterr()
    assert "regis_mni must be 1 or 2 [mm], OR None / False." in captured.out

    # Test handling of temporary cache dir
    dataset.get_size_of_prospective_mri_set(estimate_with_n=1, cache_dir="temp_cache_dir")
    captured = capsys.readouterr()
    assert "Estimated size of all pre-processed" in captured.out

    # Test fast version
    dataset.get_size_of_prospective_mri_set(estimate_processing_time=False)
    captured = capsys.readouterr()
    assert "Estimated size of all pre-processed" in captured.out

    # Test fast version for other prune modes, too
    dataset.get_size_of_prospective_mri_set(estimate_processing_time=False, prune_mode="cube")
    captured = capsys.readouterr()
    assert "Estimated size of all pre-processed" in captured.out

    dataset.get_size_of_prospective_mri_set(estimate_processing_time=False, prune_mode=None)
    captured = capsys.readouterr()
    assert "Estimated size of all pre-processed" in captured.out


def test_create_data_split_and_current_split_dict(dataset, capsys):
    split_dict, train_data_gen, val_data_gen, test_data_gen = dataset.create_data_split(target="fake_age")

    n_splits = 3
    assert isinstance(split_dict, dict)
    assert len(split_dict) == n_splits
    for subset in [train_data_gen, val_data_gen, test_data_gen]:
        assert isinstance(subset, _DataSetGenerator)

    assert dataset.current_split_dict == split_dict

    # Test saving of split_dict
    path_to_split_dict = dataset.save_split_dict()
    captured = capsys.readouterr()
    assert "Saved split dictionary to" in captured.out
    assert isinstance(path_to_split_dict, str)
    assert Path(path_to_split_dict).exists()

    # Test loading of split_dict
    loaded_split_dict = dataset.load_split_dict(split_dict_path=path_to_split_dict)

    for subset in split_dict:
        assert subset in loaded_split_dict
        assert set(split_dict[subset]) == set(loaded_split_dict[subset])


def test_create_data_split_invalid(dataset):
    """Check if the invalid split is caught correctly."""
    with pytest.raises(ValueError, match=r"Either split_dict or split_ratio must be provided. split_ratio must sum *"):
        dataset.create_data_split(target="fake_age", split_ratio=None, split_dict=None, force=True)

    # Invalid subject IDs
    with pytest.raises(ValueError, match=r"All SID's in split_dict must be part of the dataset!"):
        dataset.create_data_split(
            target="fake_age",
            split_ratio=None,
            split_dict={"train": ["invalid-sub-1"], "validation": ["invalid-sub-2"], "test": ["invalid-sub-3"]},
            force=True,
        )

    # Invalid split (duplicate SIDs in various splits)
    with pytest.raises(ValueError, match=r"SID's must only appear once in the split!"):
        dataset.create_data_split(
            target="fake_age",
            split_ratio=None,
            # duplicate SIDs in train & validation
            split_dict={
                "train": dataset.sid_list[0:1],
                "validation": dataset.sid_list[0:2],
                "test": dataset.sid_list[2:3],
            },
            force=True,
        )
    # Invalid key names
    with pytest.raises(ValueError, match=r"split_dict must have keys 'train', 'validation', 'test'!"):
        dataset.create_data_split(
            target="fake_age",
            split_ratio=None,
            split_dict={
                "train": dataset.sid_list[0:1],
                "validation": dataset.sid_list[1:2],
                "invalid-key": dataset.sid_list[2:3],
            },
            force=True,
        )

    # Invalid target
    with pytest.raises(ValueError, match=r"target variable must be in study table!"):
        dataset.create_data_split(
            target="invalid_target",
            split_ratio=None,
            split_dict={
                "train": dataset.sid_list[0:1],
                "validation": dataset.sid_list[1:2],
                "test": dataset.sid_list[2:3],
            },
            force=True,
        )


def test_save_split_dict_invalid(dataset):
    """Check if the split_dict is saved correctly."""
    dataset._split_dict = None  # overwrite
    with pytest.raises(ValueError, match=r"No split dictionary provided!"):
        _ = dataset.save_split_dict(split_dict=None, save_path="this/is/a/invalid/path/to/split_dict.npy")


def test_load_split_dict_extension(dataset):
    """Check if the extension is added automatically."""
    with pytest.raises(FileNotFoundError):
        dataset.load_split_dict(split_dict_path="this/is/a/invalid/path/to/split_dict_without_extension")


def test_load_split_dict_invalid_file(dataset, monkeypatch):
    """Check if the extension is added automatically."""

    def mock_np_load(file, allow_pickle):  # noqa: ARG001
        """Mock np.load to return an invalid type."""
        # return an invalid type
        return np.array([1])  # has .item() similar as np.load

    monkeypatch.setattr(np, "load", mock_np_load)

    with pytest.raises(TypeError, match=r"split_dict should be a dict, but is of type*"):
        dataset.load_split_dict(split_dict_path="this/is/not/a/split_dict.npy")


def test_get_unpruned_mri(dataset):
    """Check if the unpruned MRI is returned correctly."""
    for sid in dataset.sid_list:
        assert isinstance(dataset.get_unpruned_mri(sid=sid), nib.Nifti1Image)

    # Check exception if MNI NIfTI files were not cached during processing
    dataset._regis_mni = 2
    previous_cache_files = dataset._cache_files
    dataset._cache_files = False
    with pytest.raises(ValueError, match=r"MNI NIfTI files were not cached during processing!"):
        dataset.get_unpruned_mri(sid=np.random.choice(dataset.sid_list))  # noqa: NPY002
    dataset._regis_mni = REGISTER_MNI  # reset
    dataset._cache_files = previous_cache_files  # reset


def test_check_mri_path_constructor(dataset):
    """Check if the check for the MRI path constructor works."""
    dataset.mri_path_constructor = lambda sid: Path(f"this/is/a/wrong/path/to/mri/{sid}.t1.nii.gz")

    # Expect FileNotFoundError
    with pytest.raises(FileNotFoundError):
        dataset._check_mri_path_constructor()


@pytest.mark.parametrize("wrong_path", [2.5, None, [1, 2, 3], (1, 2, 3)])
def test_study_table_path_type_fails(dataset, wrong_path):
    """Check if invalid paths to the study table are captured correctly."""
    with pytest.raises(TypeError):
        dataset.study_table_path = wrong_path


@pytest.mark.parametrize("wrong_prefix", ["", "only/path/to/table/dir", "wrong/table/suffix/table.npz"])
def test_study_table_path_suffix_fails(dataset, wrong_prefix):
    """Check if invalid paths to the study table are captured correctly."""
    with pytest.raises(ValueError, match=r"must be either \*.csv or \*.tsv!"):
        dataset.study_table_path = wrong_prefix


def test_study_table_path_exists_fails(dataset):
    """Check if invalid paths to the study table are captured correctly."""
    with pytest.raises(FileNotFoundError):
        dataset.study_table_path = "this/is/a/wrong/path/to/the/study_table.csv"


def test_study_table_path_csv_tsv(dataset):
    """Check if valid paths to the study table are captured correctly."""
    for suffix in (".csv", "_nr.tsv"):
        dataset.study_table_path = PATH_TO_TEST_DATA / ("sid_tab" + suffix)
        assert Path(dataset.study_table_path).exists()


@pytest.mark.parametrize("wrong_table_object", [2.5, None, (1, 2, 3), np.empty(shape=(3, 10))])
def test_study_table_invalid_type(dataset, wrong_table_object):
    """Check if valid paths to the study table are captured correctly."""
    with pytest.raises(TypeError):
        dataset.study_table = wrong_table_object


@pytest.mark.parametrize("index_col", ["SID", "", "2", "None", 3, None])
def test_study_table_invalid_columns(dataset, index_col):
    """Check if valid paths to the study table are captured correctly."""
    wrong_table_object = pd.DataFrame(data=np.arange(0, 15).reshape(5, 3), columns=["col1", "col2", index_col])
    wrong_table_object = wrong_table_object.set_index(keys=index_col)
    wrong_table_object.index = wrong_table_object.index.astype(str)

    # # Wrong index name
    with pytest.raises(ValueError, match=r"The study table must have \'sid\' \(subject IDs\) as its index column\!"):
        dataset.study_table = wrong_table_object

    # Wrong index dtype
    wrong_table_object.index = np.arange(5).astype(int)  # wrong type
    wrong_table_object.index.name = "sid"
    with pytest.raises(TypeError, match=r"The index \'sid\' \(subject IDs\) of the study table must be of dtype*"):
        dataset.study_table = wrong_table_object

    # Set non-unique index
    wrong_table_object.index = np.ones(shape=(5,)).astype(str)
    wrong_table_object.index.name = "sid"
    with pytest.raises(ValueError, match=r"The study table must have unique subject IDs!"):
        dataset.study_table = wrong_table_object


@pytest.mark.parametrize("wrong_sid_list", ["SID", 3, 3.14, None, {"sub-1", "sub-2", "sub-3"}])
def test_sid_list_invalid_type(dataset, wrong_sid_list):
    """Check if valid paths to the study table are captured correctly."""
    with pytest.raises(TypeError, match=r"sid_list must be a list or np.ndarray, but is of type*"):
        dataset.sid_list = wrong_sid_list


@pytest.mark.parametrize("wrong_sid_list", [[1, 2, 3], [None, "sub-1", "sub-2"], ("sub-1", "sub-2", 3)])
def test_sid_list_invalid_content(dataset, wrong_sid_list):
    """Check if valid paths to the study table are captured correctly."""
    with pytest.raises(TypeError, match=r"sid_list must only contain strings\!"):
        dataset.sid_list = wrong_sid_list


@pytest.mark.parametrize("wrong_sid_list", [np.array(["sub-1", "sub-2", "0ACD5E1BB1", "009"])])
def test_sid_list_no_match(dataset, wrong_sid_list):
    """Check if valid paths to the study table are captured correctly."""
    with pytest.raises(ValueError, match=r"sid_list must only contain subject IDs that are in the study table\!"):
        dataset.sid_list = wrong_sid_list


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
