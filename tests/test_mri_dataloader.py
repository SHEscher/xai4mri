"""Running the `test_mri_dataloader` script."""

# %% Import
import re

import pytest
from xai4mri.dataloader.mri_dataloader import (
    _check_regis,
    _prepare_metadata,
    get_nifti,
    load_file_paths_from_metadata,
    load_files_from_metadata,
    mgz2nifti,
)

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_check_regis():
    """Test the _check_regis function."""
    # No registration
    assert not _check_regis(regis_mni=None)
    assert not _check_regis(regis_mni=False)

    # Valid mm values for registration
    assert _check_regis(regis_mni=1)
    assert _check_regis(regis_mni=2)

    # Invalid values
    error_msg = re.escape("regis_mni must be 1 or 2 [mm], OR None / False.")
    with pytest.raises(ValueError, match=error_msg):
        _check_regis(regis_mni=99)

    with pytest.raises(ValueError, match=error_msg):
        _check_regis(regis_mni="wrong-value")

    with pytest.raises(ValueError, match=error_msg):
        _check_regis(regis_mni=True)


@pytest.mark.skip(reason="not implemented yet!")
def test_load_file_paths_from_metadata():
    """Test the load_file_paths_from_metadata function."""
    load_file_paths_from_metadata()


@pytest.mark.skip(reason="not implemented yet!")
def test_load_from_metadata():
    """Test the load_files_from_metadata function."""
    load_files_from_metadata()


@pytest.mark.skip(reason="not implemented yet!")
def test__prepare_metadata():
    """Test the _prepare_metadata function."""
    _prepare_metadata()


@pytest.mark.skip(reason="not implemented yet!")
def test_mgz2nifti():
    """Test the mgz2nifti function."""
    mgz2nifti()


@pytest.mark.skip(reason="not implemented yet!")
def test_get_nifti():
    """Test the get_nifti function."""
    get_nifti()


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
