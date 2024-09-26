"""Running the test_transformation script."""

# %% Import
import pytest
from xai4mri.dataloader.transformation import (
    _check_norm,
    apply_mask,
    clip_data,
    compress_and_norm,
    determine_min_max_clip,
    file_to_ref_orientation,
    get_orientation_transform,
    mri_to_ref_orientation,
    save_ants_warpers,
)

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@pytest.mark.skip(reason="not implemented yet!")
def test_check_norm():
    """Test the _check_norm function."""
    _check_norm()


@pytest.mark.skip(reason="not implemented yet!")
def test_clip_data():
    """Test the clip_data function."""
    clip_data()


@pytest.mark.skip(reason="not implemented yet!")
def test_get_orientation_transform():
    """Test the get_orientation_transform function."""
    get_orientation_transform()


@pytest.mark.skip(reason="not implemented yet!")
def test_file_to_ref_orientation():
    """Test the file_to_ref_orientation function."""
    file_to_ref_orientation()


@pytest.mark.skip(reason="not implemented yet!")
def test_mri_to_ref_orientation():
    """Test the mri_to_ref_orientation function."""
    mri_to_ref_orientation()


@pytest.mark.skip(reason="not implemented yet!")
def test_save_ants_warpers():
    """Test the save_ants_warpers function."""
    save_ants_warpers()


@pytest.mark.skip(reason="not implemented yet!")
def test_apply_mask():
    """Test the apply_mask function."""
    apply_mask()


@pytest.mark.skip(reason="not implemented yet!")
def test_determine_min_max_clip():
    """Test the determine_min_max_clip function."""
    determine_min_max_clip()


@pytest.mark.skip(reason="not implemented yet!")
def test_compress_and_norm():
    """Test the compress_and_norm function."""
    compress_and_norm()


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
