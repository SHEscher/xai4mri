"""Running the test_prune_image.py script."""

# %% Import
import numpy as np
import pytest
from xai4mri.dataloader.prune_image import (
    # _PruneConfig,
    PruneConfig,  # instance of _PruneConfig
    _place_small_in_middle_of_big,
    find_brain_edges,
    get_brain_axes_length,
    get_global_max_axes,
    permute_array,
    permute_nifti,
    prune_mri,
    reverse_pruning,
)

# from xai4mri.model.mrinets import MRInet  # noqa: ERA001
# from xai4mri.model.interpreter import analyze_model  # noqa: ERA001
from .test_datasets import dataset  # noqa: F401

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# dataset is a fixture from tests/test_datasets.py
pass


# %% Test functions o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_pruneconfig():
    """Test PruneConfig()."""
    assert PruneConfig.orientation_space == "LIA"  # == GLOBAL_ORIENTATION_SPACE

    mri_n_dim: int = 3
    # For sequences
    assert len(PruneConfig.largest_brain_max_axes) == mri_n_dim
    assert (isinstance(e, (int, np.int_)) for e in PruneConfig.largest_brain_max_axes)

    new_max_axes = (182, 182, 218)
    PruneConfig.largest_brain_max_axes = new_max_axes
    assert np.all(PruneConfig.largest_brain_max_axes == new_max_axes)

    # For single integer
    new_max_axis = 198
    PruneConfig.largest_brain_max_axes = new_max_axis
    assert np.all(PruneConfig.largest_brain_max_axes == new_max_axis)  # == np.array([198, 198, 198])
    assert len(PruneConfig.largest_brain_max_axes) == mri_n_dim

    # Test error handling
    # Value(s) too small
    with pytest.raises(ValueError, match="For 'largest_brain_max_axes' a 1 mm isotropic resolution is assumed."):
        PruneConfig.largest_brain_max_axes = np.array([91, 91, 109])

    part_error_msg = "largest_brain_max_axes must be a sequence of positive ints "

    # For sequences
    with pytest.raises(ValueError, match=part_error_msg):
        PruneConfig.largest_brain_max_axes = (0, -100, 218)

    with pytest.raises(ValueError, match=part_error_msg):
        PruneConfig.largest_brain_max_axes = [198, 218]

    with pytest.raises(ValueError, match=part_error_msg):
        PruneConfig.largest_brain_max_axes = np.array([198.5, 198.2, 218])

    # If a single integer is passed
    with pytest.raises(ValueError, match=part_error_msg):
        PruneConfig.largest_brain_max_axes = 198.5

    with pytest.raises(TypeError, match=part_error_msg):
        PruneConfig.largest_brain_max_axes = None

    with pytest.raises(TypeError, match=part_error_msg):
        PruneConfig.largest_brain_max_axes = dict(x=198, y=198, z=218)


def test_reverse_pruning(dataset):  # noqa: F811
    """Test reverse_pruning()."""
    # Get pruned MRI data
    split_dict, _, _, test_data_gen = dataset.create_data_split(target="fake_age", batch_size=1, force=True)

    input_img, y = next(iter(test_data_gen))  # take the first image and label

    # Get a random model
    # model = MRInet.create(name="test_model", n_classes=False, input_shape=input_img.squeeze().shape)  # noqa: ERA001

    # Do something with input_img and y, create a relevance map
    # TODO: Solve the issue with eageer execution and the tf.gradients  # noqa: FIX002
    #  (see: https://stackoverflow.com/questions/66221788/tf-gradients-is-not-supported-when-eager-execution-is-enabled-use-tf-gradientta)
    # relevance_map = analyze_model(model=model, ipt=input_img, norm=True)  # noqa: ERA001
    relevance_map = input_img.astype(float) - input_img.max() / 2  # temporary simulate relevance map

    # Get current SID in the test set
    current_sid = dataset.sid_list[test_data_gen.current_indices][0]

    # Get the original MRI of the SID (as nib.Nifti1Image)
    org_img_nii = dataset.get_unpruned_mri(sid=current_sid)

    # Reverse pruning for the input image
    input_img_nii = reverse_pruning(
        original_mri=org_img_nii,  # alternatively, an np.ndarray can be passed
        pruned_mri=input_img.squeeze(),  # (1, x, y, z, 1) -> (x, y, z)
        pruned_stats_map=None,
    )
    # if np.ndarray is passed, then reverse_pruning will return a np.ndarray of the original MRI

    # Reverse pruning for the relevance map
    relevance_map_nii = reverse_pruning(
        original_mri=org_img_nii,  # reverse pruning for heatmap
        pruned_mri=input_img.squeeze(),
        pruned_stats_map=relevance_map.squeeze(),  # ← this must be given here
    )

    assert input_img_nii.shape == org_img_nii.shape
    assert relevance_map_nii.shape == org_img_nii.shape
    assert input_img.max() == input_img_nii.get_fdata().max()
    assert input_img.min() == input_img_nii.get_fdata().min()
    assert relevance_map.max() == relevance_map_nii.get_fdata().max()
    assert relevance_map.min() == relevance_map_nii.get_fdata().min()


@pytest.mark.skip("Not implemented yet!")
def test_prune_mri():
    """Test prune_mri()."""
    prune_mri()


@pytest.mark.skip("Not implemented yet!")
def test__place_small_in_middle_of_big():
    """Test _place_small_in_middle_of_big()."""
    _place_small_in_middle_of_big()


@pytest.mark.skip("Not implemented yet!")
def test_find_brain_edges():
    """Test find_brain_edges()."""
    find_brain_edges()


@pytest.mark.skip("Not implemented yet!")
def test_get_brain_axes_length():
    """Test get_brain_axes_length()."""
    get_brain_axes_length()


@pytest.mark.skip("Not implemented yet!")
def test_get_global_max_axes():
    """Test get_global_max_axes()."""
    get_global_max_axes()


@pytest.mark.skip("Not implemented yet!")
def test_permute_array():
    """Test permute_array()."""
    permute_array()


@pytest.mark.skip("Not implemented yet!")
def test_permute_nifti():
    """Test permute_nifti()."""
    permute_nifti()


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
