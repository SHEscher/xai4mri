"""Running the `test_mri_registration` script."""

# %% Import
import nibabel as nib
import numpy as np
import pytest
from xai4mri.dataloader.mri_dataloader import get_nifti
from xai4mri.dataloader.mri_registration import get_mni_template, is_mni, register_to_mni

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_get_mni_template(capsys):
    """Test the test_get_mni_template function."""
    # low_res
    for low_res in [True, False]:
        mni_template = get_mni_template(low_res=low_res, as_nii=True, prune_mode=None)
        resolution = np.round(mni_template.header["pixdim"][1:4], decimals=3)
        assert (resolution == np.array((1.0, 1.0, 1.0)) + int(low_res)).all()

    # reorient
    for reorient in [True, False]:
        mni_template = get_mni_template(reorient=reorient, prune_mode=None, as_nii=True, original_template=True)
        assert mni_template.shape == (91, 91, 109) if reorient else (91, 109, 91)

    # prune_mode
    mni_template = get_mni_template(low_res=True, prune_mode="max", original_template=True, reorient=True)
    assert mni_template.shape in ((78, 83, 102), (80, 82, 99))
    mni_template = get_mni_template(low_res=True, prune_mode="cube", original_template=True, reorient=True)
    assert mni_template.shape in ((102, 102, 102), (99, 99, 99))
    mni_template = get_mni_template(low_res=True, prune_mode=None, original_template=True, reorient=True)
    assert mni_template.shape == (91, 91, 109)

    with pytest.raises(ValueError, match=r"prune_mode must be 'cube' OR 'max' OR None"):
        get_mni_template(low_res=True, prune_mode="NonSense", original_template=True, reorient=True)

    # norm
    mni_template = get_mni_template(norm=(0, 1), low_res=True, prune_mode=None, original_template=True, as_nii=False)
    assert mni_template.min() == 0.0
    assert mni_template.max() == 1.0

    max_value = 255.0
    mni_template = get_mni_template(
        norm=(0, max_value), low_res=True, prune_mode=None, original_template=True, as_nii=False
    )
    assert mni_template.min() == 0.0
    assert mni_template.max() == max_value

    for norm in [(-23.5, 1), (5, 255)]:
        with pytest.raises(ValueError, match=r"Function works only for zero-background "):
            get_mni_template(norm=norm, low_res=True, prune_mode=None, original_template=True, as_nii=False)

    mni_template = get_mni_template(mask=True, low_res=True, prune_mode=None, original_template=True, as_nii=False)
    assert (np.unique(mni_template) == np.array([0, 1])).all()
    mni_template = get_mni_template(mask=False, low_res=True, prune_mode=None, original_template=True, as_nii=False)
    min_n_intensity_values = 50
    assert len(np.unique(mni_template)) > min_n_intensity_values

    # original_template
    mni_template = get_mni_template(original_template=True, low_res=True, prune_mode=None, as_nii=False, reorient=True)
    assert mni_template.shape == (91, 91, 109)
    mni_template = get_mni_template(
        original_template=False, low_res=True, prune_mode=None, as_nii=False, reorient=True
    )
    assert mni_template.shape == (99, 95, 117)

    # as_nii
    mni_template = get_mni_template(as_nii=False, low_res=True, prune_mode=None, reorient=True)
    assert isinstance(mni_template, np.ndarray)
    mni_template_as_nii = get_mni_template(as_nii=True, low_res=True, prune_mode=None, reorient=True)
    assert isinstance(mni_template_as_nii, nib.Nifti1Image)
    assert mni_template_as_nii.shape == mni_template.shape
    # Check no pruning when as_nii=True
    mni_template = get_mni_template(as_nii=True, prune_mode="max", mask=True)
    captured = capsys.readouterr()
    assert "No pruning or masking is done for MNI templates that are returned as NIfTI!" in captured.out
    assert mni_template.shape == (91, 91, 109)


def test_register_to_mni_and_is_mni(capsys, tmp_path):
    """Test the register_to_mni and is_mni function."""
    save_path_mni = tmp_path / "mri_mni.nii.gz"
    moving_mri = get_nifti(mri_path="data/demo/LIFE/0ACD5E1BB1/mri/brain.finalsurfs.mgz", reorient=False)
    mri_mni = register_to_mni(
        moving_mri=moving_mri,
        resolution=2,
        type_of_transform="Rigid",  # "SyN"
        save_path_mni=save_path_mni,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "Given image with original shape " in captured.out
    assert mri_mni.shape == (91, 109, 91)

    # Check loading of already registered MRI
    mri_mni2 = register_to_mni(
        moving_mri=moving_mri,
        resolution=2,
        type_of_transform="Rigid",  # "SyN"
        save_path_mni=save_path_mni,
        verbose=True,
    )
    assert (mri_mni.get_fdata() == mri_mni2.get_fdata()).all()

    # Wrong resolution
    with pytest.raises(ValueError, match=r"resolution must be either 1 or 2"):
        register_to_mni(
            resolution=55,
            moving_mri=moving_mri,
            type_of_transform="Rigid",  # "SyN"
            save_path_mni=None,
            verbose=False,
        )

    # test is_mni()
    assert not is_mni(img=moving_mri)
    assert is_mni(img=mri_mni)
    assert is_mni(img=mri_mni2)


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
