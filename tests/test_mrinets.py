"""Running the test_MRInets script."""

# %% Import
import pytest
import requests
from tensorflow import keras
from xai4mri.model.mrinets import (
    SFCN,
    MRInet,
    OutOfTheBoxModels,
    PretrainedModelsMRInet,
    PretrainedMRInetFLAIR,
    PretrainedMRInetSWI,
    PretrainedMRInetT1,
    _check_n_classes,
    _PretrainedModel,
    _PretrainedModels,
    _PretrainedMRInet,
    get_model,
)

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Test Functions o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def test_check_n_classes():
    """Test the _check_n_classes function."""
    # Bool True / False
    assert _check_n_classes(n_classes=False) is None
    with pytest.raises(ValueError, match="For classification tasks n_classes must be "):
        _check_n_classes(n_classes=True)

    # not int
    with pytest.raises(TypeError, match="For classification tasks n_classes must be an integer!"):
        _check_n_classes(n_classes="2")

    with pytest.raises(TypeError, match="For classification tasks n_classes must be an integer!"):
        _check_n_classes(n_classes=5.0)

    # ints
    assert _check_n_classes(n_classes=0) is None  # this assumes a regression task
    with pytest.raises(ValueError, match="For classification tasks n_classes must be "):
        _check_n_classes(n_classes=1)
    assert _check_n_classes(n_classes=2) is None  # this assumes a regression task
    assert _check_n_classes(n_classes=99) is None  # this assumes a regression task


def test_pretrainedmodels():
    """Test the test_PretrainedModels enum."""
    assert _PretrainedModels.MRINET.value == PretrainedModelsMRInet
    assert len(_PretrainedModels) == 1  # currently, there is only one pretrained model type


def test_pretrainedmodelsmrinet():
    """Test the PretrainedModelsMRInet enum."""
    assert PretrainedModelsMRInet.T1_MODEL.value == PretrainedMRInetT1()
    assert PretrainedModelsMRInet.FLAIR_MODEL.value == PretrainedMRInetFLAIR()
    assert PretrainedModelsMRInet.SWI_MODEL.value == PretrainedMRInetSWI()
    assert len(PretrainedModelsMRInet) == 3  # noqa: PLR2004

    # Check wether the enum can be called by string via _missing_
    assert PretrainedModelsMRInet("t1") == PretrainedModelsMRInet.T1_MODEL
    assert PretrainedModelsMRInet("fLAir") == PretrainedModelsMRInet.FLAIR_MODEL
    assert PretrainedModelsMRInet("SWI_MOdeL") == PretrainedModelsMRInet.SWI_MODEL


def test_pretrainedmrinett1(tmp_path):
    """Test the PretrainedMRInetT1 class."""
    pretrained_model = PretrainedMRInetT1()
    assert isinstance(pretrained_model, _PretrainedModel)
    assert isinstance(pretrained_model, _PretrainedMRInet)
    assert "t1_model" in pretrained_model.name

    # Test URL
    response = requests.get(pretrained_model.url)  # noqa: S113
    assert response.status_code == 200  # website is available  # noqa: PLR2004

    tempdir_model = tmp_path / "models"
    model = pretrained_model.load_model(
        parent_folder=tempdir_model,
        custom_objects=None,
        compile_model=False,
    )
    assert isinstance(model, keras.Sequential)
    assert "t1" in model.name.lower()


def test_pretrainedmrinetflair(tmp_path):
    """Test the PretrainedMRInetFLAIR class."""
    pretrained_model = PretrainedMRInetFLAIR()
    assert isinstance(pretrained_model, _PretrainedModel)
    assert isinstance(pretrained_model, _PretrainedMRInet)
    assert "flair_model" in pretrained_model.name

    # Test URL
    response = requests.get(pretrained_model.url)  # noqa: S113
    assert response.status_code == 200  # website is available  # noqa: PLR2004

    tempdir_model = tmp_path / "models"
    model = pretrained_model.load_model(
        parent_folder=tempdir_model,
        custom_objects=None,
        compile_model=False,
    )
    assert isinstance(model, keras.Sequential)
    assert "flair" in model.name.lower()


def test_pretrainedmrinetswi(tmp_path):
    """Test the PretrainedMRInetSWI class."""
    pretrained_model = PretrainedMRInetSWI()
    assert isinstance(pretrained_model, _PretrainedModel)
    assert isinstance(pretrained_model, _PretrainedMRInet)
    assert "swi_model" in pretrained_model.name

    # Test URL
    response = requests.get(pretrained_model.url)  # noqa: S113
    assert response.status_code == 200  # website is available  # noqa: PLR2004

    tempdir_model = tmp_path / "models"
    model = pretrained_model.load_model(
        parent_folder=tempdir_model,
        custom_objects=None,
        compile_model=False,
    )
    assert isinstance(model, keras.Sequential)
    assert "swi" in model.name.lower()


def test_mrinet():
    """Test the MRInet class."""
    mrinet = MRInet()

    assert "Hofmann et al." in mrinet.reference()
    assert PretrainedModelsMRInet == mrinet.pretrained_models()

    # Test creation of the model
    model = mrinet.create(
        name="test_model",
        n_classes=False,
        input_shape=(198, 198, 198),
        target_bias=57.5,
        learning_rate=5e-4,
        leaky_relu=False,
        batch_norm=False,
    )
    assert isinstance(model, keras.Sequential)
    assert "test_model" in model.name.lower()

    # Test via get_model
    model = get_model(
        model_type=OutOfTheBoxModels.MRINET,
        name="test_model2",
        input_shape=(198, 198, 198),
        n_classes=False,
    )
    assert isinstance(model, keras.Sequential)
    assert "test_model2" in model.name.lower()

    # TODO: test inference  # noqa: FIX002
    pass


def test_sfcn():
    """Test the SFCN class."""
    sfcn = SFCN()
    assert "Peng et al." in sfcn.reference()
    assert sfcn.pretrained_models() is None

    model = sfcn.create(
        name="test_sfcn_model",
        n_classes=40,
        input_shape=(160, 192, 160),
        learning_rate=0.01,
        dropout=True,
    )
    assert isinstance(model, keras.Sequential)
    assert "test_sfcn_model" in model.name.lower()

    # Test via get_model
    model = get_model(
        model_type=OutOfTheBoxModels.SFCN,
        name="test_sfcn_model2",
        n_classes=40,
        input_shape=(160, 192, 160),
    )
    assert isinstance(model, keras.Sequential)
    assert "test_sfcn_model2" in model.name.lower()

    # TODO: test inference  # noqa: FIX002
    pass


def test_get_model_error():
    wrong_type = "Wrong Model Type"
    with pytest.raises(ValueError, match=f"'{wrong_type}' is not a valid {OutOfTheBoxModels.__name__}"):
        get_model(
            model_type=wrong_type,
            name="WrongModel",
            n_classes=40,
            input_shape=(160, 192, 160),
        )


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
