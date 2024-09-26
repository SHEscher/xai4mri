"""
Train & load deep learning models for MRI-based predictions.

This module provides a set of out-of-the-box models for MRI-based predictions.
The models are designed for 3D MRI data and can be used for regression or classification tasks.
Moreover, there are pretrained models available for some model architectures.

Models are built on top of the [`TensorFlow` `Keras` API](https://www.tensorflow.org/guide/keras).
This is necessary to ensure compatibility with [`iNNvestigate`](https://github.com/albermax/innvestigate)
for model interpretability.

    Author: Simon M. Hofmann
    Years: 2022-2024
"""

# %% Import
from __future__ import annotations

import urllib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeVar

from tensorflow import keras

from ..utils import _experimental, cprint

# %% Global vars << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
T = TypeVar("T")  # Any type


# %% Define model creators o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class _ModelCreator(Protocol):
    """
    Model creator protocol.

    This protocol defines the expected signature for model creator functions.
    """

    @staticmethod
    @abstractmethod
    def create(
        name: str,
        n_classes: bool | int,
        input_shape: tuple[int, int, int],
        learning_rate: float,
    ) -> keras.Sequential:
        """
        Create a Keras Sequential model.

        :param name: The name of the model.
        :param n_classes: The number of classes for classification tasks.
                          If False, the model is for regression.
        :param input_shape: The shape of the input data as a tuple (height, width, depth).
        :param learning_rate: The learning rate for the model's optimizer [Default is 5e-4]
        :return: A compiled Keras Sequential model.
        """
        ...

    @staticmethod
    @abstractmethod
    def reference() -> str:
        """
        Get the reference for the model.

        Format:
            "Author et al. (Year). Journal. DOI-URL"
        """
        ...

    @staticmethod
    @abstractmethod
    def pretrained_models() -> type[Enum] | None:
        """
        List of pretrained models.

        If no pretrained models are available, return an empty list.
        """
        ...


def _check_n_classes(n_classes: int | bool) -> None:
    """
    Check the n_classes variable.

    :param n_classes: number of classes for classification tasks or False for regression tasks
    :raises TypeError: If n_classes is not an integer for classification tasks or False for regression tasks
    """
    minimum_n_classes: int = 2
    if n_classes and not isinstance(n_classes, int):
        msg = "For classification tasks n_classes must be an integer!"
        raise TypeError(msg)
    if n_classes and n_classes < minimum_n_classes:
        msg = f"For classification tasks n_classes must be {minimum_n_classes} or more!"
        raise ValueError(msg)


class MRInet(_ModelCreator):
    """
    `MRInet` model creator.

    `MRInet` is the basemodel architecture (3D-ConvNet) used within the multi-level ensembles in
    [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).
    The models were trained to predict age from MR images of different sequences (`T1`, `FLAIR`, `SWI`)
    in the [`LIFE Adult` study](https://doi.org/10.1186/s12889-015-1983-z).

    This model creator class provides a hard-coded Keras implementation of the 3D-CNN model.
    Creating a model with this class is as simple as calling the `create` method.
    This will return a fresh (i.e., untrained) and compiled `Keras` model ready to be trained:

    !!! example "Create a new instance of `MRInet`"
        ```python
        # Create a new instance of MRInet
        mrinet = MRInet.create(name="MyMRInet", n_classes=False, input_shape=(91, 109, 91))

        # Train on MRI dataset
        mrinet.fit(X_train, y_train, ...)
        ```

    Pretrained models are available for `T1`, `FLAIR`, and `SWI` images.
    !!! tip "Get an overview of pretrained models by calling:"
        ```python
        MRInet.pretrained_models().show()
        ```
    """

    @staticmethod
    def create(
        name: str,
        n_classes: bool | None | int,
        input_shape: tuple[int, int, int],
        learning_rate: float = 5e-4,
        target_bias: float | None = None,
        leaky_relu: bool = False,
        batch_norm: bool = False,
    ) -> keras.Sequential:
        """
        Create a new instance of `MRInet`, a 3D-convolutional neural network (CNN) for predictions on MRIs.

        This is a hard-coded Keras implementation of 3D-CNN model as reported in
        [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).

        The model can be trained for regression (`n_classes=False`) or classification (`n_classes: int >= 2`) tasks.

        :param name: Model name, which, for instance, could refer to the project it is applied for.
        :param target_bias: Model output bias.
                            For classification tasks with this can be left blank [`None`].
                            For regression tasks, it is recommended to set this bias to the average of the
                            prediction target distribution in the dataset.
        :param input_shape: Shape of the input to the model.
                            This should be the shape of a single MRI (e.g., (91, 91, 109).
        :param learning_rate: Learning rate which is used for the model's optimizer (here, `Adam`).
        :param batch_norm: Use batch normalization or not.
                           Batch normalization should only be used if the model is fed with larger batches.
                           For model interpretability provided by `xai4mri`,
                           it is recommended to not use batch normalization.
        :param leaky_relu: Using leaky or vanilla ReLU activation functions.
                           Leaky ReLU is recommended for better performance.
                           However, `iNNvestigate`, which is used for model interpretability,
                           does not support leaky ReLU currently.
        :param n_classes: Number of classes.
                          For regression tasks set to `False` or `0`;
                          For classification tasks, provide integer >= 2.
        :return: Compiled `MRInet` model (based on `Keras`), ready to be trained.
        """
        if target_bias is not None:
            cprint(string=f"\nGiven target bias is {target_bias:.3f}\n", col="y")

        actfct = None if leaky_relu and not batch_norm else "relu"

        _check_n_classes(n_classes=n_classes)

        k_model = keras.Sequential(name=name)  # OR: Sequential([keras.layer.Conv3d(....), layer...])

        # 3D-Conv
        if batch_norm:
            k_model.add(keras.layers.BatchNormalization(input_shape=(*input_shape, 1)))
            k_model.add(keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), padding="SAME", activation=actfct))
        else:
            k_model.add(
                keras.layers.Conv3D(
                    filters=16,
                    kernel_size=(3, 3, 3),
                    padding="SAME",
                    activation=actfct,
                    input_shape=(*input_shape, 1),
                )
            )
            # auto-add batch:None, last: channels
        if leaky_relu:
            k_model.add(keras.layers.LeakyReLU(alpha=0.2))  # lrelu
        k_model.add(keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="SAME"))

        if batch_norm:
            k_model.add(keras.layers.BatchNormalization())
        k_model.add(keras.layers.Conv3D(filters=16, kernel_size=(3, 3, 3), padding="SAME", activation=actfct))
        if leaky_relu:
            k_model.add(keras.layers.LeakyReLU(alpha=0.2))
        k_model.add(keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="SAME"))

        if batch_norm:
            k_model.add(keras.layers.BatchNormalization())
        k_model.add(keras.layers.Conv3D(filters=32, kernel_size=(3, 3, 3), padding="SAME", activation=actfct))
        if leaky_relu:
            k_model.add(keras.layers.LeakyReLU(alpha=0.2))
        k_model.add(keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="SAME"))

        if batch_norm:
            k_model.add(keras.layers.BatchNormalization())
        k_model.add(keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding="SAME", activation=actfct))
        if leaky_relu:
            k_model.add(keras.layers.LeakyReLU(alpha=0.2))
        k_model.add(keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="SAME"))

        # 3D-Conv (1x1x1)
        if batch_norm:
            k_model.add(keras.layers.BatchNormalization())
        k_model.add(keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), padding="SAME", activation=actfct))
        if leaky_relu:
            k_model.add(keras.layers.LeakyReLU(alpha=0.2))

        k_model.add(keras.layers.MaxPool3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="SAME"))

        if batch_norm:
            k_model.add(keras.layers.BatchNormalization())

        # FC
        k_model.add(keras.layers.Flatten())
        k_model.add(keras.layers.Dropout(rate=0.5))
        k_model.add(keras.layers.Dense(units=64, activation=actfct))
        if leaky_relu:
            k_model.add(keras.layers.LeakyReLU(alpha=0.2))

        # Output
        if n_classes:
            k_model.add(
                keras.layers.Dense(
                    units=n_classes,
                    activation="softmax",  # in binary case. also: 'sigmoid'
                    use_bias=False,
                )
            )  # default: True

        else:
            k_model.add(
                keras.layers.Dense(
                    units=1,
                    activation="linear",
                    # add target bias (recommended: mean of target distribution)
                    use_bias=True,
                    bias_initializer=keras.initializers.Constant(value=target_bias) if target_bias else "zeros",
                )
            )

        # Compile
        k_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),  # ="adam",
            loss="mse",
            metrics=["accuracy"] if n_classes else ["mae"],
        )

        # Summary
        k_model.summary()

        return k_model

    @staticmethod
    def reference() -> str:
        """Get the reference for the `MRInet` model."""
        return "Hofmann et al. (2022). NeuroImage. https://doi.org/10.1016/j.neuroimage.2022.119504"

    @staticmethod
    def pretrained_models() -> type[PretrainedModelsMRInet]:
        """
        Return enum of pretrained `MRInet` models.

        !!! tip "Call the `show` method to get an overview of available pretrained models:"
            ```python
            MRInet.pretrained_models().show()
            ```
        """
        return PretrainedModelsMRInet


class SFCN(_ModelCreator):
    """
    `SFCN` model creator.

    `SFCN` is the fully convolutional neural network (`SFCN`) model by
    [Peng et al. (2021, *Medical Image Analysis*)](https://doi.org/10.1016/j.media.2020.101871)
    for age prediction from MRI data of the [`ukbiobank`](https://www.ukbiobank.ac.uk).

    The architecture won the first place in the brain-age competition `PAC 2019`.
    """

    @staticmethod
    @_experimental
    def create(
        name: str,
        n_classes: int,
        input_shape: tuple[int, int, int] = (160, 192, 160),
        learning_rate: float = 0.01,
        dropout: bool = True,
    ) -> keras.Sequential:
        """
        Create the fully convolutional neural network (`SFCN`) model.

        The `SFCN` was introduced in
        [Peng et al. (2021, *Medical Image Analysis*)](https://doi.org/10.1016/j.media.2020.101871).

        The original open-source implementation is done in `PyTorch` and can be found at:
        https://github.com/ha-ha-ha-han/UKBiobank_deep_pretrain

        For training:

        !!! quote "from [Peng et al. (2021, p.4)](https://doi.org/10.1016/j.media.2020.101871):"
            > "The L2 weight decay coefficient was 0.001.
            The batch size was 8.
            The learning rate for the SGD optimiser was initialized as 0.01,
            then multiplied by 0.3 every 30 epochs unless otherwise specified.
            The total epoch number is 130 for the 12,949 training subjects.
            The epoch number is adjusted accordingly for the experiments with
            smaller training sets so that the training steps are roughly the
            same."

        ???+ warning "Use this version of the `SFCN` model with caution"
            - The **implementation is still experimental**,
            is not tested properly yet, and might be updated in the future.
            - The authors used "Gaussian soft labels" (`sigma=1, mean=*true age*`) for their loss function.
            This is not implemented here yet,
            and might require additional adjustments of the `xai4mri.dataloader` module.

        :param name: Model name, which, for instance, could refer to the project it is applied for.
        :param input_shape: Shape of the input to the model.
                            This should be the shape of a single MRI.
        :param n_classes: Number of classes.
                          In [Peng et al. (2021)](https://doi.org/10.1016/j.media.2020.101871)
                          there were 40 age classes, representing 40 age strata.
        :param learning_rate: Learning rate which is used for the optimizer of the model.
        :param dropout: Use dropout or not.
        :return: Compiled `SFCN` model (based on `Keras`), ready to be trained.
        """
        conv_ctn = 0  # conv block counter
        _check_n_classes(n_classes=n_classes)

        def conv_block(
            _out_channel: int,
            max_pool: bool = True,
            kernel_size: int = 3,
            padding: str = "same",
            max_pool_stride: int = 2,
            in_shape: tuple[int, int, int] | None = None,
        ) -> keras.Sequential:
            """Define a convolutional block for SFCN."""
            c_block = keras.Sequential(name=f"conv3D_block_{conv_ctn}")

            conv_kwargs = {} if in_shape is None else {"input_shape": (*in_shape, 1)}
            c_block.add(keras.layers.Conv3D(_out_channel, kernel_size=kernel_size, padding=padding, **conv_kwargs))
            c_block.add(keras.layers.BatchNormalization())
            if max_pool:
                c_block.add(keras.layers.MaxPooling3D(pool_size=2, strides=max_pool_stride))
            c_block.add(keras.layers.ReLU())
            return c_block

        # Build the model
        k_model = keras.Sequential(name=name)

        # Feature extractor
        channel_number = (32, 64, 128, 256, 256, 64)
        n_layer = len(channel_number)

        for i in range(n_layer):
            out_channel = channel_number[i]
            if i < n_layer - 1:
                k_model.add(
                    conv_block(
                        out_channel,
                        max_pool=True,
                        kernel_size=3,
                        padding="same",
                        in_shape=input_shape if i == 0 else None,
                    )
                )
            else:
                k_model.add(conv_block(out_channel, max_pool=False, kernel_size=1, padding="valid"))
            conv_ctn += 1

        # Classifier [in: (bs, 5, 6, 5, 64)]
        avg_shape = [5, 6, 5]
        k_model.add(keras.layers.AveragePooling3D(pool_size=avg_shape))
        if dropout:
            k_model.add(keras.layers.Dropout(rate=0.5))

        k_model.add(keras.layers.Conv3D(filters=n_classes, kernel_size=1, padding="valid"))

        # Output
        k_model.add(keras.layers.Activation(activation="log_softmax", name="log_softmax"))

        # Compile
        # Define the learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=30,  # for the total number of 130 epochs in Peng et al. (2021)
            decay_rate=0.3,
            staircase=True,
        )
        k_model.compile(
            optimizer=keras.optimizers.SGD(  # SGD best in Peng et al. (2021)
                learning_rate=lr_schedule,
                weight_decay=keras.regularizers.l2(0.001),
            ),
            loss="kl_divergence",  # Peng et al. (2021) use K-L Divergence loss:
            # no one-hot encoding required: loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            # no log_softmax when using from_logits=True, however, KL-Div. loss expects log_softmax
            metrics=["accuracy"],  # ... if n_classes else ["mae"],
        )
        # ... to minimize a Kullback-Leibler divergence loss function between the predicted probability and a
        # Gaussian distribution (the mean is the true age,
        # and the distribution sigma is 1 year for UKB) for each training subject.
        # This soft-classification loss encourages the model to predict age as accurately as possible.

        # Summary
        k_model.summary()
        return k_model

    @staticmethod
    def reference() -> str:
        """Get the reference for the `SFCN` model."""
        return "Peng et al. (2021). Medical Image Analysis. https://doi.org/10.1016/j.media.2020.101871"

    @staticmethod
    def pretrained_models() -> None:
        """
        Return enum of pretrained `SFCN` models.

        !!! note
            There are no pretrained models available for `SFCN` yet.
        """
        cprint(string="There are no pretrained models available for SFCN (yet).", col="y")


class OutOfTheBoxModels(Enum):
    """
    Enum class for out-of-the-box models for MRI-based predictions.

    Call the `show` method to get an overview of available models:
    ```python
    OutOfTheBoxModels.show()
    ```

    !!! example "Get a model by selecting it"
        ```python
        mrinet_creator = OutOfTheBoxModels.MRINET.value
        mrinet = mrinet_creator.create(...)

        # Alternatively, get the model by string
        sfcn = OutOfTheBoxModels("sfcn").value.create(...)
        ```
    """

    MRINET: _ModelCreator = MRInet
    SFCN: _ModelCreator = SFCN

    @classmethod
    def _missing_(cls, value: str) -> OutOfTheBoxModels | None:
        """
        Get the missing value.

        This could be used also in the following way:
            value = "sfcn"
            OutOfTheBoxModels(value) == OutOfTheBoxModels.SFCN
            # Output: True
        """
        value = value.lower()
        for model in cls:
            if value == model.value.__name__.lower():
                return cls(model)
        return None

    def reference(self):
        """
        Get the reference for the model.

        !!! example "Get the reference for `MRInet`"
            ```python
            OutOfTheBoxModels.MRINET.reference()
            ```
        Ultimately, this is just a small wrapper avoiding calling the `value` attribute.
        """
        return self.value.reference()

    def pretrained_models(self):
        """
        Get the pretrained models for the model.

        !!! example "Get information about pretrained `MRInet` models"
            ```python
            OutOfTheBoxModels.MRINET.pretrained_models().show()
            ```
        Ultimately, this is just a small wrapper avoiding calling the `value` attribute.
        """
        return self.value.pretrained_models()

    @classmethod
    def default(cls):
        """Get the default model, which is `MRInet`."""
        return cls.MRINET

    @classmethod
    def show(cls):
        """Show available models."""
        return print(*[(model, model.reference()) for model in cls], sep="\n")


def get_model(model_type: OutOfTheBoxModels | str, **kwargs) -> keras.Sequential | None:
    """
    Get a freshly initiated out-of-the-box model.

    !!! example "Example of how to get an `MRInet` model"
        ```python
        mrinet = get_model(
                     model_type=OutOfTheBoxModels.MRINET,
                     name="MyMRInet",
                     n_classes=False,
                     input_shape=(91, 109, 91),
                     )

        # Alternatively, get the model by string
        sfcn = get_model(
                   model_type="sfcn",
                   name="MySFCN",
                   n_classes=40,
                   input_shape=(160, 192, 160),
                   )

        This is a wrapper for the `create` method of the model creator classes.
        ```
    :param model_type: model of type `OutOfTheBoxModels` or `string` (e.g., 'mrinet')
    :param kwargs: keyword arguments for model creation
    """
    if OutOfTheBoxModels(model_type) == OutOfTheBoxModels.MRINET:
        return MRInet.create(**kwargs)
    if OutOfTheBoxModels(model_type) == OutOfTheBoxModels.SFCN:
        return SFCN.create(**kwargs)
    return None


# %% Loaders of pretrained models o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@dataclass(frozen=True)
class _PretrainedModel(ABC):
    """
    Abstract class for pretrained models.

    :param name: name of the model
    :param url: URL to download the model
    """

    name: str
    url: str

    @abstractmethod
    def load_model(
        self,
        parent_folder: str | Path,
        custom_objects: T | None,
        compile_model: bool,
    ) -> keras.Sequential:
        """Enforce function that loads a pretrained Keras model."""
        ...


@_experimental
def _load_pretrained_mrinet_model(
    pretrained_model_type: type[_PretrainedMRInet],
    parent_folder: str | Path,
    custom_objects: T | None = None,
    compile_model: bool = False,
) -> keras.Sequential:
    """
    Load a pretrained MRInet Keras model.

    For details see [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504)

    The function retrieves the model from a server (URL) if it is not present on the local machine.

    :param parent_folder: path to parent folder of the model
    :param pretrained_model_type: the class of the pretrained MRInet model (e.g., 'PretrainedMRInetT1')
    :param custom_objects: custom objects to load (e.g., loss functions)
    :param compile_model: compile model or not
    """
    # Prepare the path to the model
    path_to_model = Path(parent_folder, pretrained_model_type.name).with_suffix(".h5")

    # Load model
    if not path_to_model.is_file():
        # TODO: remove warning after final update  # noqa: FIX002
        warnings.warn(message="Model download URLs are not final and  will be changed!", stacklevel=2)
        # Retrieve model from server if not present locally
        cprint(string="Downloading model from server...", col="b")
        path_to_model.parent.mkdir(parents=True, exist_ok=True)
        try:
            _, _ = urllib.request.urlretrieve(url=pretrained_model_type.url, filename=path_to_model)  # noqa: S310
        except urllib.error.HTTPError as e:
            cprint(string=f"HTTP Error: {e.code} - {pretrained_model_type.url} {e.reason}", col="r")

    return keras.models.load_model(path_to_model, custom_objects=custom_objects, compile=compile_model)


@dataclass(frozen=True)
class _PretrainedMRInet(_PretrainedModel):
    """Abstract class for pretrained MRInet models (3D-CNNs)."""

    @classmethod
    def load_model(
        cls, parent_folder: str | Path, custom_objects: T | None = None, compile_model: bool = False
    ) -> keras.Sequential:
        """
        Load a pretrained `MRInet` model.

        If the model is not present on the local machine, it will be downloaded from the server
        and saved in the provided parent folder.

        :param parent_folder: Path to parent folder of the model.
        :param custom_objects: Custom objects to load (e.g., loss functions).
        :param compile_model: Compile the model or not.
        :return: Pretrained `MRInet` model.
        """
        return _load_pretrained_mrinet_model(
            pretrained_model_type=cls,
            parent_folder=parent_folder,
            custom_objects=custom_objects,
            compile_model=compile_model,
        )


@dataclass(frozen=True)
class PretrainedMRInetT1(_PretrainedMRInet):
    """
    Pretrained `MRInet` model for `T1`-weighted images.

    This is a basemodel in the `T1` sub-ensemble as reported in
    [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).

    The training data stem from the [`LIFE Adult` study](https://doi.org/10.1186/s12889-015-1983-z),
    and were preprocessed using FreeSurfer (`brain.finalsurfs.mgz`), and
    subsequently pruned to `(198, 198, 198)` voxels (for more details,
    see [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).
    """

    name: str = "t1_model"
    url: str = "https://keeper.mpdl.mpg.de/f/3be6fed59b4948aca699/?dl=1"


@dataclass(frozen=True)
class PretrainedMRInetFLAIR(_PretrainedMRInet):
    """
    Pretrained `MRInet` model for `FLAIR` images.

    This is a basemodel in the `FLAIR` sub-ensemble as reported in
    [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).

    The training data stem from the [`LIFE Adult` study](https://doi.org/10.1186/s12889-015-1983-z),
    and were normalized and registered to the `T1w`-`FreeSurfer` file (`brain.finalsurfs.mgz`), and
    subsequently pruned to `(198, 198, 198)` voxels (for more details,
    see [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).
    """

    name: str = "flair_model"
    url: str = "https://keeper.mpdl.mpg.de/f/8481f5906f3d4192ab12/?dl=1"


@dataclass(frozen=True)
class PretrainedMRInetSWI(_PretrainedMRInet):
    """
    Pretrained `MRInet` model for `SWI`.

    This is a basemodel in the `SWI` sub-ensemble as reported in
    [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).

    The training data stem from the [`LIFE Adult` study](https://doi.org/10.1186/s12889-015-1983-z),
    and were normalized and registered to the `T1w`-`FreeSurfer` file (`brain.finalsurfs.mgz`), and
    subsequently pruned to `(198, 198, 198)` voxels (for more details,
    see [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).
    """

    name: str = "swi_model"
    url: str = "https://keeper.mpdl.mpg.de/f/f53d43b723274687b6e2/?dl=1"


class PretrainedModelsMRInet(Enum):
    """
    Enum class for pretrained `MRInet` models (`T1`, `FLAIR`, `SWI`).

    !!! tip "Call the `show` method to get an overview of available models"
        ```python
        PretrainedModelsMRInet.show()
        ```

    !!! example  "Get a model by selecting it"
        ```python
        t1_mrinet = PretrainedModelsMRInet.T1_MODEL.value.load_model(...)

        # Alternatively, get the model by string
        swi_mrinet = PretrainedModelsMRInet("swi_model").value.load_model(...)
        ```
    """

    T1_MODEL = PretrainedMRInetT1()
    FLAIR_MODEL = PretrainedMRInetFLAIR()
    SWI_MODEL = PretrainedMRInetSWI()

    @classmethod
    def _missing_(cls, value: str) -> PretrainedModelsMRInet | None:
        """Get the missing value."""
        # Check if value is found in lowercase with (e.g., "t1_model") or without "_model" (e.g., "t1")
        value = value.lower()
        for model in cls:
            if value == model.value.name or value == model.value.name.split("_model")[0]:
                return cls(model)
        return None

    @classmethod
    def show(cls):
        """Show available pretrained `MRInet` models."""
        return print(*[model.value for model in cls], sep="\n")


# _Hide as long there is only one type of pretrained models
class _PretrainedModels(Enum):
    """Enum class for pretrained models."""

    MRINET = PretrainedModelsMRInet  # currently, there is only one type with pretrained models
    # add more in the future

    @classmethod
    def _missing_(cls, value: str) -> _PretrainedModels | None:
        """Get the missing value."""
        value = value.lower()
        fit = []
        for model in cls:
            if value == model.name.lower():
                fit.append(model)
        if len(fit) == 1:
            return cls(fit.pop())
        if len(fit) > 1:
            cprint(string=f"Too many pretrained models fit the value '{value}': {fit} ", col="r")
        return None

    @classmethod
    def show(cls):
        """Show available models."""
        return print(*list(cls), sep="\n")


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
