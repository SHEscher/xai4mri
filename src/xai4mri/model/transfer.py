"""
Scripts for transfer learning of deep learning models for MRI-based predictions.

The goal is to transfer models such as
the 3D-CNNs reported in [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504)
to new and smaller datasets and other prediction tasks.

    Author: Simon M. Hofmann
    Years: 2023

!!! note "This module is experimental and still in development."
"""

# %% Import
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tensorflow import keras

from ..utils import _experimental, cprint

if TYPE_CHECKING:
    from ..dataloader.datasets import BaseDataSet

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# TODO: What do we want:  # noqa: FIX002
#  1) freeze weights at different layers
#  2) estimate of how much data is needed, then suggest data split accordingly
#  3) allow for multiple regression
#  4) ...
#  Check for some tips: https://www.tensorflow.org/tutorials/images/transfer_learning


@_experimental
def adapt_model(
    model: keras.Sequential,
    learning_rate: float = 5e-4,
    target_bias: float | None = None,
    n_classes: bool | int = False,
) -> keras.Sequential:
    """
    Adapt a pretrained model to a new dataset and/or task.

    This function adapts the output layer of a pretrained model to a new dataset and/or task
    by replacing the output layer.
    The model is recompiled with a new learning rate and loss function.

    !!! warning "This function is experimental and still in development."

    :param model: Pretrained `Keras` model.
    :param learning_rate: Learning rate for the optimizer.
    :param target_bias: Model output bias.
                        For classification tasks with this can be left blank [`None`].
                        For regression tasks, it is recommended to set this bias to the average of the
                        prediction target distribution in the dataset.
    :param n_classes: Number of classes.
                      For regression tasks set to `False` or `0`;
                      For classification tasks, provide integer >= 2.
    :return: The adapted model.
    """
    if isinstance(n_classes, int) and n_classes == 1:
        msg = "For a regression model, choose n_classes = 0, for a classification task n_classes >= 2."
        raise ValueError(msg)
    n_outputs = n_classes if n_classes else 1

    if model.layers[-1].units != n_outputs:
        model.pop()
        model.add(
            keras.layers.Dense(
                units=n_outputs,
                activation="softmax" if n_classes else None,
                use_bias=not n_classes,
                bias_initializer=keras.initializers.Constant(target_bias) if target_bias else "zeros",
            )
        )

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),  # ="adam",
        loss="mse",
        metrics=["accuracy"] if n_classes else ["mae"],
    )

    # Summary
    model.summary()

    return model


# TODO: Implement  # noqa: FIX002
def analyse_model_and_data(model: keras.Sequential, data: BaseDataSet, target: str):  # noqa: ARG001
    """
    Analyze model and data to estimate training parameters for transfer learning.

    !!! warning "This function is not implemented yet."

    :param model: Pre-trained `Keras` model.
    :param data: Dataset, which should be used for model fine-tuning.
    :param target: Variable in the dataset, which is supposed to be predicted.
    :return: Recommended number of training epochs, number of training samples, freeze weights, ...
    """
    msg = "This function is not implemented yet."
    raise NotImplementedError(msg)


def mono_phase_model_training(
    model: keras.Sequential,
    epochs: int,
    data: BaseDataSet,
    target: str,
    model_parent_path: str | Path,
    split_dict: dict | None = None,
    callbacks: list[keras.callbacks.Callback] | None = None,
    **kwargs,
) -> keras.Sequential | None:
    """
    Train / finetune all model weights at once.

    This simply trains the model on the provided dataset for the given number of epochs.

    !!! note "When used for transfer learning"
        This is a naive approach to transfer learning, and can lead to issues such as catastrophic forgetting.

    :param model: Compiled `Keras` model to be trained on the provided dataset.
    :param epochs: Number of training epochs.
    :param data: Dataset for training and evaluation. This must be a subclass of `BaseDataSet`.
    :param target: Variable to be predicted. This must be in the 'study_table` of the dataset (`data').
    :param model_parent_path: The path to the parent folder of the given model, where the model will be saved.
    :param split_dict: Data split dictionary for training, validation, and test data.
    :param callbacks: A list of `Keras`'s `callbacks` (except of `ModelCheckpoint`).
    :param kwargs: Additional keyword arguments for `data.create_data_split()`.
    :return: Trained model.
    """
    # Set path to model and split dictionary
    path_to_model = Path(model_parent_path) / model.name
    path_to_checkpoints = path_to_model / "checkpoints"
    path_to_split_dict = path_to_model / f"{model.name}_split_dict"

    # Check if model has been trained already
    if list(path_to_checkpoints.glob("*")):
        cprint(string="Model has been trained already. Skipping training.", col="y")
        return None

    # Create data splits
    split_dict, train_data_gen, val_data_gen, test_data_gen = data.create_data_split(
        target=target,
        batch_size=kwargs.pop("batch_size", 1),
        split_ratio=kwargs.pop("split_ratio", (0.8, 0.1, 0.1)),
        split_dict=split_dict,
    )

    # Define callbacks
    callbacks = [] if callbacks is None else callbacks
    for c in callbacks:
        if isinstance(c, keras.callbacks.ModelCheckpoint):
            break
    else:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=path_to_checkpoints / "cp-{epoch:04d}.ckpt",
                save_weights_only=True,  # TODO: revisit  # noqa: FIX002
                monitor="val_loss",
                mode="auto",
                save_best_only=True,  # TODO: revisit  # noqa: FIX002
                save_freq="epoch",
            )
        )

    # Train model
    model.fit(
        x=train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
        callbacks=callbacks,
    )

    # Save split dictionary and model
    data.save_split_dict(save_path=path_to_split_dict)
    model.save(path_to_model / f"{model.name}.h5")

    # Model evaluation
    model.evaluate(test_data_gen)

    return model


@_experimental
def dual_phase_model_training(
    model: keras.Sequential,
    epochs: tuple[int, int],
    data: BaseDataSet,
    target: str,
    model_parent_path: str | Path,
    split_dict: dict | None = None,
    callbacks: list[keras.callbacks.Callback] | None = None,
    **kwargs,
) -> keras.Sequential | None:
    """
    Train / finetune a model in two phases.

    First, train all layers of the model.
    Then, freeze the first layers, and only finetune the last layers.

    !!! warning "This function is experimental and is still in development."

    :param model: Compiled `Keras` model to be trained on the provided dataset.
    :param epochs: Number of training epochs.
    :param data: Dataset for training and evaluation.
    :param target: Variable to be predicted.
    :param model_parent_path: The path to the parent folder of the given model, where the model will be saved.
    :param split_dict: Data split dictionary for training, validation, and test data.
    :param callbacks: A list of `Keras`'s `callbacks` (except of `ModelCheckpoint`).
    :param kwargs: Additional keyword arguments for `data.create_data_split()`.
    :return: trained model
    """
    # Set path to model and split dictionary
    path_to_model = Path(model_parent_path) / model.name
    path_to_checkpoints = path_to_model / "checkpoints"
    path_to_split_dict = path_to_model / f"{model.name}_split_dict"

    # Check if model has been trained already
    if list(path_to_checkpoints.glob("*")):
        cprint(string="Model has been trained already. Skipping training.", col="y")
        return None

    # TODO: Create data splits  # noqa: FIX002
    #   This should be done in a separate function
    split_dict, train_data_gen, val_data_gen, test_data_gen = data.create_data_split(
        target=target,
        batch_size=kwargs.pop("batch_size", 1),
        split_ratio=kwargs.pop("split_ratio", (0.8, 0.1, 0.1)),
        split_dict=split_dict,
    )

    # Define callbacks
    callbacks = [] if callbacks is None else callbacks
    # Add model checkpoint
    for c in callbacks:
        if isinstance(c, keras.callbacks.ModelCheckpoint):
            break
    else:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=path_to_checkpoints / "cp-{epoch:04d}.ckpt",
                save_weights_only=True,  # TODO: revisit  # noqa: FIX002
                monitor="val_loss",
                mode="auto",
                save_best_only=True,  # TODO: revisit  # noqa: FIX002
                save_freq="epoch",
            )
        )
    # Add early stopping
    for c in callbacks:
        if isinstance(c, keras.callbacks.EarlyStopping):
            break
    else:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="auto",
                min_delta=0.001,
                patience=10,
                verbose=1,
            )
        )

    # First training loop over all layers
    model.fit(
        x=train_data_gen,
        epochs=epochs[0],
        validation_data=val_data_gen,
        callbacks=callbacks,
    )

    # Fine-tuning of later layers while freezing earlier layers
    # Freeze all but fully connected layers
    for layer in model.layers:
        if not isinstance(layer, keras.layers.Dense):
            layer.trainable = False
    # TODO: revisit this later, since this is a naive approach & has specific architectures in mind  # noqa: FIX002

    # Recompile the model after making any changes to the `trainable` attribute of any inner layer,
    # so that changes are taken into account
    model.compile()

    # Fine-tuning
    model.fit(
        x=train_data_gen,
        epochs=epochs[1],
        validation_data=val_data_gen,
        callbacks=callbacks,
    )

    # Save split dictionary and model
    data.save_split_dict(save_path=path_to_split_dict)
    model.save(path_to_model / f"{model.name}.h5")

    # Model evaluation
    model.evaluate(test_data_gen)

    return model


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
