# Deep learning models for MRI prediction

`xai4mri` provides deep learning models for MRI-based prediction tasks.

You can find out-of-the-box models in the submodule `xai4mri.model.mrinets`.
The models are designed for 3D MRI data specifically (future support for 4D fMRI data is planned),
and can be used for both regression or classification tasks.

Moreover, there are pretrained models available for some model architectures.

Models are built on top of the [`TensorFlow` `Keras` API](https://www.tensorflow.org/guide/keras).
This is necessary to ensure compatibility with the [`iNNvestigate`](https://github.com/albermax/innvestigate) toolbox
for model interpretability.

## Choose a model architecture

Get an overview of the available deep learning model architectures for MRI prediction tasks:

```python
from xai4mri.model.mrinets import OutOfTheBoxModels

# List available model architectures
OutOfTheBoxModels.show()
```

The `xai4mri` toolbox is an offspring from the work in
[Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).
While several model architectures are provided, the most stable model type is `MRInet`.
which was introduced in this work.

To get a fresh, untrained model use the following:

```python
from xai4mri.model.mrinets import get_model, OutOfTheBoxModels, MRInet

MODEL_NAME = "MyMRInet"  # name your model, keep for later reference

# Create fresh model
model = MRInet.create(
                name="MyMRInet",
                n_classes=40,  # number of classes for classification tasks
                input_shape=(91, 91, 109),
                # other parameters are optional
)

# Alternatively, you can use the get_model function
model = get_model(
                model_type=OutOfTheBoxModels.MRINET,  # or pass a string "mrinet"
                name=MODEL_NAME,
                n_classes=False,  # False for regression tasks
                input_shape=(91, 91, 109),  # shape of one (pruned) MRI in your dataset after processing
                target_bias=None,  # in a regression task, this could be set to the mean of the target variable
                learning_rate=5e-4,  # learning rate for the optimizer
                leaky_relu=False,  # use leaky ReLU instead (but, currently interference with `iNNvestigate`)
                batch_norm=False,  # use batch normalization (only usefully for models trained on larger batches)
)
```

!!! note "Shape of the model input"
    The `input_shape` parameter should be the shape of one MRI in your dataset after processing (including pruning).

All models are `TensorFlow` / `Keras` models (`keras.Sequential`) and can be used as such.

## Train a model for MRI prediction

To train a model, we follow the standard procedure of `TensorFlow` / `Keras` models.
Using the `xai4mri` data loader, you can easily load your MRI data and train the model
(see [Create a data split for model training and evaluation](dataloading.md#create-a-data-split-for-model-training-and-evaluation)).

```python
from pathlib import Path

import tensorflow as tf

PATH_TO_MODEL: str = "PATH/TO/SAVE/MODEL/TO/{model_name}"
EPOCHS: int = 100

# Define checkpoint callbacks (see tf.Keras documentation for details)
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(
        Path(
            PATH_TO_MODEL.format(model_name=model.name),
            "ckpt",
            "cp-{epoch:03d}",
        )
    ),
    save_weights_only=True,
    monitor='val_loss',
    mode='min',  # 'min' for loss, 'max' for accuracy, OR 'auto'
    save_best_only=True,
    save_freq="epoch"
)

csv_logger_callback = tf.keras.callbacks.CSVLogger(
    filename=Path(
        PATH_TO_MODEL.format(model_name=model.name), "training_log.csv"
    ),
    separator=",",
)

# Train model
model.fit(
    train_data_gen,
    epochs=EPOCHS,
    validation_data=val_data_gen,
    callbacks=[model_checkpoint_callback, csv_logger_callback]
)
```

??? tip "Simplify model training with `mono_phase_model_training()`"
    To simplify the training process, you can use the `mono_phase_model_training()` function in `xai4mri.model`.

    ```python
    from xai4mri.model.mrinets import MRInet
    from xai4mri.model import mono_phase_model_training
    from xai4mri.dataloader.data import BaseDataSet

    # Define your dataset
    class MyDataset(BaseDataSet):
        ...

    mydata = MyDataset()

    # Create fresh model
    model = MRInet.create(name=MODEL_NAME, n_classes=False, input_shape=(91, 91, 109))

    # Train model
    trained_model = mono_phase_model_training(
                        model=model,
                        epochs=40,
                        data=mydata,
                        target="age",
                        model_parent_path="PATH/TO/SAVE/MODEL/",
                        split_dict=None,
                        callbacks=None,
    )
    ```


### Load a trained model

After training a model, you can load it from memory using the following code:

```python

# Create the same model architecture as it was used for model training
model = get_model(
    model_type=OutOfTheBoxModels.MRINET,
    name=MODEL_NAME,
    input_shape=(91, 91, 109),  # shape of one MRI
    n_classes=False,
    target_bias=None,
    learning_rate=5e-4,
    leaky_relu=False,
    batch_norm=False
)

# Load weights of the trained model (see `tf.keras` documentation for details)
latest = tf.train.latest_checkpoint(
    Path(PATH_TO_MODEL.format(model_name=model.name), "ckpt")
)
model.load_weights(latest)
```

## Get a pretrained model

`xai4mri` ships with pretrained deep learning models for MRI predictions.

To check which model architectures have pretrained models available, use the following code:

```python
from xai4mri.model.mrinets import OutOfTheBoxModels

OutOfTheBoxModels.MRINET.pretrained_models().show()
```

!!! note "Available pretrained models"
    Currently, there are only pretrained models for `MRInet`
    from [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).
    There are more pretrained models planned for future releases of `xai4mri`.

To load a pretrained model, use the following code:

```python
from xai4mri.model.mrinets import PretrainedMRInetFLAIR

# Load pretrained model
model = PretrainedMRInetFLAIR.load_model(parent_folder="PATH/TO/STORE/MODEL/")
```

!!! tip "Loading pretrained models"
    `load_model()` will first check the `parent_folder` for the model.
    If the model is not found, it will download the model from the `xai4mri` repository.
    Hence, you can use the same code to download and load a pretrained model from memory.
    However, note, in case you apply the model to a new prediction task,
    the transferred model should be saved at a different location
    (see the `tf.keras.callback` approach in
    [Train a model for MRI prediction](#train-a-model-for-mri-prediction)).

See [Model transfer](transfer.md) for more details how to transfer a pretrained model to a new dataset.
