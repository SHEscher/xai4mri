# Transfer pretrained models to new MRI datasets

## Motivation

One major motivator to develop `xai4mri` is to enable MRI analysis using deep learning models
on relatively small datasets.

To this end, **transfer learning** is a promising technique to leverage models,
which have been pretrained on large MRI datasets
(N >> 2,000, such as the `ukbiobank`, `LIFE Adult study`, `NAKO`, etc.),
and apply them to new datasets with fewer samples.

!!! tip "Foundation models"
    One major issue with transfer learning is that deep learning models often do not generalize
    well to new datasets using different MR scanners and recording sequences.
    To overcome this, the imaging community should develop foundation models.
    That is, large models that are trained on diverse MRI datasets.

## Application

!!! warning inline end "Under construction"
    Note that functions for transfer learning are not fully implemented yet and still in their infancy.

    Already existing functionality has been developed using the
    [`MRInet`](models.md#get-a-pretrained-model) architecture.
    The application to other models architectures is not yet tested.

`xai4mri` has some initial functions to transfer pretrained models to new datasets.


### Analyze a dataset and a candidate model

The training of deep learning models can be a time-consuming process,
and involves a lot of trial-and-errors to find the best hyperparameters,
and training strategies.

Building upon heuristics, experiences, and empirical evidence from the literature,
there are the following goals for `xai4mri`:

1. to estimate the success of transfer learning to new and small MRI datasets, and
2. to suggest training strategies for model transfer

To analyze models and datasets, with the goal to estimate training parameters for transfer learning,
use the following function:

```python
from xai4mri.model.transfer import analyse_model_and_data
from xai4mri.dataloader.datasets import BaseDataSet

# Get a promising pretrained model
pretrained_model: tf.keras.Sequential = ...

# Define your dataset
class MyDataset(BaseDataSet):
    ...

mydata = MyDataset()

# Now analyze the model and the dataset
analyse_model_and_data(model=pretrained_model, data=mydata)
```

!!! warning "Not implemented yet"
    This is, unfortunately, not implemented yet.
    But since it is a promising feature, it is mentioned here already.
    Feel invited to contribute to this feature, see [Contributing](contributing.md).

### Reconstruct an existing model

To apply a pretrained model to a new dataset might require reconstructing the model.
That is, to adapt its output layer to the requirements of the new prediction task.

For this, use the following function:

```python
from xai4mri.model.transfer import adapt_model

# Get a pretrained model
pretrained_model: tf.keras.Sequential = ...
# for instance, this could be a model trained to predict 40 classes in a large MRI dataset

adapted_model = adapt_model(
    model=pretrained_model,
    learning_rate=0.01,
    n_classes=2,  # number of classes in the new dataset
)
```

### Run transfer learning

!!! tip inline end "Mono-phase model training"
    The function `mono_phase_model_training()` in `xai4mri.model.transfer`
    can also be used to train a new (i.e., untrained) model on a dataset
    (see [Train a model for MRI prediction](models.md#train-a-model-for-mri-prediction)).

Currently, there are two experimental ways to run transfer learning:
*Mono-phase* and *dual-phase* training.
The former classically trains a model on a dataset from the beginning to the end.
The latter trains a model in two phases:
first, all layers of the model are trained.
Then, the first layers get frozen, and only the last layers get fine-tuned.

Since these functions are still in an experimental phase,
refer to the doc-strings in the **Code** section:
See `mono_phase_model_training()` and `dual_phase_model_training ()`
in [`xai4mri.model.transfer`](reference/model/transfer.md) for more information.
