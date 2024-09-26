# Loading MRI datasets

MRI datasets can be huge.
`xai4mri` provides a simple way to load MRI datasets in a memory-efficient way.
Moreover, `xai4mri` prepares the data for training and evaluation of deep learning models.

## Prepare your research data

Your study data must be prepared to match the schema of the `xai4mri` package.

It is recommended to use the [BIDS](https://bids.neuroimaging.io) format for structuring MRI data and
derivatives, as well as to provide a corresponding study table (`*.tsv`, `*.csv`).
However, the functions in `xai4mri.dataloader.datasets` allow for other data structures as well.

### MRI data

It is expected that the background in MRI data is set to zero.
Statistical maps with negative and positive values are possible when the background remains set to zero.

In the case, multiple MRI sequences and/or derivatives (including statistical maps) are present, but not for
all subjects (`sid`), create separate dataset classes (see [below](#implement-your-own-dataset-class)) for each MRI sequence and/or derivative with their own
corresponding study table.

!!! question inline end "Why using `sid`?"
    The `sid` column will become the index of a
    [`pandas.DataFrame`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html),
    which is passed across the `xai4mri` package.

### Study table

Importantly, subject IDs (`sid`) in the study table must correspond to existing MRI data.
The first column of the study table must contain subject ID's and must be called '`sid`'.

See the example below:

??? example "Example study table"

    | sid    | age | condition |
    |--------|-----|-----------|
    | sub-01 | 25  | control   |
    | sub-02 | 30  | patient   |
    | sub-03 | 35  | control   |
    | ...    | ... | ...       |
    | sub-99 | 45  | patient   |


## Implement your own dataset class

Inherit from the `BaseDataSet` to create a new dataset class, as shown in the example below:

```python
from pathlib import Path

from xai4mri.dataloader import BaseDataSet


class MyStudy(BaseDataSet):
    def __init__(self):
        super().__init__(
            study_table_or_path="PATH/TO/STUDY_TABLE.csv",  # or `*.tsv` OR table as `pd.DataFrame`
            project_id="MyStudy",  # this can be shared across multiple dataset classes
            mri_sequence="t1w",  # this should be unique for each dataset class
            cache_dir="PATH/WHERE/TO/CACHE/PROCESSED/DATA",
            load_mode="full_array",  # use "file_path" for very large datasets (N>>2,500)
            # Optional kwargs for how to load/process MRIs
            norm=False,  # Normalize MRI data
            prune_mode="max",  # Pruning means to minimize the background in the MRIs.
        )

    # This class method must be implemented
    def mri_path_constructor(self, sid: str) -> str | Path:
        """
        Construct the path to the MRI of a given subject for MyStudy.

        MRI data will be loaded using nibabel from given paths.

        :param sid: subject ID
        :return: absolute file path to MRI of the subject with the given ID
        """
        return Path("PATH/TO/MRI", sid, f"MNI/2mm/{self.mri_sequence.upper()}w_brain.nii.gz")
```

??? note "About **kwargs in `BaseDataSet.__init__()`"
    Find details to addtional `kwargs` in docs of `_load_data_as_full_array()` or `_load_data_as_file_paths()` in
    `xai4mri.dataloader.mri_dataloader`.

`BaseDataSet` provides several attributes and methods, see details in [API reference](reference/dataloader/datasets.md)

???+ note "MRI pruning for model efficiency"
    The `prune_mode` parameter is used to minimize the background in MRIs.
    This is useful for training deep learning models, since the models can become substantially smaller,
    avoiding redundant parameters.
    Set `prune_mode` to `None` if this is not desired.
    In how far MRIs are pruned can be adjusted with the `PruneConfig` object:

    ```python
    from xai4mri.dataloader.prune_image import PruneConfig

    # Check the defeault size of the "largest brain"
    print(PruneConfig.largest_brain_max_axes)

    # Adapt the size of the "largest brain" to your dataset
    PruneConfig.largest_brain_max_axes = (185, 185, 220)  # always use 1mm isotropic resolution here
    # Note that xai4mri uses the 'LIA' orientation.
    ```
    Image pruning will adhere to the settings in `PruneConfig.largest_brain_max_axes`.
    During pruning `xai4mri` automatically adjusts the axes lengths to the resolution of the given MRI dataset.
    To reverse this process, see [Reverse pruning](interpretation.md#reverse-pruning-bringing-model-input-and-relevance-maps-back-to-the-nifti-format)

    If you are note sure, which values to use for your case, you can use the default values,
    or run the `get_brain_axes_length()` in the `xai4mri.dataloader.prune_image` submodule over your MRI dataset.
    Then, choose the largest values you found such that all brains will get pruned
    without cutting off any brain voxels.

### Instantiate your dataset class

Use your project-specific dataset class to process and load data:

```python
mydata = MyStudy()
```

Processing data might require some time.
To get an estimate of cache storage and processing time, use the following:

```python
mydata.get_size_of_prospective_mri_set()
```

Then get the data.
If the data has not been processed yet, this will take some time.
After that, the whole data set can be loaded within seconds.

```python
volume_data, sid_list = mydata.get_data(**kwargs)
```

??? note "One more time about **kwargs"
    As mentioned above, `**kwargs` are passed to the `_load_data_as_file_paths` or
    `_load_data_as_full_array` in `xai4mri.Dataloader.mri_dataloader` method.

    If values that were set at the implementation `MyStudy()` are not desired,
    they can be passed to `get_data()`;
    e.g., getting data with normalization would be archived with
    `mydata.get_data(norm=True)`, this overwrites the initially set `norm=False` (see above).

## Create a data split for model training and evaluation

For the training of deep learning models, we need to prepare the data split:

```python
split_dict, train_data_gen, val_data_gen, test_data_gen = mydata.create_data_split(
    target="age",
    batch_size=4,
    split_ratio=(0.8, 0.1, 0.1),
    split_dict=None
)
```

!!! note
    The `target` variable must be a column in the study table.

The returned data generators can be used directly for the training of deep learning models
based on `TensorFlow` / `Keras`.

Also, ideally use a small batch size (`batch_size`) when GPUs are used,
since their memory is easily exhausted with relatively big MRIs as model input.

### Keep track of data splits

`split_dict` can be used when specific subject ID's shall reside in specific splits.
For instance, if you are interested in model predictions for specific subjects, put their ID's (SID's) in
the test set:

!!! example
    ```python
    split_dict = {
        "train": ["sid_45", "sid_33", ...],
        "validation": ["sid_29", "sid_11", ...],
        "test": ["sid_1", "sid_99", ...]
    }
    ```

If `split_dict` is provided to the function above, the `split_ratio` is ignored. Therefore, choose the
respective lists of SID's (*train*, *validation*, *test*) such that a desired ratio is achieved.

#### Save data splits

If a split dictionary should be saved, use the following:

```python
# Save latest data split
save_path = mydata.save_split_dict()  # this outputs the path to the saved split dictionary

# Alternatively, save a data split to a specified path
save_path = "PATH/TO/SAVE/SPLIT/TO"  # Optional: define your own path, otherwise None for the default path
mydata.save_split_dict(split_dict=split_dict, save_path=save_path)
```

!!! tip "Keep track of your data splits"
    For later reproducibility, but also model interpretation,
    it is essential that you know which subject data was used for training and evaluation.
    Usually, the XAI-based interpretation of model predictions is done on the test set.


#### Load data splits

Loading a split dictionary is done with:

```python
split_dict = mydata.load_split_dict(split_dict_path=save_path)
```
