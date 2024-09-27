# Explaining MRI-based predictions of deep learning models

Deep learning models have become an essential tool in image analysis, including
medical image analysis, and MRI research in general. However, these models are
often considered as *black boxes*, as they are built upon huge parameter spaces
and non-linear processing steps.
These make their decision-making process hard to explain.

New methods have been developed to interpret the predictions of deep learning models
and to provide insights into their decision-making process. These methods are summerized
under the term *explainable artificial intelligence* (XAI).

The `xai4mri` package offers a set of tools to apply XAI methods to analyze
the predictions of deep learning models for MRI-based tasks.

### Apply XAI analyzer and interpret model predictions

The `model.interpreter` submodule of `xai4mri` is built around the
[`iNNvestigate`](https://github.com/albermax/innvestigate) package.
Moreover, there is a strong focus on
 [*Layer-wise Relevance Propagation* (LRP)](https://doi.org/10.1038/s41467-019-08987-4),
since it has shown to overcome some limitations of other *post-hoc* XAI methods.

The API of the `model.interpreter` submodule is designed to be straightforward and intuitive,
and hides some of the complexity behind these sophisticated methods.

To analyze the predictions of a deep learning model, use the `analyze_model()` to generate
a relevance map for a given input MRI. The relevance map highlights the regions of the MRI
that are most relevant for the model's prediction. Then, use the `plot_heatmap()` function
to visualize the relevance map:

```python
from xai4mri.model.interpreter import analyze_model
from xai4mri.visualizer import plot_heatmap

# Load your trained model
model: tf.keras.Sequential = ...

# Get your research data
split_dict = mydata.load_split_dict(split_dict_path="PATH/TO/SAVED/SPLIT")
# ideally, use the same data split as during training

_, train_data_gen, val_data_gen, test_data_gen = mydata.create_data_split(
    target="age",  # the prediction target
    batch_size=1,  # here we analyze one MRI at a time
    split_dict=split_dict,
)

# Now iterate over images in the test set
for input_img, y in test_data_gen:
    # Compute XAI-based relevance map
    relevance_map = analyze_model(
        model=model,
        ipt=input_img,  # a single MRI
        norm=True,  # normalize the relevance object
        neuron_selection=None,  # relevant for classification models
    )

    # Get model prediction
    prediction = model.predict(input_img)[0][0]
    print(f"prediction = {prediction:.2f} | ground truth = {y[0]:.2f}")

    # Plot (mid) slice of each axis
    plot_heatmap(
        ipt=input_img,
        analyser_obj=relevance_map,
        mode="triplet",  # plot one slice of each axis, see doc-string for other options
        # slice_idx=(15, 60, 45),  # uncomment: specify which slices to plot, otherwise take mid-slices
        fig_name=f"Relevance map of {y[0]:.0f}-years old, predicted as {prediction:.1f} from "
                 f"{mydata.mri_sequence.upper()}-MRIs",
    )
```

!!! note "Using test set data for explaining model predictions"
    Similar to evaluating the performance of a model,
    it is recommended to use the test set data for explaining model predictions.
    You can consider using the validation set as well. However, training set data should be avoided.
    In the end, this depends on the contex of course
    (e.g., one could be interested in analyzing the training process with the help of XAI methods).

## Reverse pruning: Bringing model input and relevance maps back to the NIfTI format

!!! inline end tip "Reverse pruning"
    *Reverse pruning* is useful to compare relevance maps with other statistical maps or atlases in form of NIfTIs.

If MR images have been pruned for more efficient model training
(see [Implement your own dataset class](dataloading.md#implement-your-own-dataset-class)),
the image pruning can be reversed in hindsight.
This can be done both for the MRI (model input) and for the computed relevance maps.

To reverse the pruning, use a combination of `reverse_pruning()` in the `xai4mri.dataloader.prune_image` submodule,
and `get_unpruned_mri()` as a method of the dataset class.
See the following for how this is done for the model input image and the relevance map:

```python
from xai4mri.dataloader.prune_image import reverse_pruning

# Iterate over images in the test set
for input_img, y in test_data_gen:  # here, a batch size of 1 is assumed
    # Do something with input_img and y, create a relevance map ...
    relevance_map = analyze_model(model=model, ipt=input_img, ...)

    # Get current SID in the test set
    current_sid = mydata.sid_list[test_data_gen.current_indices][0]

    # Get the original MRI of the SID (as nib.Nifti1Image)
    org_img_nii = mydata.get_unpruned_mri(sid=current_sid)

    # Reverse pruning for the input image
    input_img_nii = reverse_pruning(
        original_mri=org_img_nii,  # alternatively, an np.ndarray can be passed
        pruned_mri=input_img.squeeze(),  # (1, x, y, z, 1) -> (x, y, z)
        pruned_stats_map=None
    )
    # if np.ndarray is passed, then reverse_pruning will return a np.ndarray of the original MRI

    # Reverse pruning for the relevance map
    relevance_map_nii = reverse_pruning(
        original_mri=org_img_nii,  # reverse pruning for heatmap
        pruned_mri=input_img.squeeze(),
        pruned_stats_map=relevance_map.squeeze()  # ← this must be given here
    )
```

Now, both the model input image and the relevance map can be analyzed and plotted with other packages, which
can handle the data of type `nibabel.Nifti1Image`.

??? example "Exploring relevance maps outside of `xai4mri`"
    After reverse pruning, and having the relevance map in the NIfTI format,
    one can use, for instance, the [`nilearn`](https://nilearn.github.io/stable/index.html) package
    to plot the relevance map on top of the MRI:

    ```python
    from nilearn import plotting

    plotting.plot_stat_map(stat_map_img=relevance_map_nii, bg_img=input_img_nii, ...).
    ```

    Or just save the NIfTI files to disk and use other MRI software for further analysis.
