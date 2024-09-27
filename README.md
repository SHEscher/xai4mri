# xai4mri

*Explainable A.I. for MRI research using deep learning.*

![xai4mri logo](xai4mri.svg)

![Last update](https://img.shields.io/badge/last_update-Sep_27,_2024-green)
![Last update](https://img.shields.io/badge/version-v.0.0.1-blue)

## What is `xai4mri`

`xai4mri` is designed for advanced MRI analysis combining deep learning with explainable A.I. (XAI).
It offers the following key functionalities:

- **Data Integration**: Effortlessly import new MRI datasets and apply the models to generate accurate predictions.
- **Model Loading**: Load (pretrained) 3D-convolutional neural network models tailored for MRI predictions.
- **Interpretation Tools**: Utilize analyzer tools,
such as [Layer-wise Relevance Propagation (LRP)](https://doi.org/10.1038/s41467-019-08987-4),
to interpret model predictions through intuitive heatmaps.

With `xai4mri`, you can complement your MRI analysis pipeline, ensuring precise predictions and
insightful interpretations.

## Quick-start

```shell
pip install -U xai4mri
```

Get started with `xai4mri` in Python:

```python
import xai4mri as xai
```

Visit the [**documentation**](https://shescher.github.io/xai4mri/overview), for detailed information.

## Citation

When using `xai4mri`, please cite the following papers:
[`toolbox paper in prep`] and [Hofmann et al. (2022, *NeuroImage*)](https://doi.org/10.1016/j.neuroimage.2022.119504).
