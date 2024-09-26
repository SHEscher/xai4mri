"""
Model analyzer using explainable AI (XAI) methods.

Functionality is built around the [`iNNvestigate`](https://github.com/albermax/innvestigate)
toolbox to analyze predictions of deep learning models.

    Author: Simon M. Hofmann
    Years: 2023-2024
"""

# %% Import
from __future__ import annotations

from typing import TYPE_CHECKING

import innvestigate
import numpy as np

if TYPE_CHECKING:
    from tensorflow import keras

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def analyze_model(
    model: keras.Model,
    ipt: np.ndarray,
    norm: bool,
    analyzer_type: str = "lrp.sequential_preset_a",
    neuron_selection: int | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Analyze the prediction of a model with respect to a given input.

    Produce an analyzer map ('heatmap') for a given model and input image.
    The heatmap indicates the relevance of each pixel w.r.t. the model's prediction.

    :param model: Deep learning model.
    :param ipt: Input image to model, shape: `[batch_size: = 1, x, y, z, channels: = 1]`.
    :param norm: True: normalize the computed analyzer map to [-1, 1].
    :param analyzer_type: Type of model analyzers [default: "lrp.sequential_preset_a" for ConvNets].
                          Check documentation of `iNNvestigate` for different types of analyzers.
    :param neuron_selection: Index of the model's output neuron [int], whose activity is to be analyzed;
                             Or take the 'max_activation' neuron [if `None`]
    :param kwargs: Additional keyword arguments for the `innvestigate.create_analyzer()` function.
    :return: The computed analyzer map.
    """
    # Create analyzer
    disable_model_checks = kwargs.pop("disable_model_checks", True)
    analyzer = innvestigate.create_analyzer(
        name=analyzer_type,
        model=model,
        disable_model_checks=disable_model_checks,
        neuron_selection_mode="index" if isinstance(neuron_selection, int) else "max_activation",
        **kwargs,
    )

    # Apply analyzer w.r.t. maximum activated output-neuron
    a = analyzer.analyze(ipt, neuron_selection=neuron_selection)

    if norm:
        # Normalize between [-1, 1]
        a /= np.max(np.abs(a))

    return a


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
