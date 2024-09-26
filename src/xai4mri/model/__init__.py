"""
Init `model` submodule of `xai4mri`.

    Author: Simon M. Hofmann
    Years: 2023-2024
"""

from .interpreter import analyze_model
from .mrinets import MRInet, OutOfTheBoxModels, get_model
from .transfer import mono_phase_model_training

# TODO: when finalised, do: from .transfer import ...  # noqa: FIX002
# TODO: when finalised, do: from .mrinets import [Pretrained, SFCN] # noqa: FIX002

__all__ = [
    "analyze_model",
    "get_model",
    "mono_phase_model_training",
    "OutOfTheBoxModels",
]
