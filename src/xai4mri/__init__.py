"""
Init of the `xai4mri` package.

This package provides tools for explainable AI (XAI) in MRI research using deep learning.

    Author: Simon M. Hofmann
    Years: 2023-2024
"""

__version__ = "0.0.1"
__author__ = "Simon M. Hofmann"

from . import dataloader, model, utils, visualizer

__all__ = [
    "dataloader",
    "model",
    "visualizer",
]
