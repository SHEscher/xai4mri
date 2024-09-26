"""Init `dataloader` submodule of `xai4mri`."""

from .datasets import BaseDataSet
from .mri_dataloader import get_nifti, mgz2nifti, process_single_mri
from .mri_registration import get_mni_template, is_mni, register_to_mni
from .prune_image import PruneConfig, get_global_max_axes, reverse_pruning
from .transformation import GLOBAL_ORIENTATION_SPACE, apply_mask, clip_data, file_to_ref_orientation

__all__ = [
    "BaseDataSet",
    "PruneConfig",
    "reverse_pruning",
    "GLOBAL_ORIENTATION_SPACE",
]
