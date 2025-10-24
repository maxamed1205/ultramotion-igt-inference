"""Unified inference package entrypoint.

This module re-exports the main inference functions from the refactored
`core.inference.engine` subpackage via the compatibility façade
`core.inference.detection_and_engine`.
"""

from .detection_and_engine import (
    initialize_models,
    run_detection,
    run_segmentation,
    compute_mask_weights,
    prepare_inference_inputs,
    run_inference,
    fuse_outputs,
    process_inference_once,
)

__all__ = [
    "initialize_models",
    "run_detection",
    "run_segmentation",
    "compute_mask_weights",
    "prepare_inference_inputs",
    "run_inference",
    "fuse_outputs",
    "process_inference_once",
]
