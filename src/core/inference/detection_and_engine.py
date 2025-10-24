"""Compatibility façade — legacy import redirection

This file preserves the legacy import path `core.inference.detection_and_engine`.
New code should import from `core.inference.engine.*` modules. The functions
are re-exported here so existing imports in the codebase continue to work.
"""

from core.inference.engine.model_loader import initialize_models
from core.inference.engine.inference_dfine import run_detection
from core.inference.engine.inference_sam import run_segmentation
from core.inference.engine.postprocess import compute_mask_weights
from core.inference.engine.orchestrator import (
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
