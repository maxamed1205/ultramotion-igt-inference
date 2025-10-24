"""Engine subpackage for inference components.

This package contains modular components extracted from
`core.inference.detection_and_engine` to improve clarity and testability.

Public API: import the functions from their modules directly. This
file intentionally keeps things minimal; users should import
`core.inference.engine.<module>` symbols.
"""

__all__ = [
    "gpu_optim",
    "model_loader",
    "inference_dfine",
    "inference_sam",
    "postprocess",
    "orchestrator",
]
