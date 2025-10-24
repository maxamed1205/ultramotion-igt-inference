# src/core/inference/MobileSAM/mobile_sam/__init__.py
"""MobileSAM local package wrapper (modeling, utils).

This repository keeps a vendored copy of MobileSAM under
`src/core/inference/MobileSAM/mobile_sam/`. Upstream code and tests import
`mobile_sam.modeling`, but the vendored package exposes the subpackage as
`mobile_sam.Modeling` (capital M). To be compatible with imports that expect
lowercase `mobile_sam.modeling`, create a module alias on import.
"""

from __future__ import annotations

from importlib import import_module
import sys

# Expose `mobile_sam.modeling` as an alias to the vendored `Modeling` package.
try:
	# Import the local Modeling package (relative import) and register it under
	# the lowercase name so `import mobile_sam.modeling` works.
	_modeling = import_module(".Modeling", package=__name__)
	sys.modules[__name__ + ".modeling"] = _modeling
except Exception:
	# Best-effort: if aliasing fails, let normal import errors bubble up later.
	pass

__all__ = ["modeling"]