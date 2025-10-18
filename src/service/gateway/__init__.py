"""Gateway package: modular components for IGTGateway orchestration.

This package contains lightweight, testable components used by the
public IGT gateway manager.
"""

from .manager import IGTGateway

__all__ = ["IGTGateway"]
