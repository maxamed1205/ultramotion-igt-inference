"""State machine for visibility decisions."""

from .visibility_fsm import (
    evaluate_visibility,
    update_state_machine,
    is_frame_valid_for_gpu,
)

__all__ = ["evaluate_visibility", "update_state_machine", "is_frame_valid_for_gpu"]
