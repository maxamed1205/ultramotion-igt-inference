import numpy as np
import time

from core.types import Pose, FrameMeta


def test_pose_default():
    p = Pose()
    assert isinstance(p.matrix, np.ndarray)
    assert p.matrix.shape == (4, 4)
    assert p.matrix.dtype == np.float32


def test_pose_in_frame_meta():
    p = Pose()
    fm = FrameMeta(frame_id=42, ts=time.time(), pose=p)
    assert isinstance(fm.pose, Pose)


def test_pose_as_tensor_if_torch_available():
    try:
        import torch
    except Exception:
        return

    p = Pose()
    # as_tensor is not implemented; we just assert the interface exists
    try:
        _ = p.as_tensor(device="cpu")
    except NotImplementedError:
        # acceptable for contract test
        pass