import time
import numpy as np

from core.types import FrameMeta, RawFrame, GpuFrame, ResultPacket, Pose


def test_dataclasses_instantiation():
    meta = FrameMeta(frame_id=1, ts=time.time(), pose=Pose(), spacing=(0.5, 0.5), device_name="sim")
    img = np.zeros((512, 512), dtype=np.uint8)

    raw = RawFrame(image=img, meta=meta)
    assert raw.meta.frame_id == 1
    assert raw.image.shape == (512, 512)

    # GpuFrame: tensor placeholder (None for contract test)
    gpu = GpuFrame(tensor=None, meta=meta)
    assert gpu.meta is raw.meta

    res = ResultPacket(mask=img, score=0.5, state="VISIBLE", meta=meta)
    assert res.score == 0.5
    assert res.state == "VISIBLE"


def test_trace_id():
    meta = FrameMeta(frame_id=1, ts=time.time())
    assert meta.trace_id == "1"
