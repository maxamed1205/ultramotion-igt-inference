import time
import numpy as np
import logging
import builtins

from core.types import FrameMeta, RawFrame

from core.preprocessing.cpu_to_gpu import prepare_frame_for_gpu

LOG = logging.getLogger("igt.gpu.test")


def test_prepare_uint8_cpu_fallback():
    # Force CPU by passing device="cpu"
    img = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    meta = FrameMeta(frame_id=7, ts=time.time())
    raw = RawFrame(image=img, meta=meta)
    gf = prepare_frame_for_gpu(raw, device="cpu")
    t = gf.tensor
    assert t is not None
    assert t.ndim == 4
    assert t.shape == (1, 1, 512, 512)
    assert str(t.device).startswith("cpu")


def test_prepare_float32_preserve_range_cpu():
    img = (np.random.rand(256, 256).astype(np.float32) * 0.8)  # within [0,1)
    meta = FrameMeta(frame_id=8, ts=time.time())
    raw = RawFrame(image=img, meta=meta)
    gf = prepare_frame_for_gpu(raw, device="cpu", config={"normalize": {"mode": "unit", "clip": [0.0, 1.0]}})
    t = gf.tensor
    assert t.dtype == __import__("torch").float32
    arr = t.cpu().numpy()
    assert arr.max() <= 1.0 + 1e-6


def test_bad_channels_raises():
    img = np.zeros((3, 128, 128), dtype=np.uint8)
    meta = FrameMeta(frame_id=9, ts=time.time())
    raw = RawFrame(image=img, meta=meta)
    try:
        prepare_frame_for_gpu(raw, device="cpu")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_idempotent_shapes():
    img1 = np.random.randint(0, 256, (320, 240), dtype=np.uint8)
    img2 = img1.reshape((320, 240, 1))
    meta = FrameMeta(frame_id=10, ts=time.time())
    gf1 = prepare_frame_for_gpu(RawFrame(image=img1, meta=meta), device="cpu")
    gf2 = prepare_frame_for_gpu(RawFrame(image=img2, meta=meta), device="cpu")
    assert gf1.tensor.shape == gf2.tensor.shape
