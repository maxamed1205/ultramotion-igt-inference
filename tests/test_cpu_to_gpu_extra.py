import time
import numpy as np
import logging

import pytest

from core.types import FrameMeta, RawFrame
from core.preprocessing.cpu_to_gpu import prepare_frame_for_gpu


def test_zscore_without_mean_std_warns(caplog):
    caplog.set_level(logging.WARNING)
    img = (np.random.rand(64, 64).astype(np.float32) * 0.8)
    meta = FrameMeta(frame_id=123, ts=time.time())
    raw = RawFrame(image=img, meta=meta)

    # Request zscore without providing mean/std -> should warn and skip
    gf = prepare_frame_for_gpu(raw, device="cpu", config={"normalize": {"mode": "zscore"}})
    assert any("zscore sans mean/std" in rec.message for rec in caplog.records)

    # Ensure values unchanged (no zscore applied)
    arr_out = gf.tensor.cpu().numpy()
    expected = img.reshape((1, 1, img.shape[0], img.shape[1])).astype(np.float32)
    assert np.allclose(arr_out, expected)


def test_oom_fallback_emits_single_kpi(monkeypatch):
    # Small image
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    meta = FrameMeta(frame_id=999, ts=time.time())
    raw = RawFrame(image=img, meta=meta)

    import core.preprocessing.cpu_to_gpu as cpu_mod

    # Stub safe_log_kpi to count calls
    calls = {"n": 0}

    def fake_safe_log_kpi(msg):
        calls["n"] += 1

    monkeypatch.setattr("core.monitoring.kpi.safe_log_kpi", fake_safe_log_kpi, raising=False)
    monkeypatch.setattr("core.monitoring.kpi.format_kpi", lambda x: x, raising=False)

    # Force CUDA available and make Tensor.to raise OOM
    if hasattr(cpu_mod, "torch") and cpu_mod.torch is not None:
        # Force code path to believe CUDA is available
        monkeypatch.setattr(cpu_mod.torch.cuda, "is_available", lambda: True)

        # Provide a dummy Stream implementation and a matching context manager
        from contextlib import contextmanager

        class DummyStream:
            def __init__(self, *a, **k):
                self.cuda_stream = None

        @contextmanager
        def dummy_stream_cm(s):
            yield s

        monkeypatch.setattr(cpu_mod.torch.cuda, "Stream", DummyStream, raising=False)
        monkeypatch.setattr(cpu_mod.torch.cuda, "stream", dummy_stream_cm, raising=False)

        # Make Tensor.to raise OOM to trigger fallback
        def raise_oom(self, *args, **kwargs):
            raise cpu_mod.torch.cuda.OutOfMemoryError()

        monkeypatch.setattr(cpu_mod.torch.Tensor, "to", raise_oom, raising=False)

    # Call with device cuda to trigger the path
    gf = prepare_frame_for_gpu(raw, device="cuda")

    # We expect safe_log_kpi to have been called exactly once
    assert calls["n"] == 1
    # And tensor should be on CPU after fallback
    assert str(gf.tensor.device).startswith("cpu")
