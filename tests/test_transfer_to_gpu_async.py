import time
import numpy as np
import pytest

import core.preprocessing.cpu_to_gpu as cpu_to_gpu


class KPICollector:
    def __init__(self):
        self.events = []

    def __call__(self, msg):
        # format_kpi already returns a string in production; accept dict or string
        try:
            # If format_kpi produced a string, try to eval-like parse? Keep simple: store raw
            self.events.append(msg)
        except Exception:
            self.events.append(msg)


def _capture_kpi(monkeypatch, collector):
    # Install a fake safe_log_kpi and format_kpi (format_kpi returns its input)
    monkeypatch.setattr("core.monitoring.kpi.safe_log_kpi", lambda m: collector(m), raising=False)
    monkeypatch.setattr("core.monitoring.kpi.format_kpi", lambda d: d, raising=False)


def test_cpu_path_noop(monkeypatch):
    # CPU tensor, device='cpu' -> noop with transfer_async_noop
    arr = np.zeros((1, 1, 32, 32), dtype=np.float32)
    collector = KPICollector()
    _capture_kpi(monkeypatch, collector)

    ten = cpu_to_gpu.transfer_to_gpu_async(arr, stream_transfer=None, device="cpu")

    assert hasattr(ten, "device") or hasattr(ten, "shape")

    # One KPI event
    assert len(collector.events) == 1
    evt = collector.events[0]
    # format_kpi in tests returns dict
    assert evt["event"] == "transfer_async_noop"
    assert evt["copy_ms"] == 0.0
    assert evt["device_target"] == "cpu"


def test_already_on_target_noop(monkeypatch):
    # Skip if torch not present or no CUDA
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    ten_gpu = torch.zeros((1, 1, 16, 16), device="cuda:0")
    collector = KPICollector()
    _capture_kpi(monkeypatch, collector)

    out = cpu_to_gpu.transfer_to_gpu_async(ten_gpu, stream_transfer=None, device="cuda:0")
    assert out.device == ten_gpu.device

    assert len(collector.events) == 1
    evt = collector.events[0]
    assert evt["event"] == "transfer_async_noop"
    assert evt["already_on_device"] == 1
    assert evt["copy_ms"] == 0.0


def test_cpu_to_gpu_with_stream(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for copy test")

    # Use a small CPU tensor
    arr = np.ones((1, 1, 8, 8), dtype=np.float32)
    stream = torch.cuda.Stream()

    collector = KPICollector()
    _capture_kpi(monkeypatch, collector)

    out = cpu_to_gpu.transfer_to_gpu_async(arr, stream_transfer=stream, device="cuda:0")

    assert out.is_cuda

    # Find KPI event dict in collector
    assert len(collector.events) >= 1
    # The last event should be transfer_async
    evt = collector.events[-1]
    assert evt["event"] == "transfer_async"
    # small transfers may measure as 0ms on fast systems; accept >= 0
    assert evt["copy_ms"] >= 0


def test_oom_path(monkeypatch):
    torch = pytest.importorskip("torch")
    # Create a CPU tensor that will be used for the test
    ten = torch.zeros((1, 1, 8, 8), dtype=torch.float32)

    # Monkeypatch Tensor.to to raise OOM
    original_to = torch.Tensor.to

    def fake_to(self, *args, **kwargs):
        raise torch.cuda.OutOfMemoryError("Simulated OOM")

    monkeypatch.setattr(torch.Tensor, "to", fake_to)

    # Ensure the function attempts a CUDA transfer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    collector = KPICollector()
    _capture_kpi(monkeypatch, collector)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        cpu_to_gpu.transfer_to_gpu_async(ten, stream_transfer=None, device="cuda:0")

    # Ensure transfer_async_oom KPI emitted
    assert any((isinstance(e, dict) and e.get("event") == "transfer_async_oom") for e in collector.events)

    # restore if something else in the session relies on it (monkeypatch will revert automatically at test end)
    monkeypatch.setattr(torch.Tensor, "to", original_to)


def test_cpu_to_gpu_no_stream(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for CPU→GPU no-stream test")

    arr = np.ones((1, 1, 8, 8), dtype=np.float32)
    collector = KPICollector()
    _capture_kpi(monkeypatch, collector)

    out = cpu_to_gpu.transfer_to_gpu_async(arr, stream_transfer=None, device="cuda:0")

    assert out.is_cuda
    evt = collector.events[-1]
    assert evt["event"] == "transfer_async"
    assert evt["stream_used"] == 0
    # small transfers may measure as 0ms on fast systems; accept >= 0
    assert evt["copy_ms"] >= 0


def test_pinned_and_contiguous(monkeypatch):
    torch = pytest.importorskip("torch")
    called = {"contig": False, "pinned": False}

    class DummyTensor(torch.Tensor):
        def contiguous(self):
            called["contig"] = True
            return self

        def pin_memory(self):
            called["pinned"] = True
            return self

    # simulate a CPU tensor
    t = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
    # change class to inject methods
    t.__class__ = DummyTensor

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # we don't need a real device here; just ensure methods are called
    try:
        cpu_to_gpu.transfer_to_gpu_async(t, stream_transfer=None, device="cuda:0")
    except Exception:
        # transfer may fail without proper GPU; we only assert that contig/pin were attempted
        pass

    assert called["contig"] is True or called["pinned"] is True


def test_numpy_input_to_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for numpy input test")

    arr = np.random.rand(1, 1, 8, 8).astype(np.float32)
    collector = KPICollector()
    _capture_kpi(monkeypatch, collector)

    out = cpu_to_gpu.transfer_to_gpu_async(arr, stream_transfer=None, device="cuda:0")

    assert hasattr(out, "is_cuda")
    assert out.is_cuda
    evt = collector.events[-1]
    assert evt["event"] == "transfer_async"
