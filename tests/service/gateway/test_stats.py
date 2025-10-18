import threading
import time
import logging

from service.gateway.stats import GatewayStats

LOG = logging.getLogger("test.gateway.stats")


def test_snapshot_defaults():
    s = GatewayStats()
    snap = s.snapshot()
    assert isinstance(snap, dict)
    assert snap["fps_rx"] == 0.0
    assert snap["fps_tx"] == 0.0
    assert snap["uptime_s"] >= 0.0


def test_rolling_window_avg():
    s = GatewayStats(rolling_window_size=5)
    # insert 25 values; average should reflect last 5
    for i in range(25):
        s.update_rx(float(i), ts=time.time())
    snap = s.snapshot()
    # last five values are 20..24 -> avg = 22
    assert abs(snap["avg_fps_rx"] - 22.0) < 1e-6


def test_reset_and_cold_start():
    s = GatewayStats()
    s.update_rx(10.0, ts=time.time(), bytes_count=100)
    s.mark_rx(1, time.time())
    s.mark_tx(1, time.time() + 0.01)
    s.reset()
    snap = s.snapshot()
    assert snap["fps_rx"] == 0.0
    assert snap["latency_samples"] == 0
    # cold start resets uptime
    s.reset(cold_start=True)
    snap2 = s.snapshot()
    assert snap2["uptime_s"] <= 0.1


def test_latency_recording_and_bounds():
    s = GatewayStats(latency_window_size=3)
    t0 = time.time()
    s.mark_rx(10, t0)
    s.mark_tx(10, t0 + 0.025)
    snap = s.snapshot()
    assert abs(snap["latency_ms_avg"] - 25.0) < 1.0
    # add more than window size and check truncation
    for i in range(20):
        s.mark_rx(100 + i, t0 + i * 0.001)
        s.mark_tx(100 + i, t0 + i * 0.001 + 0.005)
    snap2 = s.snapshot()
    assert snap2["latency_samples"] <= 3


def test_orphan_tx_and_inverted_ts(caplog):
    caplog.set_level(logging.WARNING)
    s = GatewayStats()
    # orphan tx should not raise
    s.mark_tx(9999, time.time())
    # inverted timestamps
    s.mark_rx(1234, time.time() + 1.0)
    s.mark_tx(1234, time.time())
    assert any("clamped to 0" in r.message for r in caplog.records)


def test_thread_safety_basic():
    s = GatewayStats()
    def updater():
        for i in range(1000):
            s.update_rx(30.0, time.time())
            s.update_tx(30.0)
    threads = [threading.Thread(target=updater) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # snapshot should not raise
    snap = s.snapshot()
    assert "fps_rx" in snap and "fps_tx" in snap
