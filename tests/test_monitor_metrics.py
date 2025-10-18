def test_monitor_collects_metrics():
    from core.monitoring.monitor import get_pipeline_metrics
    data = get_pipeline_metrics()
    assert "queues" in data and "timestamp" in data
    assert isinstance(data["timestamp"], float)
    q = data["queues"]
    assert all(name in q for name in ("Queue_Raw", "Queue_RT_dyn", "Queue_GPU", "Queue_Out"))
