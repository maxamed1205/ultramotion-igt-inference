import sys
sys.path.insert(0, 'src')
from service.dashboard_service import MetricsCollector, DashboardConfig

c = MetricsCollector(DashboardConfig())
snapshot = c.collect()

print("=" * 60)
print("SNAPSHOT COMPLET")
print("=" * 60)
print(f"proc_count: {snapshot.get('proc_count', 'MISSING')}")
print(f"tx_count: {snapshot.get('tx_count', 'MISSING')}")
print(f"fps_proc: {snapshot.get('fps_proc', 'MISSING')}")
print(f"fps_tx: {snapshot.get('fps_tx', 'MISSING')}")
print(f"last_frame_proc: {snapshot.get('last_frame_proc', 'MISSING')}")
print(f"last_frame_tx: {snapshot.get('last_frame_tx', 'MISSING')}")
print(f"sync_txproc: {snapshot.get('sync_txproc', 'MISSING')}")
print(f"latency_proctx_avg: {snapshot.get('latency_proctx_avg', 'MISSING')}")
print(f"fps_rx_kpi: {snapshot.get('fps_rx_kpi', 'MISSING')}")
print("=" * 60)
