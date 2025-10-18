import time
from core.monitoring.monitor import log_kpi_tick


LOG_N = 10000


def bench(n: int = LOG_N):
    t0 = time.time()
    for _ in range(n):
        log_kpi_tick(25, 25, 40, 0.0)
    t1 = time.time()
    elapsed = t1 - t0
    print(f"{n} KPI logs in {elapsed:.3f}s => {(n / elapsed):.1f} logs/sec")


if __name__ == "__main__":
    bench()
