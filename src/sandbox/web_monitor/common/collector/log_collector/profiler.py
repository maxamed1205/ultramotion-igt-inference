"""
profiler.py
------------
Surveille la latence interne du collecteur (durée parsing+fusion).
"""

from collections import deque
import statistics

from . import logger

class CollectorProfiler:
    def __init__(self, max_samples=200):
        self.samples = deque(maxlen=max_samples)

    def add_sample(self, dt):
        self.samples.append(dt)

    def stats(self):
        if not self.samples:
            return {"avg_ms": 0, "p95_ms": 0}
        avg_ms = round(statistics.mean(self.samples)*1000,2)
        # statistics.quantiles requires at least two data points; fallback to avg
        if len(self.samples) < 2:
            return {"avg_ms": avg_ms, "p95_ms": avg_ms}
        return {
            "avg_ms": avg_ms,
            "p95_ms": round(statistics.quantiles(self.samples, n=20)[18]*1000,2),
        }
