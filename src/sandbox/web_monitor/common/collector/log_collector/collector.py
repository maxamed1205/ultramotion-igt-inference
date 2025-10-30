"""
collector.py
-------------
Coordonne la lecture, le parsing et la fusion des logs pipeline.log et kpi.log.
"""

import threading
import time
from queue import Queue, Empty
from .tailer import FileTailer
from .parser import LogParser
from .aggregator import FrameAggregator
from .profiler import CollectorProfiler
from . import logger


class LogCollector:
    """Service central : lecture, parsing, agrégation et snapshot."""

    def __init__(self, pipeline_path: str, kpi_path: str, history_size: int = 300):
        self.pipeline_path = pipeline_path
        self.kpi_path = kpi_path

        # Queues partagées (threads -> consumer)
        self.q_pipeline = Queue(maxsize=500)
        self.q_kpi = Queue(maxsize=500)

        # Composants internes
        self.parser = LogParser()
        self.aggregator = FrameAggregator(max_history=history_size)
        self.profiler = CollectorProfiler()

        # Threads de lecture (tailers)
        self.t_pipeline = FileTailer(pipeline_path, self.q_pipeline)
        self.t_kpi = FileTailer(kpi_path, self.q_kpi)

        # Thread de consommation (fusion)
        self.t_consumer = threading.Thread(target=self._consume_loop, daemon=True)

        self.running = False

    # ------------------------------------------------------------------ #
    def start(self):
        """Démarre le système de collecte multi-thread."""
        logger.info("[Collector] Starting tailers and consumer thread")
        self.running = True
        self.t_pipeline.start()
        self.t_kpi.start()
        self.t_consumer.start()

    # ------------------------------------------------------------------ #
    def _consume_loop(self):
        """Boucle principale : lit les queues, parse et agrège."""
        # Allow a single manual drain when the collector isn't running (useful for tests
        # that inject lines directly into queues and call this method). When the
        # collector is started via `start()` (self.running == True) this becomes a
        # continuous loop as before.
        while True:
            t0 = time.perf_counter()
            self._drain_queue(self.q_pipeline, "pipeline")
            self._drain_queue(self.q_kpi, "kpi")
            self.profiler.add_sample(time.perf_counter() - t0)
            # If not running, perform a single iteration and exit (manual mode)
            if not self.running:
                break
            time.sleep(0.01)

    def _drain_queue(self, q, source: str):
        """Vide une queue jusqu’à épuisement."""
        while True:
            try:
                line = q.get_nowait()
            except Empty:
                break
            parsed = self.parser.parse_line(line, source)
            if parsed:
                self.aggregator.update(parsed)

    # ------------------------------------------------------------------ #
    def stop(self):
        """Arrête proprement tous les threads."""
        self.running = False
        logger.info("[Collector] Stopping tailers")
        self.t_pipeline.stop()
        self.t_kpi.stop()

    # ------------------------------------------------------------------ #
    def snapshot(self):
        """Retourne un snapshot agrégé complet (pour WS / tests)."""
        stats = self.profiler.stats()
        return self.aggregator.as_snapshot(profiler_stats=stats)

    # ------------------------------------------------------------------ #
    def get_latest(self):
        return self.aggregator.get_latest()

    def get_history(self, n=100):
        return self.aggregator.get_history(n)
