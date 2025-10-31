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

        # Queues partagées
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

        # Flag d’exécution thread-safe
        self.running = threading.Event()

    # ------------------------------------------------------------------ #
    def start(self):
        """Démarre le système de collecte multi-thread."""
        logger.info("[Collector] Démarrage des tailers et du thread consumer")
        self.running.set()
        self.t_pipeline.start()
        self.t_kpi.start()
        self.t_consumer.start()

    # ------------------------------------------------------------------ #
    def _consume_loop(self):
        """Boucle principale : lit les queues, parse et agrège."""
        logger.info("[Collector] Thread consumer démarré")
        try:
            while self.running.is_set():
                t0 = time.perf_counter()
                self._drain_queue(self.q_pipeline, "pipeline")
                self._drain_queue(self.q_kpi, "kpi")
                self.profiler.add_sample(time.perf_counter() - t0)
                time.sleep(0.01)
        except Exception as e:
            logger.error(f"[Collector] Erreur dans _consume_loop: {e}")
        finally:
            logger.info("[Collector] Thread consumer arrêté proprement (finally)")

    # ------------------------------------------------------------------ #
    def _drain_queue(self, q, source: str):
        """Vide une queue jusqu’à épuisement."""
        drained = 0
        while not q.empty():
            try:
                line = q.get_nowait()
            except Empty:
                break

            drained += 1
            parsed = self.parser.parse_line(line, source)

            if parsed:
                logger.debug(f"[Collector] {source}: parsed frame_id={parsed.get('frame_id')} event={parsed.get('event')}")
                self.aggregator.update(parsed)
            else:
                logger.debug(f"[Collector] {source}: ligne ignorée (non parsable)")

        if drained > 0:
            logger.debug(f"[Collector] {source}: {drained} lignes traitées dans cette itération.")


    # ------------------------------------------------------------------ #
    def stop(self, timeout: float = 2.0):
        """Arrête proprement tous les threads du LogCollector."""
        if not self.running.is_set():
            logger.debug("[Collector] stop() appelé alors que le collector n’était pas actif.")
            return

        logger.info("[Collector] 🔻 Arrêt demandé du LogCollector")
        self.running.clear()

        # 🧹 Étape 1 — stoppe les tailers
        try:
            self.t_pipeline.stop()
            self.t_kpi.stop()
            logger.debug("[Collector] Tailers stoppés.")
        except Exception as e:
            logger.warning(f"[Collector] Erreur lors de l’arrêt des tailers: {e}")

        # 🕒 Étape 2 — attendre leur terminaison propre
        for t in (self.t_pipeline, self.t_kpi):
            if t.is_alive():
                t.join(timeout=timeout)
                if t.is_alive():
                    logger.warning(f"[Collector] ⚠️ Thread {t.name} n’a pas terminé à temps")
                else:
                    logger.info(f"[Collector] ✅ Thread {t.name} arrêté proprement")

        # 🧩 Étape 3 — arrêter la boucle consumer
        if self.t_consumer.is_alive():
            self.t_consumer.join(timeout=timeout)
            if self.t_consumer.is_alive():
                logger.warning("[Collector] ⚠️ Thread consumer bloqué (timeout)")
            else:
                logger.info("[Collector] ✅ Thread consumer arrêté proprement")

        logger.info("[Collector] ✅ Tous les threads terminés proprement (shutdown complet)")


    # ------------------------------------------------------------------ #
    def snapshot(self):
        """Retourne un snapshot agrégé complet (pour WS / tests)."""
        stats = self.profiler.stats()
        return self.aggregator.as_snapshot(profiler_stats=stats)

    def get_latest(self):
        """Retourne la dernière frame agrégée."""
        latest = self.aggregator.get_latest()
        if latest:
            logger.debug(f"[Collector] get_latest() → frame#{latest.frame_id} total={getattr(latest.interstage, 'total', None)}ms")
        else:
            logger.debug("[Collector] get_latest() → None (aucune frame complète encore disponible)")
        return latest


    def get_history(self, n=100):
        """Retourne l’historique récent."""
        return self.aggregator.get_history(n)
