"""
runner.py
----------
Point central du simulateur de pipeline.
"""

import threading
import logging

from sandbox.web_monitor.common.utils.mock_gateway_runner.log_utils import clean_old_logs, setup_logging
from sandbox.web_monitor.common.utils.mock_gateway_runner.gpu_runtime import init_gpu_if_available
from sandbox.web_monitor.common.utils.mock_gateway_runner.threads_manager import create_threads
from sandbox.web_monitor.common.utils.mock_gateway_runner.state_monitor import supervise_threads
from service.gateway.manager import IGTGateway
from core.monitoring.monitor import set_active_gateway

LOG = logging.getLogger("igt.mock.runner")


def run_mock_gateway():
    """Orchestre le pipeline complet RX → PROC → TX."""

    # ──────────────────────────────────────────────
    # Étape 1 : Nettoyage + Logging
    # ──────────────────────────────────────────────
    clean_old_logs()
    setup_logging()

    # ──────────────────────────────────────────────
    # Étape 2 : Initialisation GPU / CPU
    # ──────────────────────────────────────────────
    device = init_gpu_if_available()
    use_gpu = device != "cpu"

    # ──────────────────────────────────────────────
    # Étape 3 : Timer haute résolution (Windows only)
    # ──────────────────────────────────────────────
    try:
        from utils.win_timer_resolution import enable_high_resolution_timer
        enable_high_resolution_timer()
        LOG.info("[TIMER] Haute résolution activée (1ms)")
    except ImportError:
        LOG.warning("[TIMER] Module win_timer_resolution non disponible — précision timer limitée.")
    except Exception as e:
        LOG.warning(f"[TIMER] Échec d’activation du timer haute résolution: {e}")

    # ──────────────────────────────────────────────
    # Étape 4 : Création Gateway + Enregistrement global
    # ──────────────────────────────────────────────
    LOG.info(f"[MOCK] Device actif: {device}")
    gateway = IGTGateway("127.0.0.1", 18944, 18945, target_fps=100.0)
    gateway._running = True
    set_active_gateway(gateway)

    # ──────────────────────────────────────────────
    # Étape 5 : Création des threads
    # ──────────────────────────────────────────────
    stop_event = threading.Event()
    frame_ready = threading.Event()
    threads = create_threads(gateway, stop_event, frame_ready, use_gpu, device)

    for t in threads:
        t.start()

    LOG.info("🚀 Threads RX/PROC/TX démarrés")
    LOG.info("📊 Dashboard disponible sur http://localhost:8050 (si activé)")
    LOG.info("=" * 80)

    # ──────────────────────────────────────────────
    # Étape 6 : Supervision et arrêt propre
    # ──────────────────────────────────────────────
    supervise_threads(stop_event, threads)
    LOG.info("✅ Simulation terminée proprement.")
