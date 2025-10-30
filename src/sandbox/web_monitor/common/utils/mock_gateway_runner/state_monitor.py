"""
state_monitor.py
----------------
Surveille les threads et gère un arrêt propre via SIGINT/SIGTERM.
"""
import time, signal, logging
LOG = logging.getLogger("igt.mock.supervisor")

def supervise_threads(stop_event, threads):
    """Boucle de supervision principale."""
    def signal_handler(sig, frame):
        LOG.info(f"🛑 Signal {sig} reçu — arrêt du simulateur")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, signal_handler)

    try:
        while not stop_event.is_set():
            time.sleep(1.0)
            for t in threads:
                if not t.is_alive():
                    LOG.warning(f"Thread {t.name} stoppé prématurément")
                    stop_event.set()
                    break
    except KeyboardInterrupt:
        LOG.info("🧤 Ctrl+C détecté, arrêt manuel")
        stop_event.set()
