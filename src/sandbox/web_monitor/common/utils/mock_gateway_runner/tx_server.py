"""
tx_server.py
-------------
Simule le thread TX (envoi vers Slicer).
Lit les masques dans la outbox du Gateway et met à jour les métriques TX.
"""

import time
import logging
import numpy as np

LOG = logging.getLogger("igt.mock.tx")


def run_tx_server(gateway, stop_event, tx_interval=0.01):
    """
    Thread TX simulé — envoie périodiquement les masques du gateway._outbox.

    Args:
        gateway: instance IGTGateway
        stop_event: threading.Event signalant l’arrêt
        tx_interval: délai minimum entre envois (s)
    """
    # LOG.info("[TX] Thread TX simulé démarré")
    outbox = gateway._outbox
    stats = gateway.stats
    events = gateway.events

    sent_count = 0
    last_tx_time = time.perf_counter()

    while not stop_event.is_set():
        try:
            # Vérifie s’il y a quelque chose à envoyer
            if not outbox:
                time.sleep(tx_interval)
                continue

            mask, meta = outbox.popleft()
            frame_id = meta.get("frame_id", -1)
            ts_proc_done = meta.get("ts", time.perf_counter())

            # 🔹 Timestamp TX
            ts_tx = time.perf_counter()

            # 📊 Instrumentation : latence inter-étape PROC → TX
            stats.mark_interstage_tx(frame_id, ts_tx)
            stats.mark_tx(frame_id, ts_tx)

            # Mise à jour FPS TX
            elapsed = ts_tx - last_tx_time
            if elapsed > 0:
                fps = 1.0 / elapsed
                gateway.update_tx_stats(fps)
                last_tx_time = ts_tx

            sent_count += 1
            if sent_count % 10 == 0:
                LOG.info(f"[TX] {sent_count} masques envoyés (Frame #{frame_id})")

            # 🔹 Événement Dashboard temps réel
            try:
                events.emit("tx_mask_sent", {"frame_id": frame_id, "timestamp": ts_tx})
            except Exception:
                pass

            # 💡 Simulation réseau → pause légère
            time.sleep(tx_interval)

        except Exception as e:
            LOG.warning(f"[TX] Erreur envoi masque : {e}")
            time.sleep(0.05)

    LOG.info(f"[TX] Thread TX arrêté proprement après {sent_count} masques envoyés ✅")
