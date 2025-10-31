"""
⚠️ [Deprecated / Legacy Notice]
================================
Ce module n’est **plus utilisé** dans l’architecture actuelle d’Ultramotion IGT Inference.

⏳ Ancien rôle :
----------------
`slicer_sender.py` servait de thread d’envoi vers 3D Slicer.
Il lisait les masques produits par la pipeline (`Queue_Out`)
et les transmettait via un serveur pyigtl externe.

🚀 Nouvelle architecture :
--------------------------
Depuis la refonte du Gateway (`IGTGateway`), cette tâche est entièrement
prise en charge par :
    → `service/slicer_server.py`  (fonction `run_slicer_server()`)

Le pipeline TX fonctionne désormais ainsi :
    Segmentation → Gateway._outbox → run_slicer_server() → 3D Slicer

Les threads RX/TX sont lancés automatiquement depuis `IGTGateway.start()`
via le `THREAD_REGISTRY` (défini dans `service/registry.py`).

💡 Ce fichier est conservé uniquement à titre historique / référence de design.
Il n’est plus importé ni exécuté dans la version actuelle.
"""



"""Slicer server thread: reads masks from outbox and emits IGTLink IMAGE messages.

This module tries to use `pyigtl` when available; otherwise it provides a
placeholder that consumes the outbox and simulates sends while reporting fps.
"""
from typing import Optional, Callable, Any
import time
import logging
import numpy as np

LOG = logging.getLogger("igt.slicer_server")


def run_slicer_server(
    outbox,
    stop_event,
    port,
    stats_cb: Optional[Callable] = None,
    event_cb: Optional[Callable] = None,
    tx_ready: Optional[Any] = None,  # 🔬 OPTIMISATION : Event signalant qu'une frame est disponible dans outbox
    gateway_stats: Optional[Any] = None  # ✅ INSTRUMENTATION : GatewayStats pour mark_tx() (latency tracking)
) -> None:
    """Thread TX : serveur IGTLink pour Slicer.

    - Lit les masques dans outbox.
    - Émet IMAGE (labelmap) vers Slicer.
    - Appelle stats_cb(fps) toutes les 2s si fourni.
    - Appelle event_cb('tx_client_connected', {...}) / ('tx_stopped', {...}) si fournis.
    - Si tx_ready est fourni, utilise wait() pour un réveil instantané au lieu de polling.
    - Si gateway_stats est fourni, appelle mark_tx(frame_id, ts) pour calcul de latence.
    """
    # print("[DEBUG] run_slicer_server() started")
    LOG.info("[TX-SIM] run_slicer_server() started — debug check")
    try:
        import pyigtl  # type: ignore
    except Exception:
        pyigtl = None
        LOG.debug("pyigtl not available; using simulator for slicer_server")

    # emit started/connected event
    try:
        if event_cb:
            event_cb("tx_server_started", {"port": port})
    except Exception:
        LOG.exception("event_cb failed on start")

    sent_timestamps = []
    last_stats = time.time()
    send_count = 0

    # Hypothetical server setup when pyigtl available (best-effort)
    server = None
    if pyigtl:
        try:
            server = pyigtl.OpenIGTLinkServer(port)  # hypothetical
            server.start()
        except Exception:
            LOG.exception("Failed to start pyigtl server; falling back to simulator")
            server = None
            pyigtl = None

    while not stop_event.is_set():
        now = time.time()
        try:
            # 🔬 OPTIMISATION : Attendre signal Event au lieu de polling avec sleep()
            if tx_ready is not None:
                # Mode optimisé : wait() bloque jusqu'à ce qu'une frame soit disponible
                if not tx_ready.wait(timeout=0.01):
                    continue  # Timeout → revérifier stop_event
                tx_ready.clear()  # Reset l'event pour la prochaine frame
            
            # 🔬 INSTRUMENTATION : Timestamp AVANT lecture outbox
            t_start = time.time()
            
            # Try to consume one item from outbox
            try:
                item = outbox.popleft()
            except Exception:
                item = None
            
            # 🔬 INSTRUMENTATION : Timestamp APRÈS lecture outbox
            t_after_read = time.time()

            if item is None:
                # nothing to send (seulement si pas d'Event ou timeout)
                if tx_ready is None:
                    time.sleep(0.005)  # Fallback polling si pas d'Event
                continue
            else:
                mask, meta = item
                
                # 🔬 INSTRUMENTATION : Timestamp AVANT sérialisation/envoi
                t_before_send = time.time()
                
                # Send via pyigtl if available
                if pyigtl and server:
                    try:
                        msg = pyigtl.ImageMessage(mask)
                        server.send(msg)
                    except Exception:
                        LOG.exception("pyigtl send failed; dropping message")
                else:
                    # Debug : afficher chaque envoi simulé
                    pass  # Log déplacé après pour mesurer la vraie latence
                
                # 🔬 INSTRUMENTATION : Timestamp APRÈS envoi
                t_after_send = time.time()
                
                # ✅ LATENCY TRACKING : Enregistrer timestamp TX pour calcul RX→TX et inter-étapes
                if gateway_stats is not None and meta and "frame_id" in meta:
                    try:
                        frame_id = int(meta["frame_id"])
                        # 1️⃣ Latence RX→TX globale
                        gateway_stats.mark_tx(frame_id, t_after_send)
                        # 2️⃣ Latences inter-étapes GPU résidentes
                        gateway_stats.mark_interstage_tx(frame_id, t_after_send)
                    except Exception as e:
                        LOG.debug(f"mark_tx/mark_interstage_tx() failed for frame {meta.get('frame_id', -1)}: {e}")
                        pass  # Ignorer erreurs non critiques
                
                # 🔬 LOG avec détails de timing
                t_read = (t_after_read - t_start) * 1000  # ms
                t_send = (t_after_send - t_before_send) * 1000  # ms
                t_total = (t_after_send - t_start) * 1000  # ms
                
                LOG.info(
                    f"[TX-SIM] Sent frame #{meta.get('frame_id', -1):03d} "
                    f"| read={t_read:.2f}ms send={t_send:.2f}ms total={t_total:.2f}ms"
                )


                # ✅ ENREGISTREMENT DANS LE MONITOR GLOBAL (Phase 4)
                try:
                    from core.monitoring import monitor
                    monitor.record_interstage("gpu_cpu_to_tx", t_total)
                except Exception:
                    LOG.debug("Monitor TX latency record failed")


                send_count += 1
                sent_timestamps.append(now)
                cutoff = now - 5.0
                while sent_timestamps and sent_timestamps[0] < cutoff:
                    sent_timestamps.pop(0)

            # emit stats every 2s
            if now - last_stats >= 2.0:
                duration = now - (sent_timestamps[0] if sent_timestamps else now)
                fps = (len(sent_timestamps) / duration) if duration > 0 else 0.0
                try:
                    # compute bytes_count from last mask if available
                    bytes_count = 0
                    try:
                        bytes_count = getattr(mask, "nbytes", 0)
                    except Exception:
                        bytes_count = 0
                    if stats_cb:
                        stats_cb(fps, bytes_count)
                except Exception:
                    LOG.exception("stats_cb failed in slicer_server")
                last_stats = now

        except Exception:
            LOG.exception("Exception in run_slicer_server loop")
            time.sleep(0.1)

    # cleanup
    try:
        if event_cb:
            event_cb("tx_stopped", {"port": port})
    except Exception:
        LOG.exception("event_cb failed on stop")

    try:
        if server:
            server.stop()
    except Exception:
        LOG.exception("Failed to stop pyigtl server")

    LOG.info("run_slicer_server stopped")
