"""Slicer sender — Thread D

Description
-----------
Module responsable de la lecture asynchrone de `Queue_Out` et de l'envoi
des résultats (mask, score, état) vers Slicer via OpenIGTLink (pyigtl).

Responsabilités
----------------
- consommer les résultats produits par la pipeline (Queue_Out),
- sérialiser la sortie en message OpenIGTLink IMAGE/TRANSFORM si besoin,
- envoyer de façon asynchrone via le serveur pyigtl fourni.

Dépendances attendues
---------------------
- pyigtl (optionnel pour tests) — import local quand nécessaire
- la structure `result_dict` attendue contient au minimum:
    {
        'frame_id': str/int,
        'mask': numpy.ndarray ou torch.Tensor,
        'score': float,
        'state': str,
        'timestamp': float,
        ...
    }

Fonctions principales
---------------------
- start_sending_thread(pyigtl_server)
- _send_result_to_slicer(result_dict)
- stop_sending_thread()

Note
----
Ce module expose uniquement les signatures et docstrings. Pas d'implémentation
réseau ici.
"""

from typing import Any, Dict, Optional

from core.types import ResultPacket
from core.queues.buffers import get_queue_out, try_dequeue
import logging
import time

LOG = logging.getLogger("igt.slicer")
LOG_KPI = logging.getLogger("igt.kpi")


def start_sending_thread(pyigtl_server: Any, config: Optional[Dict] = None) -> None:
    """Démarre le thread qui lit `Queue_Out` et envoie vers Slicer.

    Args:
        pyigtl_server: instance serveur pyigtl déjà initialisée (ou équivalent)
        config: options (intervalle d'envoi, batching, retry policy)
    """
    # Minimal loop-style helper (non-blocking). Real implementation should
    # run in a background thread and call _send_result_to_slicer.
    q_out = get_queue_out()
    result = try_dequeue(q_out)
    if result is None:
        return
    _send_result_to_slicer(result)
    return


def _send_result_to_slicer(result: ResultPacket) -> None:
    """Construit le message OpenIGTLink à partir de `result` et envoie.

    Args:
        result: ResultPacket (mask, score, state, meta)
    """
    meta_dict = getattr(result, "meta", None)
    if meta_dict is None:
        return
    try:
        md = result.meta.to_igt_dict()
    except Exception:
        md = {}

    # Try to use pyigtl if available; do not raise if not.
    try:
        import pyigtl

        # pyigtl.ImageMessage creation is demo-like here; real API may differ
        msg = pyigtl.ImageMessage(
            device_name=md.get("DeviceName", result.meta.device_name),
            timestamp=md.get("Timestamp", result.meta.ts),
            image=result.mask,
            metadata=md,
        )
        if LOG.isEnabledFor(logging.DEBUG):
            try:
                LOG.debug("Prepared ImageMessage for frame %s", getattr(result.meta, "frame_id", None))
            except Exception:
                pass
        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi

            q_out = get_queue_out()
            kmsg = format_kpi({"ts": time.time(), "event": "out_event", "frame": getattr(result.meta, "frame_id", None), "q_out": q_out.qsize()})
            safe_log_kpi(kmsg)
        except Exception:
            LOG.debug("Failed to emit KPI out_event")
        # In real code: pyigtl_server.send_message(msg) or similar
    except Exception as e:
        LOG.exception("Failed to prepare/send message to Slicer: %s", e)
        return


def stop_sending_thread() -> None:
    """Arrête proprement le thread d'envoi et libère les ressources."""
    return
