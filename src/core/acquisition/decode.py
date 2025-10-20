"""
⚠️ [Deprecated / Legacy Notice]
================================
Ce module n’est **plus utilisé** dans l’architecture actuelle d’Ultramotion IGT Inference.

⏳ Ancien rôle :
----------------
`decode.py` servait à convertir les messages IGTLink (`IMAGE`, `TRANSFORM`, etc.)
en structures internes `RawFrame(image, meta)` via la fonction `decode_igt_image()`.

Cela permettait autrefois d’isoler la logique de décodage du thread réseau.

🚀 Nouvelle architecture :
--------------------------
Depuis la refonte du pipeline, cette étape est directement intégrée dans :
    → `service/plus_client.py`  (fonction `run_plus_client()`)

Désormais :
  - Le thread RX lancé par `IGTGateway.start()` via `THREAD_REGISTRY["rx"]`
    appelle `run_plus_client()`.
  - Cette fonction reçoit les messages IGTLink (via pyigtl),
    convertit immédiatement le buffer en `np.ndarray`,
    crée un `FrameMeta` et un `RawFrame`,
    puis les empile dans la file `mailbox` du Gateway.

En d’autres termes :
    PlusServer → run_plus_client() → RawFrame(image, meta) → mailbox
et non plus :
    PlusServer → decode_igt_image() → RawFrame → mailbox

💡 Ce fichier est conservé uniquement :
  - pour compatibilité ascendante (anciens scripts/tests),
  - comme référence pour le décodage IGTLink pur (hors thread réseau).

"""


"""IGTLink message decoding helpers.

Ce module contient des fonctions pures pour convertir les messages IGTLink
(image + tracking) en structures internes RawFrame + FrameMeta.
Cela permet de tester le décodage indépendamment du thread réseau.
"""

from typing import Any, Optional
import numpy as np
import logging

from core.types import RawFrame, FrameMeta

LOG = logging.getLogger("igt.decode")


def decode_igt_image(image_message: Any, tracking_message: Any) -> Optional[RawFrame]:
    """Convertit un message IGTLink en RawFrame (zero-copy si possible).

    Args:
        image_message: message image IGTLink (doit exposer .image, .height, .width)
        tracking_message: message tracking associé (non utilisé pour l'instant)

    Returns:
        RawFrame ou None si la conversion échoue.
    """
    try:
        buf = getattr(image_message, "image", None)
        if buf is None:
            return None

        arr = None
        try:
            arr = np.frombuffer(buf, dtype=np.uint8)
            h = getattr(image_message, "height", None)
            w = getattr(image_message, "width", None)
            if h is not None and w is not None and arr.size == h * w:
                arr = arr.reshape((h, w))
        except Exception:
            try:
                arr = np.array(buf, copy=True)
            except Exception:
                return None

        meta = FrameMeta.from_igt(image_message)
        rf = RawFrame(image=arr, meta=meta)

        # KPI debug facultatif (même que dans _igt_callback actuel)
        try:
            from core.monitoring.kpi import is_kpi_enabled
            if is_kpi_enabled():
                import logging as _logging
                KPI = _logging.getLogger("igt.kpi")
                KPI.debug(
                    "acq.frame_id=%s bytes=%d shape=%s",
                    getattr(meta, "frame_id", -1),
                    int(arr.nbytes) if hasattr(arr, "nbytes") else -1,
                    getattr(arr, "shape", None),
                )
        except Exception:
            pass

        return rf
    except Exception as e:
        LOG.debug("decode_igt_image failed: %r", e)
        return None
