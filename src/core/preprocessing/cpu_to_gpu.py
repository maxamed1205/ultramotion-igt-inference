"""
CPU → GPU Transfer Utility (Process B)
======================================

💡 Mise à jour 20/10/2025 — version “single copy_async”
-------------------------------------------------------

Ce module correspond au **Process B** dans la pipeline Ultramotion (A→B→C→D) :

    A. Acquisition         → service/plus_client.py (PlusServer → RawFrame)
    👉 B. Préprocessing     → core/preprocessing/cpu_to_gpu.py (RawFrame → GpuFrame)
    C. Inférence           → core/inference/detection_and_engine.py (GpuFrame → ResultPacket)
    D. Sortie vers Slicer  → service/slicer_server.py (ResultPacket → 3D Slicer)

Rôle du module
--------------
Assurer **un seul transfert CPU→GPU par frame**, de manière **asynchrone** et **non bloquante**,
avant l’inférence D-FINE + MobileSAM. Ce module effectue :

1. la conversion numpy → torch.Tensor,  
2. la normalisation et mise en forme (channels_first, dtype),  
3. l’allocation en mémoire “pinned” (fixée CPU pour transfert rapide),  
4. la copie asynchrone vers le GPU (copy_async sur un stream dédié).

Une fois la frame convertie en `GpuFrame`, elle est déposée dans `Queue_GPU`
et utilisée directement par `detection_and_engine.py` sans nouvelle copie.

Avantages :
-----------
- ✅ Une seule copie CPU→GPU pour tout le pipeline.
- ✅ Transfert asynchrone (copy + compute se chevauchent).
- ✅ Compatible CUDA streams pour D-FINE / MobileSAM.
- ✅ Facile à synchroniser en aval via `stream.wait_stream()`.

Flux schématique :
------------------
    RawFrame (numpy)
         │
         ▼
    prepare_frame_for_gpu()
         │
     copy_async CUDA stream
         │
         ▼
    GpuFrame(tensor, meta, stream)
         │
         ▼
    Queue_GPU  → detection_and_engine.py

Fonctions principales :
-----------------------
- `prepare_frame_for_gpu(frame, device)` : prépare et transfert CPU→GPU (asynchrone)
- `transfer_to_gpu_async(tensor, stream)` : copie GPU dédiée (optionnelle)
- `process_one_frame()` : consomme une frame RT et alimente Queue_GPU

Ce fichier définit les squelettes des fonctions, à implémenter dans la phase GPU
du Process B. Aucun calcul réel n’est effectué ici.
"""

from typing import Any, Optional
import logging
import time

from core.types import RawFrame, GpuFrame
from core.queues.buffers import (
    get_queue_rt_dyn,
    get_queue_gpu,
    try_dequeue,
    enqueue_nowait_gpu,
)


LOG = logging.getLogger("igt.gpu")
LOG_KPI = logging.getLogger("igt.kpi")


# ======================================================================
# 1. Préparation et transfert CPU → GPU
# ======================================================================

def prepare_frame_for_gpu(frame: RawFrame, device: str = "cuda", config: Optional[dict] = None) -> GpuFrame:
    """
    Prépare une frame CPU (numpy) pour le GPU, via un **seul transfert asynchrone**.

    Étapes attendues (à implémenter) :
      1. Validation de la forme et du dtype (float32, [C,H,W]).
      2. Normalisation des intensités (ex : /255.0 ou mean/std).
      3. Allocation mémoire CPU en pinned memory.
      4. Transfert asynchrone vers GPU via `torch.cuda.Stream`.

    Args:
        frame: RawFrame contenant l’image CPU (numpy array ou buffer équivalent).
        device: cible du transfert ('cuda', 'cuda:0', etc.)
        config: dictionnaire optionnel (normalisation, dtype, scale, etc.)

    Returns:
        GpuFrame : objet contenant le tensor GPU, les métadonnées et le stream CUDA associé.

    Notes :
        - L’objectif est d’assurer que **toute la pipeline C (inférence)** travaille
          sur ce tensor unique sans reconversion CPU→GPU.
        - Le transfert asynchrone permet le chevauchement avec le calcul du frame précédent.
    """
    raise NotImplementedError


def transfer_to_gpu_async(tensor: Any, stream_transfer: Optional[Any] = None, device: str = "cuda") -> Any:
    """
    Effectue un transfert asynchrone CPU→GPU sur un stream CUDA fourni.

    Args:
        tensor: tensor CPU (torch.Tensor sur device='cpu').
        stream_transfer: stream CUDA dédié au transfert.
        device: cible GPU ('cuda', 'cuda:0', etc.).

    Returns:
        tensor GPU (torch.Tensor sur le device cible).

    Comportement attendu :
        - Utiliser `with torch.cuda.stream(stream_transfer):`
          puis `tensor.to(device, non_blocking=True)`.
        - Retourner le tensor GPU pour réutilisation immédiate.
    """
    raise NotImplementedError


# ======================================================================
# 2. Intégration dans la boucle pipeline A→B
# ======================================================================

def process_one_frame(config: Optional[dict] = None) -> None:
    """
    Consomme une RawFrame depuis la queue temps réel, effectue la préparation GPU,
    et dépose le résultat dans Queue_GPU (étape B du pipeline).

    Étapes :
        1. Défile une frame de Queue_RT_dyn (non bloquant).
        2. Appelle prepare_frame_for_gpu().
        3. Envoie le GpuFrame vers Queue_GPU (non bloquant).

    Notes :
        - Les métriques de saturation Queue_GPU sont publiées via KPI logs.
        - Les transferts asynchrones permettent le chevauchement CPU/GPU.
    """
    q_rt = get_queue_rt_dyn()
    raw = try_dequeue(q_rt)
    if raw is None:
        return

    if LOG.isEnabledFor(logging.DEBUG):
        fid = getattr(getattr(raw, "meta", None), "frame_id", None)
        LOG.debug("Dequeued RawFrame %s from RT queue", fid)

    # Préparation GPU (asynchrone)
    gpu_frame = prepare_frame_for_gpu(raw)

    # Envoi vers la queue GPU
    q_gpu = get_queue_gpu()
    ok = enqueue_nowait_gpu(q_gpu, gpu_frame)

    if not ok:
        LOG.warning("GPU queue full, frame %s dropped", getattr(raw, "meta", None) and getattr(raw.meta, "frame_id", None))
        try:
            from core.monitoring.kpi import safe_log_kpi, format_kpi
            msg = format_kpi({
                "ts": time.time(),
                "event": "gpu_saturation",
                "frame": getattr(raw, "meta", None) and getattr(raw.meta, "frame_id", None),
                "q_gpu": q_gpu.qsize(),
            })
            safe_log_kpi(msg)
        except Exception:
            LOG.debug("Failed to emit KPI gpu_saturation")

    return