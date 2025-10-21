# -*- coding: utf-8 -*-
"""
core/inference/dfine_infer.py
=============================

Fast D-FINE Inference Engine (Process C1)
-----------------------------------------
Moteur d’inférence **D-FINE** sur GPU, conçu pour s’intégrer avec :
- core.preprocessing.cpu_to_gpu (Process B, fournit GpuFrame.tensor déjà sur GPU)
- core.inference.detection_and_engine (orchestration + FSM + SAM)
- core.monitoring.kpi (KPI logs)

Contraintes :
- Entrée = torch.Tensor [1,1,H,W] float32, device='cuda' (mono-canal, déjà normalisé)
- Zéro recopie CPU (sauf scalars pour KPI / sorties)
- Support des streams CUDA pour overlap CPU↔GPU
"""

# ⚠️ TODO: [Phase 2] Exporter le modèle D-FINE en ONNX (mono-canal, 512x512) pour préparation TensorRT FP16.
# ⚠️ TODO: [Phase 2] Implémenter infer_dfine_trt() basé sur un moteur TensorRT (.engine) une fois la pipeline stabilisée.
# ⚠️ TODO: [Phase 2] Ajouter un batching opportuniste (2–4 frames) quand la scène est stable pour augmenter le throughput.

from __future__ import annotations

import time
import logging
from typing import Any, Optional, Tuple

import numpy as np
import torch

LOG = logging.getLogger("igt.dfine")
LOG_KPI = logging.getLogger("igt.kpi")

# -----------------------------
# Feature flags (V3.1)
# -----------------------------
ENABLE_FP16: bool = True              # Active l'autocast FP16
USE_CHANNELS_LAST: bool = False       # Bascule mémoire channels_last (optionnel, à activer si tout le chemin est compatible)
SAFE_CLAMP_INPLACE: bool = True       # Clamp [0,1] en sécurité numérique (in-place)
STRICT_SHAPE_CHECK: bool = True       # Assertions strictes sur la forme des tenseurs
# Hooks V4
ENABLE_CUDA_GRAPH: bool = False       # Capture CUDA Graphs quand H=W=fixe et modèle figé
# ⚠️ TODO [Phase 4] : Implémenter la capture d'un CUDA Graph fixe
#  - Nécessite un warmup (quelques frames) avec H=W constants (ex: 512x512)
#  - Capturer preprocess_frame_for_dfine() + model() dans un torch.cuda.CUDAGraph
#  - Rejouer ensuite via graph.replay() pour éviter le surcoût Python
#  - Gain attendu: +5–15 % selon GPU

# =============================
# 0. Utilitaires internes
# =============================

def _xywh_to_xyxy_xy_abs(box_xywh: torch.Tensor) -> torch.Tensor:
    """Convertit xywh → xyxy (absolu)."""
    x, y, w, h = box_xywh.unbind(-1)
    return torch.stack((x, y, x + w, y + h), dim=-1)

def _cxcywh_norm_to_xyxy_abs(box_cxcywh: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Convertit cxcywh normalisé [0..1] → xyxy absolu (pixels)."""
    cx, cy, w, h = box_cxcywh.unbind(-1)
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return torch.stack((x1, y1, x2, y2), dim=-1)

def _clip_xyxy_inplace(box_xyxy: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
    """Clippe la bbox dans les bornes de l’image (in-place si possible)."""
    box_xyxy[..., 0].clamp_(0, img_w - 1)
    box_xyxy[..., 1].clamp_(0, img_h - 1)
    box_xyxy[..., 2].clamp_(0, img_w - 1)
    box_xyxy[..., 3].clamp_(0, img_h - 1)
    return box_xyxy


# ============================================================
# 1. Prétraitement minimal GPU
# ============================================================

def preprocess_frame_for_dfine(frame_gpu: torch.Tensor) -> torch.Tensor:
    """Prépare le tensor [1,1,H,W] pour D-FINE (mono-canal natif, pas de duplication).
    - Vérifie device / forme.
    - Optionnel : channels_last, clamp de sécurité.
    - Zéro copie CPU.
    """
    if STRICT_SHAPE_CHECK:
        assert isinstance(frame_gpu, torch.Tensor), "frame_gpu doit être un torch.Tensor"
        assert frame_gpu.is_cuda, "frame_gpu doit être sur CUDA"
        assert frame_gpu.ndim == 4 and frame_gpu.shape[0] == 1 and frame_gpu.shape[1] == 1, \
            f"Attendu [1,1,H,W], reçu {tuple(frame_gpu.shape)}"

    x = frame_gpu
    if USE_CHANNELS_LAST:
        # Attention : ne pas casser upstream ; s'assurer que le modèle accepte channels_last.
        x = x.contiguous(memory_format=torch.channels_last)

    if SAFE_CLAMP_INPLACE:
        x = x.clamp_(0.0, 1.0)

    return x


# ============================================================
# 2. Inférence principale (asynchrone)
# ============================================================

@torch.inference_mode()
def infer_dfine(model: torch.nn.Module,
                frame_mono: torch.Tensor,
                stream: Optional[torch.cuda.Stream] = None) -> dict:
    """Exécute le forward D-FINE sur GPU de manière non-bloquante (côté CPU).
    - Respecte le stream fourni (overlap CPU↔GPU).
    - Active l'autocast FP16 si ENABLE_FP16.
    - Pas de synchronize() ici; la synchro arrive quand on lit les scalars en post-process.
    """
    if stream is None:
        stream = torch.cuda.current_stream()

    outputs: dict
    try:
        with torch.cuda.stream(stream):
            if ENABLE_FP16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(frame_mono)
            else:
                outputs = model(frame_mono)
        # Pas de sync ici : on laisse vivre la latence GPU pour overlap avec le CPU amont.
        return outputs

    except torch.cuda.OutOfMemoryError:
        LOG.warning("D-FINE OOM sur GPU → tentative fallback CPU (dégradé).")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        frame_cpu = frame_mono.detach().to("cpu", non_blocking=False)
        # En CPU, pas d’autocast
        outputs = model(frame_cpu)
        return outputs


# ============================================================
# 3. Post-traitement (décodage prédictions)
# ============================================================

def postprocess_dfine(outputs: dict,
                      img_size: Tuple[int, int],
                      conf_thresh: float = 0.3) -> Tuple[Optional[np.ndarray], float]:
    """Extrait la bbox la plus confiante (bbox_t en xyxy absolu, conf_t).
    Supporte 2 conventions d'outputs :
      - Style DETR: outputs['pred_logits'] (N,C), outputs['pred_boxes'] (N,4) en cxcywh normalisé.
      - Style torchvision: outputs['scores'] (N,), outputs['boxes'] (N,4) en xyxy absolu.
    """
    W, H = img_size  # img_size = (W, H)

    # Branches de compatibilité
    if "scores" in outputs and "boxes" in outputs:
        # torchvision-like
        scores: torch.Tensor = outputs["scores"]  # (N,)
        boxes_xyxy: torch.Tensor = outputs["boxes"]  # (N,4) absolu
        if scores.numel() == 0:
            return None, 0.0
        idx: int = int(torch.argmax(scores))
        conf = float(scores[idx].detach().item())
        if conf < conf_thresh:
            return None, conf
        box_xyxy = boxes_xyxy[idx].detach()
        _clip_xyxy_inplace(box_xyxy, W, H)
        return box_xyxy.to(dtype=torch.float32).cpu().numpy(), conf

    # DETR-like
    if "pred_logits" in outputs and "pred_boxes" in outputs:
        logits: torch.Tensor = outputs["pred_logits"]  # (N,C)
        boxes: torch.Tensor = outputs["pred_boxes"]    # (N,4) (souvent cxcywh normalisé)
        if logits.ndim == 2:
            # score = max sigmoid(logits) par instance
            scores = torch.sigmoid(logits).amax(dim=1)  # (N,)
        else:
            raise ValueError(f"pred_logits shape inattendue: {tuple(logits.shape)}")

        if scores.numel() == 0:
            return None, 0.0
        idx: int = int(torch.argmax(scores))
        conf = float(scores[idx].detach().item())
        if conf < conf_thresh:
            return None, conf

        b = boxes[idx].detach()
        # Heuristique de format : si <=1.5 → considéré comme normalisé (cxcywh)
        if torch.max(b) <= 1.5:
            box_xyxy = _cxcywh_norm_to_xyxy_abs(b, W, H)
        else:
            # Sinon on suppose xywh absolu (fallback générique)
            box_xyxy = _xywh_to_xyxy_xy_abs(b)

        _clip_xyxy_inplace(box_xyxy, W, H)
        return box_xyxy.to(dtype=torch.float32).cpu().numpy(), conf

    # Si format non reconnu
    LOG.error("Format d'outputs D-FINE non reconnu. Clés: %s", list(outputs.keys()))
    return None, 0.0


# ============================================================
# 4. Routine unifiée
# ============================================================

def run_dfine_detection(model: torch.nn.Module,
                        frame_gpu: torch.Tensor | Any,
                        stream: Optional[torch.cuda.Stream] = None,
                        conf_thresh: float = 0.3) -> Tuple[Optional[np.ndarray], float]:
    """
    Pipeline complet : frame_gpu → preprocess → infer → postprocess
    Entrées :
        - model : module D-FINE déjà chargé/patché mono-canal.
        - frame_gpu : soit un torch.Tensor [1,1,H,W] (CUDA), soit un objet GpuFrame {tensor, stream}.
        - stream : stream CUDA cible pour l'inférence D-FINE (peut être distinct du stream de copy_async).
    Sortie :
        (bbox_t [x1,y1,x2,y2] en pixels, conf_t float)
    """
    # Support GpuFrame (objet) ou tenseur direct
    in_tensor = getattr(frame_gpu, "tensor", frame_gpu)
    in_stream = getattr(frame_gpu, "stream", None)

    if STRICT_SHAPE_CHECK:
        assert isinstance(in_tensor, torch.Tensor) and in_tensor.is_cuda, "frame_gpu.tensor doit être un Tensor CUDA [1,1,H,W]"

    # Déduis H,W pour clipping et conversion
    _, _, H, W = in_tensor.shape

    # Prépare le stream
    if stream is None:
        stream = torch.cuda.current_stream()

    # ⚠️ TODO: s'assurer que frame_gpu.stream (copy_async) est bien propagé jusqu'ici 
    # Dans la pipeline complète, D-FINE agit comme un "validateur" de visibilité.
    # Il détermine si une bbox/conf valide existe (→ état VISIBLE). 
    # Le calcul MobileSAM (sam_encoder/sam_decoder) ne doit donc pas être lancé
    # tant que l'état est LOST ou que la bbox_t est absente.
    # En phase d'optimisation (V3.2), on pourra lancer sam_encoder en parallèle
    # sur stream_sam *uniquement* quand une prédiction D-FINE est plausible,
    # tout en respectant stream_dfine.wait_stream(copy_stream) pour la cohérence.
    if in_stream is not None and isinstance(in_stream, torch.cuda.Stream):
        stream.wait_stream(in_stream)

    # Temps CPU global (la lecture des scalars forceront la sync partielle)
    t0 = time.time()

    # Prétraitement minimal (no-op logique en mono-canal natif)
    x = preprocess_frame_for_dfine(in_tensor)

    # Inférence (asynchrone côté CPU)
    outputs = infer_dfine(model, x, stream=stream)

    # Post-traitement (l’accès aux scalars déclenche la sync nécessaire)
    bbox_t, conf_t = postprocess_dfine(outputs, img_size=(W, H), conf_thresh=conf_thresh)

    # KPI minimal (non-bloquant – l’accès à conf_t a déjà synchronisé le strict nécessaire)
    t1 = time.time()
    infer_ms = (t1 - t0) * 1000.0
    try:
        mem_mb = torch.cuda.memory_allocated() / 1e6
    except Exception:
        mem_mb = 0.0

    try:
        LOG_KPI.info(
            "event=infer_dfine ts=%.3f infer_ms=%.2f gpu_mem_MB=%.1f conf_max=%.3f H=%d W=%d",
            time.time(), infer_ms, mem_mb, float(conf_t) if conf_t is not None else -1.0, H, W
        )
    except Exception:
        pass

    return bbox_t, conf_t


__all__ = [
    "preprocess_frame_for_dfine",
    "infer_dfine",
    "postprocess_dfine",
    "run_dfine_detection",
]
