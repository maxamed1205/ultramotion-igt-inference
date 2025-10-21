# ⚠️ TODO [Phase 3] : ROI crop 100 % GPU
# Utiliser torch.nn.functional.grid_sample() ou roi_align pour découper la ROI
# directement sur frame_t.cuda(), sans .cpu() ni numpy slicing.
# Attention : effectuer le crop seulement si state_hint == "VISIBLE".
# Cela remplacera la version CPU actuelle (plus lente).

"""
Module d’inférence combinée D-FINE + MobileSAM (Process C)
==========================================================

💡 Mise à jour 20/10/2025 — intégration du pré-processing “FSM mask-aware”
---------------------------------------------------------------------------
Ce module gère désormais **les étapes 0 → 3** du pipeline de visibilité :

    0. Détection globale via D-FINE (bbox_t, conf_t)
    1. Gating macro : si conf_t < τ_conf → LOST direct
    2. Crop ROI autour de bbox_t
    3. Segmentation fine via MobileSAM
    4. Calcul des pondérations spatiales (W_edge, W_in, W_out)

Il constitue le moteur d’inférence (Process C) entre la réception GPU
et le module FSM (`core.state_machine.visibility_fsm`).

Flux typique :
    RawFrame (CPU) → cpu_to_gpu.py → GpuFrame (torch.Tensor)
    → detection_and_engine.prepare_inference_inputs()
    → visibility_fsm.evaluate_visibility()
    → ResultPacket → Gateway._outbox → Slicer

Rôle général
------------
- charger et initialiser les modèles IA (D-FINE et MobileSAM)
- exécuter l’inférence D-FINE → bbox/conf
- effectuer le crop ROI et la segmentation MobileSAM
- générer les pondérations spatiales utilisées pour les scores S₁–S₄
- fournir un dictionnaire normalisé pour le module FSM

Entrées attendues
-----------------
frame_tensor : image 2D échographique (numpy | torch)
model_paths  : dict avec au moins les clés {"dfine", "mobilesam"}

Sortie type
-----------
{
    "state_hint": "VISIBLE" | "LOST",
    "bbox_t": <tuple[int,int,int,int]> | None,
    "conf_t": <float>,
    "roi": <ndarray> | None,
    "mask_t": <ndarray> | None,
    "W_edge": <ndarray> | None,
    "W_in": <ndarray> | None,
    "W_out": <ndarray> | None,
}

Implémentations à prévoir
-------------------------
- initialize_models()        → chargement GPU des deux modèles
- run_detection()            → inférence D-FINE (bbox/conf)
- run_segmentation()         → inférence MobileSAM
- compute_mask_weights()     → calcul morphologique des pondérations
- prepare_inference_inputs() → orchestration globale (étapes 0 → 3)

Le reste du Process C (run_inference/fuse_outputs/process_inference_once)
reste valide pour la compatibilité pipeline et KPI.
"""

from typing import Any, Dict, Tuple, Optional, TYPE_CHECKING
import threading
import io
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
else:
    try:
        import torch
        import torch.nn as nn
    except Exception:  # pragma: no cover - environment may not have torch in tests
        torch = None  # type: ignore
        nn = None  # type: ignore
import logging
import numpy as np
import time

from core.types import GpuFrame, ResultPacket
from core.queues.buffers import get_queue_gpu, get_queue_out, try_dequeue, enqueue_nowait_out

LOG = logging.getLogger("igt.inference")
LOG_KPI = logging.getLogger("igt.kpi")


# Global cache and lock for loaded models
_MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
_MODEL_LOCK = threading.Lock()


def resolve_precision(req: str, device_obj: "torch.device") -> str:
    """Résout dynamiquement la précision effective ('fp32', 'fp16', 'bf16').

    Règles:
    - device CPU -> 'fp32'
    - req == 'auto' -> fp16 si CC >= 7, sinon fp32
    - req == 'bf16' -> vérifie le support bf16 sinon fallback fp32
    - sinon retourne la requête
    """
    # use the module-level `torch` which may be None in some test environments
    if device_obj.type == "cpu":
        return "fp32"
    if req == "auto":
        try:
            if torch is None:
                return "fp32"
            cc = torch.cuda.get_device_capability(device_obj) if hasattr(torch.cuda, "get_device_capability") else (6, 0)
            major = cc[0] if isinstance(cc, (list, tuple)) else 6
            return "fp16" if major >= 7 else "fp32"
        except Exception:
            return "fp32"
    if req == "bf16" and hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
        LOG.warning("bf16 not supported; fallback to fp32")
        return "fp32"
    return req


# ============================================================
# 1. Chargement et initialisation des modèles
# ============================================================

def initialize_models(model_paths: Dict[str, str], device: str = "cuda") -> Dict[str, Any]:
    """Charge et initialise les modèles D-FINE et MobileSAM.

    Args:
        model_paths: dictionnaire contenant les chemins {'dfine': ..., 'mobilesam': ...}
        device: 'cuda' ou 'cpu'

    Returns:
        dict {'dfine': model_dfine, 'mobilesam': model_sam}
    """
    if not isinstance(model_paths, dict) or "dfine" not in model_paths or "mobilesam" not in model_paths:
        raise ValueError("model_paths must contain 'dfine' and 'mobilesam'")

    if torch is None:
        raise ImportError("torch is required to initialize models")

    # Resolve device availability
    req_device = device or "cpu"
    if req_device.startswith("cuda") and not torch.cuda.is_available():
        LOG.warning("CUDA unavailable, fallback to CPU")
        req_device = "cpu"

    torch_device = torch.device(req_device)

    # Resolve precision
    precision_req = model_paths.get("precision", "auto")

    precision_effective = resolve_precision(precision_req, torch_device)

    # Build cache key
    cache_key = f"{torch_device.type}:{precision_effective}:{model_paths['dfine']}:{model_paths['mobilesam']}"

    with _MODEL_LOCK:
        if cache_key in _MODEL_CACHE:
            LOG.debug("Using cached models for %s", cache_key)
            return _MODEL_CACHE[cache_key]

    # Helper: load model from path into CPU memory using BytesIO
    def load_model_async(path: str):
        try:
            with open(path, "rb") as f:
                buf = f.read()
            bio = io.BytesIO(buf)
            return torch.load(bio, map_location="cpu")
        except Exception:
            LOG.exception("Failed to load model from %s", path)
            return make_stub_model(path)

    def make_stub_model(name: str):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 1, 3, padding=1)
            def forward(self, x):
                return torch.sigmoid(self.conv(x))
        m = Dummy()
        try:
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)
        except Exception:
            LOG.debug("Failed to set eval/grad flags on Dummy stub")
        LOG.warning("Using Dummy stub model for %s", name)
        return m

    # Parallel CPU loading
    t_start = time.perf_counter()
    dfine_load_ms = sam_load_ms = 0.0
    with ThreadPoolExecutor(max_workers=2) as executor:
        fut_dfine = executor.submit(load_model_async, model_paths["dfine"])
        fut_sam = executor.submit(load_model_async, model_paths["mobilesam"])
        try:
            t0 = time.perf_counter()
            dfine = fut_dfine.result()
            t1 = time.perf_counter()
            dfine_load_ms = (t1 - t0) * 1000.0
        except Exception:
            LOG.exception("Exception while loading dfine model")
            dfine = make_stub_model("dfine")
            dfine_load_ms = 0.0
        try:
            t0 = time.perf_counter()
            sam = fut_sam.result()
            t1 = time.perf_counter()
            sam_load_ms = (t1 - t0) * 1000.0
        except Exception:
            LOG.exception("Exception while loading mobilesam model")
            sam = make_stub_model("mobilesam")
            sam_load_ms = 0.0

    # Prepare models on CPU (inference-ready)
    try:
        with torch.inference_mode():
            for m in (dfine, sam):
                try:
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad_(False)
                    # keep CPU in fp32
                    m.float()
                except Exception:
                    LOG.debug("Model %s does not support parameters iteration", type(m))
    except Exception:
        LOG.debug("torch.inference_mode not available, skipping")

    # Transfer to GPU with streams if requested
    compiled_flag = bool(model_paths.get("compile", False))
    ts_flag = bool(model_paths.get("torchscript", False))

    if torch_device.type == "cuda":
        try:
            # ⚠️ TODO [Phase 3] : ajuster les priorités de streams
            # D-FINE = haute priorité (priority=-1)
            # SAM    = priorité normale (priority=0)
            stream_dfine = torch.cuda.Stream()
            stream_sam = torch.cuda.Stream()
            with torch.inference_mode():
                with torch.cuda.stream(stream_dfine):
                    dfine = dfine.to(torch_device, non_blocking=True)
                with torch.cuda.stream(stream_sam):
                    sam = sam.to(torch_device, non_blocking=True)
            torch.cuda.synchronize()

            # cast precision on device
            if precision_effective == "fp16":
                try:
                    dfine = dfine.half()
                    sam = sam.half()
                except Exception:
                    LOG.debug("Half cast failed")
            if precision_effective == "bf16":
                try:
                    dfine = dfine.bfloat16()
                    sam = sam.bfloat16()
                except Exception:
                    LOG.debug("BFloat16 cast failed")
        except Exception:
            LOG.exception("GPU transfer failed, falling back to CPU models")
            dfine = dfine.to("cpu") if hasattr(dfine, "to") else dfine
            sam = sam.to("cpu") if hasattr(sam, "to") else sam
            torch_device = torch.device("cpu")
    else:
        # define streams as None on CPU so out dict always has the key
        stream_dfine = None
        stream_sam = None

    # Optional TorchScript / compile - grouped for readability
    try:
        if ts_flag:
            for name, mdl in [("dfine", dfine), ("mobilesam", sam)]:
                try:
                    scripted = torch.jit.script(mdl)
                    if name == "dfine":
                        dfine = scripted
                    else:
                        sam = scripted
                except Exception:
                    LOG.warning("TorchScript failed for %s", name)

        if compiled_flag and hasattr(torch, "compile"):
            for name, mdl in [("dfine", dfine), ("mobilesam", sam)]:
                try:
                    compiled = torch.compile(mdl, mode="reduce-overhead")
                    if name == "dfine":
                        dfine = compiled
                    else:
                        sam = compiled
                except Exception:
                    LOG.debug("torch.compile failed for %s", name)
    except Exception as e:
        LOG.debug("Optional optimization block failed: %s", e)

    t_end = time.perf_counter()
    total_ms = (t_end - t_start) * 1000.0

    # KPI logging
    try:
        from core.monitoring.kpi import safe_log_kpi, format_kpi
        # try to collect GPU memory consumption when available
        gpu_mem_MB = 0.0
        if torch_device.type == "cuda":
            try:
                gpu_mem_MB = torch.cuda.memory_allocated(torch_device) / 1e6
            except Exception:
                gpu_mem_MB = -1.0
        LOG.debug("GPU memory allocated after init: %.1f MB", gpu_mem_MB)

        kdata = {
            "ts": time.time(),
            "event": "init_models",
            "device": str(torch_device),
            "precision": precision_effective,
            "compiled": compiled_flag,
            "torchscripted": ts_flag,
            "dfine_load_ms": round(dfine_load_ms, 2),
            "sam_load_ms": round(sam_load_ms, 2),
            "total_ms": round(total_ms, 2),
            "gpu_mem_MB": round(gpu_mem_MB, 1),
            "dfine_path": model_paths["dfine"],
            "mobilesam_path": model_paths["mobilesam"],
        }
        safe_log_kpi(format_kpi(kdata))
    except Exception:
        LOG.debug("Failed to emit KPI init_models")

    out = {
        "device": torch_device,
        "dfine": dfine,
        "mobilesam": sam,
        "meta": {
            "dfine_path": model_paths["dfine"],
            "mobilesam_path": model_paths["mobilesam"],
            "precision": precision_effective,
            "compiled": compiled_flag,
            "torchscripted": ts_flag,
            "ts": time.time(),
        },
        "streams": {
            "dfine": stream_dfine,
            "mobilesam": stream_sam,
        },
    }

    with _MODEL_LOCK:
        _MODEL_CACHE[cache_key] = out

    return out


# ============================================================
# 2. Étapes d’inférence mask-aware
# ============================================================

def run_detection(dfine_model: Any, frame_tensor: Any) -> Tuple[Tuple[int, int, int, int], float]:
    """Exécute le modèle D-FINE et renvoie (bbox_t, conf_t)."""
    raise NotImplementedError


def run_segmentation(sam_model: Any, roi: np.ndarray) -> Optional[np.ndarray]:
    """Exécute MobileSAM sur la ROI et retourne le mask binaire."""
    raise NotImplementedError


def compute_mask_weights(mask_t: np.ndarray, width_edge: int = 3, width_out: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit les trois pondérations spatiales W_edge, W_in, W_out à partir du mask."""
    raise NotImplementedError


def prepare_inference_inputs(frame_t: np.ndarray, dfine_model: Any, sam_model: Any, tau_conf: float = 0.5) -> Dict[str, Any]:
    """
    Orchestration complète des étapes 0 → 3.

    0. Passe l’image dans D-FINE → bbox/conf.
    1. Si conf < τ_conf → renvoie state_hint='LOST'.
    2. Crop ROI autour de la bbox.
    3. Passe la ROI dans MobileSAM.
    4. Calcule les pondérations spatiales (W_edge/W_in/W_out).

    Returns:
        dictionnaire prêt pour visibility_fsm.evaluate_visibility().
    """
    raise NotImplementedError


# ============================================================
# 3. Compatibilité Process C (inférence générique)
# ============================================================

def run_inference(frame_tensor: GpuFrame, stream_infer: Any = None) -> Tuple[ResultPacket, float]:
    """Exécute (mock) l’inférence GPU et retourne un ResultPacket minimal."""
    raise NotImplementedError


def fuse_outputs(mask: Any, score: float, state: str) -> ResultPacket:
    """Fusionne les sorties et renvoie un ResultPacket standardisé."""
    raise NotImplementedError


# ============================================================
# 4. Routine de boucle unique (mock actuelle)
# ============================================================

def process_inference_once(models: Any = None) -> None:
    """Consomme une GpuFrame, exécute une inférence (mock) et place le résultat en sortie."""
    q_gpu = get_queue_gpu()
    gf = try_dequeue(q_gpu)
    if gf is None:
        return

    if LOG.isEnabledFor(logging.DEBUG):
        fid = getattr(getattr(gf, "meta", None), "frame_id", None)
        LOG.debug("Dequeued GpuFrame for inference: %s", fid)

    # Simulation minimale d’inférence
    t0 = time.time()
    result: ResultPacket = {
        "frame_id": getattr(getattr(gf, "meta", None), "frame_id", None),
        "mask": None,
        "score": 1.0,
        "state": "OK",
        "timestamp": getattr(getattr(gf, "meta", None), "ts", None),
    }  # type: ignore[assignment]
    t1 = time.time()
    latency_ms = (t1 - t0) * 1000.0

    try:
        from core.monitoring.kpi import safe_log_kpi, format_kpi
        msg = format_kpi({"ts": t1, "event": "infer_event", "frame": result.get("frame_id"), "latency_ms": f"{latency_ms:.1f}"})
        safe_log_kpi(msg)
    except Exception:
        LOG.debug("Failed to emit KPI infer_event")

    try:
        q_out = get_queue_out()
        ok = enqueue_nowait_out(q_out, result)  # type: ignore[arg-type]
        if not ok:
            LOG.warning("Out queue full, result for frame %s dropped", result.get("frame_id"))
    except Exception as e:
        LOG.exception("Failed to enqueue result: %s", e)
