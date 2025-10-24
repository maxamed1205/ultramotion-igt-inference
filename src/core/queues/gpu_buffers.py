# ================================================================
# core/queues/gpu_buffers.py
# ================================================================
#
# ⚠️ TODO [Phase 4] — GPU Memory Pool (pré-allocation persistante)
# ---------------------------------------------------------------
# Objectif :
#   Réduire la fragmentation CUDA et le jitter en remplaçant
#   les torch.empty() répétés par un pool de tensors réutilisables.
#
# Contexte :
#   - utilisé après cpu_to_gpu.py et dfine_infer.py
#   - appelé dans detection_and_engine.initialize_models()
#   - gère uniquement la mémoire GPU (CUDA)
#
#   Inspiré du _PIN_POOL CPU (pinned memory) mais pour GPU
#   avec interface simple : init / acquire / release / clear / metrics
#
#   Aucun multi-GPU pour l’instant (device='cuda:0' fixe)
#
# ----------------------------------------------------------------

"""
core.queues.gpu_buffers
=======================

🧩 Rôle :
    Gérer un pool de tensors CUDA pré-alloués et réutilisables.

🧠 Motivation :
    Éviter les torch.empty() ou torch.zeros() à chaque frame
    pour réduire la fragmentation mémoire et la latence.

💡 Utilisation typique :
    from core.queues.gpu_buffers import init_gpu_pool, acquire_buffer, release_buffer

    init_gpu_pool(device='cuda:0', defaults={'roi': (1,1,512,512)})
    roi_buf = acquire_buffer('roi')
    # ... traitement sur GPU ...
    release_buffer('roi')

📦 Statut :
    Phase 4 — squelette initial (à implémenter dans étapes suivantes)
"""

from typing import Dict, Optional, Tuple, Any
import logging

try:
    import torch
except ImportError:
    torch = None  # type: ignore

# ----------------------------------------------------------------
# Globals
# ----------------------------------------------------------------
LOG = logging.getLogger("igt.queues.gpu_buffers")

# Registre principal du pool GPU : mapping name → {tensor, in_use, shape, device}
_GPU_POOL: Dict[str, Dict[str, Any]] = {}

# Optionnel : verrou global si accès multi-thread futur
_LOCK = None  # remplacé plus tard par threading.Lock()


# ----------------------------------------------------------------
# 1️⃣ Initialisation du pool GPU
# ----------------------------------------------------------------
def init_gpu_pool(device: str = "cuda:0", defaults: Optional[Dict[str, Tuple[int, ...]]] = None) -> None:
    """Initialise le pool GPU avec des buffers persistants standards.

    Args:
        device: identifiant du GPU ('cuda:0')
        defaults: dictionnaire optionnel {name: shape}

    Cette fonction :
        - crée le dictionnaire global _GPU_POOL
        - pré-alloue quelques tensors (roi, mask, logits, bbox)
        - logge la taille totale allouée
    """
    # ✅ Enhanced: lock + debug shapes
    global _GPU_POOL, _LOCK

    # Vérifier que torch est disponible
    if torch is None:
        LOG.warning("torch not installed; cannot initialize GPU pool")
        return

    # Déterminer si CUDA est disponible
    use_cuda = False
    try:
        use_cuda = torch.cuda.is_available()
    except Exception:
        use_cuda = False

    device_str = device
    if not use_cuda:
        LOG.warning("CUDA non disponible, fallback CPU")
        device_str = "cpu"

    dev = torch.device(device_str)

    # Si déjà initialisé, avertir et vider
    if _GPU_POOL:
        LOG.warning("GPU pool already initialized; clearing and re-allocating")
        _GPU_POOL.clear()

    # Initialiser le verrou global si nécessaire (futur accès multi-thread)
    import threading
    if _LOCK is None:
        _LOCK = threading.Lock()

    # Defaults si non fourni
    if defaults is None:
        defaults = {
            "roi": (1, 1, 512, 512),
            "mask": (1, 1, 512, 512),
            "logits": (1, 64, 32, 32),
            "bbox": (1, 4),
        }

    total_alloc_MB = 0.0

    for name, shape in defaults.items():
        try:
            tensor = torch.empty(shape, device=dev, dtype=torch.float32)
        except Exception as exc:
            LOG.warning("Failed to allocate tensor %s on %s: %s", name, dev, exc)
            # On allocation failure, try CPU as fallback
            try:
                cpu_dev = torch.device("cpu")
                tensor = torch.empty(shape, device=cpu_dev, dtype=torch.float32)
                LOG.warning("Allocated %s on CPU as fallback", name)
            except Exception as exc2:
                LOG.error("Cannot allocate buffer %s (shape=%s): %s", name, shape, exc2)
                continue

        _GPU_POOL[name] = {
            "tensor": tensor,
            "shape": tuple(shape),
            "in_use": False,
            "device": str(dev),
        }

        total_alloc_MB += tensor.numel() * tensor.element_size() / 1e6

    # Debug détaillé : shapes et devices alloués
    for name, meta in _GPU_POOL.items():
        LOG.debug("Allocated %s %s on %s", name, meta["shape"], meta["device"])

    # Logging résumé
    LOG.info(
        "GPU pool initialized (%d buffers, %.2f MB on %s)",
        len(_GPU_POOL),
        total_alloc_MB,
        str(dev),
    )

    # Si CUDA activé, logguer la mémoire GPU utilisée
    if use_cuda:
        try:
            allocated_mb = torch.cuda.memory_allocated(dev) / 1e6
            LOG.info("CUDA memory allocated on %s: %.2f MB", str(dev), allocated_mb)
        except Exception:
            # Non critique — seulement diagnostic
            LOG.debug("Unable to query torch.cuda.memory_allocated for %s", str(dev))
    return


# ----------------------------------------------------------------
# 2️⃣ Acquisition d’un buffer GPU
# ----------------------------------------------------------------
def acquire_buffer(
    name: str,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional["torch.dtype"] = None,
    device: Optional["torch.device"] = None,
    stream: Optional[Any] = None,
) -> Optional["torch.Tensor"]:
    """Renvoie un tensor GPU depuis le pool (ou le crée à la volée).

    Args:
        name: identifiant logique ('roi', 'mask', 'logits', ...)
        shape: dimensions désirées
        dtype: type des données (torch.float16, torch.float32, etc.)
        device: GPU cible
        stream: flux CUDA optionnel

    Returns:
        torch.Tensor prêt à l’emploi (déjà alloué sur le GPU)
    """
    global _GPU_POOL, _LOCK

    # Vérifier que torch est disponible
    if torch is None:
        LOG.warning("torch not available")
        return None

    # Optionnel lock pour accès thread-safe
    lock = _LOCK

    def _alloc_tensor(target_shape, target_device, target_dtype):
        try:
            t = torch.empty(target_shape, device=target_device, dtype=target_dtype)
            return t
        except Exception as exc:
            LOG.error("Allocation failed for %s shape=%s on %s: %s", name, target_shape, target_device, exc)
            return None

    # If pool empty, log and continue (do not auto-init full pool here)
    if not _GPU_POOL:
        LOG.debug("GPU pool empty; initializing minimal entry for %s", name)

    # Use lock if available
    if lock is not None:
        lock.acquire()

    try:
        # Case: buffer exists in pool
        if name in _GPU_POOL:
            meta = _GPU_POOL[name]

            if meta.get("in_use", False):
                LOG.warning("%s already in use; returning None", name)
                return None

            # Determine requested shape/dtype/device
            current_shape = tuple(meta.get("shape", ()))
            req_shape = tuple(shape) if shape is not None else current_shape

            # Determine device to use: prefer provided, else meta device
            try:
                dev_obj = device if device is not None else torch.device(meta.get("device", "cuda:0"))
            except Exception:
                dev_obj = torch.device(meta.get("device", "cuda:0"))

            req_dtype = dtype if dtype is not None else meta.get("tensor").dtype

            # If shape differs, reallocate (log warning)
            if req_shape != current_shape:
                LOG.warning("Shape mismatch for '%s': requested %s != existing %s — reallocating", name, req_shape, current_shape)
                new_t = _alloc_tensor(req_shape, dev_obj, req_dtype)
                if new_t is None:
                    return None
                meta["tensor"] = new_t
                meta["shape"] = tuple(req_shape)
                meta["device"] = str(dev_obj)

            # Optionally bind to stream (no-op placeholder)
            if stream is not None and hasattr(torch.cuda, "stream"):
                try:
                    with torch.cuda.stream(stream):
                        _ = meta["tensor"]
                except Exception:
                    # ignore stream binding failures for now
                    LOG.debug("Stream association skipped for %s", name)

            meta["in_use"] = True
            return meta["tensor"]

        # Case: buffer does not exist -> create new
        # Need a shape to create
        if shape is None:
            LOG.error("Cannot create buffer '%s' without shape", name)
            return None

        # Determine device to allocate on
        if device is not None:
            dev_obj = device
        else:
            # prefer CUDA if available
            try:
                dev_obj = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            except Exception:
                dev_obj = torch.device("cuda:0")

        req_dtype = dtype if dtype is not None else torch.float32

        new_tensor = _alloc_tensor(tuple(shape), dev_obj, req_dtype)
        if new_tensor is None:
            return None

        _GPU_POOL[name] = {
            "tensor": new_tensor,
            "shape": tuple(shape),
            "in_use": True,
            "device": str(dev_obj),
        }

        LOG.info("Allocated new buffer '%s' %s on %s", name, tuple(shape), str(dev_obj))

        # Stream association placeholder
        if stream is not None and hasattr(torch.cuda, "stream"):
            try:
                with torch.cuda.stream(stream):
                    _ = new_tensor
            except Exception:
                LOG.debug("Stream association skipped for new buffer %s", name)

        return new_tensor
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


# ----------------------------------------------------------------
# 3️⃣ Libération d’un buffer
# ----------------------------------------------------------------
def release_buffer(name: str) -> None:
    """Marque un buffer comme libre dans le pool (réutilisable).

    Args:
        name: nom du buffer à libérer
    """
    global _GPU_POOL, _LOCK

    # Vérifier que torch est disponible
    if torch is None:
        LOG.warning("torch not available; cannot release buffer")
        return

    # Si le pool est vide
    if not _GPU_POOL:
        LOG.debug("GPU pool empty, nothing to release")
        return

    # Acquérir le verrou si défini
    lock = _LOCK
    if lock is not None:
        lock.acquire()

    try:
        if name not in _GPU_POOL:
            LOG.warning("release_buffer: '%s' not found in pool", name)
            return

        meta = _GPU_POOL[name]

        if not meta.get("in_use", False):
            LOG.debug("release_buffer: '%s' already free", name)
            return

        # Marquer comme libre
        meta["in_use"] = False
        LOG.debug("Released buffer '%s' (%s)", name, meta.get("shape"))

        # Placeholder for future GPU sync (phase later)
        # if torch.cuda.is_available():
        #     try:
        #         torch.cuda.synchronize()
        #     except Exception:
        #         pass

        return
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


# ----------------------------------------------------------------
# 4️⃣ Nettoyage complet du pool
# ----------------------------------------------------------------
def clear_gpu_pool() -> None:
    """Libère explicitement tous les tensors GPU et vide le pool."""
    global _GPU_POOL, _LOCK

    # Vérifier que torch est disponible
    if torch is None:
        LOG.warning("torch not available; cannot clear GPU pool")
        return

    # Si already empty
    if not _GPU_POOL:
        LOG.debug("GPU pool already empty")
        return

    # Acquérir verrou si défini
    lock = _LOCK
    if lock is not None:
        lock.acquire()

    try:
        # Parcourir une copie des items pour pouvoir supprimer en sécurité
        for name, meta in list(_GPU_POOL.items()):
            t = meta.get("tensor")
            if t is not None:
                try:
                    # supprimer la référence au tensor pour permettre le GC
                    del t
                except Exception:
                    pass
            LOG.debug("Cleared buffer '%s' from pool", name)

        # Synchronisation GPU et vidage du cache si disponible
        try:
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    LOG.debug("torch.cuda.synchronize failed during clear_gpu_pool")
                try:
                    torch.cuda.empty_cache()
                    LOG.info("CUDA cache cleared after GPU pool cleanup")
                except Exception as exc:
                    LOG.debug("torch.cuda.empty_cache failed: %s", exc)
        except Exception:
            # Si la vérification torch.cuda.is_available échoue, ignorer
            pass

        # Vider le registre global
        _GPU_POOL.clear()
        LOG.info("GPU pool cleared and all buffers released")
        return
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


# ----------------------------------------------------------------
# 5️⃣ Métriques et diagnostic
# ----------------------------------------------------------------
def collect_gpu_metrics() -> Dict[str, Any]:
    """Retourne un dictionnaire résumant l’état du pool GPU.

    Exemple de sortie :
        {
            'roi': {'shape': (1,1,512,512), 'in_use': False, 'allocated_MB': 0.5},
            'mask': {'shape': (1,1,512,512), 'in_use': True, 'allocated_MB': 0.5},
        }
    """
    global _GPU_POOL, _LOCK

    # Vérifier que torch est disponible
    if torch is None:
        LOG.warning("torch not available; cannot collect GPU metrics")
        return {}

    # Si pool vide
    if not _GPU_POOL:
        return {}

    metrics: Dict[str, Any] = {}
    total_alloc_MB = 0.0
    in_use_count = 0

    lock = _LOCK
    if lock is not None:
        lock.acquire()

    try:
        for name, meta in _GPU_POOL.items():
            t = meta.get("tensor")
            if t is None:
                continue
            try:
                alloc_mb = float(t.numel() * t.element_size() / 1e6)
            except Exception:
                alloc_mb = None

            in_use = bool(meta.get("in_use", False))
            if in_use:
                in_use_count += 1

            if alloc_mb:
                try:
                    total_alloc_MB += float(alloc_mb)
                except Exception:
                    pass

            metrics[name] = {
                "shape": meta.get("shape"),
                "in_use": in_use,
                "device": meta.get("device"),
                "allocated_MB": alloc_mb,
            }

        # Summary
        metrics["_summary"] = {
            "total_buffers": len(_GPU_POOL),
            "in_use_count": in_use_count,
            "total_alloc_MB": float(total_alloc_MB),
        }

        return metrics
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception:
                pass


# ----------------------------------------------------------------
# ✅ Fin du squelette Phase 4
# ----------------------------------------------------------------
