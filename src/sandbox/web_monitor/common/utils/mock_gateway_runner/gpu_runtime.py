"""
gpu_runtime.py
---------------
Gère la détection GPU et l'initialisation du runtime CUDA / CPU fallback.
"""
import logging
from core.preprocessing.cpu_to_gpu import init_transfer_runtime

LOG = logging.getLogger("igt.mock.gpu")

def init_gpu_if_available():
    """Initialise le GPU si disponible, sinon utilise CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            LOG.info(f"[OK] CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            LOG.info("[WARNING] CUDA non disponible, utilisation CPU")
        
        # Initialiser le runtime de transfert
        init_transfer_runtime(device=device, pool_size=2, shape_hint=(512, 512))
        LOG.info(f"[OK] Runtime GPU initialisé (device={device})")
        return device
    except ImportError:
        LOG.warning("[WARNING] PyTorch non installé, utilisation CPU seulement")
        return "cpu"
    except Exception as e:
        LOG.warning(f"[WARNING] Erreur GPU, fallback CPU: {e}")
        return "cpu"