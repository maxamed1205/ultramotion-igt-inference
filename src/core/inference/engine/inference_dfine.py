from typing import Any, Tuple, Union, Optional
import torch
import numpy as np
from core.inference.dfine_infer import run_dfine_detection

def run_detection(dfine_model: Any, 
                  frame_tensor: Any, 
                  allow_cpu_fallback: bool = False,
                  return_gpu_tensor: bool = True,
                  stream: Optional[torch.cuda.Stream] = None,
                  conf_thresh: float = 0.3) -> Tuple[Union[np.ndarray, torch.Tensor], Union[float, torch.Tensor]]:    
    """Exécute le modèle D-FINE et renvoie (bbox_t, conf_t).
    
    Args:
        allow_cpu_fallback: Si True, permet fallback CPU en cas d'OOM (défaut False)
        return_gpu_tensor: Si True, retourne torch.Tensor GPU au lieu de np.ndarray (défaut True)
        stream: Stream CUDA optionnel pour inférence asynchrone
        conf_thresh: Seuil de confiance pour détection
    
    Returns:
        (bbox_t, conf_t) où bbox_t peut être torch.Tensor GPU ou np.ndarray selon return_gpu_tensor
    """
    bbox_t, conf_t = run_dfine_detection(
        dfine_model, 
        frame_tensor,
        stream=stream,
        conf_thresh=conf_thresh,
        allow_cpu_fallback=allow_cpu_fallback,
        return_gpu_tensor=return_gpu_tensor
    )
    return bbox_t, conf_t