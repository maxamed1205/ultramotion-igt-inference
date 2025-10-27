from typing import Any, Tuple
from core.inference.dfine_infer import run_dfine_detection

def run_detection(dfine_model: Any, frame_tensor: Any) -> Tuple[Tuple[int, int, int, int], float]:    
    """Exécute le modèle D-FINE et renvoie (bbox_t, conf_t).
    Currently a stub — implement model-specific logic here.
    """
    bbox_t, conf_t = run_dfine_detection(dfine_model, frame_tensor)
    return bbox_t, conf_t