import numpy as np
from typing import Tuple

def compute_mask_weights(mask_t: np.ndarray, width_edge: int = 3, width_out: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit les trois pondérations spatiales W_edge, W_in, W_out à partir du mask.

    Currently a stub — implement morphological operations (distance transforms,
    dilation/erosion) to generate edge, inside and outside weight maps.
    """
    raise NotImplementedError
