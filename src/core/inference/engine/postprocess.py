import numpy as np
from typing import Tuple
import cv2

def compute_mask_weights(mask_t: np.ndarray, width_edge: int = 3, width_out: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit les trois pondérations spatiales W_edge, W_in, W_out à partir du mask.
    Currently a stub — implement morphological operations (distance transforms,
    dilation/erosion) to generate edge, inside and outside weight maps.
    """
    mask = (mask_t > 0.5).astype(np.uint8)
    kernel_edge = np.ones((width_edge, width_edge), np.uint8)
    kernel_out = np.ones((width_out, width_out), np.uint8)

    dilated = cv2.dilate(mask, kernel_edge, iterations=1)
    eroded = cv2.erode(mask, kernel_edge, iterations=1)
    edge = (dilated - eroded).astype(np.float32)
    inner = eroded.astype(np.float32)
    outer = ((cv2.dilate(mask, kernel_out, iterations=1) - mask) > 0).astype(np.float32)

    return edge, inner, outer
