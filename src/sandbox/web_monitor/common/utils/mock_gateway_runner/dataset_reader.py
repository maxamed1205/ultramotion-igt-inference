"""
dataset_reader.py
-----------------
Lit un dossier d'images et alimente la pipeline.
"""
import time
import glob
import threading
import logging
import numpy as np
from pathlib import Path
from PIL import Image

from core.types import RawFrame, FrameMeta, Pose
from service.gateway.manager import IGTGateway 

LOG = logging.getLogger("igt.mock.rx")

# ──────────────────────────────────────────────
#  DATASET PATH (à adapter si nécessaire)
# ──────────────────────────────────────────────
DATASET_PATH = Path(r"C:\Users\maxam\Desktop\TM\dataset\HUMERUS LATERAL XG SW_cropped\JPEGImages\Video_001")

# ──────────────────────────────────────────────
#  Lecteur d'images réelles (remplace simulate_frame_source)
# ──────────────────────────────────────────────
def read_dataset_images(
    gateway: IGTGateway,
    stop_event: threading.Event,
    frame_ready: threading.Event,
    fps: int = 100,
    target_size: tuple = (512, 512),
    loop_mode: bool = False  # Nouveau paramètre : boucler ou s'arrêter après toutes les images
):
    """Lit séquentiellement les images du dataset et les injecte dans la pipeline.
    
    Args:
        gateway: Instance IGTGateway
        stop_event: Signal d'arrêt
        frame_ready: Signal de synchronisation avec PROC thread
        fps: Cadence d'envoi (Hz)
        target_size: Taille de redimensionnement (H, W)
        loop_mode: Si True, boucle indéfiniment. Si False, s'arrête après avoir envoyé toutes les images.
    """
    # ─────────────────────────────────────────────
    # 1. Charger la liste des images
    # ─────────────────────────────────────────────
    if not DATASET_PATH.exists():
        LOG.error(f"Dataset path not found: {DATASET_PATH}")
        LOG.error("Please update DATASET_PATH in the script")
        return
    
    image_files = sorted(glob.glob(str(DATASET_PATH / "*.jpg")))
    
    if not image_files:
        LOG.error(f"No JPEG images found in {DATASET_PATH}")
        return
    
    num_images = len(image_files)
    LOG.info(f"[DATASET-RX] Loaded {num_images} images from {DATASET_PATH.name}")
    LOG.info(f"[DATASET-RX] First image: {Path(image_files[0]).name}")
    LOG.info(f"[DATASET-RX] Last image: {Path(image_files[-1]).name}")
    LOG.info(f"[DATASET-RX] Target FPS: {fps} Hz (interval: {1000.0/fps:.1f} ms)")
    
    # ─────────────────────────────────────────────
    # 2. Boucle d'envoi à cadence régulière
    # ─────────────────────────────────────────────
    frame_id = 0
    image_idx = 0  # Index dans la liste d'images
    interval = 1.0 / fps
    next_frame_time = time.perf_counter()
    
    while not stop_event.is_set():
        # Lire l'image courante
        img_path = image_files[image_idx]
        
        try:
            # Charger et redimensionner l'image
            with Image.open(img_path) as pil_img:
                # Convertir en niveaux de gris (comme les images simulées étaient mono-canal)
                pil_img = pil_img.convert("L")  # Grayscale
                pil_img = pil_img.resize(target_size, Image.BILINEAR)
                img = np.array(pil_img, dtype=np.uint8)
            
            # Créer la frame avec métadonnées
            pose = Pose()
            ts = time.time()
            meta = FrameMeta(
                frame_id=frame_id,
                ts=ts,
                pose=pose,
                spacing=(0.3, 0.3, 1.0),
                orientation="UN",
                coord_frame="Image",
                device_name="Dataset",  # Identifier la source
            )
            frame = RawFrame(image=img, meta=meta)
            
            # Log toutes les 10 frames pour éviter spam
            if frame_id % 10 == 0:
                LOG.info(
                    f"[DATASET-RX] Frame #{frame_id:03d} | Image: {Path(img_path).name} | Shape: {img.shape}"
                )
            
            # Injection dans la pipeline
            gateway._inject_frame(frame)
            frame_ready.set()  # Signal pour PROC thread
            
            # ✅ INSTRUMENTATION : Enregistrer RX timestamp pour calcul de latence
            gateway.stats.mark_rx(frame_id, ts)
            
            # Avancer les compteurs
            frame_id += 1
            image_idx += 1
            
            # Si on a envoyé toutes les images
            if image_idx >= num_images:
                # Mode boucle : recommencer à 0
                LOG.info(f"[DATASET-RX] Fin du dataset atteinte, redémarrage en boucle...")
                break
            
        except Exception as e:
            LOG.error(f"[DATASET-RX] Failed to load {img_path}: {e}")
            # Passer à l'image suivante
            image_idx = (image_idx + 1) % num_images
            continue
        
        # ─────────────────────────────────────────────
        # Sleep compensé pour maintenir FPS constant
        # ─────────────────────────────────────────────
        next_frame_time += interval
        now = time.perf_counter()
        sleep_duration = next_frame_time - now
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        # Si en retard, continuer immédiatement
    
    LOG.info(f"[DATASET-RX] Stopped after {frame_id} frames")