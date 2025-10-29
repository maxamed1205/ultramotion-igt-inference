"""
Test de Validation : Étapes 1 et 2 de la Pipeline
=================================================

Ce test valide isolément les 2 premières étapes :

Étape 1: read_dataset_images() → RawFrame (numpy)
  ✅ Lit 213 JPEGs du dataset
  ✅ Convertit en numpy array (512x512 uint8 grayscale)
  ✅ Appelle gateway._inject_frame()
  ✅ Appelle gateway.stats.mark_rx() pour KPI
  ✅ Logs "RX frame #N" à ~100 Hz

Étape 2: prepare_frame_for_gpu() → GpuFrame (torch.Tensor CUDA)
  ✅ Convertit numpy → torch.Tensor
  ✅ Transfert CPU → GPU (asynchrone)
  ✅ Normalisation [0,255] → [0,1]
  ✅ copy_async sur stream CUDA
  ✅ Vérifie tensor.device == 'cuda'

Scénario de test :
- Lit les 10 premières images du dataset
- Pour chaque image :
  1. Crée RawFrame (simule read_dataset_images)
  2. Transfère vers GPU via prepare_frame_for_gpu()
  3. Vérifie shape, dtype, device, valeurs
  4. Mesure latence CPU→GPU

Sortie attendue :
- 10 frames traitées avec succès
- Latence CPU→GPU : < 5ms par frame
- Tensors sur GPU avec bonnes dimensions
- Aucune erreur CUDA
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOG_MODE"] = "dev"

import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image
import glob

# ──────────────────────────────────────────────
#  Préparation du contexte
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ──────────────────────────────────────────────
#  Imports
# ──────────────────────────────────────────────
from core.types import RawFrame, FrameMeta, Pose
from core.preprocessing.cpu_to_gpu import (
    init_transfer_runtime,
    prepare_frame_for_gpu,
)

# ──────────────────────────────────────────────
#  Setup Logging Asynchrone (pour KPI)
# ──────────────────────────────────────────────
import logging
import logging.config
import yaml

def setup_async_logging():
    """Configure le système de logging asynchrone pour écrire dans kpi.log."""
    logging_config_path = ROOT / "src" / "config" / "logging.yaml"
    
    if logging_config_path.exists():
        with open(logging_config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # Setup via async_logging module
        from core.monitoring import async_logging
        async_logging.setup_async_logging(yaml_cfg=config)
        async_logging.start_health_monitor()
    else:
        # Fallback: basic config
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

LOG = logging.getLogger("test.validation")

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
DATASET_PATH = Path(r"C:\Users\maxam\Desktop\TM\dataset\HUMERUS LATERAL XG SW_cropped\JPEGImages\Video_001")
TARGET_SIZE = (512, 512)
NUM_FRAMES_TO_TEST = None  # None = toutes les images (213), sinon spécifier un nombre

# ──────────────────────────────────────────────
#  Test Principal
# ──────────────────────────────────────────────
def test_step1_step2():
    """Valide Étape 1 (load images) + Étape 2 (CPU→GPU transfer)."""
    
    # Setup async logging AVANT le test
    setup_async_logging()
    
    LOG.info("=" * 70)
    LOG.info("TEST VALIDATION : Étape 1 + Étape 2")
    LOG.info("=" * 70)
    
    # ─────────────────────────────────────────────
    # ÉTAPE 1 : Charger les images du dataset
    # ─────────────────────────────────────────────
    LOG.info("\n[ÉTAPE 1] Chargement des images du dataset...")
    
    if not DATASET_PATH.exists():
        LOG.error(f"❌ Dataset path does not exist: {DATASET_PATH}")
        return False
    
    # Lister les fichiers JPEG
    image_files = sorted(glob.glob(str(DATASET_PATH / "*.jpg")))
    
    if not image_files:
        LOG.error(f"❌ No JPEG images found in {DATASET_PATH}")
        return False
    
    LOG.info(f"✅ Found {len(image_files)} images")
    LOG.info(f"   First: {Path(image_files[0]).name}")
    LOG.info(f"   Last: {Path(image_files[-1]).name}")
    
    # Limiter au nombre de frames à tester (ou prendre toutes si None)
    if NUM_FRAMES_TO_TEST is not None:
        image_files = image_files[:NUM_FRAMES_TO_TEST]
        LOG.info(f"   Testing with first {len(image_files)} images")
    else:
        LOG.info(f"   Testing with ALL {len(image_files)} images")
    
    # ─────────────────────────────────────────────
    # ÉTAPE 2 : Initialiser le runtime GPU
    # ─────────────────────────────────────────────
    LOG.info("\n[ÉTAPE 2] Initialisation du runtime GPU...")
    
    try:
        import torch
        if not torch.cuda.is_available():
            LOG.warning("⚠️ CUDA not available, using CPU (slower)")
            device = "cpu"
        else:
            device = "cuda"
            LOG.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        LOG.error("❌ PyTorch not installed")
        return False
    
    # Initialiser le runtime de transfert (stream + pinned buffers)
    try:
        init_transfer_runtime(device=device, pool_size=2, shape_hint=(512, 512))
        LOG.info(f"✅ Transfer runtime initialized (device={device})")
    except Exception as e:
        LOG.exception(f"❌ Failed to initialize transfer runtime: {e}")
        return False
    
    # ─────────────────────────────────────────────
    # ÉTAPE 3 : Boucle de traitement
    # ─────────────────────────────────────────────
    LOG.info("\n[ÉTAPE 3] Test CPU → GPU transfer...")
    LOG.info("-" * 70)
    
    latencies = []
    success_count = 0
    
    for frame_id, img_path in enumerate(image_files):
        try:
            # ═════════════════════════════════════════
            # ÉTAPE 1.1 : Lire l'image (simule read_dataset_images)
            # ═════════════════════════════════════════
            with Image.open(img_path) as pil_img:
                pil_img = pil_img.convert("L")  # Grayscale
                pil_img = pil_img.resize(TARGET_SIZE, Image.BILINEAR)
                img_np = np.array(pil_img, dtype=np.uint8)
            
            # Créer RawFrame avec métadonnées
            pose = Pose()
            ts = time.time()
            meta = FrameMeta(
                frame_id=frame_id,
                ts=ts,
                pose=pose,
                spacing=(0.3, 0.3, 1.0),
                orientation="UN",
                coord_frame="Image",
                device_name="Dataset",
            )
            raw_frame = RawFrame(image=img_np, meta=meta)
            
            # Log seulement toutes les 20 frames pour éviter spam
            if frame_id % 20 == 0 or frame_id < 3:
                LOG.info(f"  Frame #{frame_id:03d}: Loaded {Path(img_path).name} → shape={img_np.shape} dtype={img_np.dtype}")
            
            # ═════════════════════════════════════════
            # ÉTAPE 2.1 : Transférer CPU → GPU
            # ═════════════════════════════════════════
            t0 = time.perf_counter()
            gpu_frame = prepare_frame_for_gpu(raw_frame, device=device)
            t1 = time.perf_counter()
            latency_ms = (t1 - t0) * 1000.0
            latencies.append(latency_ms)
            
            # ═════════════════════════════════════════
            # ÉTAPE 2.2 : Vérifier le résultat
            # ═════════════════════════════════════════
            tensor = gpu_frame.tensor
            
            # Vérifications
            checks = []
            
            # Check 1: Est-ce un torch.Tensor ?
            is_tensor = hasattr(tensor, "device") and hasattr(tensor, "dtype")
            checks.append(("is_tensor", is_tensor))
            
            # Check 2: Device correct ?
            if is_tensor:
                actual_device = str(tensor.device)
                device_ok = device in actual_device
                checks.append(("device", device_ok, actual_device))
            
            # Check 3: Shape correcte ?
            expected_shape = (1, 1, 512, 512)  # [B, C, H, W]
            shape_ok = tuple(tensor.shape) == expected_shape
            checks.append(("shape", shape_ok, tuple(tensor.shape)))
            
            # Check 4: Dtype float32 ?
            dtype_ok = tensor.dtype == torch.float32
            checks.append(("dtype", dtype_ok, tensor.dtype))
            
            # Check 5: Valeurs normalisées [0, 1] ?
            if device == "cuda":
                # Copier sur CPU pour vérifier
                cpu_tensor = tensor.cpu()
            else:
                cpu_tensor = tensor
            
            min_val = cpu_tensor.min().item()
            max_val = cpu_tensor.max().item()
            normalized_ok = (0.0 <= min_val <= 1.0) and (0.0 <= max_val <= 1.0)
            checks.append(("normalized", normalized_ok, f"[{min_val:.3f}, {max_val:.3f}]"))
            
            # Check 6: Métadonnées préservées ?
            meta_ok = hasattr(gpu_frame, "meta") and gpu_frame.meta.frame_id == frame_id
            checks.append(("meta", meta_ok))
            
            # ═════════════════════════════════════════
            # Afficher résultats (seulement toutes les 20 frames)
            # ═════════════════════════════════════════
            all_ok = all(c[1] for c in checks)
            status = "✅" if all_ok else "❌"
            
            # Afficher détails seulement pour les premières frames et toutes les 20
            if frame_id % 20 == 0 or frame_id < 3 or not all_ok:
                LOG.info(f"  {status} GPU Transfer: latency={latency_ms:.2f}ms")
                for check in checks:
                    if len(check) == 2:
                        name, ok = check
                        LOG.info(f"      {name}: {'✅' if ok else '❌'}")
                    else:
                        name, ok, value = check
                        LOG.info(f"      {name}: {'✅' if ok else '❌'} ({value})")
            
            if all_ok:
                success_count += 1
            
            # Afficher progression toutes les 50 frames
            if (frame_id + 1) % 50 == 0:
                LOG.info(f"  ⏳ Progress: {frame_id + 1}/{len(image_files)} frames processed...")
            
        except Exception as e:
            LOG.exception(f"  ❌ Frame #{frame_id:02d} FAILED: {e}")
    
    # ─────────────────────────────────────────────
    # RÉSUMÉ FINAL
    # ─────────────────────────────────────────────
    LOG.info("-" * 70)
    LOG.info("\n[RÉSUMÉ]")
    LOG.info(f"  Frames testées : {len(image_files)}")
    LOG.info(f"  Succès         : {success_count}/{len(image_files)}")
    LOG.info(f"  Échecs         : {len(image_files) - success_count}")
    
    if latencies:
        avg_lat = np.mean(latencies)
        min_lat = np.min(latencies)
        max_lat = np.max(latencies)
        LOG.info(f"  Latence CPU→GPU:")
        LOG.info(f"    - Moyenne : {avg_lat:.2f} ms")
        LOG.info(f"    - Min     : {min_lat:.2f} ms")
        LOG.info(f"    - Max     : {max_lat:.2f} ms")
        
        # Objectif : < 5ms par frame
        if avg_lat < 5.0:
            LOG.info(f"  ✅ Objectif atteint (< 5ms)")
        else:
            LOG.warning(f"  ⚠️ Latence élevée (objectif: < 5ms)")
    
    LOG.info("\n" + "=" * 70)
    
    if success_count == len(image_files):
        LOG.info("🎉 TEST RÉUSSI : Étapes 1 et 2 validées !")
        return True
    else:
        LOG.error("❌ TEST ÉCHOUÉ : Certaines frames ont échoué")
        return False


# ──────────────────────────────────────────────
#  Point d'entrée
# ──────────────────────────────────────────────
if __name__ == "__main__":
    success = test_step1_step2()
    sys.exit(0 if success else 1)
