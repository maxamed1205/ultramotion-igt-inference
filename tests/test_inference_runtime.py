"""
Test d’intégration pour le moteur d’inférence D-FINE + MobileSAM.

Ce test charge les deux modèles, exécute une inférence de segmentation
sur une image simulée (aléatoire) et affiche les temps d’exécution.

Chemins attendus :
- core/model/Dfine_last_mono.pth
- core/model/mobile_sam.pt
"""

import sys
import os

# Racine du projet
ROOT = os.path.dirname(os.path.dirname(__file__))

# Ajoute à sys.path le dossier qui contient 'mobile_sam'
MOBILE_SAM_PARENT = os.path.join(ROOT, "src", "core", "inference", "MobileSAM")
if MOBILE_SAM_PARENT not in sys.path:
    sys.path.insert(0, MOBILE_SAM_PARENT)

# Vérification facultative (debug)
if not os.path.exists(os.path.join(MOBILE_SAM_PARENT, "mobile_sam")):
    print(f"⚠️ Dossier 'mobile_sam' introuvable dans {MOBILE_SAM_PARENT}")



import time
import numpy as np
import logging
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from core.inference.engine.model_loader import initialize_models
from core.inference.engine.inference_sam import run_segmentation

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("test.inference")

# ==============================================================
# 1. Configuration des chemins
# ==============================================================

ROOT = os.path.dirname(os.path.dirname(__file__))
DFINE_PATH = os.path.join(ROOT, "src", "core", "model", "Dfine_last_mono.pth")
MOBILESAM_PATH = os.path.join(ROOT, "src", "core", "model", "mobile_sam.pt")

assert os.path.exists(DFINE_PATH), f"DFINE non trouvé: {DFINE_PATH}"
assert os.path.exists(MOBILESAM_PATH), f"MobileSAM non trouvé: {MOBILESAM_PATH}"

# ==============================================================
# 2. Chargement des modèles
# ==============================================================

model_paths = {
    "dfine": DFINE_PATH,
    "mobilesam": MOBILESAM_PATH,
    "precision": "auto",
}

t0 = time.perf_counter()
models = initialize_models(model_paths, device="cuda")
t1 = time.perf_counter()

LOG.info("✅ Modèles chargés en %.2f s", t1 - t0)
LOG.info("Device utilisé: %s", models["device"])
LOG.info("Précision effective: %s", models["meta"]["precision"])

# ==============================================================
# 3. Test d’inférence MobileSAM
# ============================================================== 

sam_model = models["mobilesam"]

# Création d'une image factice 256×256
# Chargement d'une image réelle (00157.jpg) fournie à la racine du projet
IMAGE_PATH = os.path.join(ROOT, "00157.jpg")
assert os.path.exists(IMAGE_PATH), f"Image 00157.jpg non trouvée: {IMAGE_PATH}"

# Ouvre l'image, force RGB puis redimensionne à 512x512 (bilinear)
with Image.open(IMAGE_PATH) as im:
    im = im.convert("RGB")
    im = im.resize((512, 512), Image.BILINEAR)
    image = np.array(im, dtype=np.uint8)

LOG.info("🧠 Lancement segmentation test (MobileSAM)...")
t0 = time.perf_counter()
mask = run_segmentation(sam_model, image)
t1 = time.perf_counter()

if mask is not None:
    LOG.info("✅ Segmentation réussie en %.2f ms, masque shape=%s", (t1 - t0) * 1000, mask.shape)
else:
    LOG.warning("⚠️ Aucune sortie de segmentation reçue")

# ==============================================================
# 3.5 Visualisation
# ============================================================== 

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Colonne 1: Image originale
axes[0].imshow(image)
axes[0].set_title("Image Originale", fontsize=14, fontweight='bold')
axes[0].axis('off')

# Colonne 2: Image avec détection DFINE (placeholder - DFINE retourne des boîtes)
# Pour l'instant, on affiche l'image de base en attendant l'intégration DFINE complète
axes[1].imshow(image)
axes[1].set_title("Détections DFINE (À implémenter)", fontsize=14, fontweight='bold')
axes[1].axis('off')

# Colonne 3: Image avec segmentation MobileSAM par-dessus
if mask is not None:
    # Créer une overlay avec alpha blending
    segmentation_colored = np.zeros_like(image)
    segmentation_colored[mask > 0] = [0, 255, 0]  # Vert pour les zones segmentées
    
    # Mélanger l'image originale et la segmentation
    overlay = image.copy().astype(float)
    overlay[mask > 0] = 0.5 * image[mask > 0].astype(float) + 0.5 * segmentation_colored[mask > 0].astype(float)
    overlay = overlay.astype(np.uint8)
    
    axes[2].imshow(overlay)
    axes[2].set_title("Segmentation MobileSAM", fontsize=14, fontweight='bold')
else:
    axes[2].imshow(image)
    axes[2].set_title("Segmentation (Échec)", fontsize=14, fontweight='bold', color='red')

axes[2].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(ROOT, "inference_visualization.png"), dpi=150, bbox_inches='tight')
LOG.info("📊 Visualisation sauvegardée dans %s", os.path.join(ROOT, "inference_visualization.png"))
plt.show()

# ==============================================================
# 4. Résumé global
# ==============================================================

LOG.info("Test terminé. MobileSAM = %s | DFINE = %s", type(sam_model).__name__, type(models["dfine"]).__name__)

if __name__ == "__main__":
    print("\n✅ Test complet exécuté.\n")


def test_inference_runtime():
    """Vérifie que MobileSAM produit bien un masque sur une image factice."""
    # Réutilise les objets calculés au module-level: `mask` doit exister
    assert 'mask' in globals(), "Le test d'inférence n'a pas initialisé la variable 'mask'"
    assert mask is not None, "La segmentation n’a pas produit de masque"
    # Le masque doit avoir au moins 1x1
    assert hasattr(mask, 'shape'), "Le masque n'a pas d'attribut shape"
    assert mask.shape[0] > 0 and mask.shape[1] > 0, "Le masque est vide"
    LOG.info("✅ test_inference_runtime: masque valide de shape=%s", mask.shape)
