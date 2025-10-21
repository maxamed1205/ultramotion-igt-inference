"""
test_dfine_single_infer.py
===========================
Test unitaire d’inférence D-FINE (mono-canal)
sur une seule image échographique.
"""

import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# --- Configuration ---
IMG_PATH = r"C:\Users\maxam\Desktop\TM\PA\dataset\HUMERUS LATERAL XG SW_cropped\JPEGImages\Video_001\00135.jpg"
MODEL_PATH = r"C:\Users\maxam\Desktop\TM\custom_d_fine\output\models\dfine_fingers_2025-07-27\last_mono.pth"
CUSTOM_DFINE_SRC = Path(r"C:\Users\maxam\Desktop\TM\custom_d_fine\src")

# --- Préparer les imports ---
if str(CUSTOM_DFINE_SRC) not in sys.path:
    sys.path.insert(0, str(CUSTOM_DFINE_SRC))

# patch virtuel pour "src.d_fine"
import types
if "src" not in sys.modules:
    fake_src = types.ModuleType("src")
    sys.modules["src"] = fake_src
    import d_fine
    sys.modules["src.d_fine"] = d_fine

from d_fine.dfine import build_model

# --- Charger modèle ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Chargement modèle patché sur {device}...")



state_dict = torch.load(MODEL_PATH, map_location="cpu")

# 🧠 Déterminer nb de canaux à partir du checkpoint
first_key = next(iter(state_dict))
first_weight = state_dict[first_key]
in_ch = first_weight.shape[1] if first_weight.ndim == 4 else 3
print(f"🧩 Le modèle patché attend {in_ch} canal(s) en entrée")

# 🏗️ Construire modèle avec bon nb de canaux
model = build_model("s", num_classes=1, device=device, img_size=(640, 640))
if in_ch == 1:
    # Patch dynamique de la première couche (RGB→mono)
    from torch import nn
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=m.out_channels,
                kernel_size=m.kernel_size,
                stride=m.stride,
                padding=m.padding,
                bias=(m.bias is not None),
            )
            model.backbone.stem.stem1.conv = new_conv
            print(f"✅ Première Conv2d patchée en 1 canal : {name}")
            break

model.load_state_dict(state_dict, strict=False)

model.eval().to(device)

# --- Charger image ---
image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"❌ Image introuvable : {IMG_PATH}")
image = cv2.resize(image, (640, 640))
tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float() / 255.0
tensor = tensor.to(device)

# --- Inférence ---
with torch.inference_mode():
    output = model(tensor)

print("✅ Inférence terminée !")

# --- Résumé des sorties ---
if isinstance(output, dict):
    print(f"🧩 Sortie dict : {list(output.keys())}")
    for k, v in output.items():
        if torch.is_tensor(v):
            print(f" - {k}: {tuple(v.shape)}")
elif torch.is_tensor(output):
    print(f"🧩 Sortie tensor: {tuple(output.shape)}")
else:
    print(f"🧩 Sortie type: {type(output)}")


# --- Visualisation simple ---
if isinstance(output, dict) and "pred_boxes" in output and "pred_logits" in output:
    boxes = output["pred_boxes"][0].detach().cpu().numpy()
    scores = torch.sigmoid(output["pred_logits"][0]).detach().cpu().numpy().flatten()

    # garder la box avec le score max
    idx = np.argmax(scores)
    box = boxes[idx]
    score = scores[idx]

    # D-FINE renvoie les boxes normalisées (cx, cy, w, h) dans [0,1]
    cx, cy, w, h = box
    x0 = int((cx - w/2) * image.shape[1])
    y0 = int((cy - h/2) * image.shape[0])
    x1 = int((cx + w/2) * image.shape[1])
    y1 = int((cy + h/2) * image.shape[0])

    color = (0, 255, 0)
    disp = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(disp, (x0, y0), (x1, y1), color, 2)
    cv2.putText(disp, f"{score:.2f}", (x0 + 5, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("D-FINE inference", disp)
    print(f"📦 BBox: ({x0},{y0})→({x1},{y1}), score={score:.3f}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
