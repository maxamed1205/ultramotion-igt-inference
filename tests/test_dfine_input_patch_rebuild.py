"""
tests/test_dfine_input_patch_rebuild.py
=======================================

Reconstruit le modèle D-FINE à partir de la classe DFINE,
charge les poids 'last.pt', inspecte et patch la 1ère couche Conv2d.
"""

import sys
import torch
import argparse
import logging
from pathlib import Path
import types

# -------------------------------------------------------------------
# 🔧 Configuration des chemins (import du dépôt custom_d_fine)
# -------------------------------------------------------------------
CUSTOM_DFINE_ROOT = Path(r"C:\Users\maxam\Desktop\TM\custom_d_fine\src")
if str(CUSTOM_DFINE_ROOT) not in sys.path:
    sys.path.insert(0, str(CUSTOM_DFINE_ROOT))
print(f"[DEBUG] Import path ajouté : {CUSTOM_DFINE_ROOT}")

# ⚙️ Patch virtuel pour simuler 'src.d_fine'
if "src" not in sys.modules:
    fake_src = types.ModuleType("src")
    sys.modules["src"] = fake_src
    import d_fine
    sys.modules["src.d_fine"] = d_fine
    print("[DEBUG] Module virtuel 'src' injecté → redirigé vers d_fine")

# Import du constructeur principal
from d_fine.dfine import build_model

# -------------------------------------------------------------------
# 🧠 Logger
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOG = logging.getLogger("dfine.inspect")

# -------------------------------------------------------------------
# 🔍 Inspection de la première couche Conv2d
# -------------------------------------------------------------------
def inspect_first_conv(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            LOG.info(f"🔍 Première couche Conv2d trouvée : {name}")
            LOG.info(
                f"   in_channels={module.in_channels}, out_channels={module.out_channels}, "
                f"kernel_size={module.kernel_size}, stride={module.stride}, padding={module.padding}"
            )
            return name, module
    LOG.warning("⚠️ Aucune couche Conv2d trouvée dans ce modèle.")
    return None, None

# -------------------------------------------------------------------
# 🧩 Patch 3→1
# -------------------------------------------------------------------
def patch_first_conv_to_mono(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
            new_conv = torch.nn.Conv2d(
                in_channels=1,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                bias=(module.bias is not None),
            )
            with torch.no_grad():
                new_conv.weight[:] = module.weight.mean(dim=1, keepdim=True)
                if module.bias is not None:
                    new_conv.bias[:] = module.bias
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, layer_name, new_conv)
            LOG.info(f"✅ Patch appliqué : Conv2d {name} 3→1 canaux")
            return model
    LOG.info("ℹ️ Aucun patch nécessaire (déjà mono-canal).")
    return model

# -------------------------------------------------------------------
# ⚙️ Test d’un forward mono-canal
# -------------------------------------------------------------------
def test_forward(model: torch.nn.Module, img_size=512, device="cuda"):
    device = "cuda" if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu"
    model = model.to(device)
    dummy = torch.randn(1, 1, img_size, img_size, device=device)
    try:
        model.eval()
        with torch.inference_mode():
            out = model(dummy)
        LOG.info(f"✅ Forward réussi sur {device}")
        if isinstance(out, dict):
            LOG.info(f"   Clés sorties : {list(out.keys())}")
        elif torch.is_tensor(out):
            LOG.info(f"   Tensor sortie : shape={tuple(out.shape)}")
        else:
            LOG.info(f"   Type sortie : {type(out)}")
    except Exception as e:
        LOG.error(f"❌ Forward échoué : {e}")

# -------------------------------------------------------------------
# 🚀 Script principal
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inspect & patch D-FINE model")
    parser.add_argument("--weights", type=str, required=True, help="Chemin du state_dict (last.pt)")
    parser.add_argument("--model_name", type=str, default="s", help="Taille du modèle (n, s, m, l, x)")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--patch", action="store_true")
    parser.add_argument("--save_patched", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG.info(f"🧠 Reconstruction du modèle D-FINE ({args.model_name}) sur {device}...")

    # 1️⃣ Création du modèle vide
    img_size_tuple = (args.img_size, args.img_size)
    try:
        model = build_model(args.model_name, num_classes=1, device=device, img_size=img_size_tuple)
    except Exception as e:
        LOG.warning(f"⚠️ Échec avec tuple ({img_size_tuple}), tentative fallback int : {e}")
        model = build_model(args.model_name, num_classes=1, device=device, img_size=args.img_size)

    # 2️⃣ Chargement du checkpoint
    ckpt = torch.load(args.weights, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        LOG.info("📦 Détection de clé 'state_dict' — chargement standard.")
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        LOG.info("📦 Fichier = dict pur (state_dict direct).")
        state_dict = ckpt
    else:
        LOG.error("❌ Format inattendu (pas un dict ni state_dict).")
        return

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    LOG.info(f"💾 Poids chargés : {args.weights}")
    LOG.info(f"   Clés manquantes: {len(missing)}, inattendues: {len(unexpected)}")

    # 3️⃣ Inspection + patch si demandé
    name, conv = inspect_first_conv(model)
    if conv and conv.in_channels == 3 and args.patch:
        model = patch_first_conv_to_mono(model)
        if args.save_patched:
            save_path = Path(args.weights).with_name("last_mono.pth")
            torch.save(model.state_dict(), save_path)
            LOG.info(f"💾 Modèle patché sauvegardé : {save_path}")

    # 4️⃣ Test forward
    test_forward(model, img_size=args.img_size, device=device)

# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
