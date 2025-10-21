"""
tools/inspect_dfine_input.py
============================

Diagnostic D-FINE input layer (Conv2d)
--------------------------------------
Inspecte et teste la compatibilité mono-canal du modèle D-FINE.

Usage :
    python tools/inspect_dfine_input.py --model "C:/Users/maxam/Desktop/TM/custom_d_fine/output/models/dfine_fingers_2025-07-27/last.pt" --img_size 512
"""

import torch
import argparse
import logging
from pathlib import Path

# -------------------------------------------------------------
# Configuration du logger
# -------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
LOG = logging.getLogger("dfine.inspect")


# -------------------------------------------------------------
# Inspecter la première couche de convolution
# -------------------------------------------------------------
def inspect_first_conv(model: torch.nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            LOG.info(f"🔍 Première couche Conv2d trouvée : {name}")
            LOG.info(f"   in_channels={module.in_channels}, out_channels={module.out_channels}, "
                     f"kernel_size={module.kernel_size}, stride={module.stride}, padding={module.padding}")
            return name, module
    LOG.warning("⚠️ Aucune couche Conv2d trouvée dans ce modèle.")
    return None, None


# -------------------------------------------------------------
# Patcher Conv2d(3→1)
# -------------------------------------------------------------
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
            # Moyenne pondérée des 3 canaux → 1 canal
            with torch.no_grad():
                new_conv.weight[:] = module.weight.mean(dim=1, keepdim=True)
                if module.bias is not None:
                    new_conv.bias[:] = module.bias
            # Remplacement dans le modèle
            parent_name = ".".join(name.split(".")[:-1])
            layer_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, layer_name, new_conv)
            LOG.info(f"✅ Patch appliqué : Conv2d {name} 3→1 canaux")
            return model
    LOG.info("ℹ️ Aucun patch nécessaire (déjà mono-canal).")
    return model


# -------------------------------------------------------------
# Test d'un forward sur une image mono-canal
# -------------------------------------------------------------
def test_forward(model: torch.nn.Module, img_size: int = 512, device: str = "cuda"):
    if torch.cuda.is_available() and device.startswith("cuda"):
        model = model.to(device)
        dummy = torch.randn(1, 1, img_size, img_size, device=device)
    else:
        device = "cpu"
        model = model.to(device)
        dummy = torch.randn(1, 1, img_size, img_size)

    try:
        model.eval()
        with torch.inference_mode():
            out = model(dummy)
        LOG.info(f"✅ Forward réussi sur {device}")
        if isinstance(out, dict):
            LOG.info(f"   Clés sorties : {list(out.keys())}")
            for k, v in out.items():
                if torch.is_tensor(v):
                    LOG.info(f"   {k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
        elif torch.is_tensor(out):
            LOG.info(f"   Tensor sortie : shape={tuple(out.shape)}, dtype={out.dtype}")
        else:
            LOG.info(f"   Type sortie : {type(out)}")
    except Exception as e:
        LOG.error(f"❌ Échec du forward : {e}")


# -------------------------------------------------------------
# Script principal
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inspect and patch D-FINE input layer")
    parser.add_argument("--model", type=str, required=True, help="Chemin vers le modèle .pt/.pth")
    parser.add_argument("--img_size", type=int, default=512, help="Taille d'image d'entrée (H=W)")
    parser.add_argument("--patch", action="store_true", help="Appliquer le patch 3→1 automatiquement")
    parser.add_argument("--save_patched", action="store_true", help="Sauvegarder le modèle patché")
    parser.add_argument("--device", type=str, default="cuda", help="cuda ou cpu")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        LOG.error(f"Modèle introuvable : {model_path}")
        return

    LOG.info(f"🧠 Chargement du modèle : {model_path}")

    # Safe loading: prefer weights_only=True when available to avoid
    # unpickling arbitrary objects. Fall back for older torch versions.
    def safe_torch_load(path):
        import inspect

        load_kwargs = {"map_location": "cpu"}
        try:
            sig = inspect.signature(torch.load)
            if "weights_only" in sig.parameters:
                load_kwargs["weights_only"] = True
        except Exception:
            # If we can't introspect, just try without weights_only
            pass

        try:
            return torch.load(path, **load_kwargs)
        except TypeError:
            # Older torch: retry without weights_only
            return torch.load(path, map_location="cpu")

    loaded = None
    try:
        loaded = safe_torch_load(model_path)
    except Exception as e:
        LOG.error(f"❌ Échec du chargement du fichier : {e}")
        return

    # Handle common checkpoint formats
    model = None
    if isinstance(loaded, torch.nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        # Common keys: 'model' (module or state_dict), 'state_dict'
        if "model" in loaded:
            if isinstance(loaded["model"], torch.nn.Module):
                model = loaded["model"]
            elif isinstance(loaded["model"], dict):
                LOG.error("⚠️ Le checkpoint contient un 'model' (state_dict) mais pas la classe du modèle."
                          " Charger requires the model class and do model.load_state_dict(...).")
                LOG.info(f"   Clés présentes dans 'model': {list(loaded['model'].keys())[:10]}")
                return
        elif "state_dict" in loaded:
            LOG.error("⚠️ Le fichier contient un 'state_dict' seul. Veuillez fournir la classe du modèle pour charger les poids.")
            return
        else:
            LOG.error(f"⚠️ Le fichier chargé est un dict, clés: {list(loaded.keys())}. Aucune clé 'model' ni 'state_dict' détectée.")
            return
    else:
        LOG.error("⚠️ Le fichier ne contient pas un nn.Module valide.")
        return

    name, conv = inspect_first_conv(model)
    if conv is None:
        return

    if conv.in_channels == 3 and args.patch:
        model = patch_first_conv_to_mono(model)
        if args.save_patched:
            patched_path = model_path.with_name(model_path.stem + "_mono.pth")
            torch.save(model, patched_path)
            LOG.info(f"💾 Modèle patché sauvegardé : {patched_path}")

    test_forward(model, img_size=args.img_size, device=args.device)


if __name__ == "__main__":
    main()
