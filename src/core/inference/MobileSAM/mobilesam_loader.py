import os
import sys

# ✅ S'assure que le dossier parent (contenant mobile_sam/) est visible
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)


import torch
from mobile_sam.Modeling.sam import Sam
from mobile_sam.Modeling.tiny_vit_sam import TinyViT
from mobile_sam.Modeling.prompt_encoder import PromptEncoder
from mobile_sam.Modeling.mask_decoder import MaskDecoder
from mobile_sam.utils.transformer import TwoWayTransformer

def build_mobilesam_model(checkpoint: str | None = None) -> torch.nn.Module:
    """Construit le modèle MobileSAM tiny (vit_t) et charge les poids depuis checkpoint."""
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    model = Sam(
        image_encoder=TinyViT(
            img_size=image_size, in_chans=3, num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2, embedding_dim=prompt_embed_dim,
                mlp_dim=2048, num_heads=8
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    model.eval()
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu")
        # Validate loaded checkpoint to avoid silent failures if the file is empty
        # or contains an unexpected object (e.g. nested structures or bad serialization).
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f"Checkpoint {checkpoint} is not a valid state_dict")
    return model
