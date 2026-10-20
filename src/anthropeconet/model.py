"""AnthropEco-Net: shared SegFormer (MiT-B2) encoder, Head A (6-class semantic / damage map)
and Head B (natural / anthropogenic, a single 1x1 convolution on the deepest encoder feature).

Inference-only extract of the training code: the class definitions, colours, ImageNet statistics and
evaluation transform are copied unchanged; training-only code (datasets, augmentation, losses) is removed.
"""
import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

DAMAGE_NAMES = ["background", "building-intact", "building-damaged", "road", "water", "tree"]
MACRO_NAMES = ["natural", "anthropogenic"]
N_DAMAGE = len(DAMAGE_NAMES)   # 6
N_MACRO = len(MACRO_NAMES)     # 2

# Colours for the overlays (index = class id)
DAMAGE_COLORS = np.array([[0, 0, 0], [0, 255, 0], [255, 0, 0],
                          [128, 128, 128], [0, 0, 255], [0, 128, 0]], dtype=np.uint8)
MACRO_COLORS = np.array([[34, 139, 34], [178, 34, 34]], dtype=np.uint8)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EVAL_SIZE = 1024          # AnthropEco-Net input resolution (square resize)


def _resize(size):
    try:
        return A.Resize(height=size, width=size)
    except Exception:
        return A.Resize(size=(size, size))


def build_eval_tf(size=EVAL_SIZE):
    return A.Compose([
        _resize(size),                                            # deterministic, comparable metrics
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class FrugalMacroHead(nn.Module):
    """Head B — extremely minimalist: a single 1x1 convolution. Nothing else."""
    def __init__(self, in_channels, n_macro=N_MACRO):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, n_macro, kernel_size=1)

    def forward(self, x):
        return self.classifier(x)


def build_base(classes, encoder="mit_b2"):
    # encoder_weights=None: every weight is loaded from the AnthropEco-Net checkpoint afterwards
    return smp.Segformer(encoder_name=encoder, encoder_weights=None, classes=classes, activation=None)


class AnthropecoNet(nn.Module):
    def __init__(self, n_damage=N_DAMAGE, n_macro=N_MACRO, encoder="mit_b2"):
        super().__init__()
        self.base = build_base(n_damage, encoder)
        deep_ch = self.base.encoder.out_channels[-1]   # deepest encoder feature channels
        self.macro_head = FrugalMacroHead(deep_ch, n_macro)

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = self.base.encoder(x)                   # list of feature maps (stable API)
        dec = self.base.decoder(feats)                 # Head A: SegFormer MLP decoder
        pred_damage = self.base.segmentation_head(dec)
        macro = self.macro_head(feats[-1])             # Head B on deepest feature
        pred_macro = F.interpolate(macro, size=(H, W), mode="bilinear", align_corners=False)
        return {"pred_damage": pred_damage, "pred_macro": pred_macro}


def predict_probs(model, x):
    """Return (damage_probs, macro_probs), both softmax-normalized at input res (no test-time augmentation)."""
    o = model(x)
    return F.softmax(o["pred_damage"], 1), F.softmax(o["pred_macro"], 1)
