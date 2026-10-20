"""End-to-end inference: luminance rule -> (enhancer, night only) -> AnthropEco-Net -> ecological expert.

Same calls, order and settings as the full pipeline: the enhancer runs at the input's native resolution
(tile by tile by default: same output up to float rounding, bounded memory) and its output is quantised
to uint8; AnthropEco-Net sees a 1024x1024 resize; its Head A / Head B probabilities are resized back to
the native resolution, where the ecological expert and the turbidity classifier run.
"""
from pathlib import Path

import numpy as np
import torch

from ..classifier import TAU, luma256
from ..enhancer import DCENet, apply_enhancer_tiled
from .experts import (GRID_DEFAULTS, compute_turbidity, load_classifier, load_turbidity,
                      postprocess_eco_map, process_image)
from .model import EVAL_SIZE, AnthropecoNet, build_eval_tf

WEIGHT_FILES = {"anthropeconet": "anthropeconet_mitb2.pth", "eco": "eco_classifier.pth",
                "turbidity": "turbidity_classifier.pth", "enhancer": "enhancer_dcenet.pt"}


def weight_paths(weights_dir):
    paths = {k: Path(weights_dir) / v for k, v in WEIGHT_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("missing checkpoint(s): " + ", ".join(missing)
                                + " (see weights/README.md)")
    return paths


class Pipeline:
    def __init__(self, weights_dir, device="cpu", use_enhancer=True, tau=TAU, eval_size=EVAL_SIZE,
                 enhancer_tile=1024):
        p = weight_paths(weights_dir)
        self.device, self.tau, self.use_enhancer, self.enhancer_tile = device, tau, use_enhancer, enhancer_tile
        seg = AnthropecoNet()
        seg.load_state_dict(torch.load(p["anthropeconet"], map_location="cpu", weights_only=True)["model"], strict=True)
        self.seg = seg.eval().to(device)
        self.tf = build_eval_tf(eval_size)
        self.cls, self.cls_classes, self.cls_size, self.vbc = load_classifier(p["eco"], device)
        self.turb, self.turb_classes, self.turb_size = load_turbidity(p["turbidity"], device)
        self.enh = None
        if use_enhancer:
            enh = DCENet()
            enh.load_state_dict(torch.load(p["enhancer"], map_location="cpu", weights_only=True)["model"], strict=True)
            self.enh = enh.eval().to(device)

    def seg_input(self, rgb):
        """Resize to the evaluation size + ImageNet normalisation."""
        x = self.tf(image=rgb, mask=np.zeros(rgb.shape[:2], np.uint8))["image"]
        return x.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def __call__(self, rgb, patch=GRID_DEFAULTS["patch"], stride=GRID_DEFAULTS["stride"],
                 conf=GRID_DEFAULTS["conf"], gate_mode=GRID_DEFAULTS["gate_mode"],
                 water_buffer=GRID_DEFAULTS["water_buffer"], min_area=GRID_DEFAULTS["min_area"]):
        """rgb: uint8 HxWx3. Returns a dict with the routing decision, the image AnthropEco-Net saw and
        four native-resolution maps (uint8): head_a (0..5), head_b (0 natural, 1 anthropogenic),
        eco (0..4, 255 = not natural) and turbidity (0..4, 255 = not scored)."""
        luma = luma256(rgb)
        night = luma < self.tau
        enhanced = bool(night and self.enh is not None)
        x = apply_enhancer_tiled(self.enh, rgb, self.device, self.enhancer_tile) if enhanced else rgb
        eco, a, b, is_water = process_image(
            self.seg, self.cls, self.cls_classes, self.cls_size, self.vbc, x, self.seg_input(x),
            self.device, patch, stride, conf, gate_mode, water_buffer)
        eco = postprocess_eco_map(eco, is_water, water_buffer=water_buffer, min_area=min_area, do_morph=True)
        turb = compute_turbidity(a, x, self.turb, self.turb_classes, self.turb_size,
                                 patch, stride, conf, self.device)
        return {"luma256": luma, "night": bool(night), "enhanced": enhanced, "input": x,
                "head_a": a.astype(np.uint8), "head_b": b.astype(np.uint8), "eco": eco, "turbidity": turb}
