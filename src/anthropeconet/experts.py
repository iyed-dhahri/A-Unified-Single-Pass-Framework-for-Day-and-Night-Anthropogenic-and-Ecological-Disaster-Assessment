"""Ecological expert of AnthropEco-Net: context-gated patch classifier (4 ecological categories) and the
exploratory water-turbidity classifier (5 levels), both applied at the image's native resolution.

Inference-only extract of the full code: functions copied unchanged apart from imports, logging and the
removal of the training / evaluation / EEDI code.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from scipy import ndimage

from .model import DAMAGE_NAMES, IMAGENET_MEAN, IMAGENET_STD, predict_probs

# Head-A class indices
WATER_IDX = DAMAGE_NAMES.index("water")
BG_IDX = DAMAGE_NAMES.index("background")
TREE_IDX = DAMAGE_NAMES.index("tree")
DAMAGED_IDX = DAMAGE_NAMES.index("building-damaged")

# Eco classifier classes + gating (loaded/confirmed from the checkpoint)
ECO_CLASSES = ["clean_river", "debris_water", "flood_water", "debris"]
VALID_BY_CONTEXT = {"water": ["clean_river", "debris_water", "flood_water"],
                    "background": ["debris", "flood_water"], "tree": []}
DISTURBED = {"debris_water", "flood_water", "debris"}   # eco classes = disturbed natural
# eco map paint ids: 0 undisturbed-natural, then +1 per eco class, 255 = not natural
ECO_MAP_IDS = {"undisturbed": 0, "clean_river": 1, "debris_water": 2,
               "flood_water": 3, "debris": 4}
ECO_MAP_COLORS = np.array([[34, 139, 34],   # 0 undisturbed natural -> green
                           [46, 120, 210],  # 1 clean_river -> blue
                           [140, 90, 40],   # 2 debris_water -> brown
                           [200, 40, 140],  # 3 flood_water -> magenta
                           [150, 150, 30]], dtype=np.uint8)  # 4 debris -> olive
ECO_MAP_LABELS = ["undisturbed natural", "clean_river", "debris_water", "flood_water", "debris"]

# Sequential palette, low -> high turbidity (clear blue -> muddy brown).
TURBIDITY_COLORS = np.array([[150, 205, 235], [120, 175, 165], [160, 165, 110],
                             [165, 120, 70], [110, 70, 40]], dtype=np.uint8)

# Default settings of the full pipeline
GRID_DEFAULTS = dict(patch=128, stride=64, conf=0.5, gate_mode="dilated",
                     water_buffer=30, min_area=200)


# ---------------------------------------------------------------------------
class EcoClassifier(nn.Module):
    def __init__(self, n_cls=4):
        super().__init__()
        self.backbone = tv.models.resnet18(weights=None)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.cls_head = nn.Linear(feat, n_cls)

    def forward(self, x):
        return self.cls_head(self.backbone(x))


def load_classifier(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    m = EcoClassifier(n_cls=len(ck.get("classes", ECO_CLASSES)))
    # keep only backbone + cls_head weights (the domain-adversarial head is used in training only)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("dom_head")}
    m.load_state_dict(sd, strict=True)
    m.eval().to(device)
    img_size = ck.get("img_size", 160)
    classes = ck.get("classes", ECO_CLASSES)
    vbc = ck.get("valid_by_context", VALID_BY_CONTEXT)
    return m, classes, img_size, vbc


class TurbidityNet(nn.Module):
    def __init__(self, n_cls, backbone="resnet18"):
        super().__init__()
        if backbone == "efficientnet_b0":
            self.backbone = tv.models.efficientnet_b0(weights=None)
            feat = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            self.backbone = tv.models.resnet18(weights=None)
            feat = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        self.head = nn.Linear(feat, n_cls)

    def forward(self, x):
        return self.head(self.backbone(x))


def load_turbidity(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    classes = ck.get("classes", [f"level{i}" for i in range(5)])
    m = TurbidityNet(len(classes), ck.get("backbone", "resnet18"))
    m.load_state_dict(ck["model"], strict=True)
    m.eval().to(device)
    return m, classes, ck.get("img_size", 224)


def _grid(H, W, patch, stride):
    ys = list(range(0, max(1, H - patch + 1), stride))
    xs = list(range(0, max(1, W - patch + 1), stride))
    if ys and ys[-1] != H - patch:
        ys.append(max(0, H - patch))
    if xs and xs[-1] != W - patch:
        xs.append(max(0, W - patch))
    return ys, xs


# ---------------------------------------------------------------------------
@torch.no_grad()
def process_image(seg, cls, classes, img_size, vbc, rgb_native, x_seg, device,
                  patch, stride, conf, gate_mode, water_buffer):
    """Return eco_map (native-res int, ids per ECO_MAP_IDS, 255=not-natural),
    plus Head A / Head B maps and the water mask at native res.

    Per-pixel SOFT VOTING: overlapping patches accumulate class probabilities per
    pixel; the decision (context mask + confidence + argmax) is made per pixel at
    the end.

    gate_mode controls where flood_water may be assigned:
      strict  -> only where Head A predicts water
      dilated -> water, or background within water_buffer px of water (default)
      loose   -> water or background anywhere
    """
    dmg_prob, mac_prob = predict_probs(seg, x_seg)
    Hn, Wn = rgb_native.shape[:2]
    a = F.interpolate(dmg_prob, size=(Hn, Wn), mode="bilinear", align_corners=False).argmax(1)[0]
    b = F.interpolate(mac_prob, size=(Hn, Wn), mode="bilinear", align_corners=False).argmax(1)[0]
    a = a.cpu().numpy(); b = b.cpu().numpy()

    is_water = (a == WATER_IDX)
    is_bg = (a == BG_IDX)
    natural_pix = (b == 0) | is_water
    eligible = natural_pix & (a != TREE_IDX)

    cls_idx = {c: i for i, c in enumerate(classes)}
    n_eco = len(classes)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)

    # accumulate per-pixel classifier probabilities over eligible pixels
    prob_accum = np.zeros((n_eco, Hn, Wn), np.float32)
    count = np.zeros((Hn, Wn), np.float32)
    ys, xs = _grid(Hn, Wn, patch, stride)
    batch, coords = [], []

    def flush():
        if not batch:
            return
        xb = torch.stack(batch).to(device)
        xb = F.interpolate(xb, size=(img_size, img_size), mode="bilinear", align_corners=False)
        xb = (xb - mean) / std
        probs = F.softmax(cls(xb), 1).cpu().numpy()                 # [B, n_eco]
        for (yy, xx), pr in zip(coords, probs):
            se = eligible[yy:yy + patch, xx:xx + patch].astype(np.float32)
            prob_accum[:, yy:yy + patch, xx:xx + patch] += pr[:, None, None] * se[None]
            count[yy:yy + patch, xx:xx + patch] += se
        batch.clear(); coords.clear()

    for yy in ys:
        for xx in xs:
            if eligible[yy:yy + patch, xx:xx + patch].mean() < 0.5:
                continue
            crop = rgb_native[yy:yy + patch, xx:xx + patch]
            batch.append(torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0)
            coords.append((yy, xx))
            if len(batch) >= 64:
                flush()
    flush()

    avg = prob_accum / np.maximum(count, 1e-6)[None]                # [n_eco,H,W]
    eco_map = np.full((Hn, Wn), 255, dtype=np.uint8)
    eco_map[natural_pix] = ECO_MAP_IDS["undisturbed"]

    # extended-water region for the 'dilated' flood gate
    if gate_mode == "dilated" and water_buffer > 0:
        ext_water = ndimage.binary_dilation(is_water, iterations=water_buffer)
    else:
        ext_water = is_water

    id_of = np.array([ECO_MAP_IDS[classes[c]] for c in range(n_eco)])

    def paint(pix_mask, valid_names):
        valid_idx = [cls_idx[c] for c in valid_names if c in cls_idx]
        idxs = np.where(pix_mask.ravel())[0]
        if len(idxs) == 0 or not valid_idx:
            return
        sub = avg.reshape(n_eco, -1)[np.ix_(valid_idx, idxs)]       # [V, N]
        s = sub.sum(0)
        subn = sub / np.maximum(s, 1e-9)
        j = subn.argmax(0)
        conf_val = subn.max(0)
        chosen = id_of[np.array(valid_idx)[j]]
        accept = (s > 0) & (conf_val >= conf)
        flat = eco_map.reshape(-1)
        flat[idxs[accept]] = chosen[accept]

    water_valid = vbc.get("water", [])
    bg_valid = list(vbc.get("background", []))
    bg_no_flood = [c for c in bg_valid if c != "flood_water"]

    paint(is_water, water_valid)                                    # water pixels
    if gate_mode == "loose":
        paint(is_bg, bg_valid)                                      # flood allowed anywhere in bg
    elif gate_mode == "strict":
        paint(is_bg, bg_no_flood)                                   # no flood in bg
    else:  # dilated: flood only in bg near water
        paint(is_bg & ext_water, bg_valid)
        paint(is_bg & ~ext_water, bg_no_flood)

    return eco_map, a, b, is_water


def postprocess_eco_map(eco_map, is_water, water_buffer=30, min_area=200, do_morph=True):
    """Clean the ecological map: water-connectivity filter for flood, small-blob
    removal, and morphological open/close. Operates in place; returns eco_map."""
    UND = ECO_MAP_IDS["undisturbed"]
    disturbed_ids = [ECO_MAP_IDS[c] for c in DISTURBED]
    flood_id = ECO_MAP_IDS["flood_water"]

    # 1) flood must connect to (or sit near) a real Head-A water body
    flood = eco_map == flood_id
    if flood.any():
        ext_water = ndimage.binary_dilation(is_water, iterations=water_buffer) if water_buffer > 0 else is_water
        lbl, n = ndimage.label(flood)
        for c in range(1, n + 1):
            m = lbl == c
            if not (m & ext_water).any():
                eco_map[m] = UND                                   # dry-land flood -> drop

    # 2) remove tiny disturbed blobs (isolated single-patch false positives)
    if min_area > 0:
        for did in disturbed_ids:
            m = eco_map == did
            if not m.any():
                continue
            lbl, n = ndimage.label(m)
            for c in range(1, n + 1):
                comp = lbl == c
                if comp.sum() < min_area:
                    eco_map[comp] = UND

    # 3) morphological open (despeckle) then close (fill pinholes)
    if do_morph:
        for did in disturbed_ids:
            m = eco_map == did
            if not m.any():
                continue
            opened = ndimage.binary_opening(m, iterations=1)
            closed = ndimage.binary_closing(opened, iterations=1)
            eco_map[m & ~opened] = UND                              # specks removed
            eco_map[~m & closed & (eco_map == UND)] = did          # holes filled (into natural only)
    return eco_map


@torch.no_grad()
def compute_turbidity(a_map, rgb_native, model, turb_classes, img_size, patch, stride, conf, device):
    """Score turbidity on Head-A WATER pixels via soft-voting patch classification.
    Returns turb_map: 0..n-1 on scored water, 255 elsewhere (incl. unscored water)."""
    Hn, Wn = rgb_native.shape[:2]
    is_water = (a_map == WATER_IDX)
    n_t = len(turb_classes)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    prob = np.zeros((n_t, Hn, Wn), np.float32); count = np.zeros((Hn, Wn), np.float32)
    ys, xs = _grid(Hn, Wn, patch, stride)
    batch, coords = [], []

    def flush():
        if not batch:
            return
        xb = torch.stack(batch).to(device)
        xb = F.interpolate(xb, size=(img_size, img_size), mode="bilinear", align_corners=False)
        xb = (xb - mean) / std
        pr = F.softmax(model(xb), 1).cpu().numpy()
        for (yy, xx), p in zip(coords, pr):
            sw = is_water[yy:yy + patch, xx:xx + patch].astype(np.float32)
            prob[:, yy:yy + patch, xx:xx + patch] += p[:, None, None] * sw[None]
            count[yy:yy + patch, xx:xx + patch] += sw
        batch.clear(); coords.clear()

    for yy in ys:
        for xx in xs:
            if is_water[yy:yy + patch, xx:xx + patch].mean() < 0.5:
                continue
            crop = rgb_native[yy:yy + patch, xx:xx + patch]
            batch.append(torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0)
            coords.append((yy, xx))
            if len(batch) >= 64:
                flush()
    flush()

    avg = prob / np.maximum(count, 1e-6)[None]
    turb_map = np.full((Hn, Wn), 255, dtype=np.uint8)
    idxs = np.where(is_water.ravel())[0]
    if len(idxs):
        sub = avg.reshape(n_t, -1)[:, idxs]; s = sub.sum(0); subn = sub / np.maximum(s, 1e-9)
        j = subn.argmax(0); cv = subn.max(0); acc = (s > 0) & (cv >= conf)
        flat = turb_map.reshape(-1); flat[idxs[acc]] = j[acc].astype(np.uint8)
    return turb_map
