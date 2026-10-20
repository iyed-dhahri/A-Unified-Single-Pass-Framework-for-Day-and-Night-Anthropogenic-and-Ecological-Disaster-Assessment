"""Overlays and the summary panel (palettes of the full code; 255 = not scored is left bare)."""
import cv2
import numpy as np

from .anthropeconet.experts import ECO_MAP_COLORS, ECO_MAP_LABELS, TURBIDITY_COLORS
from .anthropeconet.model import DAMAGE_COLORS, DAMAGE_NAMES, MACRO_COLORS, MACRO_NAMES


def overlay_classmap(rgb, class_map, palette, alpha=0.5):
    """Blend palette colours over rgb for each class id present in class_map."""
    out = rgb.astype(np.float32).copy()
    for cid in range(len(palette)):
        m = class_map == cid
        if m.any():
            out[m] = (1 - alpha) * out[m] + alpha * palette[cid].astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def make_legend_handles(palette, labels):
    from matplotlib.patches import Patch
    return [Patch(facecolor=palette[i] / 255.0, edgecolor="k", linewidth=0.3, label=labels[i])
            for i in range(len(labels))]


def overlays(res, turb_labels, alpha=0.5):
    """{name: (title, overlay RGB on the image AnthropEco-Net saw, legend handles)}"""
    x = res["input"]
    specs = [("head_a", "Head A: semantic / damage", res["head_a"], DAMAGE_COLORS, DAMAGE_NAMES),
             ("head_b", "Head B: natural / anthropogenic", res["head_b"], MACRO_COLORS, MACRO_NAMES),
             ("eco", "Ecological map", res["eco"], ECO_MAP_COLORS, ECO_MAP_LABELS),
             ("turbidity", "Water turbidity (exploratory)", res["turbidity"], TURBIDITY_COLORS, turb_labels)]
    return {k: (t, overlay_classmap(x, m, pal, alpha), make_legend_handles(pal, names))
            for k, t, m, pal, names in specs}


def save_panel(out_path, rgb, res, ovl, suptitle, max_side=1000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("Input", rgb, None)]
    if res["enhanced"]:
        panels.append(("Enhanced (AnthropEco-Net input)", res["input"], None))
    panels += [ovl[k] for k in ("head_a", "head_b", "eco", "turbidity")]
    ncols = 3
    nrows = -(-len(panels) // ncols)
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows * h / w + 0.5), squeeze=False,
                             layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, (title, img, handles) in zip(axes.flat, panels):
        if s < 1:
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        ax.imshow(img); ax.set_title(title, fontsize=10)
        if handles:
            ax.legend(handles=handles, fontsize=6, loc="lower right", framealpha=0.75,
                      ncol=2 if len(handles) > 4 else 1)
    fig.suptitle(suptitle, fontsize=11)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
