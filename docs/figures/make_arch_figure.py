#!/usr/bin/env python
"""Draw the AnthropEco-Net (Block B) architecture figure -> anthropeconet.svg / anthropeconet.png.

    python docs/figures/make_arch_figure.py

Tensor shapes are for a 1024 x 1024 network input (batch 1); H x W is the native image size.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

C = {"io": "#eef2f7", "enc": "#dce9f7", "a": "#fde2cf", "b": "#d9f0dd", "gate": "#f3e8fb",
     "exp": "#fff4c7", "fuse": "#f9dede", "edge": "#3a3f47", "group": "#8a8f98"}


def box(ax, x, y, w, h, title, body="", fc="#ffffff", fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.06",
                                fc=fc, ec=C["edge"], lw=0.9))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=fs + 0.6, weight="bold")
    if body:
        ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=fs - 1.0, linespacing=1.35)


def arrow(ax, p, q, label="", rad=0.0, lpos=0.5, dy=0.09):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=0.9, color=C["edge"],
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if label:
        x, y = p[0] + (q[0] - p[0]) * lpos, p[1] + (q[1] - p[1]) * lpos
        ax.text(x, y + dy, label, ha="center", va="bottom", fontsize=6.6, color="#30343a", style="italic")


def main():
    fig, ax = plt.subplots(figsize=(15.65, 8.6))
    ax.set_xlim(0, 15.65); ax.set_ylim(0, 8.6); ax.axis("off")

    # group outlines
    ax.add_patch(FancyBboxPatch((2.35, 4.05), 10.05, 4.35, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=C["group"], lw=1.0, ls=(0, (4, 2))))
    ax.text(2.5, 8.28, "AnthropEco-Net (single forward pass at 1024 × 1024)", fontsize=10, weight="bold",
            color="#4a4f57", va="top")
    ax.add_patch(FancyBboxPatch((5.55, 0.2), 9.95, 3.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="none", ec=C["group"], lw=1.0, ls=(0, (4, 2))))
    ax.text(5.7, 3.86, "Ecological expert (native resolution H × W)", fontsize=10, weight="bold",
            color="#4a4f57", va="top")
    ax.text(0.15, 8.45, "Block B", fontsize=12, weight="bold", va="top")

    # input
    box(ax, 0.15, 5.0, 1.95, 2.3, "Input", "RGB uint8, H × W\n(enhanced if routed\nto night)\n\nresize 1024 × 1024\nImageNet mean / std\n→ 1 × 3 × 1024 × 1024",
        C["io"])

    # encoder
    box(ax, 2.6, 4.25, 2.6, 3.65, "Shared encoder", "SegFormer MiT-B2\n(24.20 M params)\n\n"
        "stage 1: 64 × 256 × 256  (s4)\nstage 2: 128 × 128 × 128  (s8)\nstage 3: 320 × 64 × 64  (s16)\n"
        "stage 4: 512 × 32 × 32  (s32)", C["enc"])
    arrow(ax, (2.1, 6.15), (2.6, 6.15))

    # Head A
    box(ax, 5.75, 5.85, 3.0, 2.05, "Head A", "SegFormer MLP decoder (4 stages)\n→ 256 × 256 × 256 (0.53 M)\n"
        "1 × 1 conv → 6 × 256 × 256\n×4 bilinear → 6 × 1024 × 1024", C["a"])
    arrow(ax, (5.2, 6.9), (5.75, 6.9), "all 4 stages", dy=0.05)
    # Head B
    box(ax, 5.75, 4.25, 3.0, 1.35, "Head B", "1 × 1 conv on stage 4 (1,026 params)\n512 × 32 × 32 → 2 × 32 × 32\n"
        "bilinear → 2 × 1024 × 1024", C["b"])
    arrow(ax, (5.2, 4.9), (5.75, 4.9), "stage 4", dy=0.05)

    # outputs of heads (native)
    box(ax, 9.3, 5.85, 2.85, 2.05, "Head A map", "softmax → bilinear to H × W\n→ argmax\nbackground, building-intact,\n"
        "building-damaged, road,\nwater, tree", C["a"])
    arrow(ax, (8.75, 6.9), (9.3, 6.9))
    box(ax, 9.3, 4.25, 2.85, 1.35, "Head B partition", "softmax → bilinear to H × W\n→ argmax\nnatural / anthropogenic",
        C["b"])
    arrow(ax, (8.75, 4.9), (9.3, 4.9))

    # gates
    box(ax, 12.75, 4.25, 2.6, 3.65, "Gates", "water = A is water\n\nnatural = B is natural\n           or A is water\n\n"
        "eligible = natural\n           and A is not tree\n\nflood allowed in background\nonly ≤ 30 px from water", C["gate"])
    arrow(ax, (12.15, 6.9), (12.75, 6.9))
    arrow(ax, (12.15, 4.9), (12.75, 4.9))

    # expert row
    box(ax, 12.75, 0.45, 2.6, 3.0, "Patch extraction", "128 × 128 crops of the\nnative image, stride 64\n\n"
        "eco: kept if ≥ 50 %\neligible pixels\n\nturbidity: kept if ≥ 50 %\nwater pixels", C["gate"])
    arrow(ax, (14.05, 4.25), (14.05, 3.45))
    box(ax, 9.3, 2.0, 2.85, 1.45, "Patch classifier", "ResNet-18, 160 × 160 input\n4 classes: clean_river,\n"
        "debris_water, flood_water, debris", C["exp"], fs=8.3)
    box(ax, 9.3, 0.45, 2.85, 1.35, "Turbidity classifier", "ResNet-18, 224 × 224 input\n5 levels (Class 1 … 5)\n"
        "exploratory: lab-trained", C["exp"], fs=8.3)
    arrow(ax, (12.75, 2.7), (12.15, 2.7))
    arrow(ax, (12.75, 1.1), (12.15, 1.1))

    box(ax, 5.75, 0.45, 3.0, 3.0, "Fusion", "per-pixel soft voting of patch\nprobabilities\n\ncontext mask: water → {clean_river,\n"
        "debris_water, flood_water};\nbackground → {debris, flood_water};\ntree → none; confidence ≥ 0.5\n\n"
        "flood must touch water (30 px),\nblobs < 200 px removed,\nopening + closing", C["fuse"], fs=8.3)
    arrow(ax, (9.3, 2.7), (8.75, 2.2))
    arrow(ax, (9.3, 1.1), (8.75, 1.6))

    # outputs
    box(ax, 2.6, 0.45, 2.6, 3.0, "Three masks (H × W)", "1. Head A map\n   (6 classes)\n\n2. Ecological map\n"
        "   undisturbed natural +\n   4 categories, 255 = not natural\n\n3. Turbidity map\n   5 levels, 255 = not scored",
        C["io"], fs=8.3)
    arrow(ax, (5.75, 1.95), (5.2, 1.95))
    ax.text(0.2, 2.0, "Head B partition is\nalso returned and\nused by the gates", fontsize=7.5, color="#4a4f57", va="center")

    fig.savefig(HERE / "anthropeconet.svg", bbox_inches="tight")
    fig.savefig(HERE / "anthropeconet.png", dpi=170, bbox_inches="tight")
    print("wrote", HERE / "anthropeconet.svg", HERE / "anthropeconet.png")


if __name__ == "__main__":
    main()
