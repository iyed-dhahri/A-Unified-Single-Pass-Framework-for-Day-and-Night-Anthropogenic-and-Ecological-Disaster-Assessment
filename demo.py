#!/usr/bin/env python
"""Day-and-night AnthropEco-Net demo.

    python demo.py examples/images                    # a folder
    python demo.py examples/images/floodnet_7253.jpg  # one image
    python demo.py examples/images --device cpu --no-enhancer

Per image, written to <out>/<image stem>/:
    head_a.png, head_b.png, eco.png, turbidity.png   class-index masks at the input resolution
                                                     (255 = not natural / not scored)
    head_a_overlay.jpg ... turbidity_overlay.jpg     colour overlays on the image AnthropEco-Net saw
    panel.png                                        all of the above in one figure
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.anthropeconet import Pipeline  # noqa: E402
from src.anthropeconet.experts import ECO_MAP_LABELS  # noqa: E402
from src.anthropeconet.model import DAMAGE_NAMES  # noqa: E402
from src.classifier import TAU  # noqa: E402
from src.visualize import overlays, save_panel  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
IMG_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
HERE = Path(__file__).resolve().parent


def list_inputs(p):
    p = Path(p)
    if p.is_dir():
        return sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS)
    return [p]


def fractions(m, names):
    n = m.size
    return ", ".join(f"{nm} {100 * (m == k).sum() / n:.1f}%" for k, nm in enumerate(names) if (m == k).any())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="image file or folder")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--weights", default=str(HERE / "weights"), help="folder with the four checkpoints")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--no-enhancer", action="store_true", help="never enhance (night images go straight to AnthropEco-Net)")
    ap.add_argument("--tau", type=float, default=TAU, help="luminance threshold of the day/night rule")
    ap.add_argument("--enhancer-tile", type=int, default=1024,
                    help="enhance in overlapping tiles of this size (same output up to float rounding, much less memory); "
                         "0 = whole image in one pass")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = PyTorch default)")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if a.threads:
        torch.set_num_threads(a.threads)
    paths = list_inputs(a.input)
    if not paths:
        raise SystemExit(f"no images found in {a.input}")
    t0 = time.perf_counter()
    pipe = Pipeline(a.weights, device=device, use_enhancer=not a.no_enhancer, tau=a.tau,
                    enhancer_tile=a.enhancer_tile)
    print(f"[load] models on {device} in {time.perf_counter() - t0:.1f} s"
          f"{'' if pipe.enh is not None else ' (enhancer disabled)'}")

    for p in paths:
        rgb = np.array(Image.open(p).convert("RGB"))
        t = time.perf_counter()
        res = pipe(rgb)
        dt = time.perf_counter() - t
        route = "night" if res["night"] else "day"
        enh = "applied" if res["enhanced"] else ("not applied (--no-enhancer)" if res["night"] else "not applied")
        print(f"\n{p.name}  {rgb.shape[1]}x{rgb.shape[0]}")
        print(f"  luma256 = {res['luma256']:.1f}  (tau = {a.tau:g})  -> routed: {route}, enhancer {enh}")
        print(f"  Head A : {fractions(res['head_a'], DAMAGE_NAMES)}")
        print(f"  Head B : natural {100 * (res['head_b'] == 0).mean():.1f}%, anthropogenic {100 * (res['head_b'] == 1).mean():.1f}%")
        print(f"  eco    : {fractions(res['eco'], ECO_MAP_LABELS)} (rest: not natural)")
        tv = res["turbidity"][res["turbidity"] != 255]
        print(f"  turbidity (exploratory): {tv.size} water px scored"
              + (f", " + fractions(tv, pipe.turb_classes) if tv.size else ""))
        print(f"  time   : {dt:.2f} s")

        od = Path(a.out) / p.stem
        od.mkdir(parents=True, exist_ok=True)
        for k in ("head_a", "head_b", "eco", "turbidity"):
            Image.fromarray(res[k]).save(od / f"{k}.png")
        ovl = overlays(res, pipe.turb_classes)
        for k, (_, img, _) in ovl.items():
            cv2.imwrite(str(od / f"{k}_overlay.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if res["enhanced"]:
            cv2.imwrite(str(od / "enhanced.jpg"), cv2.cvtColor(res["input"], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        save_panel(od / "panel.png", rgb, res, ovl,
                   f"{p.name} | luma {res['luma256']:.1f} -> {route}, enhancer {'on' if res['enhanced'] else 'off'}")
        print(f"  saved  : {od}/")


if __name__ == "__main__":
    main()
