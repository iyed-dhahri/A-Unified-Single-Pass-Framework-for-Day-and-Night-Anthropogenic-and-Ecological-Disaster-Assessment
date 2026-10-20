"""End-to-end check of the demo on CPU.

    python -m pytest tests/test_demo.py -s          # ~2-3 min on a recent CPU, needs the four checkpoints

1. The tiled enhancer gives the same uint8 image as the whole-image pass, up to floating-point rounding
   (the CPU convolution kernels depend on the tensor size, so a few values can differ by 1 grey level;
   on the full-size example images the two are bit-identical).
2. demo.py runs on examples/images and its four maps agree with examples/expected (produced by the full
   research code on CPU). Agreement is 100 % in our environment (requirements.txt); a different CPU or
   library build can flip a few pixels through floating-point rounding, so the test requires >= 99.9 %.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.anthropeconet.pipeline import WEIGHT_FILES  # noqa: E402
from src.enhancer import DCENet, apply_enhancer, apply_enhancer_tiled  # noqa: E402

WEIGHTS = ROOT / "weights"
MAPS = ("head_a", "head_b", "eco", "turbidity")
MIN_AGREEMENT = 0.999

pytestmark = pytest.mark.skipif(not all((WEIGHTS / f).exists() for f in WEIGHT_FILES.values()),
                                reason="checkpoints not found in weights/ (see weights/README.md)")


def test_tiled_enhancer_matches_whole_image():
    enh = DCENet()
    enh.load_state_dict(torch.load(WEIGHTS / WEIGHT_FILES["enhancer"], map_location="cpu", weights_only=True)["model"])
    enh.eval()
    rgb = np.array(Image.open(ROOT / "examples/images/floodnet_7253_synthetic_night.jpg").convert("RGB"))[:900, :1300]
    rgb = np.ascontiguousarray(rgb)
    d = np.abs(apply_enhancer_tiled(enh, rgb, "cpu", tile=256).astype(int) - apply_enhancer(enh, rgb, "cpu"))
    print(f"tiled vs whole image: {int((d > 0).sum())} of {d.size} values differ, max {d.max()}")
    assert d.max() <= 1 and (d > 0).mean() < 1e-3


def test_demo_matches_expected(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "demo.py"), str(ROOT / "examples/images"),
                          "--device", "cpu", "--out", str(tmp_path)], cwd=ROOT, capture_output=True, text=True)
    print(out.stdout)
    assert out.returncode == 0, out.stderr
    assert "floodnet_7253_synthetic_night.jpg" in out.stdout
    night_block = out.stdout.split("floodnet_7253_synthetic_night.jpg")[1]
    assert "routed: night, enhancer applied" in night_block
    for exp in sorted((ROOT / "examples/expected").iterdir()):
        for k in MAPS:
            ref = np.array(Image.open(exp / f"{k}.png"))
            got = np.array(Image.open(tmp_path / exp.name / f"{k}.png"))
            assert got.shape == ref.shape, (exp.name, k)
            agree = float((got == ref).mean())
            print(f"{exp.name:32s} {k:10s} agreement {agree:.6f}")
            assert agree >= MIN_AGREEMENT, (exp.name, k, agree)
