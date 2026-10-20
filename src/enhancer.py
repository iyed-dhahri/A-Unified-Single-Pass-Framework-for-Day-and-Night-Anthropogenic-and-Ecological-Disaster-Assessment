"""Low-light enhancer: Zero-DCE-style curve estimator (DCE-Net) trained through the frozen AnthropEco-Net.

7 conv layers (3x3, 32 ch, ReLU), symmetric skip concatenations (conv5 <- [x3,x4], conv6 <- [x2,x5],
conv7 <- [x1,x6]); conv7 outputs n=8 RGB curve-parameter maps alpha = tanh(.) in [-1, 1];
LE(x) = x + alpha * x * (1 - x), applied 8 times. Runs at the input's native resolution
(optionally tile by tile, with the same result up to floating-point rounding).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ITER = 8


class DCENet(nn.Module):
    def __init__(self, nf=32, n_iter=N_ITER):
        super().__init__()
        self.n_iter = n_iter
        self.c1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.c2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.c3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.c4 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.c5 = nn.Conv2d(2 * nf, nf, 3, 1, 1)
        self.c6 = nn.Conv2d(2 * nf, nf, 3, 1, 1)
        self.c7 = nn.Conv2d(2 * nf, 3 * n_iter, 3, 1, 1)

    def forward(self, x):
        """x in [0,1], NCHW. Returns (enhanced, alpha (N, 3*n_iter, H, W))."""
        x1 = F.relu(self.c1(x))
        x2 = F.relu(self.c2(x1))
        x3 = F.relu(self.c3(x2))
        x4 = F.relu(self.c4(x3))
        x5 = F.relu(self.c5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.c6(torch.cat([x2, x5], 1)))
        alpha = torch.tanh(self.c7(torch.cat([x1, x6], 1)))
        y = x
        for a in torch.split(alpha, 3, dim=1):
            y = y + a * y * (1 - y)
        return y, alpha


@torch.no_grad()
def apply_enhancer(enh, rgb_u8, device):
    """Enhancer at the input's native resolution; uint8 RGB in, uint8 RGB out."""
    x = torch.from_numpy(rgb_u8).to(device).permute(2, 0, 1)[None].float() / 255
    y = enh(x)[0].clamp(0, 1)
    return (y[0].permute(1, 2, 0) * 255).round().byte().cpu().numpy()


RECEPTIVE_RADIUS = 7      # 7 stacked 3x3 convolutions; the curve updates are point-wise


@torch.no_grad()
def apply_enhancer_tiled(enh, rgb_u8, device, tile=1024, halo=16):
    """Same output as apply_enhancer, computed tile by tile to bound memory (a 14 MP image needs ~20 GB
    in one pass on CPU). Each tile is enhanced with a `halo`-pixel margin of real neighbouring pixels
    (> RECEPTIVE_RADIUS) and only its centre is kept, so every output pixel sees exactly the same input
    as in the full-image pass; tiles touching the image border get the same zero padding. The result is
    identical up to floating-point rounding (CPU kernels depend on the tensor size; at most 1 grey level
    on rare pixels); on the example images it is bit-identical."""
    assert halo >= RECEPTIVE_RADIUS
    H, W = rgb_u8.shape[:2]
    if tile <= 0 or (H <= tile and W <= tile):
        return apply_enhancer(enh, rgb_u8, device)
    out = np.empty_like(rgb_u8)
    for y0 in range(0, H, tile):
        for x0 in range(0, W, tile):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            ya, xa, yb, xb = max(0, y0 - halo), max(0, x0 - halo), min(H, y1 + halo), min(W, x1 + halo)
            e = apply_enhancer(enh, np.ascontiguousarray(rgb_u8[ya:yb, xa:xb]), device)
            out[y0:y1, x0:x1] = e[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
    return out
