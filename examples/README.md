# Example images

| File | Source | Content |
|---|---|---|
| `images/floodnet_7253.jpg` | FloodNet, test split, image 7253 (unchanged) | flooded houses and yard, day |
| `images/floodnet_11483.jpg` | FloodNet, test split, image 11483 (unchanged) | river, field and houses, day |
| `images/floodnet_7253_synthetic_night.jpg` | derived from FloodNet image 7253 | synthetic night rendering (contrast ×0.6, brightness −20, blue cast, vignetting; the generator used for the paper's synthetic-night test set, applied here at full resolution; JPEG quality 95) |

Neither FloodNet image contributed patches to the training, validation or test sets of the ecological
patch classifier.

`expected/<image>/` holds the four maps the full research code produces for each image on CPU
(`head_a.png`, `head_b.png`, `eco.png`, `turbidity.png`; class indices, 255 = not natural / not scored) and
a downsized copy of the demo panel (`panel.jpg`). `tests/test_demo.py` compares the demo's output with them.

`eco_patches/` holds example patches of the ecological patch dataset (FloodNet patches only), see
[`eco_patches/README.md`](eco_patches/README.md).

## Credit and licence

The images come from **FloodNet** (Rahnemoonfar et al., 2021),
<https://github.com/BinaLab/FloodNet-Supervised_v1.0>, released under the
Community Data License Agreement – Permissive 1.0. The licence text is in
[`LICENSE-CDLA-Permissive-1.0.txt`](LICENSE-CDLA-Permissive-1.0.txt). The night image is a modification of
FloodNet image 7253.

```bibtex
@article{rahnemoonfar2021floodnet,
  author  = {Rahnemoonfar, Maryam and Chowdhury, Tashnim and Sarkar, Argho and Varshney, Debvrat and Yari, Masoud and Murphy, Robin Roberson},
  title   = {FloodNet: A High Resolution Aerial Imagery Dataset for Post Flood Scene Understanding},
  journal = {IEEE Access},
  volume  = {9},
  pages   = {89644--89654},
  year    = {2021},
  doi     = {10.1109/ACCESS.2021.3090981}
}
```
