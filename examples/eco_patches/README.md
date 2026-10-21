# Ecological patch dataset (examples)

The ecological expert of AnthropEco-Net classifies 128 × 128 patches of natural regions into four
categories. We built a dedicated patch dataset for it from post-hurricane UAV images of RescueNet and
FloodNet. This folder shows **five example patches per category**; the full dataset is not distributed here.

![Example patches](patch_grid.jpg)

## Categories

| Category | Meaning |
|---|---|
| `clean_river` | water body without visible debris (river, pond, lake) |
| `debris_water` | water carrying floating debris |
| `flood_water` | water outside the normal water bodies (flooded land, yards, roads) |
| `debris` | debris on land |

At inference, the category a pixel can receive depends on its Head A context: water pixels can only
become clean_river, debris_water or flood_water; background pixels debris or flood_water (flood_water
only near water); trees are never re-classified (see [docs/ANTHROPECONET.md](../../docs/ANTHROPECONET.md)).

## How it was built

- **Source images:** training splits only of RescueNet and FloodNet, so the two datasets' test sets stay
  untouched.
- **Labelling:** with a browser-based tool, square boxes were placed by hand on the full-resolution images
  and assigned one of the four categories; each box is cropped **at native resolution** (no resizing) and
  stored with its provenance (dataset, source image, position, size).
- **Split:** at the image level, so that overlapping patches of one scene never fall in different splits,
  stratified by dataset and by the categories an image contains.

![Labelled boxes on two FloodNet scenes](patch_context.jpg)

## Size

1,856 patches of 128 × 128 px from 109 source images (83 RescueNet, 26 FloodNet).

| Category | RescueNet | FloodNet | Total | Train | Val | Test |
|---|---|---|---|---|---|---|
| clean_river | 335 | 103 | 438 | 271 | 69 | 98 |
| debris_water | 466 | 0 | 466 | 334 | 76 | 56 |
| flood_water | 105 | 361 | 466 | 316 | 112 | 38 |
| debris | 486 | 0 | 486 | 354 | 88 | 44 |
| **Total** | **1,392** | **464** | **1,856** | **1,275** | **345** | **236** |

debris and debris_water occur only in RescueNet, while most flood_water patches come from FloodNet. To
keep the classifier from recognising the dataset rather than the category, it is trained with strong
appearance augmentation and a domain-adversarial head. On the 236 test patches it reaches an accuracy of
0.962 (0.949 on RescueNet patches, 1.000 on FloodNet patches).

## Files here

- `clean_river/`, `debris_water/`, `flood_water/`, `debris/`: 5 example patches each (PNG, 128 × 128),
  file name `<dataset>__<image>__x<left>_y<top>_s<size>.png`, i.e. the crop position in the source
  training image. They were drawn at random (seed 0), one per source image, from FloodNet for clean_river
  and flood_water and from RescueNet for debris_water and debris (the only source of these two categories).
- `patch_grid.jpg`: the 20 example patches.
- `patch_context.jpg`: all labelled boxes of FloodNet training images 7802 and 6736.

## Credit and licences

These files are excerpts of the source datasets and keep their licences (they are **not** covered by the
MIT licence of the code):

- `clean_river/`, `flood_water/`, `patch_context.jpg`: from **FloodNet** (Rahnemoonfar et al., IEEE Access
  2021), <https://github.com/BinaLab/FloodNet-Supervised_v1.0>, Community Data License Agreement –
  Permissive 1.0 ([`../LICENSE-CDLA-Permissive-1.0.txt`](../LICENSE-CDLA-Permissive-1.0.txt)).
- `debris_water/`, `debris/`: from **RescueNet** (Rahnemoonfar, Chowdhury and Murphy, Scientific Data 2023),
  <https://github.com/BinaLab/RescueNet-A-High-Resolution-Post-Disaster-UAV-Dataset-for-Semantic-Segmentation>,
  Creative Commons BY-NC-ND 4.0 (<https://creativecommons.org/licenses/by-nc-nd/4.0/>): non-commercial use
  only, pixels unmodified (crops at native resolution).
- `patch_grid.jpg` combines patches from both datasets under the same terms.

```bibtex
@article{rahnemoonfar2023rescuenet,
  title   = {RescueNet: a high resolution UAV semantic segmentation dataset for natural disaster damage assessment},
  author  = {Rahnemoonfar, Maryam and Chowdhury, Tashnim and Murphy, Robin},
  journal = {Scientific Data},
  volume  = {10},
  number  = {1},
  year    = {2023}
}
```
