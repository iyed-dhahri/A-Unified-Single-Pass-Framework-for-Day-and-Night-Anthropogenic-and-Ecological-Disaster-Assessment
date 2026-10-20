# Checkpoints

The demo needs four files in this folder (they are not stored in git):

| File | Model | Parameters | Size | SHA-256 |
|---|---|---|---|---|
| `anthropeconet_mitb2.pth` | AnthropEco-Net (SegFormer MiT-B2, Head A, Head B) | 24,724,680 | 99.0 MB | `a724841749db5f91e74217dbef7fe5f592ebbfc20c11b9939a08b7488b055227` |
| `eco_classifier.pth` | ecological patch classifier (ResNet-18, 4 classes) | 11,178,564 | 45.1 MB | `ff307f5cc4b0776e7c4e87d204a26b8e59eba32269d647f252396efdeb5cbb12` |
| `turbidity_classifier.pth` | turbidity classifier (ResNet-18, 5 levels) | 11,179,077 | 44.8 MB | `2d9927604cebcda6021a9e72576bfb9f4a9ea533ee477f3dbc36cf74b4d3a9a0` |
| `enhancer_dcenet.pt` | low-light enhancer (DCE-Net) | 79,416 | 0.3 MB | `318b08079c5e8d26ca4e2ba77fb39f6df3fed55ec614830721fdc5e504f883f7` |

The eco classifier checkpoint also stores the domain-adversarial head used during training (not used at
inference, so it is not counted above).

**Download:** the download link will be added here soon. Once it is available:

```bash
python weights/download_weights.py          # downloads and verifies the SHA-256 of each file
python weights/download_weights.py --check  # only verifies files already in weights/
```

AnthropEco-Net and the ecological classifier were trained on RescueNet (CC BY-NC-ND 4.0) and FloodNet
(CDLA-Permissive 1.0); please check these dataset licences before any use other than research.
