# Segmentation Model Development Workflow

This document outlines the end-to-end workflow for building a pixel-wise segmentation model for classes: sky, sea, and obstacle.

## 1. Environment Setup

```bash
# (Optional) create a virtual environment
conda create -n seg-env python=3.8 -y
conda activate seg-env

# install dependencies
pip install torch torchvision mmcv mmsegmentation opencv-python
```

## 2. Project Structure

```
./
├── configs/
│   └── custom_seg_config.py    # segmentation config template
├── datasets/
│   └── custom_seg_dataset.py   # dataset registration (RGB + mask)
├── models/
│   ├── backbone.py             # feature extractor (e.g., ResNet)
│   ├── neck.py                 # feature pyramid (optional)
│   └── head.py                 # segmentation head (e.g., FCNHead)
├── scripts/
│   ├── train_seg.py            # training entry point
│   └── infer_seg.py            # inference/demo script
└── data/
    ├── images/                 # RGB inputs (.jpg/.png)
    └── masks/                  # ground truth masks (.png)
```

## 3. Data Preparation

1. Place your RGB images in `data/images/` and corresponding mask PNGs in `data/masks/`.
2. Inspect mask values:
   ```python
   import numpy as np
   import cv2
   mask = cv2.imread('data/masks/0001.png', cv2.IMREAD_UNCHANGED)
   print(np.unique(mask))  # discover pixel values for classes
   ```
3. Define mapping (example):
   ```python
   CLASS_MAP = {0: 'background', 1: 'sky', 2: 'sea', 3: 'obstacle'}
   ```

## 4. Dataset Registration

- Create a class inheriting from `mmseg.datasets.CustomDataset`.
- Implement `load_annotations` to read image–mask pairs.
- Override `get_gt_seg_maps` if you need custom mapping.

```python
from mmseg.datasets import CustomDataset
from mmseg.datasets.builder import DATASETS

@DATASETS.register_module()
class MySegDataset(CustomDataset):
    CLASSES = ['sky', 'sea', 'obstacle']
    PALETTE = [[128, 128, 128], [70, 130, 180], [255, 0, 0]]

    def load_annotations(self, img_dir, seg_map_dir):
        # TODO: parse file list into dicts of img & seg_map
        return super().load_annotations(img_dir, seg_map_dir)

    def get_gt_seg_maps(self, idx):
        # optional: map raw pixel IDs to class indices
        return super().get_gt_seg_maps(idx)
```

## 5. Configuration Template

- Start from a base segmentation config (e.g., `fcn_r50-d8_512x512_20k_ade20k.py`).
- Update the following fields:
  - `dataset_type`, `data_root`, `classes`, `palette`.
  - Model `backbone`, `neck` (if any), `decode_head` settings.
  - `optimizer`, `lr_config`, `runner`.

```python
model = dict(
    type='EncoderDecoder',
    backbone=dict(type='ResNet', depth=50, init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    decode_head=dict(type='FCNHead', in_channels=2048, channels=512, num_classes=3, dropout_ratio=0.1),
)
dataset_type = 'MySegDataset'
data_root = 'data/'

data = dict(
    train=dict(type=dataset_type, data_root=data_root, img_dir='images', ann_dir='masks'),
    val=dict(type=dataset_type, data_root=data_root, img_dir='images', ann_dir='masks'),
    test=dict(type=dataset_type, data_root=data_root, img_dir='images', ann_dir='masks'),
)
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
``` 

## 6. Training

```bash
python scripts/train_seg.py \
  --config configs/custom_seg_config.py \
  --work-dir work_dirs/seg \
  --gpu-id 0
```

Monitor loss and mIoU with TensorBoard or logs.

## 7. Evaluation & Metrics

```bash
python tools/val.py configs/custom_seg_config.py \
  work_dirs/seg/latest.pth --eval mIoU
```

## 8. Inference & Visualization

```bash
python scripts/infer_seg.py \
  --config configs/custom_seg_config.py \
  --checkpoint work_dirs/seg/latest.pth \
  --image data/images/0001.jpg \
  --palette configs/palette.json
```

## 9. Deployment

- Export to ONNX/TensorRT for production.
- Integrate into web or edge applications.

---

Tailor this workflow to your dataset specifics and computational resources. 