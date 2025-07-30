# mmdet-mini

A minimal scaffolding for building MMDetection models.

## Structure

- `configs/`: configuration templates
- `datasets/`: custom dataset registration
- `models/`: backbone, neck, head stubs
- `scripts/`: training and inference scripts

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python scripts/train.py --config configs/custom_config.py --work-dir work_dirs
```

## Inference

```bash
python scripts/infer.py --config configs/custom_config.py --checkpoint path/to/checkpoint.pth --image path/to/image.jpg
``` 