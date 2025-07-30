# mmdet-mini

A minimal scaffolding for building MMDetection models.

## Create environment

```bash
git clone git@github.com:wilbur1240/mmdet-mini.git
cd mmdet-mini
conda env create -n mmopenlab -f environment_full.yml
conda activate mmopenlab
```

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