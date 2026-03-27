# Melanoma Classification (AML Project)

This project is designed to assist healthcare professionals for automatically identifying melanoma in skin lesion images. EfficientNet-based melanoma classifier adapted from the **1st Place Solution** to the SIIM-ISIC Melanoma Classification Kaggle Challenge
([arXiv:2010.05351](http://arxiv.org/abs/2010.05351)).

## Architecture

**EfficientNet-B4** with Noisy Student pretraining, using:

- Multi-sample dropout (5 masks averaged) for regularization
- 9-class output head (multi-class outperforms binary per competition findings)
- Melanoma probability extracted via softmax at inference

## Project Structure

```
melanoma_project/
├── models.py          # EfficientNetMelanoma classifier
├── dataset.py         # Dataset, augmentations, data loading
├── train.py           # Training loop with AUC tracking
├── predict.py         # Inference with Test-Time Augmentation
├── evaluate.py        # Cross-validation evaluation
├── requirements.txt   # Dependencies
├── data/              # Dataset (download separately)
├── weights/           # Saved checkpoints
├── logs/              # Training history
├── oofs/              # Out-of-fold predictions
└── subs/              # Submission CSVs
```

## Quick Start — Local (Apple Silicon Laptop)

```bash
conda activate melanomaClassificationAML
pip install -r requirements.txt

# Train fold 0
python train.py --image-size 384 --batch-size 16 --fold 0

# Evaluate
python evaluate.py --image-size 384

# Predict
python predict.py --image-size 384
```

## Quick Start — Google Colab

```python
!pip install timm albumentations
!python train.py --image-size 384 --batch-size 32 --use-amp --data-dir /content/drive/MyDrive/data/
```
