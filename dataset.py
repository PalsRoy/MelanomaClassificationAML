"""
dataset.py - Melanoma Dataset & Augmentation Pipeline
Adapted from the 1st Place SIIM-ISIC Melanoma Classification Solution
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(image_size, mode='train'):
    """
    Augmentation pipelines.
    
    Train: aggressive augmentations to combat the small dataset
    and extreme class imbalance (only 1.76% melanoma).
    
    Val/Test: just resize and normalise — no randomness, so
    results are reproducible.
    """
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(
                 size=(image_size, image_size), scale=(0.8, 1.0)
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent=0.0625,
                scale=(0.9, 1.1),
                rotate=(-15, 15),
                p=0.5
            ),
            A.OneOf([
                A.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.1
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30,
                    val_shift_limit=20
                ),
            ], p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(image_size // 16, image_size // 8),
                hole_width_range=(image_size // 16, image_size // 8),
                p=0.3
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


class MelanomaDataset(Dataset):
    """
    PyTorch Dataset for melanoma classification.
    
    Loads JPEG images, applies augmentations, returns (image, target).
    """
    def __init__(self, df, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image (OpenCV reads BGR, convert to RGB)
        image = cv2.imread(row['filepath'])
        if image is None:
            raise FileNotFoundError(f"Image not found: {row['filepath']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations
        if self.transform:
            image = self.transform(image=image)['image']
        
        if self.is_test:
            return image
        
        target = torch.tensor(row['target'], dtype=torch.long)
        return image, target