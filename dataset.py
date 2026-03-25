import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2


class MelanomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
      self.df = pd.read_csv(csv_file)
      self.img_dir = img_dir
      self.transform = transform

    def __len__(self):
      return len(self.df)
    
    def __getitem__(self, idx):
      row = self.df.iloc[idx]

      image_name = row['image_name']
      target = row['target']

      img_path = os.path.join(self.img_dir, image_name + ".jpg")
      image = cv2.imread(img_path)

      if image is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")

      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

      if self.transform:
          image = self.transform(image)

      return image, target
