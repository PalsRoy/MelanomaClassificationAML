

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

class SwinT_Melanoma(nn.Module):
  def __init__(self, pretrained=True):
    super().__init__()
    self.backbone = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=pretrained,
        num_classes=0 # remove default classifier head
    )
    in_features = self.backbone.num_features
    self.classifier = nn.Linear(in_features,1)

  def forward(self, X):
    features = self.backbone(x)
    logits = self.classifier(features)
    logits = logits.squeeze(1)
    return logits
