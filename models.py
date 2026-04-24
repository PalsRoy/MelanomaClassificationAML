"""
models.py - Melanoma Classifier
Flexible wrapper that works with any timm backbone (CNN or Transformer).
Adapted from the 1st Place SIIM-ISIC Melanoma Classification Solution.
"""

import torch
import torch.nn as nn
import timm


class MelanomaClassifier(nn.Module):
    """
    Generic melanoma classifier that works with any timm backbone.
    
    Design choices from the Kaggle 1st place solution:
    
    1. Pretrained backbone (timm): transfer learning from ImageNet
       gives us rich visual features for free.
    
    2. Multi-sample dropout: 5 different dropout masks averaged together,
       acting as a mini-ensemble inside a single model for better
       regularisation.
    
    3. 9-class output: classifying into 9 diagnosis categories
       outperformed binary classification in the winning solution.
       Melanoma probability is extracted via softmax at inference.
    
    Works with:
        - EfficientNet family (tf_efficientnet_b3_ns, b4_ns, b5_ns)
        - ConvNeXt family (convnext_small, convnext_base)
        - Swin Transformer (swin_base_patch4_window7_224)
        - Vision Transformer (vit_base_patch16_224, deit_small_patch16_224)
    """
    def __init__(self, model_name='tf_efficientnet_b4_ns', out_dim=9,
                 pretrained=True, drop_rate=0.5):
        super().__init__()
        
        # Load pretrained backbone from timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Remove the default head and get feature dimension
        in_features = self._reset_classifier()
        
        # Multi-sample dropout (5 independent masks)
        self.dropouts = nn.ModuleList([
            nn.Dropout(drop_rate) for _ in range(5)
        ])
        
        # Final classification layer
        self.fc = nn.Linear(in_features, out_dim)
    
    def _reset_classifier(self):
        """
        Handle different head attribute names across model families.
        - EfficientNet uses .classifier
        - ResNet/older models use .fc
        - ViT / Swin / ConvNeXt use .head
        """
        if hasattr(self.backbone, 'classifier'):
            in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            if hasattr(self.backbone.head, 'fc'):
                in_features = self.backbone.head.fc.in_features
                self.backbone.head.fc = nn.Identity()
            elif hasattr(self.backbone.head, 'in_features'):
                in_features = self.backbone.head.in_features
                self.backbone.head = nn.Identity()
            else:
                in_features = self.backbone.num_features
                self.backbone.head = nn.Identity()
        else:
            in_features = self.backbone.num_features
        return in_features
    
    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)
        
        # Apply each dropout mask and average the classifier outputs
        logits = torch.stack([
            self.fc(dropout(features)) for dropout in self.dropouts
        ])
        logits = logits.mean(dim=0)
        
        return logits


def build_model(model_name='tf_efficientnet_b4_ns', out_dim=9,
                pretrained=True, drop_rate=0.5):
    """Factory function to build the model."""
    return MelanomaClassifier(
        model_name=model_name,
        out_dim=out_dim,
        pretrained=pretrained,
        drop_rate=drop_rate,
    )