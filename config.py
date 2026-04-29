"""
config.py - All project configuration in one place.
"""
import os
import torch


class CFG:
    # Device — auto-detect MPS, CUDA, or CPU
    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    
    # Paths
    DATA_DIR = './siim-isic-melanoma-classification'
    JPEG_DIR = os.path.join(DATA_DIR, 'jpeg')
    
    # Model — change ONE line below to swap models
    model_name = 'tf_efficientnet_b4.ns_jft_in1k'  # Primary model: EfficientNet-B4
    image_size = 384                       # Use 224 for ViT/DeiT/Swin
    out_dim = 9
    pretrained = True
    drop_rate = 0.5
    
    # Training
    batch_size = 16
    n_epochs = 15
    init_lr = 3e-5
    
    # Cross-validation
    n_folds = 5
    fold = 0
    seed = 42
    
    # System
    num_workers = 0   # 0 for MPS, 4 for Colab
    
    # ----------------------------------------------------------------
    # Available models (change `model_name` above to use one of these):
    # 
    # CNN family:
    #   'tf_efficientnet_b3_ns'   - Lighter EfficientNet-B3
    #   'tf_efficientnet_b4_ns'   - EfficientNet-B4 (recommended)
    #   'tf_efficientnet_b5_ns'   - Larger EfficientNet-B5
    #   'convnext_small'          - ConvNeXt (modern CNN)
    # 
    # Transformer family (use image_size=224):
    #   'swin_base_patch4_window7_224'  - Swin Transformer
    #   'vit_base_patch16_224'          - Vision Transformer
    #   'deit_small_patch16_224'        - Data-efficient ViT
    # ----------------------------------------------------------------