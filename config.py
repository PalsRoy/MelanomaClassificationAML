"""
config.py - All project configuration are in all place such as file paths etc.
"""
import os

class CFG:
    # Paths
    DATA_DIR = './siim-isic-melanoma-classification'
    JPEG_DIR = os.path.join(DATA_DIR, 'jpeg')
    
    # Model
    model_name = 'tf_efficientnet_b4_ns'
    out_dim = 9           # 9-class (from the 1st winning solution)
    pretrained = True
    drop_rate = 0.5
    
    # Training
    image_size = 384      # Later increase to 512
    batch_size = 16       # Reduce to 8 if OOM
    n_epochs = 15
    init_lr = 3e-5
    
    # Cross-validation
    N_FOLDS = 5
    FOLD = 0
    SEED = 42
    
    # System
    num_workers = 0       # Use 0 for MPS, 4 for Colab