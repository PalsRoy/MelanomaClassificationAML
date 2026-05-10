"""
train.py - Training & Validation Functions
Supports CUDA (with AMP) and MPS / CPU (without AMP).
Includes loss functions for class imbalance handling.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Melanoma is class 0 in our DIAGNOSIS2IDX mapping
MEL_IDX = 0


# ===========================================================================
# Loss Functions for Class Imbalance Handling
# ===========================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017).
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    The (1 - p_t)^gamma term down-weights easy examples (high confidence),
    forcing the model to focus on hard examples — typically the minority class.
    
    Args:
        alpha: balancing factor (typical 0.25)
        gamma: focusing parameter (typical 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce)
        focal = self.alpha * (1 - p_t) ** self.gamma * ce
        return focal.mean()


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss (Cui et al. 2019).
    
    Weights each class by 1 / E_n where E_n = (1 - beta^n) / (1 - beta)
    is the "effective number" of samples — accounting for sample overlap
    in the feature space.
    
    Args:
        samples_per_class: array-like, sample count for each class
        beta: hyperparameter (typical 0.9999)
    """
    def __init__(self, samples_per_class, beta=0.9999):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weights.to(logits.device))


def get_loss_function(loss_type='ce', samples_per_class=None,
                     focal_alpha=0.25, focal_gamma=2.0, beta=0.9999):
    """
    Factory function for loss functions.
    
    Options:
        'ce'    - Standard CrossEntropyLoss (no imbalance handling)
        'wce'   - Weighted CrossEntropy (inverse frequency)
        'focal' - Focal Loss
        'cb'    - Class-Balanced Loss (Cui et al. 2019)
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'wce':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for weighted CE")
        weights = 1.0 / np.array(samples_per_class)
        weights = weights / weights.sum() * len(samples_per_class)
        weights = torch.tensor(weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weights)
    
    elif loss_type == 'focal':
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    elif loss_type == 'cb':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for class-balanced loss")
        return ClassBalancedLoss(samples_per_class, beta=beta)
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


# ===========================================================================
# Training & Validation
# ===========================================================================
def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, use_amp=False):
    """
    One pass through training data.
    
    For each batch:
      1. Forward pass
      2. Compute loss
      3. Backward pass
      4. Optimizer step (with optional AMP)
    
    Args:
        model:     PyTorch model
        loader:    Training DataLoader
        optimizer: Adam, SGD, etc.
        criterion: Loss function
        device:    'cuda', 'mps', or 'cpu'
        scaler:    GradScaler (AMP only)
        use_amp:   Mixed precision flag (CUDA only)
    
    Returns:
        avg_loss, accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='  Train', leave=False)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_amp and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix(loss=f'{loss.item():.4f}', acc=f'{correct/total:.4f}')
    
    return running_loss / total, correct / total


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """
    One pass through validation data.
    Returns (loss, AUC) — melanoma vs everything else.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    for images, targets in tqdm(loader, desc='  Valid', leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        logits = model(images)
        loss = criterion(logits, targets)
        running_loss += loss.item() * images.size(0)
        
        # Melanoma probability via softmax
        probs = torch.softmax(logits, dim=1)[:, MEL_IDX]
        
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    val_loss = running_loss / len(all_targets)
    
    binary_targets = (all_targets == MEL_IDX).astype(int)
    if binary_targets.sum() > 0 and binary_targets.sum() < len(binary_targets):
        auc = roc_auc_score(binary_targets, all_probs)
    else:
        auc = 0.0
    
    return val_loss, auc


def make_amp_components(use_amp, device):
    """
    Create AMP components if appropriate.
    
    Returns:
        scaler: GradScaler or None
        use_amp: bool (False if device doesn't support AMP)
    """
    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        return scaler, True
    
    if use_amp and device.type != 'cuda':
        print(f"⚠️  AMP requested but device is {device.type}, falling back to FP32")
    return None, False