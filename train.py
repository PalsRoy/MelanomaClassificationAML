"""
train.py - Training & Validation Functions
Supports CUDA (with AMP) and MPS / CPU (without AMP).
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Melanoma is class 0 in our DIAGNOSIS2IDX mapping
MEL_IDX = 0


def train_one_epoch(model, loader, optimizer, criterion, device,
                    scaler=None, use_amp=False):
    """
    Run one full pass through the training data.
    
    For each batch:
      1. Forward pass — get model predictions
      2. Compute loss — how wrong were we
      3. Backward pass — calculate gradients
      4. Optimizer step — nudge weights to reduce loss
    
    Args:
        model:     PyTorch model
        loader:    DataLoader for training data
        optimizer: Adam, SGD, etc.
        criterion: loss function (e.g. CrossEntropyLoss)
        device:    'cuda', 'mps', or 'cpu'
        scaler:    GradScaler (only used with AMP, can be None)
        use_amp:   if True, use mixed precision (CUDA only)
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
            # Mixed precision path (CUDA only)
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard precision path (MPS, CPU, or CUDA without AMP)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        
        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix(
            loss=f'{loss.item():.4f}',
            acc=f'{correct/total:.4f}'
        )
    
    return running_loss / total, correct / total


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """
    Run one full pass through validation data.
    Returns (loss, AUC).
    
    Note: validation does NOT use AMP — we want the most precise
    predictions for AUC computation, and validation is fast anyway.
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
        
        # Get melanoma probability via softmax
        probs = torch.softmax(logits, dim=1)[:, MEL_IDX]
        
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)
    
    # Compute binary AUC: melanoma vs everything else
    binary_targets = (all_targets == MEL_IDX).astype(int)
    if binary_targets.sum() > 0 and binary_targets.sum() < len(binary_targets):
        auc = roc_auc_score(binary_targets, all_probs)
    else:
        auc = 0.0
    
    return running_loss / len(all_targets), auc


def make_amp_components(use_amp, device):
    """
    Helper to create AMP components if and only if appropriate.
    
    Returns:
        scaler: GradScaler instance, or None if AMP is disabled
        use_amp: actual use_amp value (False if device doesn't support it)
    
    Use in your notebook like:
        scaler, use_amp = make_amp_components(use_amp=True, device=CFG.device)
        # then pass to train_one_epoch
    """
    if use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        return scaler, True
    else:
        if use_amp and device.type != 'cuda':
            print(f"⚠️  AMP requested but device is {device.type}, falling back to FP32")
        return None, False