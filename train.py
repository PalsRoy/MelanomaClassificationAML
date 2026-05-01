"""
train.py - Training & Validation Functions
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Melanoma is class 0 (from DIAGNOSIS2IDX in our data)
MEL_IDX = 0


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Run one full pass through the training data.
    
    For each batch:
      1. Forward pass — get model predictions
      2. Compute loss — how wrong were we
      3. Backward pass — calculate gradients
      4. Optimizer step — nudge weights to reduce loss
    """
    model.train()  # Enable dropout, batchnorm updates
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='  Train', leave=False)
    for images, targets in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # Reset gradients from previous batch
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)
        
        # Backward pass + optimizer step
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


@torch.no_grad()  # Disable gradient tracking - saves memory during eval
def validate_one_epoch(model, loader, criterion, device):
    """
    Run one full pass through the validation data.
    
    Returns loss, AUC, and accuracy. AUC is the key metric since
    our classes are imbalanced (1.76% melanoma).
    """
    model.eval()  # Disable dropout for deterministic predictions
    running_loss = 0.0
    all_targets = []
    all_probs = []
    
    for images, targets in tqdm(loader, desc='  Valid', leave=False):
        images = images.to(device)
        targets = targets.to(device)
        
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