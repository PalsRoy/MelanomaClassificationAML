import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al. 2017, "Focal Loss for Dense Object Detection")
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    The (1 - p_t)^gamma term down-weights easy (high-confidence) examples,
    making the model focus on hard ones — typically the minority class.
    
    Args:
        alpha: weighting factor (typical 0.25)
        gamma: focusing parameter (typical 2.0). Higher = more focus on hard examples.
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
    Class-Balanced Loss (Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of Samples")
    
    Weights each class by 1 / E_n where E_n = (1 - beta^n) / (1 - beta)
    is the "effective number" of samples for class n.
    
    For highly imbalanced data, this adjusts loss contribution so rare
    classes aren't dominated by frequent ones.
    
    Args:
        samples_per_class: list of int, sample count for each class
        beta: hyperparameter (typical 0.9999), controls how aggressively to re-weight
    """
    def __init__(self, samples_per_class, beta=0.9999):
        super().__init__()
        # Effective number of samples per class
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(samples_per_class)
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, logits, targets):
        weights = self.weights.to(logits.device)
        return F.cross_entropy(logits, targets, weight=weights)


def get_loss_function(loss_type='ce', samples_per_class=None,
                     focal_alpha=0.25, focal_gamma=2.0, beta=0.9999):
    """
    Factory function for loss functions, supporting multiple
    imbalance handling strategies.
    
    Options:
        'ce'      - Standard CrossEntropyLoss (no imbalance handling)
        'wce'     - Weighted CrossEntropy (inverse frequency)
        'focal'   - Focal Loss
        'cb'      - Class-Balanced Loss (Cui et al. 2019)
    
    Args:
        loss_type: which loss to use
        samples_per_class: required for 'wce' and 'cb' loss types
        focal_alpha, focal_gamma: hyperparameters for focal loss
        beta: hyperparameter for class-balanced loss
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'wce':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for weighted CE")
        # Inverse frequency weighting
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