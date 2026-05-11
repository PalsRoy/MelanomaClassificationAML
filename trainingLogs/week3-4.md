# Training Log — Week 7

**Date Range:** 2026-03-16 to 2026-04-29
**Module:** EEEM068 — Advanced Machine Learning  
**Project:** Melanoma Classification (SIIM-ISIC 2020)  
**Repository:** https://github.com/PalsRoy/MelanomaClassificationAML

---

## 1. Objective

Migrate training to Google Colab (CUDA) for faster iteration. Run baseline architecture comparison across 5 model families: EfficientNet (B3, B4), ConvNeXt-Small, Swin-Base, DeiT-Small. Hypothesis: transformer architectures will perform comparably or better than CNNs, but at higher computational cost.

---

## 2. Activities & Commits

| Date | Activity | Commit |
|------|----------|--------|
| Day 1 | Set up Colab Pro environment, mounted Google Drive | colab_experiments.ipynb |
| Day 1 | Created `train_with_folds.csv` (pre-computed 5-fold split, shared in repo) | prepare_data.ipynb |
| Day 2 | Built `experiments/_template.ipynb` (Colab-ready experiment template) | experiments/_template.ipynb |
| Day 2 | Added AMP support to `train.py` (CUDA mixed precision) | train.py |
| Day 3 | Refactored `models.py` — `MelanomaClassifier` works with any timm backbone | models.py |
| Day 3 | Resolved Colab `sys.path` issue with `importlib`-based module loading | template fix |
| Day 4 | Ran exp1: EfficientNet-B4 (architecture baseline) | results/exp1_efficientnet_b4.json |
| Day 4 | Ran exp2: EfficientNet-B3 | results/exp2_efficientnet_b3.json |
| Day 5 | Ran exp3: ConvNeXt-Small | results/exp3_convnext_small.json |
| Day 5 | Ran exp4: Swin-Base | results/exp4_swin_base.json |

---

## 3. Experimental Settings (Common to All Runs)

| Setting | Value |
|---------|-------|
| Hardware | Google Colab Pro, NVIDIA A100-SXM4-80GB |
| Loss | CrossEntropyLoss (9-class) |
| Optimizer | Adam, lr=3e-5 |
| LR Schedule | CosineAnnealingLR, eta_min = 3e-7 |
| Epochs | 5 |
| Mixed Precision | Enabled (AMP) |
| Random Seed | 42 |
| Cross-validation | StratifiedGroupKFold by patient_id, k=5 |
| Fold trained | 0 |
| Train images | 26,499 |
| Validation images | 6,627 |

### Per-model variations

| Experiment | Model | Image size | Batch size |
|------------|-------|------------|------------|
| exp1 | tf_efficientnet_b4_ns | 384 | 64 |
| exp2 | tf_efficientnet_b3_ns | 384 | 64 |
| exp3 | convnext_small | 384 | 64 |
| exp4 | swin_base_patch4_window7_224 | 224 | 64 |

*Note: Transformer models default to 224×224 input as per their pretraining; CNNs trained at 384 to match available pretrained weights.*

---

## 4. Results Summary

| Model | Type | Params | Img Size | Best Val AUC | Time/epoch |
|-------|------|--------|----------|--------------|------------|
| EfficientNet-B4 | CNN | 19M | 384 | 0.9085 | ~12 min |
| EfficientNet-B3 | CNN | 12M | 384 | 0.9126 | ~10 min |
| ConvNeXt-Small | CNN | 50M | 384 | 0.9289 | ~11 min |
| **Swin-Base** | **Transformer** | **88M** | **224** | **0.9363** | **~9 min** |

---

## 5. Observations

### Training Dynamics
- **All models converged smoothly** — train loss decreased monotonically, validation AUC trended upward each epoch.
- **No overfitting observed in 5 epochs** — val loss either decreased or remained stable; train/val gap remained reasonable.
- **Cosine annealing reached eta_min by epoch 5** as designed.

### Architecture-Specific Notes
- **EfficientNet-B3 outperformed B4** — counter-intuitive, since B4 is larger. Possibly because B3's smaller capacity is better matched to the dataset size (~33k images), reducing overfitting risk.
- **ConvNeXt-Small outperformed both EfficientNets** — supports the modern CNN design choices (large kernels, layer norm, GELU) borrowed from transformers.
- **Swin-Base was the strongest** — its hierarchical attention mechanism captures both local and global features in dermoscopy images effectively.
- **Despite Swin having 88M parameters (vs 12M for B3), training was faster per epoch** because images are smaller (224 vs 384).

### Class Imbalance Handling (Implicit)
The pipeline handles imbalance through multiple mechanisms even before applying explicit techniques:
1. 9-class target softens gradient signal
2. AUC metric robust to imbalance
3. Aggressive augmentation effectively oversamples
4. Stratified k-fold ensures fold-level balance
5. Multi-sample dropout regularises against minority-class memorisation

---

## 6. Analysis

The architecture comparison reveals a clear pattern: **transformer-based models outperform CNNs for this medical imaging task**, despite the relatively small dataset size that conventional wisdom would suggest favours CNNs.

**Hypothesis for transformer success:** Dermoscopy images contain diagnostic information at multiple scales — global lesion shape and asymmetry (long-range) combined with fine texture patterns (local). Self-attention naturally captures both, while CNNs are constrained to local receptive fields without explicit hierarchical aggregation.

**Hypothesis for B3 > B4:** With only ~26k training samples, increased model capacity provides diminishing returns and may even hurt generalisation. This aligns with the bias-variance trade-off — B4's higher capacity slightly overfits the training distribution.

**ConvNeXt's strong showing** validates the design philosophy of bringing transformer-style choices (LayerNorm, GELU, large kernels, fewer activations) into convolutional architectures.

---

## 7. Issues Encountered

| Issue | Resolution |
|-------|------------|
| Colab CPU runtime initially assigned (3 hr/epoch) | Switched runtime to GPU (A100), 12 min/epoch |
| Initial Drive upload incomplete (only 14k of 33k images) | Re-uploaded as single jpeg.zip (32 GB), extracted in Colab |
| `from config import CFG` failed in Colab — `ModuleNotFoundError` | Used `importlib.util.spec_from_file_location` to load modules explicitly |
| Multiple unzip cycles needed — `__MACOSX` files prompted overwrites | Added `-o` flag (`unzip -q -o`) to suppress prompts |
| Compute units depleted faster than expected on A100 | Purchased additional units; future runs use A100 sparingly |

---

## 8. Per-Epoch AUC Trajectory (Validation)

*(Values shown for visual comparison — full history saved in `results/*.json`)*

| Epoch | EffNet-B3 | EffNet-B4 | ConvNeXt-S | Swin-Base |
|-------|-----------|-----------|------------|-----------|
| 1 | 0.85 | 0.84 | 0.87 | 0.89 |
| 2 | 0.88 | 0.87 | 0.89 | 0.91 |
| 3 | 0.90 | 0.89 | 0.91 | 0.92 |
| 4 | 0.91 | 0.90 | 0.92 | 0.93 |
| 5 | **0.9126** | **0.9085** | **0.9289** | **0.9363** |

*Approximate values, see JSON files for exact figures.*

---

## 9. Next Steps (for Week 8)

1. Add **DeiT-Small** to complete the transformer comparison (5-epoch baseline)
2. Run **Swin-Base + Focal Loss** experiment for explicit imbalance handling analysis
3. Run **Swin-Base 15 epochs** as the final headline result
4. Generate comparison plots (validation AUC curves across all models)
5. Begin drafting the IEEE technical report (Methodology, Experiments sections first)
