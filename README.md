# Melanoma Classification (AML Project)

Deep learning approach to melanoma classification using the SIIM-ISIC 2020 Melanoma Classification dataset. We compare CNN and Transformer architectures and evaluate class-imbalance handling strategies under identical training conditions.

Reference: SIIM-ISIC 2020 Kaggle 1st-place solution ([arXiv:2010.05351](https://arxiv.org/abs/2010.05351)).

![Melanoma Classification Network Architecture](figures/melanoma_classification_architecture.svg?raw=true "Network Architecture")

## Results

### Architecture Comparison (5 epochs, CrossEntropy, fold 0)

| Rank | Model | Type | Parameters | Image Size | Val AUC |
|------|-------|------|------------|------------|---------|
| 1 | Swin-Base | Transformer | 88M | 224 | **0.9363** |
| 2 | ConvNeXt-Small | CNN | 50M | 384 | 0.9289 |
| 3 | DeiT-Small | Transformer | 22M | 224 | 0.9183 |
| 4 | EfficientNet-B3 | CNN | 12M | 384 | 0.9126 |
| 5 | EfficientNet-B4 | CNN | 19M | 384 | 0.9085 |

### Imbalance Handling (Swin-Base, 5 epochs)

| Loss Function | Best Val AUC | Δ vs CE |
|---|---|---|
| CrossEntropy | **0.9363** | — |
| Focal Loss (α=0.25, γ=2.0) | 0.9294 | −0.0069 |

### Training Duration (Swin-Base, CrossEntropy)

| Epochs | Best Val AUC |
|---|---|
| 5 | **0.9363** |
| 15 | 0.9275 |

### Key Findings

- **Transformers outperformed CNNs**: Swin-Base and DeiT-Small both placed above the larger EfficientNet variants, supporting the hypothesis that self-attention captures multi-scale dermoscopy features more effectively than purely local convolutional receptive fields.
- **Smaller models can win**: EfficientNet-B3 (12M params) outperformed EfficientNet-B4 (19M params), suggesting excess capacity offers no benefit under our 5-epoch training budget.
- **Focal Loss hurt performance**: Adding explicit imbalance handling to a pipeline already incorporating five implicit strategies caused a small AUC decrease — evidence that imbalance handling is best treated as a multi-strategy problem rather than a single technique.
- **Longer training did not help**: 15-epoch training underperformed the 5-epoch run by ~1 percentage point, indicating overfitting on the small (~26k) training set. Early stopping based on validation AUC would have been more robust than fixed-epoch training with our cosine schedule.

## Repository Structure

```
melanomaClassificationAML/
├── config.py                       # Centralised configuration
├── dataset.py                      # MelanomaDataset + augmentation pipeline
├── models.py                       # MelanomaClassifier (works with any timm backbone)
├── train.py                        # Training/validation + loss factory (CE, Focal, CB, weighted)
├── train_with_folds.csv            # Pre-computed patient-grouped 5-fold splits
├── melanoma_classification.ipynb   # Main local training notebook
├── compare_results.ipynb           # Aggregates all experiment JSONs into plots/tables
├── experiments/
│   ├── _template.ipynb             # Colab template — duplicate this per model
│   ├── exp1_efficientnet_b4.ipynb
│   ├── exp2_efficientnet_b3.ipynb
│   ├── exp3_convnext_small.ipynb
│   ├── exp4_swin_base.ipynb
│   └── exp5_deit_small.ipynb
├── results/                        # Per-experiment training history JSONs
├── figures/                        # Comparison plots used in the report
├── logs/                           # Weekly training logs per team member
├── docs/
│   └── melanoma_study_guide.md     # Team learning resource
├── README.md
└── requirements.txt
```

## Architecture

All models share a common pipeline: input image → backbone (varies) → multi-sample dropout head (5 masks averaged) → 9-class softmax output. Melanoma probability is extracted as the softmax score of the melanoma class at inference.

**Backbones evaluated:**

- **EfficientNet-B3 / B4** with Noisy Student pretraining
- **ConvNeXt-Small** with ImageNet pretraining
- **DeiT-Small** with ImageNet pretraining
- **Swin-Base** with ImageNet pretraining

The custom head applies five independent dropout masks (p=0.5) to backbone features, passes each through a shared linear layer, and averages the resulting logits. This provides ensemble-like regularisation within a single forward pass.

## Key Design Decisions

### Dataset & Splitting
- **Stratified Group K-Fold** by `patient_id` to prevent data leakage between train/val
- 9-class diagnosis target instead of binary (per Kaggle 1st-place insight: forces the model to learn finer distinctions, producing better-calibrated melanoma probabilities)
- 5 folds computed once, stored in `train_with_folds.csv`, used identically across all experiments (only fold 0 trained due to compute constraints)
- Heavy augmentation: random crops, flips, rotations, affine, colour jitter, coarse dropout

### Models
- All backbones loaded from `timm` (PyTorch Image Models)
- 384×384 input for CNNs, 224×224 for Vision Transformers (matches pretrained input size)
- ImageNet normalisation statistics applied
- Custom classification head with multi-sample dropout (5 masks averaged)

### Training
- Adam optimizer, initial learning rate 3×10⁻⁵
- Cosine annealing LR schedule (decays to 1% of `init_lr` by final epoch)
- Mixed precision (AMP) on CUDA, FP32 on Apple MPS
- Batch size 64 (96 for DeiT-Small)
- 5 epochs for architecture comparison; 15-epoch deep dive on best architecture
- Best checkpoint selected by maximum validation AUC, saved every epoch to Drive

### Class Imbalance Handling
The project uses **multiple complementary strategies** rather than a single technique:

1. **9-class target** — softens gradient signal across visually similar diagnoses
2. **AUC evaluation metric** — robust to imbalance (unlike accuracy)
3. **Stratified k-fold** — fold-level class balance
4. **Heavy augmentation** — effective minority oversampling
5. **Multi-sample dropout** — ensemble-like regularisation
6. **Focal Loss experiment** — evaluated, but decreased AUC by 0.7pp (see Results above)

Methods considered but not evaluated: weighted CrossEntropy, Class-Balanced Loss (Cui 2019), undersampling, SMOTE. Rationale for each is discussed in the technical report.

## Quick Start

### Local development (Apple Silicon / Mac)
```bash
conda create -n melanomaClassificationAML python=3.11 -y
conda activate melanomaClassificationAML
pip install -r requirements.txt
jupyter notebook melanoma_classification.ipynb
```

### Google Colab (recommended for full training)
1. Open `experiments/_template.ipynb` in Colab
2. Save a copy in Drive, rename for your experiment (e.g. `exp2_efficientnet_b3.ipynb`)
3. Edit Cell 4 with your model choice
4. Run all cells

Training history (per-epoch loss, AUC, learning rate, wall-clock time) is saved as JSON to `results/<experiment>.json` after every epoch, enabling fully reproducible post-hoc analysis.

See `experiments/_template.ipynb` for the full pipeline.

## Reproducing the Results

```bash
# After training all experiments, aggregate results and generate figures:
jupyter notebook compare_results.ipynb
```

This produces:
- `figures/fig_architecture_auc.png` — validation AUC over epochs across all 5 architectures
- `figures/fig_architecture_loss.png` — training/validation loss curves
- `figures/fig_imbalance_handling.png` — CrossEntropy vs Focal Loss on Swin-Base
- `figures/fig_training_duration.png` — 5 vs 15 epochs on Swin-Base
- `figures/summary_table.csv` — full results summary

## References

- Esteva, A., et al. (2017). *Dermatologist-level classification of skin cancer with deep neural networks.* Nature 542, 115–118. [DOI](https://doi.org/10.1038/nature21056)
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML. [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
- Ha, Q., Liu, B., & Liu, F. (2020). *Identifying Melanoma Images using EfficientNet Ensemble.* [arXiv:2010.05351](https://arxiv.org/abs/2010.05351)
- Liu, Z., et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV. [arXiv:2103.14030](https://arxiv.org/abs/2103.14030)
- Liu, Z., et al. (2022). *A ConvNet for the 2020s.* CVPR. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)
- Touvron, H., et al. (2021). *Training data-efficient image transformers & distillation through attention.* ICML. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- Lin, T.-Y., et al. (2017). *Focal Loss for Dense Object Detection.* ICCV. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Wightman, R. (2019). *PyTorch Image Models (timm).* [GitHub](https://github.com/huggingface/pytorch-image-models)

## Team

| Member | Lead Role | Contributing To |
|--------|-----------|-----------------|
| Pallavi Roy Sawant | Pipeline architecture, EfficientNet experiments, Focal Loss imbalance experiment | All sections |
| Giridhar | Model variants, ConvNeXt experiments | All sections |
| Thilanjan | Training optimisations, Swin experiments | All sections |
| Gopi | Imbalance handling, DeiT experiments | All sections |
| Poorna | Evaluation, comparison analysis, report | All sections |

*All members contribute to literature review, methodology, coding, training, evaluation, and report writing per assessment guidelines. See `logs/` for weekly training logs per member.*

## License

Academic project for EEEM068 — University of Surrey.
