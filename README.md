# Melanoma Classification (AML Project)

Deep learning approach to melanoma classification using the SIIM-ISIC Melanoma Classification dataset. We compare multiple CNN and Transformer architectures and apply data imbalance handling strategies.
([arXiv:2010.05351](http://arxiv.org/abs/2010.05351)).

## Architecture

**EfficientNet-B4** with Noisy Student pretraining, using:

- Multi-sample dropout (5 masks averaged) for regularization
- 9-class output head (multi-class outperforms binary per competition findings)
- Melanoma probability extracted via softmax at inference

## Key Design Decisions

### Dataset & Splitting
- **Stratified Group K-Fold** by `patient_id` to prevent data leakage between train/val
- 9-class diagnosis target instead of binary (per Kaggle 1st place insight)
- Heavy augmentation: random crops, flips, rotations, colour jitter, coarse dropout

### Models
- All backbones from `timm` (PyTorch Image Models)
- Custom classification head with **multi-sample dropout** (5 masks averaged)
- Pretrained weights (Noisy Student for EfficientNet, ImageNet for others)

### Training
- Adam optimizer, learning rate 3×10⁻⁵
- Cosine annealing LR schedule (decays to 1% of init_lr by final epoch)
- Mixed precision (AMP) on CUDA, FP32 on Apple MPS
- Batch size 64 (96 for DeiT-Small)

### Class Imbalance Handling
The project uses **multiple complementary strategies** rather than a single technique:
1. 9-class target (smooths gradient signal)
2. AUC evaluation metric (robust to imbalance)
3. Stratified k-fold (fold-level class balance)
4. Heavy augmentation (effective minority oversampling)
5. Multi-sample dropout (regularisation)
6. *Planned:* Focal Loss + Weighted Random Sampler experiments on best model

## Quick Start

### Local development (Apple Silicon)
```bash
conda create -n melanomaClassificationAML python=3.11 -y
conda activate melanomaClassificationAML
pip install -r requirements.txt
jupyter notebook melanoma_classification.ipynb
```

### Google Colab (recommended for full training)
1. Open `experiments/_template.ipynb` in Colab
2. Save a copy in Drive, rename for your experiment
3. Edit Cell 4 with your model choice
4. Run all cells

See `experiments/_template.ipynb` for the full pipeline.

## References

- Ha, Q., Liu, B., & Liu, F. (2020). *Identifying Melanoma Images using EfficientNet Ensemble.* arXiv:2010.05351
- Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* ICML.
- Liu, Z. et al. (2021). *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV.
- Liu, Z. et al. (2022). *A ConvNet for the 2020s.* CVPR.
- Touvron, H. et al. (2021). *Training data-efficient image transformers & distillation through attention.* ICML.
- Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.

## Team

| Member | Lead Role | Contributing To |
|--------|-----------|-----------------|
| Pallavi Roy Sawant | Pipeline architecture, EfficientNet experiments | All sections |
| Member B | Model variants, ConvNeXt experiments | All sections |
| Member C | Training optimisations, Swin experiments | All sections |
| Member D | Imbalance handling, DeiT experiments | All sections |
| Member E | Evaluation, comparison analysis, report | All sections |

*All members contribute to literature review, methodology, coding, training, evaluation, and report writing per assessment guidelines.*

## License

Academic project for EEEM068 module — University of Surrey.

## Experiment Results

## Results

| Rank | Model | Type | Parameters | Image Size | Val AUC | Loss |
|------|-------|------|------------|------------|---------|------|
| 1 | Swin-Base | Transformer | 88M | 224 | **0.9363** | CrossEntropy |
| 2 | ConvNeXt-Small | CNN | 50M | 384 | 0.9289 | CrossEntropy |
| 3 | EfficientNet-B3 | CNN | 12M | 384 | 0.9126 | CrossEntropy |
| 4 | EfficientNet-B4 | CNN | 19M | 384 | 0.9085 | CrossEntropy |

All models trained for 5 epochs with Adam optimizer, cosine annealing LR schedule, and identical augmentation pipeline.


