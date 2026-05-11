# Training Log — Week 8

**Date Range:** 2026-05-XX to 2026-05-XX  
**Module:** EEEM068 — Advanced Machine Learning  
**Project:** Melanoma Classification (SIIM-ISIC 2020)  
**Repository:** https://github.com/PalsRoy/MelanomaClassificationAML

---

## 1. Objective

Complete the experimental phase of the project:
- Add DeiT-Small as a second transformer architecture for comparison
- Conduct a focused class-imbalance handling experiment using Focal Loss on the winning architecture
- Run an extended 15-epoch experiment on the best model for the headline result
- Generate analysis figures and begin technical report writing

---

## 2. Activities & Commits

| Date | Activity | Commit |
|------|----------|--------|
| Day 1 | Ran exp5: DeiT-Small (5 epochs) | results/exp5_deit_small.json |
| Day 2 | Implemented `FocalLoss`, `ClassBalancedLoss`, `get_loss_function` in `train.py` | train.py update |
| Day 2 | Ran exp4b: Swin-Base + Focal Loss (5 epochs) | results/exp4b_swin_base_focal.json |
| Day 3 | Ran exp_final: Swin-Base + CrossEntropy (15 epochs) — headline result | results/exp_final_swin_15ep.json |
| Day 4 | Generated comparison figures locally from saved JSONs | report assets |
| Day 5 | Started drafting Methodology and Experiments sections of report | draft document |

---

## 3. Experimental Settings

### 3.1 DeiT-Small Run

| Setting | Value |
|---------|-------|
| Model | deit_small_patch16_224 |
| Image size | 224 × 224 |
| Batch size | 96 (smaller model allows larger batch) |
| Epochs | 5 |
| Other settings | Same as Week 7 (Adam, lr=3e-5, cosine annealing, CE loss) |

### 3.2 Focal Loss Experiment

| Setting | Value |
|---------|-------|
| Model | swin_base_patch4_window7_224 (winner from Week 7) |
| Loss | Focal Loss (Lin et al. 2017) |
| Focal alpha | 0.25 |
| Focal gamma | 2.0 |
| Image size | 224 × 224 |
| Batch size | 64 |
| Epochs | 5 |

### 3.3 Final 15-Epoch Run

| Setting | Value |
|---------|-------|
| Model | swin_base_patch4_window7_224 |
| Loss | CrossEntropyLoss |
| Image size | 224 × 224 |
| Batch size | 64 |
| Epochs | 15 |
| LR schedule | CosineAnnealingLR (T_max=15) |

---

## 4. Results

### 4.1 Complete Architecture Comparison (5 epochs)

| Rank | Model | Type | Params | Best Val AUC |
|------|-------|------|--------|--------------|
| 1 | Swin-Base | Transformer | 88M | **0.9363** |
| 2 | ConvNeXt-Small | CNN | 50M | 0.9289 |
| 3 | DeiT-Small | Transformer | 22M | 0.9183 |
| 4 | EfficientNet-B3 | CNN | 12M | 0.9126 |
| 5 | EfficientNet-B4 | CNN | 19M | 0.9085 |

### 4.2 Imbalance Handling Comparison (Swin-Base, 5 epochs)

| Loss Function | Best Val AUC | Δ vs CE |
|---------------|--------------|---------|
| CrossEntropy (baseline) | 0.9363 | — |
| Focal Loss (α=0.25, γ=2.0) | 0.9294 | **−0.0069** |

### 4.3 Final 15-Epoch Run

| Metric | Value |
|--------|-------|
| Best Val AUC | *[to fill in once run completes]* |
| Best epoch | *[to fill in]* |
| Final train loss | *[to fill in]* |
| Final val loss | *[to fill in]* |
| Total training time | *[to fill in]* |

---

## 5. Observations

### DeiT-Small
- Trained quickly (~30 min for 5 epochs) due to lower parameter count and smaller image size.
- AUC of 0.9183 placed it third overall — strong for a 22M-parameter model.
- Confirmed that transformer architecture itself contributes to performance, not just Swin-specific design (DeiT is a vanilla ViT variant).

### Focal Loss Experiment (Counterintuitive Result)
- **Focal Loss decreased AUC by ~0.7 percentage points compared to CrossEntropy.**
- Initially surprising — focal loss is widely associated with imbalanced classification.
- Hypothesis: our pipeline already handles imbalance through five complementary implicit mechanisms (9-class target, stratified k-fold, augmentation, AUC metric, multi-sample dropout). Adding focal loss on top introduced redundant down-weighting that hurt performance.
- This is a **valuable negative result** — it demonstrates that explicit imbalance handling techniques are not always additive when pipeline-level handling is already strong.

### 15-Epoch Run
- *[Observations to fill in after run completes]*
- *Expected*: AUC plateau around epochs 8-10, possibly slight overfitting in later epochs (val loss ↑ while train loss ↓).
- The best-checkpoint-saving mechanism ensures we retain peak performance regardless of late-epoch behaviour.

---

## 6. Analysis

### Class Imbalance Discussion

The Focal Loss experiment provides a strong evidence base for our **multi-strategy imbalance handling claim**:

1. **Implicit mechanisms (already present):**
   - 9-class diagnosis target softens gradient signal across diagnoses
   - Stratified group k-fold maintains fold-level class balance
   - Heavy augmentation effectively oversamples minorities
   - Multi-sample dropout regularises against minority-class memorisation
   - AUC evaluation metric is robust to imbalance

2. **Explicit mechanisms (tested):**
   - **Focal Loss**: marginally hurt performance (−0.0069 AUC). Suggests the implicit handling is already saturating the available improvement.

3. **Mechanisms considered but not tested** (with rationale):
   - **Weighted CrossEntropy**: similar mathematical form to focal loss for binary problems; expected similar or worse results
   - **Class-Balanced Loss (Cui et al. 2019)**: with β=0.9999 and our extreme imbalance, converges to weighted CE
   - **Weighted Random Sampler**: image-domain SMOTE equivalent; risks overfitting via repeated minority exposure
   - **Undersampling**: would discard ~98% of training data — wasteful with only 33k samples

This provides the report with a complete imbalance handling discussion grounded in evidence, not just hypothetical comparison.

### Final Model Selection

Swin-Base + CrossEntropy is the chosen final configuration:
- Best 5-epoch result (0.9363)
- Imbalance handling experiment confirms CE is preferred over Focal in our setup
- 15-epoch run extends this for the headline number

---

## 7. Issues Encountered

| Issue | Resolution |
|-------|------------|
| Colab disconnection wiped `/content/data` between sessions | Documented re-extraction process in template; budgeted time for re-setup |
| GitHub notebook rendering error: "missing 'state' key in 'metadata.widgets'" | Wrote cleanup script using `nbformat` to strip widget metadata before commits |
| Compute units consumed by A100 faster than budget | Purchased additional units; planned T4 fallback for non-essential runs |
| Team members had logistics issues contributing | Created study guide (`docs/melanoma_study_guide.md`) explaining each report section so they can self-direct learning |

---

## 8. Report Writing Status (End of Week 8)

| Section | Status | Notes |
|---------|--------|-------|
| Abstract | Pending | Write last |
| Introduction | Drafting | Clinical + computational motivation done |
| Literature Review | In progress | 5 papers identified, 3 written up |
| Methodology | Drafting | Sections 4.1, 4.2, 4.3 drafted |
| Experiments | In progress | Tables and figures generated; discussion pending |
| Conclusion | Pending | Awaiting final 15-epoch result |

---

## 9. Final Pre-Submission Checklist

- [ ] 15-epoch Swin-Base run complete and AUC recorded
- [ ] All 6 result JSONs in `results/` folder
- [ ] Comparison plots generated (loss curves, AUC curves, model comparison)
- [ ] All weekly logs committed to `logs/` folder
- [ ] README updated with final results table
- [ ] Notebook outputs cleared / widget metadata stripped
- [ ] IEEE 5-page report PDF
- [ ] References properly formatted (≥5 papers)
- [ ] Repository access confirmed for assigned TA

---

## 10. Reflection

This project was a strong end-to-end exercise covering data exploration, pipeline engineering, model training, and comparative analysis. The most valuable lessons:

1. **Reproducibility matters** — pre-computed folds, fixed random seeds, and version-controlled config saved hours of debugging
2. **Negative results are findings** — the focal loss experiment was scientifically informative even though it didn't improve AUC
3. **Compute is a constraint** — budgeting Colab units forced disciplined experiment design
4. **Reading the literature pays off** — adopting Ha et al.'s 9-class target and multi-sample dropout decisions saved iteration cycles
