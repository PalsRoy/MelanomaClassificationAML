# Training Log — Week 1-6

**Date Range:** 2026-04-XX to 2026-04-XX  
**Module:** EEEM068 — Advanced Machine Learning  
**Project:** Melanoma Classification (SIIM-ISIC 2020)  
**Repository:** https://github.com/PalsRoy/MelanomaClassificationAML

---

## 1. Objective

Set up the project from scratch: environment configuration on Apple Silicon mainly, dataset acquisition, exploratory data analysis (EDA), and an initial pipeline scaffold. Goal: by end of week, have a runnable end-to-end training loop on a small subset to verify correctness.

---

## 2. Activities & Commits

| Date | Activity | Commit / Output |
|------|----------|-----------------|
| Week 1 | Conda environment setup (`melanomaClassificationAML`, Python 3.11) | env config, requirements.txt |
| Week 1 | PyTorch + MPS verification on M5 Pro | device_selection.ipynb |
| Wekk 1 | Downloaded SIIM-ISIC dataset (~130 GB total: DICOM + JPEG + TFRecords) | local storage |
| Week 2 | EDA: image counts, target distribution, image size analysis | EDA cells in main notebook | Spoke to TA on JPEG choice |
| Week 2 | Reviewed Kaggle 1st-place solution (Ha et al. 2020) | reference notes |
| Week 3 | Drafted project structure: config.py, dataset.py, models.py, train.py | initial commits |
| Week 4 | Implemented Dataset class with augmentation pipeline | dataset.py |
| Week 5 | Implemented EfficientNetMelanoma model with multi-sample dropout | models.py |

---

## 3. Experimental Settings (Initial Pipeline)

| Setting | Value |
|---------|-------|
| Hardware | Apple M5 Pro 2026, 24 GB unified memory, MPS backend |
| Python | 3.11 (conda) |
| PyTorch | 2.x with MPS support |
| Image format | JPEG (faster than DICOM for training + spoke to TA) |
| Train images | 33,126 |
| Test images | 10,982 |
| Class imbalance | 1.76% melanoma (584 / 33,126) |
| Image size range | 640×480 to 6000×5184 (10 unique sizes) |

---

## 4. Key EDA Findings

| Finding | Implication |
|---------|-------------|
| 1.76% melanoma class | Severe imbalance — accuracy unsuitable as metric, must use AUC |
| 10 unique source sizes | Suggests multiple imaging devices; uniform resize required for training |
| Average dimensions ~3872×2578 | Significant downscaling needed (target 384px); will lose fine texture |
| `patient_id` field present | Multiple images per patient → patient-grouped CV essential to prevent leakage |
| 65 missing `sex`, 68 missing `age`, 527 missing `anatom_site` | Metadata incomplete; image-only approach simplifies pipeline |

---

## 5. Design Decisions

Following Ha et al. (2020) winning solution patterns:

- **9-class diagnosis target** rather than binary, softens gradient, model learns finer-grained distinctions
- **Multi-sample dropout** (5 masks averaged) instead of single dropout, mini-ensemble within one model
- **StratifiedGroupKFold** by `patient_id` prevents data leakage between train/val
- **Aggressive augmentation** (RandomResizedCrop, flips, rotations, colour jitter, CoarseDropout), combats small dataset and class imbalance
- **AUC** as primary metric, robust to class imbalance (unlike accuracy)
- **Adam** optimiser, lr=3e-5, **cosine annealing** schedule — standard configuration

---

## 6. Observations

- The dataset is smaller than expected for medical imaging — 33k images is modest. Transfer learning from ImageNet will be essential.
- Image-size variability suggests the original images came from at least 10 different camera systems. Augmentation involving colour jitter is well-justified.
- Class imbalance at 1.76% is at the extreme end, many imbalance-handling techniques designed for 5-20% minority rates may not apply directly.
- Apple MPS backend works but does not support mixed precision (AMP). Will need Colab CUDA for production training. Had to buy CUDA A100 for uninterrupted trainiing. Full time working hence time constraints played a key role. 

---

## 7. Issues Encountered

| Issue | Resolution |
|-------|------------|
| `cv2` and `albumentations` not installed in Jupyter kernel | Activated correct conda env in terminal before `pip install` |
| `albumentations 2.0` API changed — `RandomResizedCrop(height=, width=)` deprecated | Updated to `RandomResizedCrop(size=(h, w))` |
| `ShiftScaleRotate` deprecated | Replaced with `Affine(translate_percent=, scale=, rotate=)` |
| `CoarseDropout(max_holes=, max_height=, max_width=)` deprecated | Updated to `num_holes_range=, hole_height_range=, hole_width_range=` |
| Python 3.12 incompatibility with `autoreload` (missing `imp` module) | Removed `%load_ext autoreload`, restart kernel manually after .py edits |

---

## 8. Initial Smoke Test Result

Ran 1 epoch of EfficientNet-B4 on Apple MPS to verify pipeline correctness:

| Metric | Value |
|--------|-------|
| Train Loss | 0.1678 |
| Train Accuracy | 0.9732 |
| Val Loss | 0.0755 |
| Val AUC | **0.8587** |
| Time per epoch | ~34 minutes |

**Interpretation:** AUC of 0.86 after a single epoch confirms transfer learning is working. The high training accuracy (0.97) is misleading due to class imbalance — a model that always predicted "benign" would already achieve 0.98. Val AUC is the reliable signal.

---

## 9. Next Steps (for Week 7)

1. **Migrate to Google Colab** — local training too slow for 15-epoch runs (~8 hours)
2. **Set up reproducible cross-validation splits** — pre-compute folds, save as CSV
3. **Run baseline EfficientNet-B4 for 5 epochs** as the architecture comparison anchor
4. **Build experiment notebook template** for team to use
