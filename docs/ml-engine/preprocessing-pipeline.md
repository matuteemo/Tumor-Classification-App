# Preprocessing pipeline (agreed version)

**Date:** 2026-10-29
**Scope:** `app/ml_engine`, 2D brain MRI classification into glioma, meningioma, pituitary and no tumor.
**Boards alignment:** epic #103 (rev 4) and its issues #108-#114, #150, #165-#168. This document is the written reference for that epic; when the two disagree, update both in the same change.
**LaTeX version:** [`preprocessing-pipeline.tex`](preprocessing-pipeline.tex).

---

## 1. Summary

The pipeline has ten functional blocks, B1 to B10. They were chosen by two criteria: published evidence (references in section 10) and computational and implementation cost.

- Steps that **learn from data** (thresholds, augmentation, scaler, PCA, classifier, calibration) only see training rows. The official Testing split is scored once.
- Steps that **do not learn** (load, mask, geometry, tensor) are one shared function for training and inference.
- **Excluded:** steps that add cost without evidence of a gain, such as skull stripping, CLAHE, N4, denoising and SMOTE.

---

## 2. Dataset facts

Measured on the cleaned dataset versioned in `develop` (`app/ml_engine/src/data`, manifest `processed/dedup_20261028T234450903626Z/manifest.csv`), on 2026-10-29.

| Fact | Value |
|---|---|
| Images before cleaning | 7,200 (Kaggle Brain Tumor MRI Dataset, Masoud Nickparvar) |
| Images kept after cleaning (#150) | 3,434 |
| Groups of near-duplicates with members in both Training and Testing | 263 |
| Training images in those groups (removed by B2) | 495 |
| Training after B2 | 1,830 |
| Testing (official, unchanged) | 1,109 |

| Class | Training kept | Removed by B2 | Training final | Testing |
|---|---|---|---|---|
| glioma | 992 | 142 | 850 | 354 |
| meningioma | 539 | 93 | 446 | 273 |
| notumor | 275 | 15 | 260 | 125 |
| pituitary | 519 | 245 | 274 | 357 |

**Image geometry varies by class.** The source datasets differ, and so do their image shapes:

| Class | Not square | Exactly 512x512 |
|---|---|---|
| glioma | 3.9% | 1,284 / 1,346 |
| meningioma | 13.4% | 686 / 812 |
| notumor | **72.8%** | **5 / 400** |
| pituitary | 1.7% | 835 / 876 |

This means 92.5% of tumor images (2,805 of 3,034) are 512x512, against 1.25% of notumor images. A plain resize distorts almost only one class, so the model could learn the distortion instead of the anatomy. Blocks B6 and B10 address this.

> The note in #166 says "no notumor image is 512x512" and "93% (2,810 of 3,034)". The measured values are 5 of 400 and 92.5% (2,805 of 3,034). The conclusion does not change.

---

## 3. Execution order

### 3.1 Training (offline)

| Step | Block | What happens | Learns from data | Boards |
|---|---|---|---|---|
| 1 | B1 | Deduplication: SHA-256 of pixels + pHash, confirmed by thumbnail correlation. Writes the manifest with `group_id`. | No | #150 (Done) |
| 2 | B2 | Leakage-free split: keep the official Testing folder, remove from Training every `group_id` present in Testing. | No | #165 |
| 3 | B9 | Fold assignment: `StratifiedGroupKFold` (5 folds, group = `group_id`, fixed seed) on the purged Training split. | No | #112 |
| 4 | B4 | Quality thresholds: in the gap between synthetic unusable references and the lowest SNR and entropy over the purged Training split. | **Yes**, Training only | #165, #109, #175 |
| 5 | B3, B4, B5, B6, B8 | Per image: load, quality gate, brain mask, geometry, tensor. Same code as inference. | No | #108, #109, #166, #111 |
| 6 | - | Features: radiomics (#110) on the mask at original resolution, frozen CNN embedding (#111) on the tensor. Cached on disk. | No | #110, #111 |
| 7 | B7 | Augmentation: k copies (default 1) of each Training image, used only as fit rows of a fold. | No, but used only in fit rows | #167 |
| 8 | B9 | Per fold: scaler + PCA (radiomics), scaler (CNN), classifier with `class_weight="balanced"`. Select by mean CV macro-F1. | **Yes**, fit rows only | #112 |
| 9 | B9 | Calibration (`CalibratedClassifierCV`, sigmoid) on the same grouped folds. | **Yes**, fit rows only | #112 |
| 10 | B9 | Single evaluation on Testing. Report: accuracy, macro-F1, balanced accuracy, per-class precision/recall, confusion matrix, ablations. | No | #112 |
| 11 | B10 | Shortcut audit and leakage guarantees written in the report. | Yes, metadata baseline, inside the same CV | #168 |
| 12 | - | Artifact bundle and GitHub Release. | No | #114 |

### 3.2 Inference (online, per uploaded image)

```
uploaded bytes
  -> B3 load (gray float32 [0, 1])           -> UnsupportedImageError
  -> B4 quality gate (stored thresholds)     -> QualityCheckError
  -> B5 brain mask
  -> radiomics on mask ----------------------+
  -> B6 crop -> pad square -> 224            |
  -> B8 3 channels + backbone mean/std       |
  -> frozen CNN -----------------------------+
  -> stored scaler / PCA / calibrated classifier
  -> PredictionResult (#107, #113)
```

No augmentation and no refitting at inference. Every stored parameter comes from the artifact manifest (#113).

### 3.3 Diagram

```
                     TRAINING                                     INFERENCE
+---------------------------------------------------------+  +----------------+
| 1  B1 Deduplication (group_id)              #150        |  |                |
| 2  B2 Leakage-free split                    #165        |  |                |
|      Training 1,830 ------------- Testing 1,109 (held)  |  |                |
| 3  B9 StratifiedGroupKFold on Training      #112        |  |                |
| 4  B4 Calibrate thresholds on Training      #165        |  |                |
| 5  B3 load -> B4 gate -> B5 mask -> B6 geometry -> B8 <-+--+-- same code    |
| 6  radiomics + frozen CNN features (cached) #110 #111   |  |                |
| 7  B7 augmentation, fit rows only           #167        |  |   (never)      |
| 8  per fold: scaler, PCA, classifier        #112        |  |                |
| 9  calibration on the same folds            #112        |  |                |
| 10 Testing scored once + report             #112        |  |                |
| 11 B10 shortcut audit + guarantees          #168        |  |                |
| 12 artifact bundle -> Release               #114        |  |                |
+---------------------------------------------------------+  +----------------+
```

### 3.4 Why the folds come after B1 and B2

The rule "split before anything else" means before any step that learns from data, not before the cleaning.

- `StratifiedGroupKFold` needs the `group_id` that B1 creates.
- It splits the Training set that B2 defines.

Steps that do not learn (B3, B5, B6, B8) give the same output for an image whatever the split, so their position relative to the folds does not matter.

---

## 4. Blocks

Each block has one purpose, a defined input and output, and testable rules.

### B1. Deduplication

- **Input:** `Train/` and `Test/` folders. **Output:** `manifest.csv` with `path, split, label, group_id, group_size, keep, reason`.
- **Rule:** exact copies (pixel SHA-256) and near-copies (pHash distance <= 6 confirmed by thumbnail correlation >= 0.98) form one cluster, of which one image is kept. Neighbouring slices (correlation 0.95-0.98, same class) share a `group_id`. Augmented files and label conflicts are dropped. Raw images are never modified.
- **Tools:** Pillow, NumPy, SciPy. **Cost:** low, run once.
- **Evidence:** [1], [2]. **Boards:** #150 (Done). Details in `app/ml_engine/DATASET_CLEANING_REPORT.md`.

### B2. Leakage-free split

- **Input:** manifest. **Output:** Training and Testing sample lists.
- **Rule:** the official Testing folder is the held-out set. Every Training image whose `group_id` also appears in Testing is removed. Removed counts are reported per class.
- **Guarantee:** no `group_id` in both splits (unit test).
- **Tools:** Python only. **Cost:** very low.
- **Evidence:** [1] (28.8% of the official test split of this dataset has a near-twin in training), [3] (slice-level splits inflate accuracy by 29 to 55 points). **Boards:** #165.

### B3. Canonical load

- **Input:** JPG/PNG bytes. **Output:** 2D `float32` array in [0, 1].
- **Rule:** format detected by content, gray conversion, 16-bit images rescaled (not clipped), size limits, `UnsupportedImageError` for anything else.
- **Tools:** Pillow, NumPy. **Cost:** low.
- **Evidence:** convention [5]. **Boards:** #108 (Done).

### B4. Quality gate

- **Input:** 2D image. **Output:** SNR and Shannon entropy, or `QualityCheckError` with the measured values.
- **Rule:** measured on the original image, inside the brain region. SNR is the median brain intensity divided by a MAD-based noise sigma of the dark background (away from every bright region), floored at one 8-bit step; entropy uses the brain region only. Each threshold sits in the gap between synthetic unusable references (blank, noise, posterized) and the lowest value over the purged Training split, so every Training image passes. Testing images are never read during calibration. The definitions carry `metrics_version`, which serving checks.
- **Tools:** scikit-image. **Cost:** low.
- **Calibration:** `python -m src.training.thresholds --data-dir src/data --manifest <manifest.csv>`. The 2026-10-30 values (min SNR 2.74, min entropy 3.20), which reject no Training or Testing image, and their review are in `decisions.md` (D17).
- **Evidence:** no external evidence for the thresholds; they are calibrated on our data. **Boards:** #109 (gate, Done), #165 (calibration), #175 (metrics).

### B5. Brain mask

- **Input:** 2D image. **Output:** boolean mask.
- **Rule:** pixels above half the Otsu threshold, largest connected component, hole filling. **No skull stripping.** The quality gate (B4) measures on this mask. When it covers less than `min_mask_fraction` (default 0.05) of the image, the full image is used as the region downstream, without failing. Decision D18.
- **Tools:** scikit-image, SciPy. **Cost:** low.
- **Evidence:** skull stripping lowered accuracy with a patient-level split [6] and had a negligible effect in [7]. **Boards:** #166 (replaces the skull-stripping step of #109).

### B6. Geometry

- **Input:** image and mask. **Output:** 224x224 image.
- **Rule:** a square centered on the bounding box of the B5 region (after its fallback), its side the longer side of the box plus `crop_margin` (default 0.05) of that side at each end; parts outside the image are black. This is the crop to the box plus a margin, padded to a square, with the margin kept even where the head touches the border (71-87% of Training images per class). Pixels outside the mask inside the square are kept. Resize to 224 (`INTER_AREA` to shrink, `INTER_LINEAR` to enlarge). The aspect ratio of the crop is preserved. Decision D19.
- **Tools:** OpenCV, NumPy (`preprocessing/geometry.py`). **Cost:** very low.
- **Evidence:** cropping is common practice [5], [8]. The square padding fixes the class-dependent distortion measured in section 2. **Boards:** #166.

### B7. Train-only augmentation

- **Input:** a Training image after B6. **Output:** k augmented copies (default k = 1).
- **Rule:** fixed seed. Rotation +/-10 degrees, horizontal flip, scale +/-10%, brightness/contrast +/-10%. Each copy inherits the label, `group_id` and fold of its original. Copies are used only as fit rows of a fold and in the final fit; never in validation folds, Testing or inference. The feature cache key includes the transform parameters.
- **Tools:** OpenCV, NumPy (no new dependency). **Cost:** about 3 extra minutes of CPU feature extraction (1,830 images at about 90 ms).
- **Evidence:** train-only augmentation is a strong rule [4], [2]. Its gain with a frozen CNN is weak evidence [9], [5], so a +/-augmentation ablation decides whether it stays enabled. **Boards:** #167.

### B8. CNN tensor

- **Input:** 224x224 image. **Output:** `(3, 224, 224)` `float32`.
- **Rule:** gray channel replicated to 3 channels, normalized with the mean and std of the selected backbone, read from its configuration and checked against the backbone bundle. Torchvision ImageNet backbones use the ImageNet values. A timm ResNet50V2 would use 0.5 / 0.5 [11].
- **Tools:** NumPy. **Cost:** very low.
- **Evidence:** [5], [10], [11]. **Boards:** #166, #111.

### B9. Validation protocol

- **Rule:** `StratifiedGroupKFold` (5 folds, `group_id`, fixed seed) on the purged Training split. Every fitted step (scaler, PCA, classifier, calibration) is fitted on the fit rows of each fold inside one scikit-learn `Pipeline`. Classes are balanced with `class_weight="balanced"`. Selection uses mean CV macro-F1. Testing is scored once, by the calibrated model.
- **Folds:** assigned once per run from the manifest, before the quality gate, and shared by selection, ablation and calibration. An image the gate rejects keeps its fold without being a row. The assignment is written to `folds.csv` in the report folder, which #167 uses to keep each augmented copy in its original's fold (G4) and out of validation rows (G3). The calibrated model averages one calibrated pipeline per fold (D16).
- **Metrics:** accuracy, macro-F1, balanced accuracy, per-class precision and recall, confusion matrix. Expected hardest pair: glioma vs meningioma.
- **Ablations (CV macro-F1, never on Testing):** CNN-only vs radiomics-only vs hybrid, and +/-augmentation (#167).
- **Tools:** scikit-learn. **Cost:** low.
- **Evidence:** [2], [4], [9], [12]. **Boards:** #112.

### B10. Shortcut audit

- **Rule:** a metadata-only baseline (width, height, file size) is evaluated with the same grouped CV, and its balanced accuracy is written in the report. The report also gives the Testing recall of notumor split by source size (512x512 vs other) and the status of every guarantee of section 6. If the baseline reaches 0.70 balanced accuracy or more, the CLI warns and the result is presented to the team before a model is published.
- **Tools:** scikit-learn. **Cost:** very low.
- **Evidence:** [1] (file-header features with no anatomy separated tumor from no tumor at 0.959 balanced accuracy). The 0.70 level is our choice. **Boards:** #168.

---

## 5. Steps with and without state

| Type | Blocks | Fitted on |
|---|---|---|
| Without state (same output for an image whatever the split) | B3, B5, B6, B8, CNN embedding, radiomic features | Nothing |
| With state | B4 thresholds | Purged Training split |
| With state | Scaler, PCA, classifier, calibration | Fit rows of each fold |
| Training only | B1, B2, B7, B9, B10 | Not used at inference |

---

## 6. Leakage guarantees

Each guarantee is a unit test or a report entry.

| # | Guarantee | Block | Boards |
|---|---|---|---|
| G1 | No `group_id` is present in both Training and Testing | B2 | #165 |
| G2 | Quality thresholds are computed from Training images only | B4 | #165 |
| G3 | No augmented copy reaches a validation fold, Testing or inference | B7 | #167 |
| G4 | An augmented copy is always in the same fold as its original | B7 | #167 |
| G5 | B3-B6 and B8 give the same tensor at training and at inference for the same image | B3-B8 | #166, #113 |
| G6 | The output of B6 is always 224x224 and keeps the aspect ratio of the crop | B6 | #166 |
| G7 | Testing is scored exactly once per run (unit test) | B9 | #112 |

---

## 7. Excluded by cost/benefit

| Option | Why not |
|---|---|
| Skull stripping (morphological or HD-BET/SynthStrip) | Lowered accuracy with a patient-level split [6], negligible effect [7]; deep tools need GPU or large models |
| CLAHE | Contradictory evidence; harmful with a patient-level split [6] |
| N4 bias correction, histogram matching, denoising | No measurable gain [7]; images are already windowed 8-bit JPEGs |
| SMOTE | New dependency; `class_weight="balanced"` already handles the imbalance [12] |
| albumentations, torchvision transforms v2 | OpenCV covers the four transforms; preprocessing stays independent of PyTorch |
| Keras / TensorFlow | A second framework for one model |
| ResNet50V2 (timm) | Postponed: weak evidence of a gain over the torchvision backbones, and a new dependency |
| CNN fine-tuning | Out of scope of epic #103; a frozen CNN is stable with little data [10] |
| Augmentation of validation or test data | Not allowed: it leaks [4] |

---

## 8. Known limitations and open decisions

- **No patient IDs.** `group_id` is a proxy. Distant slices of one patient can still fall in different groups, so metrics can be optimistic [3].
- **B4 is not fitted per fold.** Thresholds use the whole purged Training split, so validation rows of the CV take part in a reject-only filter whose thresholds sit below every Training image. With today's values it rejects none, and Testing is not involved. Accepted to keep the pipeline simple.
- **Resolution shortcut remains.** B6 removes the aspect-ratio cue, but not the differences in source resolution and compression: the median no_tumor head is 250 px against 441-495 px for tumor classes, and 15.8% of no_tumor crops are enlarged against at most 1.1% per tumor class (D19). B10 measures what is left; only an external dataset can rule it out.
- **Feature cache invalidation.** The cache key is built from the configuration. #166 added `min_mask_fraction` and `crop_margin`; #167 must add the transform parameters so that new code never reuses stale features.
- **`skull_stripped` in the contract.** `QualityMetrics.skull_stripped` (#107) stays in the contract and is always `false` (D18). Removing it is a backend and frontend API change left for its own issue.
- **Thresholds chosen by us:** the 0.70 shortcut warning (B10), k = 1 augmented copy (B7) and the crop margin `crop_margin = 0.05` (B6) have no external source.

---

## 9. Traceability

| Block | Boards | `app/ml_engine/src` module | State |
|---|---|---|---|
| B1 | #150 | `preprocessing/deduplication/` | Done |
| B2 | #165 | `training/dataset.py` | Done |
| B3 | #108 | `preprocessing/loading.py` | Done |
| B4 | #109, #165, #175 | `preprocessing/quality.py`, `training/thresholds.py` | Done: metrics redefined, Training-only calibration applied (decision D17) |
| B5 | #166 | `preprocessing/brain_mask.py` | Done: half-Otsu mask shared with B4, full-image fallback under 5% coverage, skull stripping removed (decision D18) |
| B6 | #166 | `preprocessing/geometry.py`, `preprocessing/prepare.py` | Done: square centered on the B5 region, 5% margin, black beyond the border, resize to 224 (decision D19) |
| B7 | #167 | `training/` (new) | To do |
| B8 | #166, #111 | `preprocessing/prepare.py`, `deep_learning/` | Mostly done |
| B9 | #112 | `training/folds.py`, `training/` | Done on synthetic data: folds assigned once and shared, `folds.csv`, balanced accuracy (decision D16). Real-data run to do |
| B10 | #168 | `training/report.py` | To do |

Implementation order, from the Boards dependencies: #165 and #166, then #112 (run on the real dataset), then #167, then #168.

---

## 10. References

References follow the IEEE style. Their numbers match the evidence IDs used in Azure Boards (epic #103, issues #112 and #165-#168): [n] is Rn there. Every DOI was checked against Crossref or DataCite on 2026-10-29 (title, authors and venue).

[1] B. P. Vangala, S. Guda, L. Peddi, and N. Vangala, "Scores that hold, benchmarks that leak: Measuring dataset contamination in public brain-tumor MRI classification," arXiv:2610.00421, 2026, doi: 10.48550/arXiv.2610.00421. *Used in: B1, B2, B10.*

[2] M. Saifullah, D. Baldauf, and Y. Sakumura, "Reliable brain tumor classification without metadata: A step-by-step guideline with duplicate removal," in *Proc. IEEE 37th Int. Conf. Tools Artif. Intell. (ICTAI)*, 2025, pp. 1237-1244, doi: 10.1109/ICTAI66417.2025.00180. *Used in: B1, B7, B9.*

[3] E. Yagis *et al.*, "Effect of data leakage in brain MRI classification using 2D convolutional neural networks," *Sci. Rep.*, vol. 11, no. 1, Art. no. 22544, Nov. 2021, doi: 10.1038/s41598-021-01681-w. *Used in: B2.*

[4] P. Chakraborty, "CatBoost-stacked heterogeneous deep ensembles for explainable multi-class brain tumor MRI classification," *J. Med. Health Stud.*, vol. 7, no. 5, pp. 18-24, Mar. 2026, doi: 10.32996/jmhs.2026.7.5.4. *Used in: B7, B9.*

[5] J. Kang, Z. Ullah, and J. Gwak, "MRI-based brain tumor classification using ensemble of deep features and machine learning classifiers," *Sensors*, vol. 21, no. 6, Art. no. 2222, Mar. 2021, doi: 10.3390/s21062222. *Used in: B3, B6, B7, B8.*

[6] L. Chhetri and A. Kumar, "Optimizing deep learning for brain tumor classification: A comparative ablation study of preprocessing and augmentation strategies," in *Proc. IEEE Guwahati Subsection Conf. (GCON)*, 2026, pp. 1-6, doi: 10.1109/GCON69192.2026.11649062. *Used in: B5, exclusions.*

[7] E. Kondrateva *et al.*, "Negligible effect of brain MRI data preprocessing for tumor segmentation," *Biomed. Signal Process. Control*, vol. 96, Art. no. 106599, Oct. 2024, doi: 10.1016/j.bspc.2024.106599. *Used in: B5, exclusions.*

[8] R. D. Prayogo, N. Hamid, and H. Nambo, "Hybrid CNN-based transfer learning enhances brain tumor classification on MRI images," *IEEE Access*, vol. 13, pp. 116654-116668, 2025, doi: 10.1109/ACCESS.2025.3584376. *Used in: B6.*

[9] H. Kibriya, R. Amin, A. H. Alshehri, M. Masood, S. S. Alshamrani, and A. Alshehri, "A novel and effective brain tumor classification model using deep feature fusion and famous machine learning classifiers," *Comput. Intell. Neurosci.*, vol. 2022, Art. no. 7897669, Mar. 2022, doi: 10.1155/2022/7897669. *Used in: B7, B9.*

[10] A. H. Al-Sarairah, O. A. A. M. Al-Lasasmeh, A. Al Tawil, M. S. Alkasasbeh, and N. M. Gharaibeh, "Evaluating transfer learning for brain tumor MRI classification: Frozen VGG16 feature extraction versus end-to-end CNN training," *Egypt. J. Radiol. Nucl. Med.*, vol. 57, no. 1, Art. no. 129, Jun. 2026, doi: 10.1186/s43055-026-01782-4. *Used in: B8, exclusions.*

[11] R. Wightman, "PyTorch image models (timm)," GitHub repository, 2019, file `timm/models/resnetv2.py`. Accessed: Oct. 29, 2026. [Online]. Available: https://github.com/huggingface/pytorch-image-models, doi: 10.5281/zenodo.4414861. *Used in: B8.*

[12] S. Deepak and P. M. Ameer, "Brain tumor categorization from imbalanced MRI dataset using weighted loss and deep feature fusion," *Neurocomputing*, vol. 520, pp. 94-102, Feb. 2023, doi: 10.1016/j.neucom.2022.11.039. *Used in: B9, exclusions.*

**Quality of the evidence.** Only [1], [2] and [4] control leakage explicitly. [10] splits by slice. Several sources ([1], [4], [6], [10]) are 2026 papers with few or no citations, and [1] is a preprint. Most image-preprocessing choices therefore rest on convention, while the split and fold rules rest on strong evidence.

**Internal sources**

- `app/ml_engine/DATASET_CLEANING_REPORT.md`: cleaning method and results (B1).
- `docs/research/preprocessing-2d-mri-cnn-ml.md` and `docs/research/consensus-raw-preprocessing-2d-mri.md`: earlier literature review.
- `docs/ml-engine/decisions.md`: decision log of the implemented modules.
- Measurements of section 2, run on 2026-10-29 on the manifest `dedup_20261028T234450903626Z`.
