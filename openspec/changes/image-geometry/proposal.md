# Proposal

## Why

Block B6 of the preprocessing pipeline (`docs/ml-engine/preprocessing-pipeline.md`, Azure #166) is still "To do": `prepare_image` resizes the whole loaded image straight to 224x224. 72.8% of no_tumor images are not square against 1.7-13.4% of each tumor class, and 92.5% of tumor images are exactly 512x512 against 1.25% of no_tumor, so the plain resize distorts almost only one class and the CNN can learn the distortion instead of the anatomy. B5 (change `brain-mask`) now gives a mask with a defined fallback, which is the input B6 needs.

## What Changes

- **Crop to the brain mask, as a square padded with black.** The CNN image is a square centered on the bounding box of the B5 region of interest, whose side is the longer side of the box plus a margin on each end. Any part of the square outside the loaded image is black, so the margin is the same for every head even when it touches the border (69-97% of Training images per class, measured while planning). Pixels inside the square are kept as loaded (nothing outside the mask is blanked). When B5 fell back to the full image, the square is centered on the full image.
- **Resize to the CNN size.** The square crop is resized to `cnn_size` (224), with area interpolation when shrinking and linear when enlarging. B8 (3 channels, backbone mean/std) is unchanged and receives this image.
- **Crop margin is a preprocessing setting.** `PreprocessingConfig.crop_margin`, a fraction of the longer side of the bounding box, reaches the feature cache key and the artifact manifest. Its default (0.05) is our choice and is checked against the purged Training split.
- **BREAKING (artifact manifest and feature cache):** the CNN input of every image changes and the manifest gains `crop_margin`. Older manifests are refused by the strict preprocessing check added in `brain-mask`; old cache entries are not reused. No artifact has been published (#114 pending).
- **Radiomics unchanged.** Radiomic features stay on the loaded image at original resolution, inside the region of interest.
- **Docs.** B6 and guarantee G6 in the pipeline doc (and its `.tex`) state the margin and the module; a decision entry in `docs/ml-engine/decisions.md` records the margin check and what B6 does to the class-dependent geometry.

## Capabilities

### New Capabilities
- `image-geometry`: how the CNN image is built from the loaded image and its region of interest (square crop with margin and black padding, resize), its invariants (fixed size, aspect ratio kept) and its identity at training and inference.

### Modified Capabilities
- None. The `brain-mask` capability (in the unarchived change `brain-mask`) already states that the geometry step uses its region; its requirements do not change.

## Impact

- **Code** (`app/ml_engine/src/`): new `preprocessing/geometry.py`; `preprocessing/prepare.py` (`crop_margin` setting, CNN image built from the geometry step). `preprocessing/cropping.py` (B1 deduplication) is not touched, so deduplication output does not change.
- **Tests**: new `test_geometry.py`; `test_prepare.py`, `test_training_features.py` (cache key), `test_pipeline_artifact.py` (manifest field).
- **Retraining**: any model trained before this change must be retrained; feature caches are rebuilt.
- **Backend / frontend**: no change. `PredictionResult` is unchanged.
- **Dependencies**: none new (OpenCV, NumPy).
- **Out of scope**: B7 augmentation (#167), B10 shortcut audit (#168), Grad-CAM overlays mapped back to the original image, changing B8 normalization or the backbone, changing the B5 mask.
