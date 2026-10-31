# Design

## Context

Current code (`app/ml_engine/src/`), observed on 2026-10-30 on branch `AB-166-brain-mask`:

- `preprocessing/prepare.py`: `prepare_image` computes `foreground_mask`, runs the quality gate on it, applies `region_of_interest` (full image under `min_mask_fraction`) for the returned `mask`, and builds `cnn_input` with `_to_cnn_input(gray, config)`: a plain `cv2.resize` of the whole image to `cnn_size` (`INTER_AREA` when `min(image.shape) > size`, else `INTER_LINEAR`), gray replicated to 3 channels, normalized with `cnn_mean` / `cnn_std` (B8).
- `PreprocessingConfig` (`min_mask_fraction`, `cnn_size`, `cnn_mean`, `cnn_std`, `thresholds`) reaches the feature cache key through `asdict` (`training/features.config_fingerprint`) and the artifact manifest, where `pipelines/artifact._preprocessing` refuses missing or unexpected keys (D18).
- `deep_learning/extractor._check_preprocessing` checks `cnn_size`, mean and std against the backbone bundle. `deep_learning/explainability/` is empty: nothing maps CNN coordinates back to the image yet.
- `preprocessing/cropping.crop_to_foreground` (threshold 10 on uint8, no margin) belongs to B1: deduplication hashes cropped thumbnails with it. `classical/roi.crop_to_mask` crops radiomics windows and returns `None` under a pixel count.

Measurement made while planning (purged Training split, 1,830 images, manifest `dedup_20261028T234450903626Z`, real `foreground_mask` and `region_of_interest(0.05)` in the `ml-engine-test` Docker image; ad-hoc script, not committed):

| Class | n | Source not square | Box w/h p5 / median / p95 | Box longer side px p5 / median / p95 | Box area / frame (median) | Margin 0.05 crosses the border | Margin 0.10 crosses |
|---|---|---|---|---|---|---|---|
| glioma | 850 | 0.0% | 0.70 / 0.85 / 0.99 | 360 / 444 / 495 | 61.2% | 70.9% | 77.6% |
| meningioma | 446 | 5.4% | 0.72 / 0.87 / 1.03 | 300 / 441 / 493 | 63.8% | 76.5% | 81.8% |
| no_tumor | 260 | 69.6% | 0.73 / 0.85 / 1.07 | 183 / 250 / 708 | 77.5% | 68.5% | 85.4% |
| pituitary | 274 | 4.7% | 0.76 / 0.90 / 1.03 | 384 / 495 / 512 | 81.5% | 87.2% | 97.1% |

Read: once cropped to the mask, the head's aspect ratio is about the same in every class (median 0.85-0.90), while the source frame shape is not. The head's size in source pixels still differs (no_tumor median 250 px against 441-495 px), so after resizing no_tumor is mostly enlarged and tumor images shrunk.

## Goals / Non-Goals

**Goals:**
- A CNN image whose framing (head centered, same relative margin, same scale for the head's longer side) does not depend on the source frame.
- One pure function per step, so G6 (224x224, aspect ratio kept) is a unit test.

**Non-Goals:**
- Removing the resolution difference between classes (B10 measures it; see Risks).
- Returning the crop box in `PreprocessedImage` (no consumer until Grad-CAM overlays exist).
- Refactoring B1 `cropping.py` or `classical/roi.py`.

## Decisions

### D1. New module `preprocessing/geometry.py`, B8 stays in `prepare.py`

`geometry.py` holds the B6 steps as small pure functions on a 2D float image:
- `square_box(region, margin) -> (top, left, side)`: bounding box of the region, square side `longer + 2 * floor(margin * longer + 0.5)` (an integer margin at each end, rounded half up, which avoids float rounding such as `80 * 1.1 = 88.00000000000001`), centered on the box center.
- `crop_square(image, box) -> np.ndarray`: the square, zero where it lies outside the image (`np.pad` of the needed amount, then a slice).
- `resize_square(square, size) -> np.ndarray`: `INTER_AREA` when `side > size`, else `INTER_LINEAR`.

`prepare.py` composes them before the B8 normalization (`_to_cnn_input` keeps only channels and mean/std). This matches the CLAUDE.md "one purpose per function" rule and keeps OpenCV calls in two places.

*Alternatives:* all in `prepare.py` (the pipeline doc names it; it would mix B5 orchestration, B6 and B8 in one file; the traceability row is updated to `geometry.py` instead); one function doing all three (G6 and the border case would only be testable end to end).

### D2. Square centered on the box, black beyond the image, instead of crop-clip-then-pad

The pipeline doc says "crop to the bounding box plus a margin, pad to a square with black". Read literally (clip the margin at the image border, then pad the short side), the margin would be cut in 69-97% of Training images per class (table above), so the head's position and scale in the 224 image would depend on where it sits in the frame. Building the square around the box center and filling what falls outside the image with black gives the same result as the literal reading when nothing is clipped, and a uniform margin and scale when something is. The head's longer side always maps to `size / (1 + 2 * margin)` pixels (about 204 of 224 at 0.05).

*Alternatives:* literal crop-clip-pad (position-dependent margin, as measured); crop without margin (the half-Otsu mask can miss the darkest brain edge, D18, so tissue at the box edge could touch the frame).

### D3. Margin as a fraction of the box's longer side, default 0.05, in `PreprocessingConfig`

`PreprocessingConfig.crop_margin: float = 0.05`. A fraction, not pixels, because the head's size in source pixels differs by class (183-708 px p5-p95 for no_tumor, 300-512 px for tumors): a pixel margin would be a relative size cue of its own. 0.05 has no external source (pipeline doc §8); it leaves a thin black border (about 10 px of 224). Being a config field, it reaches the cache key and the manifest with no extra code, and the strict manifest check refuses artifacts without it.

*Alternatives:* 0.10 (more background for no gain we can measure); a constant (the cache would not notice a change).

### D4. Crop the loaded image as is, using the B5 region after its fallback

The square is cut from the loaded image, not from a masked copy: B5 removes nothing (no skull stripping, D18) and the half-Otsu mask can miss the dark brain edge. It uses `region_of_interest` (after the fallback), so a degenerate mask gives a square centered on the full image, which still removes the aspect distortion of a non-square frame. Radiomics keeps the loaded image and the same region.

*Alternatives:* blanking pixels outside the mask (a form of the skull stripping the evidence rejects); using the raw mask (empty for a black frame, so no box).

### D5. Do not reuse `crop_to_foreground` or `crop_to_mask`

`crop_to_foreground` thresholds raw uint8 values and feeds B1 hashes: changing it would change the deduplication manifest. `crop_to_mask` is radiomics-specific (returns `None` under a pixel count, crops the mask too). The two lines that find a bounding box are repeated in `square_box` rather than adding a shared helper across three unrelated blocks.

### D6. Record the check as decision D19

After implementation, re-run the planning measurement with the real code: the CNN image is 224x224 for every Training image, the share of images whose square crosses the border, and the head's source size per class. D19 records D2, the 0.05 margin with the table, and the residual resolution difference as input for B10.

## Risks / Trade-offs

- [Resolution cue remains: no_tumor heads are about half the source size of tumor heads, so they are enlarged (blurrier) while tumor heads are shrunk] → out of scope; pipeline doc §8 already lists it; D19 hands the numbers to B10 (#168), which measures a metadata baseline and notumor recall by source size.
- [Black margin beyond the border differs from the dark-gray background of some scans] → the B5 mask treats both as background; the margin is thin (about 5% each side). B10 and the ablations would show a learned edge cue.
- [Every cached feature and any local model become stale] → `crop_margin` in the config changes the cache fingerprint; the strict manifest check refuses old artifacts with `ModelUnavailableError`. No artifact is published.
- [Grad-CAM overlays (later) need the crop box to map heatmaps back] → `square_box` is a pure function of the image and settings, so the overlay code can recompute it; no field is added now.

## Migration Plan

Merge with or after `brain-mask` (same issue #166). Delete local feature caches or let the new fingerprint skip them; retrain before publishing any artifact (#114). Rollback is reverting the commits; caches keyed by the old fingerprint are reused again.
