# Tasks

Run tests from `app/ml_engine` with `pytest src/ -v` (or the `ml-engine-test` Docker image, since the host Python has no OpenCV / scikit-image). Work on `AB-166-brain-mask` (same Azure issue #166, not yet merged into `develop`) or a branch stacked on it; commits use `AB-166:`.

## 1. Geometry module (B6)

- [x] 1.1 Create `preprocessing/geometry.py` with `square_box(region, margin)` per design D1-D2 (bounding box of the region, side `longer + 2 * floor(margin * longer + 0.5)`, centered on the box). Verify with new `tests/test_geometry.py`: a 40x80 box with margin 0.05 gives side 88 centered on the box; a full-image region of a 300x400 image gives a square centered on the image; margin 0 gives the box's longer side
- [x] 1.2 Add `crop_square(image, box)` (zeros outside the image, loaded values inside, including pixels outside the region). Verify in `test_geometry.py`: a box crossing the top-left corner gives black rows/columns exactly where the image ends; a box fully inside equals the slice of the image; output dtype is `float32`
- [x] 1.3 Add `resize_square(square, size)` (`INTER_AREA` when the side is larger than `size`, else `INTER_LINEAR`). Verify in `test_geometry.py`: 512, 300 and 88 px squares all give `size x size`; a bright rectangle twice as wide as tall keeps that ratio within one pixel after crop and resize (G6)

## 2. Wire B6 into preparation

- [x] 2.1 Add `crop_margin: float = 0.05` to `PreprocessingConfig` (docstring: fraction of the box's longer side) and build `cnn_input` in `prepare_image` from `square_box` / `crop_square` / `resize_square` on the loaded image and the post-fallback `mask`, leaving only channels and mean/std in `_to_cnn_input`. Update the `PreprocessedImage` / `prepare_image` docstrings. Verify with `test_prepare.py`: 512x512, 300x400 and 630x200 inputs all give `(3, 224, 224)`; a small head on a large black frame gives a CNN input whose black border is about 5% per side; the training and inference paths give identical `cnn_input`; `radiomics_image` still equals the input; adjust `test_cnn_input_is_normalized_per_channel` to compare with the cropped square instead of the whole image
- [x] 2.2 Verify `crop_margin` reaches the cache key and the manifest: in `test_training_features.py` two configs differing only in `crop_margin` have different fingerprints; in `test_pipeline_artifact.py` a manifest without `crop_margin` is refused naming it, a manifest with `crop_margin: 0.1` loads with that value, and `test_training_run.py` finds `crop_margin` in the written manifest. Run `pytest src/tests/test_prepare.py src/tests/test_geometry.py src/tests/test_training_features.py src/tests/test_pipeline_artifact.py src/tests/test_training_run.py src/tests/test_inference_pipeline.py src/tests/test_deep_extractor.py -v`
- [x] 2.3 Verify `git diff --stat` shows no change to `preprocessing/cropping.py`, `classical/roi.py` or `preprocessing/deduplication/` (design D5)

## 3. Real-data check and docs

- [x] 3.1 With the real code, on the purged Training split (manifest `src/data/processed/dedup_20261028T234450903626Z/manifest.csv`), run an ad-hoc, uncommitted measurement per class: every `cnn_input` is `(3, 224, 224)`, share of squares crossing the image border, head longer side in source pixels (p5 / median / p95), and count of B5 fallbacks. Compare with the planning table of design.md; verify no image raises
- [x] 3.2 Add decision D19 (AB-166) to `docs/ml-engine/decisions.md`: square centered on the mask box with black beyond the border and why (design D2 with the border-crossing numbers), margin 0.05 as a fraction (our choice, design D3), loaded image cropped unmasked after the B5 fallback, B1/radiomics croppers untouched, and the residual resolution difference handed to B10. Verify the entry contains each item and its table
- [x] 3.3 Update `docs/ml-engine/preprocessing-pipeline.md` and `preprocessing-pipeline.tex` together: B6 rule (centered square, `crop_margin` 0.05 of the box's longer side, black beyond the border, interpolation), G6 unchanged, §8 "Feature cache invalidation" and "Thresholds chosen by us" mention `crop_margin`, §8 "Resolution shortcut" cites the D19 head-size numbers, traceability row B6 → `preprocessing/geometry.py`, Done (D19). Verify `grep -n "B6" docs/ml-engine/preprocessing-pipeline.md` shows no "To do"

## 4. Integration

- [x] 4.1 Run `pytest src/ -v` in `app/ml_engine` and `python -m pytest tests/ -v` in `app/backend`. Verify both pass with no skipped test caused by this change

## Workflow follow-up

- Archive `brain-mask` (and `quality-gate-metrics` before it) before archiving this change.
- Retrain and publish the artifact (#114) only after B6 is merged; any earlier local model is refused by the manifest check.
- Grad-CAM overlays can recompute `square_box` to map heatmaps back to the original image.
