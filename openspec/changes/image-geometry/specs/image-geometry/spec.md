# Spec Delta

## Purpose

Defines how the square CNN image is built from a loaded 2D MRI image and its brain region of interest, so the CNN sees the head at a fixed size without a distortion that depends on the source image shape, with the same result at training and inference.

## ADDED Requirements

### Requirement: The CNN image is a square crop centered on the region of interest
The CNN image SHALL be taken from a square centered on the bounding box of the region of interest, whose side is the longer side of the box widened at each end by the configured margin (a fraction of that longer side). Pixels of the square inside the image SHALL keep their loaded values, including those outside the region.

#### Scenario: Head on a large black background
- **WHEN** the image is a small head centered on a large black frame
- **THEN** the crop covers the head plus the margin and none of the frame beyond it

#### Scenario: Pixels outside the mask inside the square
- **WHEN** the square contains image pixels that are not in the region of interest
- **THEN** those pixels keep their loaded values

#### Scenario: Wide region
- **WHEN** the bounding box is wider than it is tall
- **THEN** the square side is set by the width and the box is centered vertically in it

### Requirement: Parts of the square outside the image are black
Any part of the square that lies outside the loaded image SHALL have zero intensity, so the margin is the same whether or not the head touches an image border. Building the square SHALL never raise for a loaded image.

#### Scenario: Head touching the border
- **WHEN** the region of interest touches an image border
- **THEN** the margin beyond that border is black and no error is raised

#### Scenario: Full-image region
- **WHEN** the region of interest is the full image of a non-square source (the brain-mask fallback, or a head filling the frame)
- **THEN** the square is centered on the image and its parts beyond the image are black

### Requirement: The CNN image has a fixed size and keeps the crop aspect ratio
The square crop SHALL be resized to the configured CNN size, so every CNN image has the same square size whatever the source shape, and the content of the crop SHALL keep its aspect ratio.

#### Scenario: Non-square source images
- **WHEN** images of 512x512, 300x400 and 630x200 pixels are prepared
- **THEN** every CNN input has the configured square size

#### Scenario: Aspect ratio preserved
- **WHEN** an image contains a bright rectangle twice as wide as it is tall
- **THEN** in the CNN input the rectangle is still twice as wide as it is tall, within one pixel

#### Scenario: Small source image
- **WHEN** the square crop is smaller than the CNN size
- **THEN** it is enlarged to the CNN size

### Requirement: The crop margin is part of the preprocessing settings
The crop margin SHALL be a preprocessing setting stored with the trained artifact and included in the feature cache key, so a model is always served with the margin it was trained with.

#### Scenario: Different margin
- **WHEN** features are requested with a margin different from the cached one
- **THEN** cached features are not reused

#### Scenario: Artifact without a margin
- **WHEN** an artifact manifest has no crop margin setting
- **THEN** loading it fails with a model-unavailable error that names the missing setting

#### Scenario: Value served from the artifact
- **WHEN** an artifact trained with a given margin is loaded
- **THEN** inference uses that same margin

### Requirement: Geometry does not change radiomics
The geometry step SHALL apply only to the CNN image. Radiomic features SHALL still be extracted from the loaded image at its original resolution inside the region of interest.

#### Scenario: Radiomics inputs after geometry
- **WHEN** an image is prepared
- **THEN** the radiomics image equals the loaded image and the region of interest has its shape

### Requirement: Same CNN image at training and inference
The CNN image SHALL be computed by the same code at training and at inference and SHALL depend only on the image and the preprocessing settings.

#### Scenario: Repeated preparation
- **WHEN** the same image bytes are prepared through the training path and through the inference path with the same settings
- **THEN** both CNN inputs are identical
