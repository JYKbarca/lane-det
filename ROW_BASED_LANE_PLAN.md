# Row-Based Lane Detection Plan

## Goal

Replace the current anchor-based lane detector with a simpler row-based formulation that is easier to train, easier to explain, and less dependent on fragile matching rules.

This direction is intended to:

- reduce dependence on `label_assigner`, `topk_per_gt`, `line_iou_width`, and anchor/NMS coupling
- align the prediction target more directly with TuSimple annotations
- provide a cleaner technical path for the thesis and follow-up development


## Why Change Direction

The current anchor-based route has these recurring problems:

- training quality is highly sensitive to matching hyperparameters
- classification, regression, and postprocess are strongly coupled
- ablations are expensive and often inconclusive
- many changes alter several mechanisms at once, making diagnosis hard

The project can run on the current route, but it is showing signs of a practical ceiling for further iteration.


## Core Idea

Use a row-based lane representation:

- keep the current backbone and FPN
- remove dense anchor generation and anchor assignment
- directly predict lane coordinates on fixed `y` rows

For each candidate lane, predict:

- whether the lane exists
- the `x` coordinate at each of the fixed `y` sample positions

This matches TuSimple naturally because TuSimple annotations already provide lane `x` values on fixed `h_samples`.


## Recommended First Version

Predict at most `K` lanes per image.

Suggested initial setting:

- `K = 5`
- `num_y = 56`

Model outputs:

- `exist_logits`: shape `[B, K]`
- `x_coords`: shape `[B, K, 56]`

Optional later extension:

- `valid_mask` or `start/end row` prediction for each lane

Do not add this in the first version unless necessary.


## Model Architecture

Reuse:

- backbone: `ResNet18`
- neck: `LaneFPN`

Replace the current anchor head with a new row-based head.

Suggested high-level structure:

1. extract image features with backbone + FPN
2. apply a compact head to produce a global lane embedding
3. branch into:
   - lane existence prediction
   - per-row coordinate regression

Possible simple implementation:

- global pooling over spatial feature map
- MLP or Conv head that outputs:
  - `K` existence logits
  - `K * 56` coordinates

This first version should prioritize simplicity over novelty.


## Label Construction

For each image:

1. parse all GT lanes
2. sort lanes from left to right using their lower visible region
3. keep up to `K` lanes
4. if fewer than `K`, pad with empty lanes

Targets:

- for real lanes:
  - `exist = 1`
  - `x_coords = lane x values on 56 y rows`
- for padded lanes:
  - `exist = 0`
  - coordinates ignored in loss

Important:

- no anchor matching
- no positive/negative assignment
- no top-k fallback logic


## Loss Design

Use the simplest stable losses first.

Recommended:

- existence loss:
  - `BCEWithLogitsLoss` or focal loss
- coordinate loss:
  - `SmoothL1Loss`
  - compute only on valid GT points of existing lanes

Total loss:

```text
loss = exist_weight * exist_loss + coord_weight * coord_loss
```

Suggested first weights:

- `exist_weight = 1.0`
- `coord_weight = 1.0`

Do not add rank loss in the first version.

Do not add extra geometric regularizers in the first version unless training is unstable.


## Inference

Inference should be much simpler than the current pipeline.

Steps:

1. run model
2. apply sigmoid to `exist_logits`
3. keep lanes above a confidence threshold
4. convert the predicted row coordinates to TuSimple output format

Initial recommendation:

- no anchor decoder
- no anchor NMS
- only optional lightweight duplicate filtering if necessary


## TuSimple Compatibility

This route is highly compatible with TuSimple because:

- TuSimple annotations already use fixed `h_samples`
- the new representation directly predicts `x` on fixed rows
- conversion to evaluation JSON is straightforward

This is one of the main reasons this route is recommended as the replacement path.


## CULane Compatibility

This route can be extended to CULane, but not as naturally as TuSimple.

Required adaptation:

- resample CULane polyline annotations onto fixed `y` rows
- produce row-based lane targets similar to TuSimple
- sort lanes left to right before assigning them into `K` slots

Implications:

- feasible for later extension
- not necessarily the best long-term architecture for CULane
- acceptable as a thesis path if TuSimple is primary and CULane is a follow-up extension


## Proposed File-Level Changes

Likely new files:

- `lane_det/models/row_lane_head.py`
- `lane_det/models/row_lane_detector.py` or adapt current detector to select head type
- `lane_det/datasets/row_target_builder.py`
- `tools/train_row.py` or modify current training entry carefully

Likely reused files:

- `lane_det/models/backbone.py`
- `lane_det/models/fpn.py`
- `lane_det/datasets/tusimple.py`
- `lane_det/metrics/tusimple_converter.py`
- `tools/evaluate.py`

Likely no longer needed in the first row-based path:

- `lane_det/anchors/*`
- current anchor-based decoder logic
- most assignment-related hyperparameters


## Recommended Development Order

1. implement the new row-based head
2. build row-based targets from TuSimple labels
3. wire a minimal training path
4. make sure loss decreases and inference runs end-to-end
5. export predictions to TuSimple format
6. evaluate
7. only then consider:
   - valid mask prediction
   - better lane ordering
   - CULane adaptation


## Thesis Framing

This path is easier to explain in a thesis than the current anchor-heavy route.

Suggested framing:

1. the anchor-based route was implemented first
2. it exposed high sensitivity to matching rules and strong coupling between modules
3. to improve interpretability and training stability, the lane representation was redesigned into a row-based coordinate regression form
4. the new formulation aligns directly with TuSimple annotation format and reduces system complexity


## Decision Summary

Recommended B route:

- keep `ResNet18 + FPN`
- discard dense anchor assignment as the main path
- predict lane existence and row-wise coordinates directly
- optimize for a clean, stable, explainable system rather than another round of fragile anchor tuning


## Next Session Starting Point

If continuing in a new session, start with:

1. create `row_lane_head.py`
2. define output tensors `[B, K]` and `[B, K, 56]`
3. build row-based targets from TuSimple labels
4. run a smoke test before full training
