---
inclusion: fileMatch
fileMatchPattern: "web/src/reader/**"
---

# Model Bundle

Rules for the tile reader in `web/src/reader/` (spec `tile-reader`).

## What the reader code reads

- The model contract is `docs/model-contract.md`. Read it before changing reader code, in particular "The path of one frame", "What the reader is given" (the flatten step and its JavaScript code) and "ONNX Runtime Web".
- The model bundle is `20261126-ba0ab4d`. Its source is `training/export/20261126-ba0ab4d/` (`reader.onnx`, `contract.json`, `fixtures/`). The app serves a copy at `web/public/models/20261126-ba0ab4d/`.
- Read the fixed numbers from `contract.json`. Do not type them into code:
  - `TILE_PX` (384)
  - `MESSAGE_BITS` (64)
  - `camera.trained_px_per_tile` (231 to 340 camera pixels per 50 mm tile)
  - `reading.frames_averaged` (3 frames)
- Show `bundle_id` in a debug view.
- `reader_input` is `flattened`: the reader gets one tile flattened to 384 x 384, with the corners in the order top-left, top-right, bottom-right, bottom-left as printed.

## Test data

- Fixtures with known answers are in `training/export/demo-fixtures/`:
  - `synthetic/<product_id>/`: clean renders, made now.
  - `photos/<product_id>/`: real phone photos, added after the paper test.
- Each product folder has `label.json`, `label.png` and, per frame, `NN_frame.png`, `NN.json` and one flattened patch per whole tile of the label (`NN_patch_r0c0.png`, `NN_patch_r1c0.png`). Both whole tiles are read and their logits averaged.
- The synthetic set is the acceptance set until the photo set exists. After that, both sets must pass.
- `training/export/` is git-ignored. It exists only in the main checkout `D:\Coding\ShadowCode`. In another worktree, point the tests at it with the environment variable `TILEMARK_DEMO_FIXTURES` (and `TILEMARK_BUNDLE_SRC` for the bundle). Do not copy the files.

## Rules

- Never create, edit or delete anything under `/training`. Read the bundle and the fixtures only.
- Copy the bundle into `web/public/models/20261126-ba0ab4d/` unchanged. Never edit a bundle file. A test checks that `reader.onnx` has the sha256 in `contract.json` (`models.reader.sha256`).
- A new bundle gets a new folder with its own `bundle_id`. Never overwrite an old bundle folder.
- Codec isolation (`.kiro/steering/structure.md`) applies to the pure parts of the reader too. Tag detection, homography, flatten and logit averaging take plain `{ width, height, data }` pixels and have no camera or DOM imports, so they run in Vitest under Node.
