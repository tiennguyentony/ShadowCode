# Eye-model loss and closer scanning: design

Status: draft for approval, 2026-11-24. Branch `training/eye-model-loss`.

## Why

The `labels_generated_fresh` run (140,000 steps, commit `a05a3bd`, kept as
`training/kept/labels_generated_fresh/20261124-a05a3bd/`) fixed the corner
patch but left the mark plainly visible: a regular dot grid over the whole
tile, period 21.5 px (about 18 cycles per tile), strongest in blue-yellow,
PSNR 29.0 dB mean and 24.3 dB on the worst tenth (acceptance, 256 held-out
labels). On flat colour it looks like a QR code or halftone screen.

Tilemark's mark has to be invisible **on a printed package at arm's length**
and readable **by a phone**. Two things work against that today:

1. **The run trades visibility for reading.** The image-loss weight rises
   only while bit accuracy is at least 0.945. It sat at 0.942 the whole run,
   so the weight crept to 0.40 and PSNR was still climbing about 0.4 dB per
   10,000 steps when the run ended.
2. **Nothing in the loss knows where the eye is sensitive.** A 50 mm tile
   at 50 cm covers 5.7 degrees, so 18 cycles per tile is about 3 cycles per
   degree: near the peak of human contrast sensitivity. Plain L2 is blind to
   spacing and colour, and LPIPS (alex) is not tied to viewing distance.

Weakening the current model doesn't help (acceptance sweep,
`training/runs/acceptance_fresh/00140000/results.md`): at strength 0.20 it
reaches 32.5 dB and reads 0.80 only with 3 simulated frames averaged; at
0.10 (38.5 dB) it reads nothing. The next run has to read from roughly a
third of today's amplitude, so the mark has to sit where the eye is weak and
the camera is not.

## Goals

- The mark cannot be seen on a printed 50 mm tile at arm's length (50 cm),
  including on flat-colour labels.
- A phone at 19 to 28 cm reads it: the error correction can recover the ID
  on at least 80 percent of held-out tiles with 3 frames averaged.
- The mark stays spread (no fixed patch: fixed place under 2.5).

## Non-goals

- Reading from 28 to 42 cm. The scan range shrinks on purpose.
- Changing the 64-bit message, BCH(63,39), the 384 px tile or the 50 mm
  print size.
- Retraining the locator.
- Direct mode.
- **The phone scanning app** (decided with the user: a separate plan). It
  doesn't exist yet: `web/src` has the message codec, the camera-measure
  page and the Vite starter page. That plan covers merging and exporting
  the locator, the ONNX runtime, the camera → locator → flatten → reader
  → 3-frame averaging → decode pipeline, and the phone reading a print.

## Anti-goals (failure even if the numbers pass)

- A run whose numbers pass while the sample sheets show a pattern. The v2
  check passed on four numbers while its sheet showed the patch. Every check
  in this design is read off the sheets as well as the numbers.
- Visibility judged on screen or by PSNR only. The judge is a print at arm's
  length.
- Reading that only works in simulation. This plan checks the phone's
  colour detail against the simulation (5b) and uses correlated frames;
  the phone reading a print is the app's plan.

## Decided with the user

| Decision | Choice |
|---|---|
| Approach | Eye-model loss with a visibility target, plus closer scanning (option 3 of 3) |
| Farthest scan distance | **28 cm**: camera range 231 to 340 px per tile, the 3 px per cycle margin kept |
| When visibility and reading conflict | **Visibility wins.** The run must meet the visibility target; reading takes what is left |
| Reading pass line | **At least 0.80 decodable with up to 3 frames averaged**; the app averages the reader's output over 3 frames. The one-frame figure is reported next to it |
| The scanning app | **A separate plan.** This one ends at the trained model, `contract.json`, the acceptance numbers and the printed blind test |

## Approaches considered

1. **Settings only**: lower strength ceiling (alpha 0.12) and a minimum
   image-loss weight. No new code. Rejected: nothing moves the mark off
   3 cycles per degree, so it likely ends as a fainter but still visible grid
   at 34 to 35 dB, with weaker reading.
2. **Eye-model loss with a visibility target**, same camera range. Moves the
   mark to spacings and colours the eye ignores, within 16 to 51 cycles per
   tile.
3. **Option 2 plus closer scanning (picked).** At 28 cm the camera resolves
   up to 77 cycles per tile (13 cycles per degree at arm's length), so the
   mark has more room where the eye is weak. Costs scan distance and an app
   change. Picked so the loss has the widest band to work with.

## Design

### 1. Camera range and band

- `camera_px_per_tile: [231, 340]` (19 to 28 cm on the Galaxy S23 Ultra:
  1,314 / (231 / 5) = 28.4 cm).
- `band.max_step: 12`. `auto` then picks step 12: 24 to 76.8 cycles per
  tile, 3.0 camera px per cycle at 231 and 4.4 at 340. That makes
  `message_grid` 192 (was 128) and the reader's texture window 32 px.
- `demo.measured`: the 35 cm row (186 px per tile) is outside the new range
  and training refuses to start with it. Remove it; keep the 20 cm row.
  *My call.* A new measurement at about 25 cm is welcome but not required.
- The embedder's message layer grows from 64 x (16 x 128 x 128) to
  64 x (16 x 192 x 192), about 37.7 M weights (was 16.8 M). Checkpoints
  grow accordingly; see section 7.

### 2. The visibility measure (S-CIELAB, with masking)

A new function in `training/losses.py`, used by the loss, the spread check
and `acceptance.py`. Differentiable, runs on the GPU, on clean and marked
tiles in sRGB [0, 1].

1. sRGB to linear to CIE XYZ to the opponent space of S-CIELAB (Zhang and
   Wandell, 1996): one luminance and two colour channels (red-green,
   blue-yellow).
2. Filter each channel with S-CIELAB's spatial filters for the viewing
   geometry: **67 pixels per degree** (384 px over the 5.72 degrees a 50 mm
   tile covers at 50 cm). Filter in the frequency domain with a circular
   FFT over the tile: the print repeats the tile, so circular is exact, not
   an approximation.
3. Back to XYZ, then CIELAB against a white of the tile paper (D65, Y = 1),
   and the per-pixel colour difference ΔE between clean and marked
   (CIE76 as in S-CIELAB; the cube root gets a small epsilon so its
   gradient stays finite).
4. **Masking**: multiply per-pixel ΔE by a masking factor from the clean
   tile, 1 on flat colour and lower on text, edges and texture: the square
   root of the existing `texture_weights` normalised so flat colour is 1,
   so busy detail gets sqrt(0.25 / 1.5) = 0.41. *My call.* ΔE grows
   linearly with the mark's amplitude while today's L2 grows with its
   square; the square root keeps the allowance the same as today: about
   2.4 times the flat amplitude on detail, not 6 times. Masking may lower
   the cost on detail, never raise it on flat colour. The calibration sheet
   (section 4) can raise the floor.
5. **Pool** per tile: the mean of the masked ΔE (the loss) and its 95th
   percentile over pixels (the tile's visibility, used by the target and
   the reports).
6. **Across tiles** *(my call)*, so that flat tiles, the most visible,
   cannot hide behind an average:
   - the controller uses the 90th percentile, over the batch's tiles, of
     the per-tile visibility;
   - acceptance requires at least 95 percent of held-out tiles at or under
     the target, and reports flat-colour tiles separately.

The unit is S-CIELAB ΔE, where about 1 is the threshold of a visible
difference. That is only roughly true for print, which is why the target is
calibrated on paper (section 4).

### 3. The loss and its schedule

- **Replace L2 with the visibility measure.** The image loss is
  `w * mean_masked_dE` with `w` set by the controller below.
- **LPIPS off** (`lpips_weight: 0`). *My call.* It is not tied to viewing
  distance and would fight the move to fine, colour-heavy spacings; one
  clear objective is easier to steer and to read.
- **Peak penalty stays** (weight 24, on the residual with the texture mask,
  as in v2) against the corner patch, scaled by the 0 to 1 trigger ramp, not
  by `w`, so a large `w` doesn't multiply it.
- **Trigger unchanged**: the image loss stays off until the smoothed bit
  accuracy passes 0.95 (reading is learned first at full strength), then
  ramps in over 5,000 steps.
- **New controller, `schedule.image_loss.follow_visibility`**, replacing
  `follow_accuracy` in this config:
  - each step, smooth the batch's visibility (section 2, item 6; EMA 0.98);
  - while it is above the target, multiply `w` by `1 + rate`; while it is
    below `0.9 x target`, divide by `1 + rate`; in between, hold;
  - `w` starts at 1, is bounded to [0.01, 1000], is saved in checkpoints and
    logged like `image_scale` today.
  - Multiplicative because the weight that reaches the target is unknown
    and may be far above 1. `rate` is the builder's to pick so `w` can move
    tenfold in about 5,000 steps (about 4.6e-4).
- Bit accuracy does not enter the controller. Reading takes what is left.
- `alpha` stays 0.3 (the per-pixel ceiling).

### 4. The visibility target, calibrated on paper

The target (95th-percentile masked ΔE) is set by the user's eye, before the
pod run:

- A new script writes a printable A4 calibration sheet: tiles printed at
  50 mm, each a held-out label marked by the 140,000-step model scaled to
  a set visibility (95th-percentile ΔE of about 0.5, 1, 2, 3, 5 and 8), each
  next to its clean original, flat-colour and busy labels both, plus a few
  pure gratings (luminance and blue-yellow at about 4 and 13 cycles per
  degree) at the same levels. Pairs are shuffled left and right, answers in
  a separate file, like `blind.html`.
- Every calibration tile is scaled to its **masked** visibility, the
  number the loss and the target use, so the sheet tests the masking too.
- The user prints it at 100 percent on the printer the demo will use and
  says, at arm's length, which pairs show a difference.
- **Target = the highest level at which no flat-colour pair is seen.** If
  every level is seen, the run waits; if none is, use the highest.
- **Busy pairs** *(my call)*: if one is seen at or below the target, the
  masking is too generous. Raise the masking floor (0.41 toward 1) until
  that pair's masked level is above the target, and redo the sheet's busy
  rows to confirm.
- **Gratings** *(my call)*: if a fine blue-yellow or luminance grating
  (about 13 cycles per degree) is seen at or below the target, the eye
  model underrates that spacing, which is where the loss will push the
  mark. Lower the target to the highest grating level not seen.

### 5. What the checks report

- **Spread check** (every 2,000 steps, 256 fixed images), new columns:
  95th-percentile and mean masked ΔE; where the mark's energy sits
  (mean frequency in cycles per tile); and the share in luminance versus
  colour. These say whether the mark is moving where it should.
- **Progress rows and emails**: smoothed ΔE and `w` next to bit accuracy.
- **`acceptance.py`**:
  - new **correlated frames** mode *(my call)*: the 3 frames of one scan
    share pose, distance, tilt, light, blur and resolution, and draw fresh
    noise, JPEG and corner jitter. The existing multi-shot table draws
    every shot independently and overstates what averaging real frames
    gains (0.855 to 0.977 for the fresh run);
  - reading pass line: at least 0.80 decodable with **3 correlated frames
    averaged**; the 1-frame figure and the independent-shot table are shown
    beside it;
  - visibility pass line: 95th-percentile masked ΔE at or under the target
    on the held-out tiles;
  - fixed place under 2.5 (unchanged);
  - PSNR stays as information.
- **Final judge**: print `blind.html` (20 pairs, 50 mm) and pick the marked
  side at arm's length. **Pass at 14 of 20 or fewer right**; 15 or more is
  unlikely by chance (about 2 percent) and means it can be seen.

### 5b. The phone's fine colour detail, before the pod *(my call)*

The eye model will push the mark toward fine blue-yellow, where phone
cameras are weakest: colour-filter demosaicing, the phone's colour noise
reduction and the video stream's compression all smooth fine colour. The
simulation models only JPEG 4:2:0 chroma subsampling, so it may promise
reading the phone can't deliver.

- A printable page of gratings in luminance, red-green and blue-yellow at
  24, 38, 51 and 77 cycles per tile, at 50 mm per tile, with flat
  reference patches.
- The user photographs it with the S23 Ultra through the web app's
  measure page at about 20 and 28 cm. A "save frame" button is added there
  so the frame is the stream's own pixels, not a phone photo.
- A script measures how much of each grating survives and compares that
  with the simulation at the same camera pixels per tile.
- If the phone keeps clearly less colour detail than the simulation (under
  half), add the missing smoothing to the simulation (a chroma blur matched
  to the measurement) before the pod run.

### 6. Laptop probe, before the pod

The new band can't load the old checkpoints (different `message_grid`), but
the loss can be tested on them:

- Resume a **copy** of the 140,000-step checkpoint with the old range and
  band (154 to 340, step 8), the new loss and controller on and the target
  from section 4, for about 3,000 steps on the RTX 4070 (about 25 minutes).
- Pass: the mark's mean frequency or colour share rises, the visibility
  falls toward the target, decodable with 3 correlated frames stays above
  0.5, and the sample sheets show the grid fading, not a new patch.
- Fails → back to this design before any pod money is spent.

Also on the laptop: `smoke_test.py` gets checks for the new measure (a
3 cycle per degree luminance grating costs more than a 13 cycle per degree
one of the same amplitude; blue-yellow costs less than luminance at the
same spacing; a mark on flat colour costs more than the same mark on text;
gradient reaches the embedder) and for the controller (up above target,
down below 0.9 x target, held in between, saved and restored). A 40-step
run of the pod config on the laptop confirms it starts, prints the band
(step 12, 24 to 76.8), the range and the target, and times a step.

### 7. The pod run

- New config `training/configs/runpod_eye.yaml`, base
  `runpod_labels_generated_fresh.yaml`, run name `eye_model`, from step 0,
  140,000 steps, the same pod type ($0.69 per hour).
- Estimate: about 10 hours and $7 at the fresh run's pace; the bigger
  message layer may slow it. The 40-step laptop run gives the real figure
  before starting.
- `checkpoint.keep`: set from the measured checkpoint size so the newest
  checkpoints fit the 10 GB volume with room for the kept bundle.
- Automatic stops: learning check at 45 minutes (bit accuracy at least
  0.75; the finer band may learn slower, so the laptop's 40 steps and the
  probe are checked first); spread stop at 2.5; 24 hours of training. New:
  a **reading check** at step 80,000: stop if 1-frame bit accuracy over the
  last 1,000 steps is below 0.80, since that run cannot pass even with
  averaging.
- The sheets at 30,000, 60,000 and 100,000 are looked at, not only the
  numbers.

### 8. Contract (the app is a separate plan)

- `export_bundle.py` writes the new trained range (231 to 340) and band
  into `contract.json`'s `camera` block, as now.
- New field `reading.frames_averaged: 3`, and a line in
  `docs/model-contract.md`: the reader's logits are averaged over the last
  3 frames of one scan before decoding, and the scan distance is 19 to
  28 cm. This is what the app's plan builds against.
- The only web change in this plan is the "save frame" button on the
  measure page (section 5b).

## Success

1. Printed blind test: 14 of 20 right or fewer at arm's length.
2. `acceptance.py`: decodable at least 0.80 with 3 correlated frames, the
   visibility target met on 95 percent of tiles, fixed place under 2.5.
3. The phone check of fine colour detail (5b) agrees with the simulation,
   or the simulation was corrected before the run.

The phone reading a printed tile is checked in the app's plan.

## Budget

Laptop work is free. One pod run, about $7 to $8. If the probe fails, no pod
money is spent.

## Left to the builder

- The S-CIELAB filter constants and opponent matrix (from the paper or a
  reference implementation), and the epsilon in the cube root.
- The controller's `rate` within the bounds above.
- Exact columns, file names and email wording.
- `checkpoint.keep` from the measured size.
- The layout of the calibration and grating pages.

## Calls I made (for the user to check)

- Arm's length = 50 cm.
- S-CIELAB as the eye model, with masking from the existing texture weights.
- LPIPS off in this run.
- Peak penalty kept, on the trigger ramp only.
- The 35 cm measurement dropped from `demo.measured`.
- The blind-test pass line of 14 of 20.
- A reading check at step 80,000 (1-frame bit accuracy at least 0.80).
- Masking as the square root of the texture weights (busy detail 0.41).
- Visibility across tiles: 90th percentile for the controller, 95 percent
  of tiles for acceptance.
- Calibration rules for busy pairs and gratings.
- Correlated frames as the reading pass line.
- A phone check of fine colour detail before the pod.
