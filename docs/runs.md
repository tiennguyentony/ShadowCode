# Training runs

What the training runs are set up for, what was checked before them, and what
has been run. Kept by whoever works in `/training`. Newest settings first.

## Status, 2026-11-21

| Run | State |
|---|---|
| Generated-label run (RunPod) | **Stopped itself** at step 38,000 of 140,000 on 2026-11-23 02:53 UTC, exit code 4: the fixed-place measure rose 1.10 (step 30,000), 1.60 (32,000), 3.20 (34,000), 3.54 (36,000), 4.20 (38,000), above the limit of 4 (`spread.stop_above`); 27.4 percent of the mark's energy in one 96 px square. Bit accuracy flat at about 0.948 from step 34,000; PSNR 23.7 to 24.8 dB and the mark still clearly visible; 54 percent of it on the busier half; image-loss weight 0.295 to 0.335. `configs/runpod_labels_generated.yaml`, commit `6cc7062`, one RTX PRO 4000 Blackwell at $0.69 per hour: **2.37 GPU hours, $1.63**. Kept: `20261123-6cc7062` (step 38,000, checksums equal to the pod's). Checkpoints 30,000 and 32,000 were deleted by the run itself when 36,000 and 38,000 were saved (`checkpoint.keep: 3`); 34,000 and 36,000 and the TensorBoard log were copied to the laptop on 2026-11-22 (`training/runs/pod/labels_generated/`; both checkpoints reproduce the pod's spread rows exactly) and are also on the pod volume. Pod stopped, not terminated. Diagnosis and the fix: "Generated-label pod run: the stop and the fix". The fix's 2,000-step check (34,000 to 36,000) ran on 2026-11-23 and failed on fixed place (2.09, line under 2); v2 (`configs/runpod_labels_generated_v2.yaml`) resumes from its step 36,000: "The fix's 2,000-step check, and v2". |
| Locator proof run (stage 2, laptop) | **Done**, 20 minutes on 2026-11-22, step 0 to 4,962, `configs/locator.yaml`. It learns: held-out mean corner error 130.11 reader px at step 0, 37.42 at the end (worst 5 percent 238.17 to 145.69). See "Locator (stage 2)". The full run is not started yet. |
| Generated-label sanity run (laptop) | **Done**, one hour on 2026-11-21/22, step 0 to 6,056, on `labelgen.py` tiles (`configs/labels_generated.yaml`). It learns: bit accuracy 0.863 over the last 1,000 steps and still rising, final eval 0.873. The image loss never switched on (needs 0.95), so PSNR is 12.5 dB. Acceptance fails, as expected after one hour. See "Generated labels". No pod run. |
| Label run (laptop) | **Done**, two hours on 2026-11-21, from step 0 to 15,087, on Open Food Facts label photos with the image-loss weight following bit accuracy. Bit accuracy held at 0.948; the image-loss weight stopped at 0.22 and PSNR at 24 dB. **Acceptance failed** on reading (0.53 of held-out tiles within reach of the error correction, line 0.80) and the mark is visible. No fixed patch (1.36). See "Label run". |
| Penalty-only run (laptop) | **Resumed** on 2026-11-21 from its step 10,000 checkpoint. It had stopped itself there on a limit that could not tell the corner patch from a mark that follows the image. The limit is now on the fixed-place measure (4.0, on 256 images); a mark that follows the image is accepted. See "Penalty-only run". |
| Long run (main reader, flattened, laptop) | **Cancelled** on 2026-11-21 at about step 26,500 of 140,000. It read well (bit accuracy 0.95 to 0.96) but had put the mark into one bright patch at the tile corner. Last checkpoint: step 26,000, kept as the fallback. See "Long run" and "Corner patch". |
| Spread run (RunPod) | **Being started** on 2026-11-21, alongside the penalty-only run, from commit `bf9c7a0`. `configs/runpod_spread.yaml`: the long run's settings with the two options against the corner patch on, from step 0. The options were checked on the CPU only. See "Spread run (RunPod)". |
| Sanity run | **Done**, one hour on the laptop GPU with the settings below. It learns; results in "Sanity run". No run has used an older camera range. |
| Direct reader | **Dropped for the gate.** Every smoke test with rotation ends at chance; no pod run was made. See "Direct mode". |
| Smoke tests | Run on 2026-11-21 with the new settings; results below. |

## Generated-label pod run: the stop and the fix

`configs/runpod_labels_generated.yaml`, RunPod, one RTX PRO 4000 Blackwell at
$0.69 per hour, 2026-11-23 00:31 to 02:53 UTC, step 0 to 38,000 (16,040 steps
per hour), 2.37 GPU hours, $1.63. It stopped itself at step 38,000 on the
fixed-place limit (exit code 4). Kept: `training/kept/labels_generated/20261123-6cc7062/`.

Everything below is measured on the laptop CPU from the checkpoints at
34,000, 36,000 and 38,000 (each reproduces the pod's spread row to the
printed digit) and from the run's TensorBoard file, unless it says
"estimate". The fixed images are the spread check's 256 held-out labels
(`labelgen.SPREAD_SEEDS`). Results and the scripts' output:
`training/runs/pod/labels_generated/diagnosis/`.

**Where.** A patch in the lower-right corner of every tile, growing from
34,000 to 38,000 (`heatmaps_32k_34k_36k_38k.png`; the 32,000 panel is from
the 4 tiles of that step's sample sheet, as the checkpoint was deleted by
`checkpoint.keep: 3`). It is the long run's "Corner patch" again, slower.
It began near step 26,000 to 28,000: training "strongest spot" 1.29, 1.36,
1.63 for the 2,000 steps ending at 26,000, 28,000, 30,000.

**Loss terms**, training means over each 2,000 steps (s is the image-loss
weight; the run's weights are L2 2, peak penalty 4, LPIPS 1, all times s):

| Steps | s | Bit | L2 | Peak | LPIPS | s x 2 x L2 | s x 4 x Peak | s x LPIPS | Strongest spot |
|---|---|---|---|---|---|---|---|---|---|
| 28,000-30,000 | 0.309 | 0.153 | 0.0053 | 0.0087 | 0.310 | 0.0033 | 0.0108 | 0.0957 | 1.63 |
| 30,000-32,000 | 0.302 | 0.145 | 0.0048 | 0.0112 | 0.251 | 0.0029 | 0.0135 | 0.0758 | 2.46 |
| 32,000-34,000 | 0.295 | 0.146 | 0.0042 | 0.0136 | 0.193 | 0.0025 | 0.0160 | 0.0569 | 3.64 |
| 34,000-36,000 | 0.302 | 0.139 | 0.0042 | 0.0147 | 0.161 | 0.0025 | 0.0178 | 0.0485 | 4.14 |
| 36,000-38,000 | 0.323 | 0.145 | 0.0037 | 0.0141 | 0.138 | 0.0024 | 0.0183 | 0.0445 | 4.45 |

**Why.** Not the image-loss weight (0.31 to 0.32): LPIPS. At step 38,000 the
trained mark was spread evenly over each tile at the same energy (a smooth
gain that flattens its 48 px window energy). Fixed place fell from 4.21 to
1.69; LPIPS rose from 0.129 to 0.204 (+0.075); the peak penalty fell from
0.0135 to 0.0060 (-0.0075); the reader's bit accuracy fell from 0.965 to
0.919 and its bit loss rose by 0.104. So piling up saves ten times more
LPIPS than the peak penalty charges for it: the penalty outweighs LPIPS only
above a weight of 10, and the run had 4.

**Visibility.** The image-loss weight follows bit accuracy (floor 0.95, band
0.01, 1/15,000 per step): it switched on at step 16,304 and was 0.044,
0.111, 0.178, 0.235, 0.266 at steps 17,000 to 21,000, 0.27 to 0.31 from
22,000 to 36,000, 0.325 and 0.335 at 37,000 and 38,000. Its ceiling is 1.0,
but training bit accuracy sat at 0.943 to 0.948, below the floor, so it was
held near 0.3 and would have stayed there. PSNR at 140,000 steps, estimate
from a log fit of the training PSNR: 29 to 31 dB (low confidence; part of
the rise was the pile-up). On the 45 mostly flat fixed images the PSNR is
27.2 dB, on the rest 24.9 dB, but on flat labels the corner patch is plainly
visible (`step_00038000/sheet_8_labels.png`: clean | marked | residual x10,
rows 1 to 3 fully flat).

**Reading at step 38,000** (eval mode, full patch, one simulated shot at full
strength, BCH(63,39) corrects 4 of the 63 codeword bits,
`/docs/message-format.md`): 225 of 256 tiles (87.9 percent) fully decode;
86.6 percent over 9 independent shots (83.2 to 89.1); 98.0 percent with the
9 shots' logits averaged. Mean 2.04 wrong bits per tile; mostly flat tiles
84.9 percent, the rest 87.0. This reader leans on the corner patch (see Why).

**The fix**, `configs/runpod_labels_generated_fix.yaml`:

| Setting | Old | New | Why |
|---|---|---|---|
| `spread.peak_penalty.weight` | 4 | 16 | Break-even with LPIPS is 10 (above); 16 leaves a margin. It stays on the image-loss weight like L2 and LPIPS, so its ratio to LPIPS is 16 whatever that weight is. |
| `loss.texture_mask` | off (new) | on: 1.5x on flat colour, 0.25x on busy detail | The L2 loss and the peak penalty weigh each pixel by how flat the clean tile is around it, so the mark may be stronger on text and detail and must be weaker on flat colour; a patch on flat colour costs more. About 70 percent of label pixels are flat by acceptance.py's level (6e-5), the rest 1e-3 to 3e-2. |
| `spread.stop_above` | 4 | 4 | unchanged |

Not taken: a faster or higher image-loss ramp (the weight is held by bit
accuracy, so it only rises by giving up reading, and it raises LPIPS, the
cause); adding LPIPS (on already, weight 1, alex); hiding in blue-yellow
colour (the simulation's JPEG chroma subsampling halves colour resolution,
which puts the top of the band at 1.5 camera px per cycle, below what can be
resolved; it needs an embedder change and a fresh start).

The texture mask is checked in `smoke_test.py` (`check_texture_mask`): flat
colour costs 1.5, an 8 px checkerboard at most 0.252, the training step's L2
and peak terms equal the texture-weighted ones computed apart, the gradient
reaches the embedder; a mask of ones, a mask without the detail term and a
training step that ignores the mask each fail it. Step 34,000 loads into the
new config. No training step has run with the fix.

**Plan.** A 2,000-step check from step 34,000 (the least piled-up checkpoint
left), then the full run if it passes; commands and pass lines in
`training/RUNPOD.md`, "The generated-label fix". Estimates at 15,990 steps
per hour and $0.69 per hour: check 0.125 h, $0.09; the rest of the run (36,000
to 140,000) 6.5 h, $4.49; a fresh start instead 8.8 h, $6.04.

### The fix's 2,000-step check, and v2

**The check ran** on 2026-11-23, 04:07 to 04:16 UTC, on the same pod, from
the old run's step 34,000 to 36,000, run name `labels_generated_fix`, commit
`a452a15`, $0.09 of GPU time. It ended normally (exit code 0), kept
`20261123-a452a15` on the pod volume (not copied to the laptop) and stopped
the pod. Its logs and console are in
`training/runs/pod/labels_generated_fix/`. Spread rows, 256 fixed images:

| Step | Fixed place | Square share | Busier half | PSNR fixed | Bit acc (last 1,000) | Image-loss weight |
|---|---|---|---|---|---|---|
| 34,000 (start) | 3.20 | 23.4% | 49% | 23.48 dB | | 0.29 |
| 35,000 | 2.17 | 21.5% | 64% | 25.02 dB | 0.928 | 0.230 |
| 36,000 | 2.09 | 21.2% | 67% | 23.51 dB | 0.941 | 0.196 |

Against its pass lines: **fixed place 2.09, failed** (line: under 2); bit
accuracy 0.941, PSNR 23.51 against 23.48, busier half 67 percent passed.
Final eval at 36,000 (eval mode, full patch): bit accuracy 0.967, within BCH
reach 0.891.

Loss terms (training means, 34,000 to 36,000, s the image-loss weight 0.234):
bit 0.171 (0.139 in the old run over the same steps), L2 with the texture
mask 0.0021, peak 0.0066 (0.0147 before), LPIPS 0.174 (0.161). Weighted:
s x 16 x peak 0.025, s x LPIPS 0.041, s x 2 x L2 0.001.

**Reading.** The penalty and the mask did spread the mark (fixed place down
by a third, busier half up from 49 to 67 percent), but spreading cost
reading (bit loss up by a quarter), the smoothed training bit accuracy sat
under 0.94 most of the time, and the image-loss weight follows it: it fell
from 0.29 to 0.20. With less image loss the mark got stronger again and the
PSNR on the fixed images went back to where it started. The fixed-place
measure was still falling at the end, but slower (2.17, 2.09).

**Step 32,000 is gone.** The old run deleted it when it saved 38,000
(`checkpoint.keep: 3`); neither the pod nor the laptop has it. The least
piled-up checkpoint that exists is the check's step 36,000 (fixed place
2.09), on the pod only: `/workspace/runs/labels_generated_fix/checkpoints/`.
v2 resumes from it.

**v2**, `configs/runpod_labels_generated_v2.yaml` (base: the fix config;
same model, message and tile, so the fix's step 36,000 loads):

| Setting | Fix | v2 | Why |
|---|---|---|---|
| `spread.peak_penalty.weight` | 16 | 24 | Measured: 16 took fixed place from 3.20 to 2.09 in 2,000 steps, slowing. 24 is a guess at what gets it under 2. It is multiplied by the image-loss weight, so with that weight held up (next rows) the push is about 1.5 to 2 times the check's. |
| `schedule.image_loss.follow_accuracy` | floor 0.95, band 0.01 | floor 0.945, band 0.005 | The weight rises at a smoothed training bit accuracy of 0.945 or more (was 0.95) and falls only below 0.94 (unchanged). Measured: with 0.95/0.01 the old run's training bit accuracy sat between the two thresholds, 0.943 to 0.948. Estimate: it now sits at about 0.9425, still above the 0.94 pass line, and the weight climbs whenever reading allows instead of waiting for 0.95. |
| `schedule.image_loss.ramp_steps` | 15,000 | 5,000 | The weight moves by 1/5,000 per step instead of 1/15,000: up to 0.4 per 2,000 steps. It also falls three times as fast below 0.94. The smoothing of the bit accuracy (0.98, about 50 steps) is much shorter than the ramp, so it does not swing. |
| `log.sheet_every` | 1,000 | 2,000 | Sample sheet, spread check and its automatic stop every 2,000 steps, with the checkpoints. Progress emails stay at one every 10,000 steps; the end, a failure and the automatic stops are emailed at once (notify.py). |
| `checkpoint.keep` | 3 | 15 | The newest 15 checkpoints, the last 30,000 steps, about 3.6 GB at 0.243 GB each, so a run can go back further than 4,000 steps (the first run deleted 32,000). Keeping all (0) from 36,000 to 140,000 is about 12.9 GB, more than the 10 GB volume holds next to the 2.7 GB already on it. |
| `loss.texture_mask` | on | on | unchanged |
| `spread.stop_above` | 4 | 4 | unchanged |

Not changed, and why: the ceiling of the image-loss weight is already 1.0; it
never got near it, because bit accuracy holds it. So "higher" can only mean
letting it rise at a lower bit accuracy, which the 0.94 pass line limits to
the 0.005 above. A minimum weight would make the mark fade whatever reading
does; it is the next step if v2 holds reading but does not fade (RUNPOD.md).

**Smoke test, 2026-11-23, laptop RTX 4070.** `python smoke_test.py`: all
checks pass, bit accuracy 0.865 (train) / 0.851 (eval) after 300 steps.
`train.py --config configs/runpod_labels_generated_v2.yaml` resumed from a
copy of the old run's step 36,000 (the same model as the fix's 36,000) for
40 steps, output in a temporary folder: it printed peak penalty weight 24,
texture mask on, "up at or above 0.945, down below 0.94, by 1/5000 per
step", resumed at step 36,000, trained and saved step 36,040.

**The v2 check passed**, 2026-11-23, ended 10:20:59 UTC, 36,000 to 38,000,
commit `25ee1bc`, 4.20 steps per second, $0.09 of GPU time. Kept on the pod
volume: `20261123-25ee1bc` (not copied to the laptop). Spread rows, 256 fixed
images:

| Step | Fixed place | Strongest spot | Square share | Busier half | PSNR fixed | Bit acc (last 1,000) | Image-loss weight |
|---|---|---|---|---|---|---|---|
| 36,000 (start) | 2.09 | 4.18 | 21.2% | 67% | 23.51 dB | | 0.196 |
| 38,000 | **1.79** | 4.04 | 20.3% | **69%** | **24.2 dB** | **0.942** | 0.196 |

All four lines pass: fixed place under 2, bit accuracy 0.94 or more, PSNR
and busier half not falling. The notify email's PSNR column holds the
training PSNR (24.67 dB) at 38,000 and the fixed-image PSNR at 36,000; the
fixed-image comparison is 23.51 to 24.2 dB (console). Final eval at 38,000:
bit accuracy 0.968, within BCH reach 0.859 (0.891 at the fix's 36,000),
0.960 with regions blanked.

The image-loss weight did not move (0.197 at 37,000, 0.196 at 38,000): the
smoothed training bit accuracy sat between 0.94 and 0.945, where it is held.
The PSNR gain came from the penalty and the mask moving the mark, not from
more image loss. If the weight is still near 0.2 at 50,000 with PSNR flat,
the next step is the minimum image-loss weight (RUNPOD.md, v2 failure table).
The full run resumes from 38,000.

### v2 from step 0

**The check passed on its numbers but not on its sheet.** The v2 full run
resumed from 38,000 on 2026-11-24 02:07 UTC. Its sample sheets at 38,000
(the check's own) and 40,000 show the mark (third column, x10) mostly in the
lower-right part of every tile: on flat tiles almost all of it, on text
tiles a part goes onto the letters and edges (the texture mask) and the
lower-right block stays. It is the first run's patch, smaller. The check was
read from the four numbers only; the 38,000 sheet already showed it.

| Step | Fixed place | Strongest spot | Square share | Busier half | PSNR fixed | Bit acc | Image-loss weight |
|---|---|---|---|---|---|---|---|
| 38,000 | 1.79 | 4.04 | 20.3% | 69% | 24.22 dB | 0.9419 | 0.196 |
| 40,000 | 1.93 | 4.26 | 21.0% | 70% | 25.47 dB | 0.9425 | 0.1995 |

A strongest spot near 4 is what a mark spread evenly over about a quarter
of the tile gives. Fixed place went back up after the check.

**Why.** Every run since the first resumed from that run's checkpoints, and
the embedder and the reader both rely on the spot it settled on near step
26,000 to 28,000. The penalty and the mask made the patch smaller and moved
part of the mark onto detail, but on a flat tile there is no detail to move
to, and keeping the mark in one place is still cheaper in LPIPS than the
penalty charges for it (measured at 38,000 in the first run: +0.075 LPIPS to
spread it evenly). Which corner is chance: the tile repeats, so any one spot
works for the reader.

**Decision, 2026-11-24:** the v2 full run is stopped by hand and v2 starts
again from step 0 (`configs/runpod_labels_generated_fresh.yaml`, run name
`labels_generated_fresh`), so the penalty (24) and the mask are on before
any patch forms. One change against v2: `spread.stop_above` 4 -> 2.5, the
acceptance line, so a patch that forms stops the run at the next spread
check. Whether it stays spread is not known: the first run formed its patch
with penalty 4 and no mask. Look at the 30,000 sheet. Estimate: 9.3 hours,
$6.40 for 140,000 steps.

Checked on the laptop GPU: the config starts from step 0, prints peak
penalty 24, texture mask on, the 0.945/0.94 image-loss thresholds and the
spread stop at 2.5, and trains 40 steps (fixed place 1.01 to 1.02).

## Locator (stage 2)

Design: `docs/model-contract.md`, "The locator: design note" (approved
2026-11-22). Code: `locator_train.py` (started through `train.py` when
`train.stage` is 2), `models/locator.py`, `simulate_view` and the new keys
in `simulate.py`, `configs/locator.yaml`. `smoke_test.py` checks rotation,
the target corners, a print that ends inside the frame and the turn-free
loss (`check_locator_view`); with each check broken on purpose it fails.
Stage 1 with `rotation_deg: 0` draws the same homographies as before, to
the last bit (checked against the previous `simulate.py`).

**Units.** Corner error is the mean distance of the four corners, in reader
pixels (384 per tile, the unit of `flatten.corner_jitter_px` = 3), after the
best of the four turns. "Worst 5%" is the mean of the worst 13 of 256
samples.

**Held-out set.** `labelgen.ACCEPTANCE_SEEDS` (seeds 0 to 255, never drawn
in training), one fixed frame each (`locator_train.eval_seed` 20261122),
rendered at camera resolution (1080 x 1080 central square, a tile 154 to 340
pixels) and shrunk to 384 as the app does. Even seeds are marked by the
frozen embedder, odd ones are plain. The same 256 frames at every
evaluation.

**Frozen embedder:** `checkpoints/labels_generated/step_00006056.pt`, the
end of the one-hour generated-label sanity run.

### Proof run, 2026-11-22

```
python train.py --config configs/locator.yaml --keep-awake --max-minutes 20
```

RTX 4070 laptop GPU, 32 samples per step (8 tiles x 4 views, half the tiles
marked), about 4.1 steps per second, peak GPU memory 1,813 MiB. Rotation 0
to 360 degrees, tilt up to 30, print edges 0.6 to 4.0 tiles from the
centre, every effect on at full strength from step 0.

| step | minutes | mean | median | 95th pct | worst 5% | max | marked | plain | best turn not 0 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 130.11 | 131.79 | 225.63 | 238.17 | 259.71 | 130.90 | 129.32 | 214 |
| 1,000 | 4.2 | 124.40 | 126.14 | 205.55 | 230.59 | 276.85 | 128.84 | 119.96 | 190 |
| 2,000 | 8.1 | 115.47 | 117.97 | 184.58 | 195.15 | 217.23 | 116.11 | 114.83 | 196 |
| 3,000 | 12.2 | 56.11 | 44.86 | 139.53 | 154.86 | 186.32 | 54.01 | 58.21 | 198 |
| 4,000 | 16.2 | 44.09 | 30.78 | 121.77 | 158.84 | 187.83 | 42.64 | 45.54 | 202 |
| 4,962 | 20.1 | 37.42 | 26.48 | 103.77 | 145.69 | 202.98 | 38.34 | 36.49 | 195 |

Held-out corner error, reader pixels, 256 samples. Step 0 is the untrained
locator, which returns the reference square.

It learns, with rotation on. For 2,000 steps it stayed near the reference
square, then dropped. At the end it is still far from the reader's
tolerance (3 reader px): the median is 26.48. Marked and plain labels are
alike, so it is not leaning on the mark. In 195 of 256 samples the best turn
is not 0: the locator picks its own corner order, as the loss allows, and
the app's four tries are needed for most frames. No reader accuracy was
measured in this run. Sample sheets: `training/runs/locator/samples/`.
Checkpoint `checkpoints/locator/step_00004962.pt`; the full run resumes from
it with the same config.

## Generated labels

From 2026-11-21 there is a second source of training tiles:
`training/labelgen.py` draws made-up packaging labels from a seed, so no
photo dataset is needed. Fonts and their licences: `training/THIRD_PARTY.md`,
"Fonts".

**What a label is.** A background (flat, two-colour gradient, stripes, dots
or checks), 1 to 4 panels (bands, boxes, rounded boxes, circles, slanted
bands) in a palette of 2 to 5 colours (saturated, pale, dark, muted or
near-grey), a made-up title, a smaller line, fine print, a logo built from
shapes, and sometimes a barcode-like box, a ruled table like a nutrition
panel, a border, small icons and text turned 90 degrees. About 1 label in 6
is mostly flat colour, the hardest case for hiding a mark. The label's long
side is 1024 layout units.

**What a tile is.** A square of the label, its side drawn log-uniformly from
256 units to the label's short side (683 to 1024 units, as the labels are
between 0.65 and 1.5 times as wide as high), at a random place. The first
version went down to 96 units; it was raised to 256 because a tile prints at
50 mm, so a 96-unit tile would stand for a label about 53 cm long. At 256
units it stands for one of about 20 cm.

**One change from the specification, approved on 2026-11-21.** The
specification said to draw each label on a canvas of about 1024 px and cut a
random 384 px crop at a random zoom. That is not what the code does: the
layout is a list of shapes and text in layout units, and each tile is drawn
straight at its own scale, twice the tile size, then averaged 2 x 2 down
(antialiasing). Reason: a crop smaller than 384 px of a rendered canvas has
to be scaled up to fill the tile, which blurs every edge, and the
specification asked for sharp edges and no blur. Drawing at the tile's scale
keeps a close-up as sharp as a whole label. Nothing is resampled except the
2 x 2 average; there is no JPEG and no noise.

**Measured on seeds 0 to 511** (flat area by `acceptance.py`'s method: the
share of 48 px windows whose local contrast is under its flat level):

| | Smallest crop 96 units (first version) | Smallest crop 256 units (used) |
|---|---|---|
| Labels of the mostly-flat kind | 84 of 512 = 16.4 percent | 84 of 512 = 16.4 percent (the labels did not change) |
| Tiles at least 90 percent flat | 27.5 percent | **17.6 percent** (90 of 512; 67 of them from mostly-flat labels) |
| Mean flat share of a tile | 0.575 | 0.506 |
| Tiles with no flat window | 4.7 percent | 1.8 percent |

For comparison, 256 crops of the label photos with `configs/labels.yaml`'s
settings: 0.4 percent of tiles at least 90 percent flat, mean flat share
0.089. Camera noise keeps a photo from ever reading as flat by this measure,
so the two sources are not alike here: the generated tiles are clean flat
colour where the photos are noisy.

**Speed** (`python labelgen.py --bench 300 --workers 1 4 8`, laptop CPU,
i7-13620H, nothing else running):

| Processes | 96-unit crops | 256-unit crops (used) |
|---|---|---|
| 1 | 15.6 tiles/s | 14.8 tiles/s |
| 4 | 49.9 tiles/s | 44.7 tiles/s |
| 8 | 45.8 tiles/s | not measured again |

The label run's fastest logged rate was 2.13 steps per second at batch 8,
that is 17.0 tiles per second. One process falls just short, four keep up
with more than twice to spare, eight were slower than four on this laptop.
So the tiles are made while training, not saved to disk first, with 4
loader processes on the laptop. Before any pod run, the same benchmark is to
be run on the pod's CPU with 1, 4 and 8 processes and the fastest taken.

**Smoke tests, 2026-11-21** (laptop GPU, `configs/smoke.yaml`: 300 steps,
batch 4, simulation off, half-width embedder):

| Source | Bit accuracy, first 25 steps | Last 25 steps (training) | Eval mode | PSNR |
|---|---|---|---|---|
| `python smoke_test.py data.source=generated_labels` | 0.499 | 0.838 | 0.839 | 12.8 dB |
| `python smoke_test.py` (24 procedural images in a folder, as before) | 0.497 | 0.853 | 0.842 | 12.7 dB |

Both pass (line 0.70). All structure checks pass, including the new one:
the same seed gives the same tile, and training seeds (1,000,000 to
1,999,999) and held-out seeds (0 to 99,999) do not overlap. The image loss
stays at weight 0 in a smoke test, so its PSNR says nothing about how
visible the mark will be.

**Sanity run, 2026-11-21 23:17 to 2026-11-22 00:17.** Code: commit
`b689e80` on `training/labels`. `configs/labels_generated.yaml`, from step
0, on the laptop's RTX 4070:

```bash
python train.py --config configs/labels_generated.yaml --max-minutes 60 --keep-awake
```

It **learns**. Stopped on the time limit at step 6,056 after 60 minutes,
exit code 0. 1.68 steps per second, against about 2.1 in the photo label
run. The GPU was at 86 to 100 percent and the CPU at 11 to 35 percent
(sampled at step 100), so the generated tiles kept up and the GPU set the
pace; this run also has the reader's patch hiding on, which the photo run
did not. The learning check at 45 minutes (step 4,495) passed with mean bit
accuracy 0.837 (line 0.75).

| | Measured |
|---|---|
| Bit accuracy, mean of the last 1,000 training steps (steps 5,001 to 6,000) | 0.863, rising every 1,000 steps (0.648, 0.770, 0.804, 0.828, 0.845, 0.863) |
| Final eval, 8 fresh batches, simulation strength 0.30 | bit accuracy 0.873; with regions of the patch blanked as in training 0.856; all 64 bits right 0.000; within reach of the error correction 0.078 |
| PSNR, marked against original | 12.5 dB (final eval); 12.53 dB mean of the last 1,000 steps |
| Image-loss weight | 0.000 for the whole run: it only starts rising once smoothed bit accuracy holds 0.95, which this run did not reach |
| Simulation strength | ramped to 0.30 by the end (schedule) |
| Spread check at step 6,000 (held-out seeds 256 to 511) | fixed place 1.00, strongest spot 1.03, 6.4 percent of the mark's energy in one 96 px square, 41 percent on the busier half |

What the numbers mean. With the image loss still at weight 0, nothing has
asked the mark to be invisible yet: PSNR of 12.5 dB is a strong mark, and
the final sample sheet (`training/runs/labels_generated/samples/
final_step_00006056.png`, original | marked | difference x10) shows an even
texture over the whole tile, flat areas included. The spread numbers sit at
their floor (fixed place 1.00): there is no patch in one place, but at this
stage that says only that the mark covers everything, not that it hides
well.

**Acceptance on the step 6,056 checkpoint** (a copy, not the run's file),
`acceptance.py --config configs/labels_generated.yaml`, on held-out seeds 0
to 255, one shot per tile, simulation at full strength (1.0, where training
had reached 0.30):

| Alpha | At most 4 wrong of the first 63 | Bit accuracy | Fixed place | PSNR mean, dB |
|---|---|---|---|---|
| 0.30 (trained) | 0.027 FAIL | 0.817 | 1.00 PASS | 12.5 |
| 0.25 | 0.016 FAIL | 0.805 | 1.00 PASS | 14.0 |
| 0.20 | 0.008 FAIL | 0.779 | 1.00 PASS | 15.8 |
| 0.15 | 0.004 FAIL | 0.736 | 1.00 PASS | 18.2 |

Averaging 9 simulated shots raises bit accuracy at alpha 0.30 from 0.817 to
0.842 and leaves the share within reach at 0.023. This checks that
acceptance works on the generated source; a one-hour run was not expected
to pass it. Results: `training/runs/acceptance_generated/00006056/`.

Spread check on held-out seeds 256 to 511, one row per 1,000 steps (PSNR
and bit accuracy after step 0 are the training means over the 1,000 steps
before the row):

<!-- labels-generated-run:start -->
| Time | Step | Fixed place | Strongest spot | Energy in one 96 px square | Energy on the busier half | PSNR, dB | Bit accuracy |
|---|---|---|---|---|---|---|---|
| 2026-11-21 23:17 | 0 | 1.01 | 1.04 | 6.5 percent | 41 percent | 21.56 (fixed images) | n/a |
| 2026-11-21 23:27 | 1,000 | 1.00 | 1.01 | 6.3 percent | 41 percent | 13.82 | 0.6476 |
| 2026-11-21 23:37 | 2,000 | 1.00 | 1.01 | 6.3 percent | 41 percent | 12.37 | 0.7703 |
| 2026-11-21 23:46 | 3,000 | 1.00 | 1.01 | 6.3 percent | 41 percent | 12.31 | 0.8042 |
| 2026-11-21 23:56 | 4,000 | 1.00 | 1.02 | 6.3 percent | 41 percent | 12.36 | 0.8282 |
| 2026-11-22 00:06 | 5,000 | 1.00 | 1.03 | 6.4 percent | 41 percent | 12.47 | 0.8446 |
| 2026-11-22 00:16 | 6,000 | 1.00 | 1.03 | 6.4 percent | 41 percent | 12.53 | 0.8628 |
<!-- labels-generated-run:end -->

## Label run (laptop)

From 2026-11-21 the models train on product labels, not general photos.

**The dataset.** Open Food Facts front-of-pack photos, fetched by
`training/fetch_labels.py` into `training/data/open-food-facts/images`
(licence, credit and open points: `training/THIRD_PARTY.md`).

| | |
|---|---|
| Fetched | 16,000 photos, one per product, shorter side at least 600 px; 13.68 GB in 22.7 minutes (measured) |
| Left out afterwards | the 50 photos over 24 megapixels, deleted with `fetch_labels.py --leave-out-over 24`; no replacements were fetched. They are listed in `left_out.csv` in the dataset folder. |
| **Final count** | **15,950 photos**, 13.37 GB. `ATTRIBUTION.csv` has the same 15,950 rows. |
| Kept back from training | 256 of them (`data.holdout: 256`), picked by a fixed seeded shuffle; `acceptance.py` measures on these. Training uses the other 15,694. |
| Crops | from the central 60 percent of the width and of the height (`data.center_fraction: 0.6`), then as before: 35 to 100 percent of that part, resized to 384 px, random mirror. |
| Upright | the loader now applies the rotation tag a phone writes (EXIF orientation). A photo without the tag, or tagged wrongly, can still be sideways. |

Of 64 crops looked at by eye (`training/runs/labels/crops_preview.png`),
about 50 were mostly printed label, 6 about half, 8 mostly something else
(clear packaging showing the food, a hand, a table). That is one person's
judgement of one sheet, not a measurement.

**The old dataset is gone.** `training/data/open-images-v7-val/images`
(16,000 photos, 4.7 GB) was deleted on 2026-11-21 after the label download
was checked. Its `ATTRIBUTION.csv` and `image_list.csv` are kept, and so is
every checkpoint and every `runs/` folder. The folder's `COMPLETE` file,
which still said 16,000, was deleted on 2026-11-21, so `fetch_dataset.py` no
longer reports that dataset as complete.

**Numbers from earlier runs are not comparable with new ones.** The spread
check and its PSNR are measured on 256 fixed images picked from the training
folder. For every run above the label run in time (sanity, long, penalty-only,
spread) those were 256 Open Images photos; from the label run on they are 256
label photos. Fixed place, strongest spot, energy shares and PSNR all depend
on the images, so a difference between an old row and a new row says nothing
by itself. Training bit accuracy and PSNR differ for the same reason.

**The run.** `training/configs/labels.yaml`, from step 0, on the laptop's RTX
4070, two hours:

```bash
python train.py --config configs/labels.yaml --max-minutes 120 --keep-awake
```

The penalty-only run's settings (peak penalty on, hiding off, stop when fixed
place goes above 4) with the label images and one change to the schedule:
**the image-loss weight follows bit accuracy**
(`schedule.image_loss.follow_accuracy: {floor: 0.95, band: 0.01}`). The
penalty-only run kept the mark spread (fixed place at most 2.01) but bit
accuracy fell from 0.964 to 0.895 as the image loss went to full weight on
its fixed ramp. Here, after the trigger, the weight goes up by 1/15,000 per
step while the smoothed bit accuracy is at or above 0.95, down by the same
while it is below 0.94, and stays where it is in between. The peak penalty
uses the same weight. Sample sheet and spread check every 1,000 steps.

<!-- labels-run:start -->
| Time | Step | Fixed place | Strongest spot | Energy in one 96 px square | Energy on the busier half | PSNR, dB | Bit accuracy |
|---|---|---|---|---|---|---|---|
| 2026-11-21 19:20 | 0 | 1.01 | 1.05 | 6.5 percent | 50 percent | 21.31 (fixed images) | n/a |
| 2026-11-21 19:28 | 1,000 | 1.00 | 1.01 | 6.3 percent | 50 percent | 14.23 | 0.6693 |
| 2026-11-21 19:36 | 2,000 | 1.00 | 1.02 | 6.3 percent | 50 percent | 12.24 | 0.8185 |
| 2026-11-21 19:44 | 3,000 | 1.00 | 1.03 | 6.4 percent | 50 percent | 12.29 | 0.8790 |
| 2026-11-21 19:52 | 4,000 | 1.00 | 1.05 | 6.5 percent | 50 percent | 12.39 | 0.9029 |
| 2026-11-21 20:01 | 5,000 | 1.00 | 1.04 | 6.4 percent | 50 percent | 12.43 | 0.9174 |
| 2026-11-21 20:09 | 6,000 | 1.00 | 1.05 | 6.5 percent | 50 percent | 12.45 | 0.9290 |
| 2026-11-21 20:16 | 7,000 | 1.01 | 1.05 | 6.5 percent | 50 percent | 12.46 | 0.9395 |
| 2026-11-21 20:24 | 8,000 | 1.01 | 1.06 | 6.5 percent | 50 percent | 12.47 | 0.9482 |
| 2026-11-21 20:32 | 9,000 | 1.02 | 1.19 | 7.0 percent | 51 percent | 15.38 | 0.9567 |
| 2026-11-21 20:40 | 10,000 | 1.04 | 1.31 | 7.5 percent | 52 percent | 19.42 | 0.9554 |
| 2026-11-21 20:48 | 11,000 | 1.07 | 1.53 | 8.5 percent | 53 percent | 22.17 | 0.9506 |
| 2026-11-21 20:56 | 12,000 | 1.11 | 1.77 | 9.8 percent | 54 percent | 23.43 | 0.9496 |
| 2026-11-21 21:04 | 13,000 | 1.29 | 2.47 | 14.1 percent | 56 percent | 23.84 | 0.9481 |
| 2026-11-21 21:12 | 14,000 | 1.30 | 2.89 | 16.0 percent | 74 percent | 24.01 | 0.9479 |
| 2026-11-21 21:19 | 15,000 | 1.36 | 3.17 | 17.6 percent | 76 percent | 24.17 | 0.9479 |
| 2026-11-21 21:28 | 15,087 | 1.46 | 3.26 | 18.1 percent | 77 percent | 24.26 (fixed images) | n/a |
| 2026-11-21 21:36 | 16,000 | 2.17 | 3.47 | 19.3 percent | 76 percent | 24.27 | 0.9477 |
| 2026-11-21 21:44 | 17,000 | 2.97 | 3.94 | 22.1 percent | 71 percent | 24.14 | 0.9484 |
| 2026-11-21 21:52 | 18,000 | 3.13 | 3.52 | 20.8 percent | 63 percent | 24.23 | 0.9481 |
| 2026-11-21 22:01 | 19,000 | 3.44 | 3.74 | 22.4 percent | 61 percent | 24.13 | 0.9474 |
<!-- labels-run:end -->

**Result, 2026-11-21, 19:20 to 21:20.** Code: commit `1a9c008` on
`training/labels`. The run stopped at its time limit at step 15,087 with exit
code 0. Every number below is measured unless it says otherwise. Bit accuracy
and PSNR are means over the 1,000 training steps before the row; the weight is
the one at that step; fixed place and strongest spot are the spread check on
the run's 256 fixed images; speed is over those 1,000 steps and includes the
sample sheet and the spread check.

| Step | Bit accuracy | PSNR, dB | Image-loss weight | Fixed place | Strongest spot | Steps per second |
|---|---|---|---|---|---|---|
| 1,000 | 0.669 | 14.2 | 0.000 | 1.00 | 1.01 | 2.00 |
| 2,000 | 0.819 | 12.2 | 0.000 | 1.00 | 1.02 | 2.17 |
| 3,000 | 0.879 | 12.3 | 0.000 | 1.00 | 1.03 | 2.17 |
| 4,000 | 0.903 | 12.4 | 0.000 | 1.00 | 1.05 | 1.99 |
| 5,000 | 0.917 | 12.4 | 0.000 | 1.00 | 1.04 | 1.88 |
| 6,000 | 0.929 | 12.4 | 0.000 | 1.00 | 1.05 | 2.18 |
| 7,000 | 0.940 | 12.5 | 0.000 | 1.01 | 1.05 | 2.18 |
| 8,000 | 0.948 | 12.5 | 0.011 | 1.01 | 1.06 | 2.17 |
| 9,000 | 0.957 | 15.4 | 0.078 | 1.02 | 1.19 | 2.10 |
| 10,000 | 0.955 | 19.4 | 0.145 | 1.04 | 1.31 | 2.10 |
| 11,000 | 0.951 | 22.2 | 0.191 | 1.07 | 1.53 | 2.10 |
| 12,000 | 0.950 | 23.4 | 0.214 | 1.11 | 1.77 | 2.10 |
| 13,000 | 0.948 | 23.8 | 0.219 | 1.29 | 2.47 | 2.11 |
| 14,000 | 0.948 | 24.0 | 0.221 | 1.30 | 2.89 | 2.10 |
| 15,000 | 0.948 | 24.2 | 0.223 | 1.36 | 3.17 | 2.11 |

- **Speed: 2.10 steps per second** over the whole process (`train.py`'s own
  figure), between 1.88 and 2.18 by 1,000 steps. The earlier runs on Open
  Images measured 1.85 to 2.1 on the same GPU. The larger label photos did
  not slow the run. The one slow stretch (steps 3,000 to 5,000) was not
  explained.
- **The image loss came on late.** Smoothed bit accuracy passed 0.95 at step
  7,691; on Open Images it was step 3,986. Labels are harder for the reader
  at full mark strength, or simply slower; one run does not say which.
- **The weight stopped at 0.22.** It rose at the full rate until about step
  10,000, then more and more slowly, and from step 12,500 it barely moved:
  bit accuracy sat at 0.948 to 0.950, inside the band where the weight holds.
  At the points logged (every 50 steps) the weight went up 90 times, held 55
  times and never went down. So the schedule did what it is for, keeping bit
  accuracy at the floor where the penalty-only run fell to 0.895, and the
  price is that the image loss never got past a quarter of its weight. Moving
  down was exercised by the smoke test only, not by this run.
- **The mark is visible.** PSNR 24 dB; the sample sheets show a dotted
  texture, strongest on flat colour.
- **No fixed patch.** Fixed place 1.36 at the end, limit 4. The strongest
  spot per image rose to 3.2 and 76 percent of the energy sits on the busier
  half of the image: the mark follows the image content.
- The simulation had reached 75 percent of its strength when the run stopped
  (its ramp ends at step 20,000). The run's own closing check at that
  strength, eval mode, 8 fresh batches of training crops: bit accuracy 0.948,
  within reach of the error correction 0.781, all 64 bits right 0.062, PSNR
  23.9 dB.

**Acceptance at step 15,087** (`python acceptance.py --config
configs/labels.yaml`, results in `training/runs/acceptance/00015087/`). The
256 held-out label images, one tile from the middle of each, fixed random
bits, eval mode, simulation at full strength, flattened reader input, one
simulated shot per tile.

| Alpha | At most 4 wrong of the first 63 | Bit accuracy | All 64 right | Fixed place | PSNR mean, dB | PSNR worst tenth, dB |
|---|---|---|---|---|---|---|
| 0.30 | 0.531 FAIL | 0.919 | 0.020 | 1.76 PASS | 24.3 | 21.8 |
| 0.25 | 0.301 FAIL | 0.887 | 0.000 | 1.76 PASS | 25.8 | 23.4 |
| 0.20 | 0.082 FAIL | 0.818 | 0.000 | 1.76 PASS | 27.8 | 25.3 |
| 0.15 | 0.000 FAIL | 0.707 | 0.000 | 1.76 PASS | 30.2 | 27.8 |

**Not accepted: it is neither read well enough nor invisible.** At the
strength it was trained with (0.30), 53 percent of the tiles are within reach
of the error correction; the line is 80. Lowering alpha makes the mark
fainter and the reading worse, so no alpha passes.

How to read the table:

- The read rate here is lower than the run's own closing check (0.781) for
  two reasons that cannot be separated from these numbers: the simulation is
  at full strength, which the models had not yet been trained at, and the
  images are ones they never saw.
- With 256 tiles and one shot each, a share near 0.5 is uncertain by about
  0.03 either way (one standard error; an estimate from the count, not a
  repeated measurement).
- Fixed place is the same at every alpha by construction: alpha scales the
  whole mark and the measure divides that out. It differs from the run's own
  1.36 because these are other images, cut differently.
- PSNR is the mean over the tiles of each tile's own PSNR.
- Only alpha 0.30 was trained. The other rows scale the mark of the same
  embedder down and ask the same reader.
- `blind.html` shows the alpha 0.30 tiles. It carries dataset photos and
  their credit line and is for private review only.

## Spread run (RunPod)

The long run's settings with `spread.peak_penalty` and `spread.hide` both on
(`training/configs/runpod_spread.yaml`), from step 0, on a RunPod pod. It
runs at the same time as the laptop's penalty-only run and shares nothing
with it. Commands: `training/RUNPOD.md`.

| | |
|---|---|
| Commit | `bf9c7a0` on `training/pipeline` (the laptop run's code is the same commit) |
| Options | penalty on (weight 4), hiding on (up to 50 percent of the reader's patch) |
| Stops | fixed place above 4 on 256 fixed images, at any sample sheet (exit code 4); bit accuracy under 0.75 after 45 minutes (3); PSNR under 14 dB at step 15,000 (5) |
| Sample sheets and table rows | every 2,000 steps |
| GPU | to fill in |
| Price per hour | to fill in |
| Started | to fill in |

The commit after `bf9c7a0` adds a fourth stop to the config: 24 hours of
training (`budget.max_hours`, exit code 6), the only one that limits time. A
pod started from `bf9c7a0` itself does not have it.

The table has the same columns as the penalty-only run's, and "The spread
check" under that heading says what each one is. Three things differ when
the two tables are read side by side:

- **Bit accuracy is read with regions of the patch blanked** in this run, so
  it runs below the penalty-only run's at the same step. That is expected
  and is not the mark reading worse.
- **This run starts at step 0.** The penalty-only run starts at step 6,000,
  from a checkpoint trained without either option. Up to the step where the
  image loss switches on, the mark covers the whole tile at full strength
  and the rows say nothing yet.
- **Time is the pod's clock.** That is UTC, 8 hours ahead of the laptop
  rows, unless `TZ` was set on the pod before the run started.

The 256 images are picked by the same seeded shuffle of the sorted file
list, so they are the same images as on the laptop if the pod holds the
same 16,000 files. That was not checked on the pod.

<!-- spread-run:start -->
| Time | Step | Fixed place | Strongest spot | Energy in one 96 px square | Energy on the busier half | PSNR, dB | Bit accuracy |
|---|---|---|---|---|---|---|---|
<!-- spread-run:end -->

**How the rows get here.** The pod does not write this file. It writes
`/workspace/runs/spread/logs/spread.csv`, one row per sample sheet. Copy
that file to `training/runs/pod/spread/spread.csv` on the laptop: `notify.py --run spread` keeps that copy up to date, or
copy it by hand (`training/RUNPOD.md`, section 6). Then, in
`training`:

```bash
python -c "from runlog import SpreadLog; SpreadLog('runs/pod/spread', {'file': '../docs/runs.md', 'marker': 'spread-run'}).update_markdown()"
```

It rebuilds the table above from the whole file, with the code the
penalty-only run uses for its own table, so it can be run again after every
copy. It loads no model and does not use the GPU.

## Penalty-only run

The long run's settings with `spread.peak_penalty` on and `spread.hide` off
(`training/configs/penalty.yaml`), on the laptop's RTX 4070:

```bash
python train.py --config configs/penalty.yaml --keep-awake
```

**It resumes from the step 6,000 checkpoint** of the sanity run (the long
run's own first hour), copied into `training/checkpoints/penalty/`. That is
the last saved point where the mark was still even: strongest spot 1.1,
6.7 percent of the energy in one 96 px square. The next one, step 7,001, was
already at 3.3. The resume was checked before it was used:

- **Optimizer state: valid.** The checkpoint holds Adam's state for 105
  parameter tensors, all at step 6,000. The penalty is a loss term and adds
  no parameters, so the state fits as it is. Its running averages were built
  without the penalty; with these settings they follow a changed loss within
  about a thousand steps.
- **Image-loss schedule: valid.** The checkpoint carries where the ramp began
  (step 3,986) and the smoothed bit accuracy (0.977). At step 6,000 the image
  loss is at 13.4 percent of its weight and goes on rising to full weight at
  step 18,986, exactly as in the long run. The simulation is at 30 percent
  and follows the step count.
- **Penalty schedule: valid, with one difference from a run started at 0.**
  The penalty is multiplied by the image-loss schedule. A run from step 0
  would have had it from step 3,986, rising from nothing. Here it comes in at
  step 6,000 at 13.4 percent at once. Because the mark is still even, the
  penalty is then about the size of the mean-square loss (roughly 0.012 added
  to a loss near 0.1), so there is no jolt. What is missed is 2,014 steps of
  a slightly stronger pull towards a fainter mark.
- Not identical to a fresh run in one more way, as for the long run: steps 1
  to 6,000 saw only the first 2,000 dataset images.

Other settings: sample sheet every 2,000 steps
(`training/runs/penalty/samples/`), and the learning check 45 minutes into
this run (bit accuracy under 0.75 stops it; the clock starts at 0 with this
run, so the check falls near step 11,000). Same GPU checks as before.

**The spread check.** With every sample sheet the run measures the mark on
256 fixed dataset images with fixed messages, in eval mode, and writes the
row below:

- **Fixed place**: is the mark at the same place in every tile? Each image's
  map of the mark's energy (48 px windows) is scaled to mean 1, the maps are
  averaged over the 256 images, and this is the mean of the strongest 2
  percent of windows of that average. A patch at one place in every tile
  keeps its full height: the unfixed run's corner patch scores 18 to 20. A
  mark that follows each image's content mostly averages out.
- **Strongest spot**: the same, per image, then averaged. 1 is even. It is
  high for the corner patch and also for a mark that follows the image, so
  it is logged and stops nothing.
- **Energy in one 96 px square**: share of the mark's energy in its strongest
  96 px square, per image, averaged. An even mark gives 6.25 percent.
- **Energy on the busier half**: share of the mark's energy on the half of
  each image with more local contrast. 50 percent means the mark ignores the
  image.

**The run stops itself, with a checkpoint, if "fixed place" goes above 4**
(`spread.stop_above: 4.0`). A mark that follows the image content is
accepted; a patch at the same place in every tile is not. The next heading
says where the 4 comes from.

PSNR and bit accuracy are training means over the 1,000 steps ending at the
row's step. The first row is the starting point, before any step of this run.
The table is written by the run itself. The rows up to step 10,000 were
recomputed on the CPU from the saved checkpoints when the check moved from 16
to 256 images; the 16-image numbers of the time are in
`training/runs/penalty/spread_16_images.csv`.

<!-- penalty-run:start -->
| Time | Step | Fixed place | Strongest spot | Energy in one 96 px square | Energy on the busier half | PSNR, dB | Bit accuracy |
|---|---|---|---|---|---|---|---|
| 2026-11-21 15:35 | 6,000 | 1.02 | 1.12 | 6.7 percent | 50 percent | 16.96 (fixed images) | n/a |
| 2026-11-21 15:52 | 8,000 | 1.05 | 1.43 | 8.2 percent | 53 percent | 24.04 | 0.9638 |
| 2026-11-21 16:08 | 10,000 | 1.59 | 3.61 | 19.2 percent | 76 percent | 26.51 | 0.9552 |
| 2026-11-21 16:43 | 12,000 | 1.58 | 3.65 | 19.7 percent | 79 percent | 27.34 | 0.9511 |
| 2026-11-21 16:59 | 14,000 | 1.64 | 2.75 | 15.6 percent | 75 percent | 27.52 | 0.9441 |
| 2026-11-21 17:15 | 16,000 | 1.61 | 2.81 | 15.9 percent | 75 percent | 27.67 | 0.9313 |
| 2026-11-21 17:33 | 18,000 | 1.79 | 2.86 | 16.2 percent | 74 percent | 27.95 | 0.9158 |
| 2026-11-21 17:51 | 20,000 | 2.01 | 3.26 | 18.2 percent | 76 percent | 28.07 | 0.8939 |
| 2026-11-21 18:10 | 22,000 | 1.73 | 2.71 | 15.6 percent | 72 percent | 28.08 | 0.8953 |
<!-- penalty-run:end -->

**First stop, at step 10,000.** The run was first started with the limit on
"strongest spot" (3, on 16 images). It stopped itself at step 10,000, 16:08,
33 minutes after its start, with a checkpoint and exit code 4: strongest
spot 3.99 on those 16 images (3.61 on 256).

What the mark looks like there
(`training/runs/penalty/samples/step_00010000.png`): there is **no patch at
the tile corner**. The mark sits on the busy parts of each image (the
jellyfish, the food, the edges of the smoke) and is nearly absent from the
flat parts (black background, a smooth leaf, sky). At step 8,000 it still
covered the whole tile. The numbers agree: fixed place 1.59, with the
strongest window of the averaged map in the middle of the image (centred at
x 200, y 192) and the window on the tile corner at 1.11; 76 percent of the
energy on the busier half of the image, up from 50.

So the limit had been tripped by a mark that follows the image, which was
then accepted. The penalty did slow the pile-up: the unfixed run was at
strongest spot 2.8 with the image loss at 20 percent of its weight (step
7,001, 20.6 dB), this run at 3.6 with it at 40 percent and a PSNR 6 dB
higher. Whether the penalty alone prevents the corner patch was not answered
at step 10,000, with the image loss at 40 percent of its weight; that is
what the resumed run is for.

**Resumed from step 10,000** on 2026-11-21 with the same command, after the
limit was moved to the fixed-place measure. Nothing else changed: same
config, same checkpoint folder, same clock (33 minutes of training before
the resume, so the 45-minute learning check comes about 12 minutes in). The
rows from step 12,000 on are from the resumed run.

**Up to step 20,000: a mild drift to the tile edge and corner.** The fixed
place number sat at 1.6 from step 10,000 to 16,000, then rose to 1.79 and
2.01. Measured on the CPU from the checkpoints (256 images): the averaged
map's energy in the band within 24 px of a tile edge (27 percent of the
tile) and in the corner region (2 percent), as multiples of the tile average.

| Checkpoint | Edge band | Corner region | Strongest window |
|---|---|---|---|
| penalty run, step 10,000 | 0.92 | 1.01 | 1.64, in the middle of the image |
| step 14,000 | 1.09 | 1.63 | 1.81, on the tile corner |
| step 16,000 | 1.11 | 1.60 | 1.77, on the tile corner |
| step 18,000 | 1.18 | 1.77 | 1.97, on the tile corner |
| step 20,000 | 1.25 | 1.99 | 2.22, on the tile corner |
| unfixed run, step 26,000 | 2.84 | 11.77 | 22.39 |

So the corner is forming here too, slowly: twice the average at step 20,000
against twelve times in the unfixed run. In the step 20,000 sheet one image
that is busy all over has its mark as a frame along the tile edges. Training
bit accuracy fell over the same stretch, from 0.955 at step 10,000 to 0.894
at 20,000 (unfixed run: 0.953 at 20,000), while PSNR rose to 28.1 dB (unfixed:
26.1). The image loss reached full weight near step 19,000 and the
simulation full strength at 20,000, so from here neither pushes harder.

### The fixed-place baseline and the limit of 4

What does "fixed place" read for a mark that only follows the image? Two
estimates, on the CPU, on the fixed images.

Made-up marks, built from the images alone, that know nothing of the tile:
all energy on the busier half of each image; energy in proportion to local
contrast; and to its square and fourth power, which pack it tighter.

| Made-up mark that only follows the image | Strongest spot per image | Fixed place, 16 images | 64 images | 256 images |
|---|---|---|---|---|
| evenly on the busier half | 2.0 | 1.82 | 1.48 | 1.40 |
| in proportion to local contrast | 4.4 | 2.33 | 1.60 | 1.63 |
| contrast squared | 9.5 | 3.68 | 2.23 | 2.07 |
| contrast to the fourth power | 20.6 | 6.14 | 3.41 | 2.71 |

Real marks with the place information destroyed: each image's energy map
shifted around the tile by its own random offset, 200 different draws, the
highest of them:

| Real mark, shifted at random | Strongest spot per image | 16 images | 64 images | 256 images |
|---|---|---|---|---|
| penalty run, step 10,000 | 3.6 | 2.21 | 1.52 | 1.26 |
| unfixed run, step 26,000 | 18.7 | 5.76 | 2.94 | 1.95 |

And the real marks as they are, 256 images:

| Checkpoint | Fixed place | Strongest spot | Energy on the busier half | Where the averaged map is strongest |
|---|---|---|---|---|
| step 6,000, the start | 1.02 | 1.12 | 50 percent | nowhere in particular |
| unfixed run, step 7,001 | 1.22 | 2.84 | 50 percent | nowhere in particular |
| penalty run, step 8,000 | 1.05 | 1.43 | 53 percent | nowhere in particular |
| penalty run, step 10,000 | 1.59 | 3.61 | 76 percent | middle of the image (x 200, y 192) |
| unfixed run, step 22,000 | 19.76 | 20.55 | | on the tile edge, 24 px from the corner |
| unfixed run, step 26,000 | 18.33 | 18.69 | 38 percent | on the tile edge, 24 px from the corner |

- **Baseline: 1.4 to 2.7 on 256 images.** It is above 1 even with many
  images because photographs are busier in the middle, so a mark that
  follows the image has a mild preferred place of its own. 2.7 is the
  extreme case, a mark packed as tightly into each image as the corner patch
  is into the tile. The mark the penalty run actually has at step 10,000
  reads 1.59, in line with "in proportion to local contrast" (1.63).
- **Limit: 4.** That is 1.5 times the highest baseline, 2.5 times what the
  image-following mark of step 10,000 reads, and 4.6 times below the unfixed
  run's 18.3.
- **16 images are too few, so the check uses 256.** On 16 the baseline
  reaches 6.1 and a tightly packed mark shifted at random reaches 5.8, which
  leaves no clear room under the limit. On 256 the number is steady: the two
  halves of the images give 1.58 and 1.62 for the penalty run and 18.24 and
  18.41 for the unfixed one.
- What the limit does not see, by design: a mark that follows the image,
  however unevenly. What it would see besides the corner patch: a mark laid
  along the tile edge in every tile, which is a fixed place too.
- Not known: how early it trips. The unfixed run read 1.22 at step 7,001 and
  19.8 at step 22,000, and no checkpoint between them was kept. Its sample
  sheet at step 10,000 already shows the patch.

A correction to the numbers first written here: the "busier half" shares
were computed with a blur that made every image look busy along its border.
With that fixed they are 50, 53 and 76 percent for the penalty run (not 50,
52, 68) and 38 percent for the unfixed run's corner patch.

## Long run

**Cancelled on 2026-11-21.** The last line in its log is step 26,500, written
at 06:34; it was stopped from outside, so no checkpoint was written at the
stop. Its last checkpoint is **step 26,000** (06:30). That file was loaded
afterwards on the CPU, models and optimizer state together, and measured:
strongest spot 19.3, 86.6 percent of the mark's energy in one 96 px square,
PSNR 25.9 dB. Training bit accuracy over steps 25,001 to 26,000 was 0.961.
The three checkpoints in `training/checkpoints/long/` (steps 22,000, 24,000
and 26,000) are kept as the fallback; nothing writes to that folder now. To
go on from step 26,000:

```bash
python train.py --config configs/long.yaml --keep-awake
```

Started 2026-11-21 03:43 on the laptop's RTX 4070, from commit `6aa5b0b`:

```bash
python train.py --config configs/long.yaml --keep-awake
```

**It resumed from the sanity run's checkpoint at step 7,001** instead of
starting at step 0. The two configs were compared key by key, and against the
config stored in the checkpoint. They differ only in the folders, the sample
sheet interval, and `simulation.direct.scale`, which the flattened run never
reads. Model, band, batch size, learning rate, losses, both schedules and the
whole simulation are identical, so the sanity hour is the long run's first
hour. The checkpoint, `progress.csv`, the clock and the TensorBoard file were
copied into `checkpoints/long` and `runs/long`; the sanity folders are
untouched.

One thing is not identical: steps 1 to 7,001 saw only the first 2,000
dataset images. From 7,001 on the run draws from all 16,000.

- Same GPU checks as the sanity run: CUDA available, `cuda:0` is the RTX
  4070, 7,068 of 8,188 MiB free; it would have stopped otherwise.
- 140,000 steps in total. At the 1.4 to 2.1 steps per second seen so far,
  about 18 to 26 more hours.
- Sample sheets every 5,000 steps in `training/runs/long/samples/`.
- Checkpoints every 2,000 steps in `training/checkpoints/long/`. The same
  command resumes after a stop or a crash.
- No other GPU job runs while it trains.

Progress, from `runs/long/progress.csv` (training means over the 1,000 steps
ending at each step):

| Step | Bit accuracy | PSNR, dB | Notes |
|---|---|---|---|
| 10,000 | 0.969 | 23.5 | |
| 13,000 | 0.971 | 25.0 | highest bit accuracy so far |
| 16,000 | 0.967 | 25.7 | |
| 19,000 | 0.958 | 26.1 | image loss at full weight from about here |
| 20,000 | 0.953 | 26.1 | simulation at full strength from here |
| 22,000 | 0.954 | 26.1 | |
| 24,000 | 0.958 | 26.1 | |
| 26,000 | 0.961 | 26.1 | last checkpoint |

The step 25,000 sheet still shows the patch.

Speed since the resume has been between 1.4 and 2.1 steps per second. CPU
checks for the spread options ran on the same machine from 06:03 to 06:32.
The thousand steps to 24,000, when those checks ran at normal priority, took
11.2 minutes, against 9.4 for the thousand before. The checks were then moved
to idle priority and the next two thousands took 10.2 and 8.8 minutes. The
run's own pace had varied as much before any check ran (12.1 minutes for
21,000 to 22,000), so the cost to the long run is a few minutes at most.

The mark is not spread over the tile: see the next section.

## Corner patch

**What the long run does.** Once the image loss is on, the embedder moves the
whole code into one bright, multicoloured patch at the tile corner (one spot,
since the tile repeats) and leaves the rest of the tile almost clean. It is
plainly visible in `training/runs/long/samples/step_00020000.png` and was
already there at step 10,000.

Measured on the CPU from the saved checkpoints, on 16 dataset images with
random bits. "Strongest spot" is the mark's energy in its strongest 2 percent
of 48 pixel windows, divided by the average over the tile; 1 is perfectly
even. A 96 pixel square is 6.25 percent of the tile.

| Checkpoint | PSNR, dB | Strongest spot | Share of the mark's energy in one 96 px square |
|---|---|---|---|
| step 4,000 (image loss just on) | 12.1 | 1.0 | 6.4 percent |
| step 6,000 | 16.8 | 1.1 | 6.7 percent |
| step 7,001 | 20.6 | 3.3 | 18.2 percent |
| step 20,000 | 25.7 | 19.8 | 86.0 percent |
| step 22,000 | 26.3 | 20.7 | 89.2 percent |

So the mark is even while the image loss is off, and piles up as the image
loss comes on. The likely reason, not established by a test: the mean-square
loss does not care where the mark's energy sits, the perceptual loss is
cheaper for one small patch than for a texture over the whole image, and the
reader always gets a full tile, so one spot is enough for it.

Two things follow. The 26 dB is mostly that patch, not a faint mark. And a
print or a camera view that loses that one spot loses the whole message.

**Two options against it**, both off by default (`training/spread.py`,
config section `spread`):

- `spread.peak_penalty`: a loss on the strongest local region of the mark,
  the same quantity as "strongest spot" above before dividing by the
  average. For an even mark it equals the mean-square loss; for the long
  run's mark it is 20 times that. Weight 4, on the image-loss schedule, so it
  acts exactly when the pile-up happens. The weight is a first guess; no run
  has tuned it.
- `spread.hide`: before the reader sees its patch in a training step, random
  regions of it are blanked with the patch's mean colour. Per sample the
  share is uniform between 0 and 50 percent, in a grid of 2 x 2 to 8 x 8
  cells shifted by a random amount, so the regions are large and connected
  for some samples and scattered for others. Evaluation reads the whole patch.

A plain per-pixel limit on the mark was considered and not added: one already
exists (`model.embedder.alpha`, 0.3), and tightening it enough to matter
would also weaken the full-strength mark the reader learns from in the first
4,000 steps. The penalty on the strongest region costs an even mark nothing.

**Checked on the CPU, 2026-11-21. No training run has used the options.**
The long run kept the GPU throughout.

1. Checks inside `smoke_test.py`, run before any training step: an even mark
   scores 1 and the same energy packed on the tile corner scores above 20,
   seen as one patch across the tile edges; the penalty's gradient falls on
   the patch and nowhere else; hiding never blanks more than the limit and
   leaves the other pixels untouched; with both on and 22 percent of the
   patch blanked, the bit loss still gives gradient to the embedder (norm
   11.7) and the reader (20.6), and the penalty gives gradient to the
   embedder (0.096). All finite.
2. The 300-step smoke test with both on, on the CPU (generated images,
   simulation off):

   | | Bit accuracy, training | Eval, whole patch |
   |---|---|---|
   | both options on (CPU) | 0.821, read with regions blanked | 0.820 |
   | plain smoke test, both off (CPU, run later the same day) | 0.891 | 0.897 |
   | plain smoke test (GPU, from the table further down) | 0.852 | 0.833 |

   So the reader still learns from the start with part of its patch hidden,
   but more slowly: on the same device the hiding costs about 7 points at
   step 300. The CPU and GPU rows differ by 4 points with identical settings,
   so compare CPU with CPU.
   The image loss stays off in a 300-step test, so this exercises the hiding
   and not the penalty.
3. 30 real training steps on the CPU from the long run's step 22,000
   checkpoint (full-size models, real images, full simulation, LPIPS, batch
   4, same images and seed in both rows):

   | 30 steps with | Strongest spot | Energy in one 96 px square | PSNR, dB | Bit accuracy, training mean |
   |---|---|---|---|---|
   | before | 20.7 | 89.2 percent | 26.3 | |
   | options off (control) | 21.0 | 89.3 percent | 26.7 | 0.949 |
   | both options on | 14.2 | 70.1 percent | 30.1 | 0.771, read with regions blanked |

   The penalty term was 0.049 at the first step (times its weight of 4: 0.20,
   against a bit loss near 0.2), and the patch began to shrink at once.

What this does and does not show. The options do what they are built to do
and push in the right direction on the real model. In those 30 steps the
patch got weaker; the mark did not yet reappear elsewhere, and bit accuracy
fell, as it must while the reader still depends on a patch that is being
hidden and penalised. Whether a run that has the options on from step 0
ends with an even mark that reads well is not known. Three things to watch
in such a run:

- The image loss only switches on when training bit accuracy passes 0.95.
  With regions blanked that may come later than the 4,000 steps of the
  sanity run, or not at all. A PSNR still near 12 dB after 15,000 steps
  means it never switched on.
- The penalty's weight is untested over a whole run. Too low and the patch
  comes back; too high and it could hold the mark down everywhere.
- Hiding up to half the patch forces the message to be readable from half a
  tile. It does not promise a quarter.

The pod config for this is `training/configs/runpod_spread.yaml`: the long
run's settings with both options on and a sample sheet every 2,000 steps.
The run saves a checkpoint and exits by itself in three cases:

| Stop | When | Exit code |
|---|---|---|
| Learning check (`early_stop`) | bit accuracy under 0.75 after 45 minutes of training | 3 |
| Mark piled up (`spread.stop_above: 4.0`) | fixed place above 4 on the 256 fixed images, at any sample sheet (first written here as strongest spot above 3 on 16 images; changed in commit `bf9c7a0`) | 4 |
| Image loss never switched on (`image_loss_check`) | mean PSNR still below 14 dB at step 15,000 | 5 |

The last two are config switches, off by default. The first of them covers
the first point above without anyone watching the PSNR. Both were dry-run on
the CPU with a tiny config: each stops with a checkpoint, and the image-loss
stop refuses to resume until it is switched off. `training/RUNPOD.md` has the
commands.

Every run started from this code prints the "strongest spot" number with
each sample sheet and logs it to TensorBoard (`image/concentration`), whether
or not the options are on. The long run was started before it and does not.


One hour of the long run's own config (`configs/sanity.yaml` changes only the
folders and requires the RTX 4070), 2026-11-21, 01:56 to 02:56.

```bash
python train.py --config configs/sanity.yaml --max-minutes 60
```

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, `cuda:0`, 7,068 of 8,188 MiB free
  at the start. Batch 8, no out-of-memory error. PyTorch's peak was 4,322 MiB;
  `nvidia-smi` showed 5,021 MiB used on the whole card and 92 to 100 percent
  utilisation.
- Data: the first 2,000 images of the dataset. The long run uses 16,000.
- 7,001 steps in 60 minutes: 1.94 steps per second overall.

| Step | Bit accuracy | PSNR, dB | Minutes for these 1,000 steps |
|---|---|---|---|
| 1,000 | 0.660 | 14.4 | 8.2 |
| 2,000 | 0.819 | 12.4 | 7.7 |
| 3,000 | 0.899 | 12.3 | 7.7 |
| 4,000 | 0.941 | 12.3 | 7.7 |
| 5,000 | 0.959 | 13.0 | 8.0 |
| 6,000 | 0.974 | 15.7 | 8.7 |
| 7,000 | 0.976 | 18.9 | 11.9 |

Bit accuracy and PSNR are training means over the 1,000 steps ending there.

At the end, in eval mode on 8 fresh batches: **bit accuracy 0.960**, all 64
bits right in 16 percent of tiles, within reach of the error correction (at
most 4 wrong bits) in 83 percent, **PSNR 20.4 dB**.

What that does and does not show:

- The run learns with the phone settings and the top band, on real images.
- The image loss switched on at about step 4,000, when bit accuracy passed
  0.95. From there the PSNR rose 7 dB in 3,000 steps while bit accuracy kept
  rising. At step 7,000 the image loss was at 20 percent of its weight, so
  the mark is still plainly visible in the samples. That is expected at this
  point and says nothing about the final look.
- The simulation was at 35 percent of full strength at the end (it ramps over
  20,000 steps). The full phone camera has not been trained against yet.
- Speed fell in the last quarter hour, from about 2.1 to 1.4 steps per
  second. The image loss is not the cause by itself: the first 1,000 steps
  with it cost 4 percent more. Unrelated timing runs straight afterwards were
  also slower than in the afternoon (0.80 s per step against 0.50 s for the
  same test), so the laptop as a whole had slowed after an hour under load.
  It was on mains power, Balanced plan. Heat is the likely reason; that was
  not verified.
- At 1.4 to 2.1 steps per second, 140,000 steps take 18 to 28 hours.

Samples: `training/runs/sanity/samples/` (original, marked, difference x10;
one sheet every 1,000 steps and `final_step_00007001.png`). The images are
dataset images under CC BY 2.0: credit the authors from `ATTRIBUTION.csv`
before publishing a sheet. Checkpoint: `training/checkpoints/sanity/`.

## Camera settings for the long run

Phone only. The 720p laptop webcam is no longer a target.

| Setting | Value | Where |
|---|---|---|
| Printed tile | 50 mm | `demo.tile_mm` |
| Camera stream | 1080 x 1920, portrait | `demo.stream_px` |
| Camera pixels per printed tile | 154 to 340, each sample draws its own, log-uniform | `camera_px_per_tile` |
| Starting band | top band: grating step 8, 16 to 51.2 cycles per tile, picked by `auto` | `band.grating_step` |
| Simulated frame | square, 768 | `simulation.frame_px` |

All in `training/configs/default.yaml`. Before this the config had
`camera_px_per_tile: null`: the loss-of-resolution step did not run at all.
No 120 to 220 range was in the repository.

The low end was first set to 150 and raised to 154 the same day, so that the
top band meets the margin everywhere (next section).

### Measurements

Galaxy S23 Ultra, 1080 x 1920 portrait stream, 50 mm tile.

| Distance | Measured | Camera pixels per tile |
|---|---|---|
| about 20 cm | 66.3 px/cm | 331.5 |
| about 35 cm | 37.2 px/cm | 186.0 |

The two agree with each other: distance times px/cm is 1,326 and 1,302, as it
should be for one lens. From them, 340 pixels per tile is about 19 cm and 154
is about 42 to 43 cm. The frame is 3.2 to 7.0 tiles wide over the range.

Training stops with an error if a measurement in `demo.measured` falls outside
`camera_px_per_tile`.

## Is the top band valid across the range?

**Yes.** With the range at 154 to 340, `band.grating_step: auto` picks step 8
by itself.

The top band's finest stripes are at 51.2 cycles per tile. The camera has to
give each cycle more than 2 pixels to sample it at all. The config asks for 3
(`band.min_camera_px_per_cycle`).

| Camera px per tile | Px per cycle at 51.2 cycles | Sampling limit (2) | Margin (3) | Amplitude left after the camera step |
|---|---|---|---|---|
| 154 (low end) | 3.01 | passes | passes | 47 percent |
| 186 (measured, 35 cm) | 3.63 | passes | passes | 60 percent |
| 331.5 (measured, 20 cm) | 6.47 | passes | passes | 86 percent |
| 340 (high end) | 6.64 | passes | passes | 86 percent |

The coarsest stripes of the band (16 cycles) keep 93 to 99 percent.

- The margin is met with almost nothing to spare at the low end. At 153
  pixels per tile `auto` drops to step 6 (12 to 38.4 cycles). Lowering the
  range again means either accepting that or setting step 8 by hand.
- The margin is for a tile seen straight on. At 154 pixels per tile and a 30
  degree tilt, the finest stripes get 2.60 pixels per cycle along the tilt:
  under the margin, above the sampling limit.
- The next band up (step 12, 24 to 76.8 cycles) is at the sampling limit at
  154 (2.01 pixels per cycle) and far under the margin. `auto` does not pick
  it; set by hand below 153.6 the code refuses it.

"Amplitude left" is measured: a stripe pattern at that frequency put through
the simulation's down-and-up resampling.

The band is only where training starts. The embedder is free to move the code.

### Smoke tests with the camera step

300 steps each on 24 generated images, RTX 4070 laptop GPU, loss of resolution
only, at full strength. One run per row, seed 0, so differences of a few
points between rows are not evidence. Pass mark is 0.70; chance is 0.50.

| Camera px per tile | Band | Bit accuracy, training | Bit accuracy, eval |
|---|---|---|---|
| simulation off | step 8 | 0.852 | 0.833 |
| 154 | step 8 (top), by auto | 0.815 | 0.794 |
| 340 | step 8 (top) | 0.851 | 0.841 |
| 154 to 340 (the long-run range) | step 8 (top), by auto | 0.818 | 0.824 |
| 150, before the low end was raised | step 8 (top), set by hand | 0.817 | 0.789 |
| 150 to 340, before the low end was raised | step 8 (top), set by hand | 0.833 | 0.804 |
| 150 | step 6, for comparison | 0.823 | 0.822 |

The top band learns at both ends of the range. This says the start works. It
does not say what a long run reaches.

```bash
python smoke_test.py --config configs/smoke_resolution.yaml camera_px_per_tile=154
```

## Portrait frames

The simulation and the flatten step did not assume landscape. The simulation
assumed a **square** frame; that is now lifted.

- The simulated frame (`simulation.frame_px`, 768) is not the camera stream.
  It is a window onto the surface at 384 pixels per tile, two tiles across,
  and the camera's resolution is modelled by the resolution step. A square is
  neutral between portrait and landscape, and the long run keeps it: a 9:16
  window would cost 78 percent more pixels per step and change nothing the
  flattened reader sees.
- `simulation.frame_px` now also takes `[width, height]`. `smoke_test.py`
  checks on every run that a 540 x 960, a 960 x 540 and a 768 x 768 frame all
  flatten back to the surface (largest error 0.0015 on a 0 to 1 scale).
- Exported fixture frames are now portrait, 768 x 1366. The JavaScript flatten
  in `docs/model-contract.md` was run on them unchanged and reproduced the
  patches (at most 2 values in 442,368 off by one level).

## Two changes to the simulation that came with the range

1. **Each sample draws its own camera resolution.** Before, the whole batch
   shared one.
2. **The range now means what it says.** In flattened mode the simulation
   shrinks each view by 3 to 20 percent (8 percent in the median) to keep two
   tiles inside the frame. That shrink used to add to the loss of resolution,
   so a low end of 154 would have been as low as 124 in effect. It is now
   divided out. Measured with a 16-cycle stripe pattern through the whole
   geometry, when the low end was still 150: 92 percent of its amplitude is
   left at 150 pixels per tile and 97 percent at 340, where the camera step
   alone predicts 93 and 99.

Neither costs measurable time: 0.50 s per step with the camera step and 0.51
without it (batch 8, full-size models, every other effect on, generated
images, no LPIPS; 30 steps timed on the RTX 4070 laptop GPU). At that rate
140,000 steps is about 19 to 20 hours on the laptop; LPIPS, once the image
loss switches on, adds to that and was not timed. Peak GPU memory in that
test: 3.9 GB.

## Direct mode

**Dropped for the gate, 2026-11-21.** The gate goes ahead with the flattened
reader only. The reason is the test results below: six smoke tests, and all
four that turn the tile end at chance.

| Direct mode with | Steps | Bit accuracy, training | Eval |
|---|---|---|---|
| no rotation, tile fixed at 340 px | 300 | 0.935 | 0.947 |
| no rotation, tile 154 to 340 px | 300 | 0.653 | 0.639 |
| any rotation, tile 154 to 340 px (what `runpod_direct.yaml` trains) | 300 | 0.502 | 0.497 |
| rotation and size range eased in over the first 700 steps | 900 | 0.507 | 0.506 |
| rotation within 30 degrees of the four right angles, from step 0 | 900 | 0.515 | 0.520 |
| 300 upright steps, then rotation and size range eased in over steps 300 to 800 | 1,000 | 0.499 | 0.501 |

Chance is 0.50. One run each, small models, generated images. The last row
reached 0.833 upright at step 300 and fell to chance as the tile began to
turn. No pod run of direct mode was made, and none is planned.
`configs/runpod_direct.yaml` and the code path stay in the repository,
unchanged and unrecommended; the pod now runs `configs/runpod_spread.yaml`
(see "Corner patch"). The details behind the table follow.

Matched to the phone since 2026-11-21.

**What changed.** The direct reader used to see a tile at 307 to 480 pixels
in its 384 pixel crop (`simulation.direct.scale: [0.8, 1.25]`), with the
camera range applied as blur only. Now `scale: auto` takes the size from
`camera_px_per_tile`: the tile is 154 to 340 pixels wide in the crop, at any
rotation, position and tilt, so the crop holds 1.1 to 2.5 tiles across. The
tile is drawn large and reduced with averaging, as a sensor does it; noise
and JPEG are applied at that size. The surface is sampled as an endless
pattern, so a small or tilted view never runs off its edge.

Checked on the CPU: tile size between 154.6 and 339.8 pixels over 320
samples; a stripe pattern lands at the expected frequency at both ends; a
pattern too fine for 154 pixels comes out flat, not as a false coarse one;
the returned homography agrees with where a dot is seen to within 0.5 pixel.

**Cost per step.** Faster, not slower. Old and new measured in turn on the
RTX 4070, twice each (batch 8, full-size models, every effect on, generated
images, no LPIPS, 60 steps timed):

| | Old direct | New direct | Flattened, as a control |
|---|---|---|---|
| Seconds per step | 0.780, 0.786 | 0.673, 0.668 | 0.797, 0.795 old code; 0.802, 0.798 new |
| Peak GPU memory | 3,849 MiB | 3,596 MiB | 3,884 MiB |

New direct takes 14 percent less time per step than old direct. The drawn
square is smaller on average than the old 768 frame, and one tile is sampled
instead of a 3 x 3 surface. The absolute times are high because the laptop
was running slow after the sanity run (see there); the ratio is what counts.
On a pod the split between CPU and GPU work differs, so expect the direction
to hold, not the exact figure.

**Does it learn?** Not shown, and the first evidence is mixed. 300-step smoke
tests, generated images, no other effect, one run each:

| Direct mode with | Bit accuracy, training | Eval |
|---|---|---|
| any rotation, tile 154 to 340 px (what the pod config trains) | 0.502 | 0.497 |
| no rotation, tile 154 to 340 px | 0.653 | 0.639 |
| no rotation, tile fixed at 340 px | 0.935 | 0.947 |

Chance is 0.50. So the direct path itself works and learns quickly when the
view is upright and one size; the size range slows it; free rotation stops
it within 300 steps.

Three longer runs on the same day, to see whether easing rotation in helps:

| Direct mode with | Steps | Bit accuracy, training | Eval |
|---|---|---|---|
| `ramp_with_strength`: rotation and size range grow from nothing to full over the first 700 steps | 900 | 0.507 | 0.506 |
| rotation within 30 degrees of the four right angles (`rotation_near_right_angles_deg: 30`), from step 0 | 900 | 0.515 | 0.520 |
| `ramp_with_strength` after 300 upright steps: growing over steps 300 to 800 | 1,000 | 0.499 | 0.501 |

The third is the telling one. Upright it reached 0.833 by step 300. As
rotation and size range grew it fell to 0.595 by step 400 (rotation up to 72
degrees) and to chance by step 500 (up to 144 degrees), and did not recover.
It learns the upright case and loses it as soon as the tile turns.

**Verdict: this does not justify launching the pod.** Every test with
rotation ends at chance, including the restricted and the eased-in ones. A
pod run with the current config would most likely be stopped by its
30-minute check. These are short runs with small models on generated images,
one run each, so they do not prove a long run cannot learn; but nothing here
points that way.

A likely reason, not established: the code is made of stripes at set
orientations and the reader's texture filters are tied to those orientations.
A turned tile presents each bit's stripes at an angle the reader has learned
for another bit or for none, and nothing in the tile says which way is up.
If so, direct mode needs a change of design (an orientation cue in the mark,
or a reader built to be indifferent to rotation), not more training time.
The flattened route does not have the problem: the locator hands the reader
an upright patch.

## contract.json

New bundles carry a `camera` block: tile size, stream size and orientation,
the trained range, the measurements and the starting band.
`docs/model-contract.md` explains it. Older bundles were left as they are.

Current wiring bundle: **`20261121-e5b5090`**. Random weights, flattened
reader, range 154 to 340, exported from commit `e5b5090` with no uncommitted
changes (`git_dirty: false`). It replaces `20261121-8f13ea4`, which was
exported from uncommitted code with the earlier 150 low end and has been
deleted.
