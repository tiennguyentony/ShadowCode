# Training runs

What the training runs are set up for, what was checked before them, and what
has been run. Kept by whoever works in `/training`. Newest settings first.

## Status, 2026-11-21

| Run | State |
|---|---|
| Long run (main reader, flattened, laptop) | **Running** since 2026-11-21 03:43, resumed from the sanity run at step 7,001. It reads well (bit accuracy 0.95 to 0.96) but has put the mark into one bright patch at the tile corner. See "Long run" and "Corner patch". |
| Spread run (RunPod) | **Not started.** `configs/runpod_spread.yaml`: the long run's settings with the two options against the corner patch on. Checked on the CPU only. See "Corner patch". |
| Sanity run | **Done**, one hour on the laptop GPU with the settings below. It learns; results in "Sanity run". No run has used an older camera range. |
| Direct reader | **Dropped for the gate.** Every smoke test with rotation ends at chance; no pod run was made. See "Direct mode". |
| Smoke tests | Run on 2026-11-21 with the new settings; results below. |

## Long run

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
   | plain smoke test, for comparison (GPU, from the table further down) | 0.852 | 0.833 |

   So the reader still learns from the start with part of its patch hidden.
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
run's settings with both options on, a sample sheet every 2,000 steps, and a
check 45 minutes in that stops the run if bit accuracy is under 0.75.
`training/RUNPOD.md` has the commands.

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
