# Training runs

What the training runs are set up for, what was checked before them, and what
has been run. Kept by whoever works in `/training`. Newest settings first.

## Status, 2026-11-21

| Run | State |
|---|---|
| Penalty-only run (laptop) | **Resumed** on 2026-11-21 from its step 10,000 checkpoint. It had stopped itself there on a limit that could not tell the corner patch from a mark that follows the image. The limit is now on the fixed-place measure (4.0, on 256 images); a mark that follows the image is accepted. See "Penalty-only run". |
| Long run (main reader, flattened, laptop) | **Cancelled** on 2026-11-21 at about step 26,500 of 140,000. It read well (bit accuracy 0.95 to 0.96) but had put the mark into one bright patch at the tile corner. Last checkpoint: step 26,000, kept as the fallback. See "Long run" and "Corner patch". |
| Spread run (RunPod) | **Not started.** `configs/runpod_spread.yaml`: the long run's settings with the two options against the corner patch on. Checked on the CPU only. See "Corner patch". |
| Sanity run | **Done**, one hour on the laptop GPU with the settings below. It learns; results in "Sanity run". No run has used an older camera range. |
| Direct reader | **Dropped for the gate.** Every smoke test with rotation ends at chance; no pod run was made. See "Direct mode". |
| Smoke tests | Run on 2026-11-21 with the new settings; results below. |

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
| Mark piled up (`spread.stop_above: 3.0`) | strongest spot above 3 on the 16 fixed images, at any sample sheet | 4 |
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
