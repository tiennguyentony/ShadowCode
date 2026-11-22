# Training

PyTorch pipeline for the neural watermark: one repeating tile carries 64 bits.

Three models are planned. Stage 1, built here, trains two of them:

- **Embedder** (`models/embedder.py`): U-Net, circular padding in every conv, writes the bits into a 384 px tile as a seamless residual. Offline only.
- **Reader** (`models/reader.py`): small CNN (2.0M parameters), returns 64 logits.
- **Locator** (`models/locator.py`): not built. The file states the contract the rest of the code assumes.

`MESSAGE_BITS = 64` and `TILE_PX = 384` are fixed in `constants.py`. The bit
layout is in `/docs/message-format.md`; training uses random bits.

## Setup

Run everything from this folder.

```bash
cd training
python -m venv .venv
```

Activate it (PowerShell: `.venv\Scripts\Activate.ps1`, Git Bash: `source .venv/Scripts/activate`, macOS/Linux: `source .venv/bin/activate`), then:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

```bash
pip install -r requirements.txt
```

The first line is the CUDA 12.6 build for an NVIDIA GPU. On a Mac or a
CPU-only machine, skip it; `requirements.txt` installs the default build.

## Smoke test

300 steps on 24 generated images with the simulation off. Asserts bit
accuracy of at least 0.70 (chance is 0.50) in training and in eval mode,
prints the PSNR between marked and original tiles, and saves
`outputs/smoke_samples.png` (original | marked | residual x10). Before
training it runs structure checks, among them that the two `spread` options
measure what they claim and pass gradients, that the image-loss weight moves
both ways with `follow_accuracy`, that `data.center_fraction` keeps crops in
the middle, that photos load upright, and that held-out images stay out of
training.

```bash
python smoke_test.py
```

It uses CUDA if present, then MPS, then CPU. About 45 s on an RTX 4070
laptop GPU, about 10 minutes on CPU. To force a device:

```bash
python smoke_test.py --device cpu
```

The image loss is off for the whole smoke test (it only switches on at 95
percent bit accuracy), so the mark in the saved sheet is at full strength and
the PSNR is low (about 13 dB). That sheet shows the raw code, not what a
trained mark looks like.

Blur-only variants, used to compare starting bands:

```bash
python smoke_test.py --config configs/smoke_blur.yaml --sheet outputs/smoke_blur_step8.png band.grating_step=8
```

```bash
python smoke_test.py --config configs/smoke_blur.yaml --sheet outputs/smoke_blur_step4.png band.grating_step=4
```

Loss of resolution only, used to check a band against one camera resolution
(results in `/docs/runs.md`):

```bash
python smoke_test.py --config configs/smoke_resolution.yaml camera_px_per_tile=154
```

The literal version of the spec (plain dense init, plain CNN reader, batch
statistics) still runs; it does not learn within 300 steps:

```bash
python smoke_test.py model.embedder.message_init=dense model.reader.texture=null model.reader.pool_norm=batch
```

## Training

**Dataset.** From 2026-11-21 the models train on product labels: Open Food
Facts front-of-pack photos, one per product, CC BY-SA 3.0. Read
`THIRD_PARTY.md` before showing any of them, or anything made from them, to
someone else: the photos need a credit, and the label artwork in them belongs
to third parties. `fetch_labels.py` downloads 16,000 (13.7 GB) from the AWS
bucket Open Food Facts names for bulk use and can be run again to continue.
It needs `pyarrow`, which `requirements.txt` installs.

```bash
python fetch_labels.py --out data/open-food-facts
```

```bash
python fetch_labels.py --out data/open-food-facts --leave-out-over 24
```

The second command deletes the photos over 24 megapixels (50 of the 16,000)
and takes them out of `ATTRIBUTION.csv`; 15,950 remain. `/docs/runs.md`,
"Label run", has the counts.

The dataset path is the config value `data.train_dir`. Any folder of images
works (searched recursively). Photos are loaded upright: the rotation tag a
phone writes is applied. Three more `data` settings, all off by default:

- `data.center_fraction`: the random crop draws only from the middle of each
  image, this fraction of the width and of the height. `labels.yaml` uses
  0.6, because the photos show a table, a hand or a shelf around the package.
- `data.holdout`: this many images are kept back from training, the same ones
  every time. `acceptance.py` measures on them.
- `data.credit`: a line printed below every sheet of dataset photos.

To look at what the models will be shown, 64 training crops on one sheet,
with a list of the files beside it:

```bash
python data.py --config configs/labels.yaml --sheet runs/labels/crops_preview.png
```

The runs up to 2026-11-21 used Open Images V7 (validation split, first
16,000 images, listed as CC BY 2.0). `fetch_dataset.py` still downloads it;
the copy on the laptop was deleted when the labels replaced it.

```bash
python train.py --config configs/default.yaml data.train_dir=data/open-food-facts/images
```

Anything in the config can be overridden on the command line as
`dotted.key=value`, for example:

```bash
python train.py data.train_dir=D:/path/to/images train.reader_input=direct checkpoint.dir=checkpoints/direct log.dir=runs/direct
```

**Resume.** Checkpoints go to `checkpoint.dir` every `checkpoint.every`
steps, and once more when the run is stopped with Ctrl-C or SIGTERM. Running
the same command again continues from the newest one; if that file cannot be
read, the one before it is used. To start over in the same directory (the old
checkpoints are moved to a `replaced-...` subfolder, not deleted):

```bash
python train.py data.train_dir=D:/path/to/images --fresh
```

**Sanity run.** One hour of the long run's own settings on the laptop GPU,
in its own folders. It prints the GPU name and free memory first and stops
if it is not on the RTX 4070 (`train.expect_gpu`); it never falls back to the
CPU. `--max-minutes` stops any run after that much training, with a
checkpoint, eval-mode metrics and a sample sheet in `<log.dir>/samples/`.
Results are in `/docs/runs.md`.

```bash
python train.py --config configs/sanity.yaml --max-minutes 60
```

**Long run.** `configs/long.yaml` is `default.yaml` with the dataset path,
its own folders (`runs/long`, `checkpoints/long`), the RTX 4070 requirement
and a sample sheet every 5,000 steps (`log.sheet_every`). `--keep-awake`
keeps a Windows machine from idle-sleeping while the run lives. The same
command resumes after a stop. `/docs/runs.md` records how it was started.

```bash
python train.py --config configs/long.yaml --keep-awake
```

**Progress and cost.** Every `log.report_every` steps (1,000) one line is
printed and appended to `<log.dir>/progress.csv`: step, bit accuracy and PSNR
(means over those steps), the image-loss weight at that step, and the cost so
far. The cost needs the machine's price:

```bash
python train.py data.train_dir=D:/path/to/images --price-per-hour 0.50
```

It is that price times the time `train.py` has run on this run, summed over
restarts (`<log.dir>/run_state.json`). It is an estimate, not a bill.

**Keeping the mark spread.** Left alone, the image loss lets the embedder
put the whole code into one bright patch at the tile corner; the long run did
(`/docs/runs.md`, "Corner patch"). Two switches in the config section
`spread` work against that, both off by default (`spread.py`):

- `spread.peak_penalty`: a loss on the strongest local region of the mark
  (the mean energy of the strongest 2 percent of 48 pixel windows, windows
  wrapping around the tile). It follows the image-loss schedule.
- `spread.hide`: before the reader sees its patch in a training step, random
  regions covering up to half of it are blanked with the patch's mean colour.

The smoke test checks that they run and that gradients flow. With every
sample sheet a run also measures the mark on 256 fixed dataset images (the
spread check): one `[spread]` line and one row in `<log.dir>/spread.csv`.
Its main number is "fixed place": high only for a patch at the same place in
every tile (the long run's corner patch: 18 to 20), low for a mark that
follows the image content (1.4 to 2.7). It also logs the strongest spot per
image, the share of the mark's energy in one 96 px square and the share on
the busier half of the image. `spread.stop_above` stops the run when "fixed
place" passes a limit (exit code 4); `log.spread_markdown` keeps the rows as
a table in a Markdown file. `/docs/runs.md` has the baseline behind the
limit of 4.

`configs/penalty.yaml` is the long run with the penalty on and the hiding
off, resumed from the step 6,000 checkpoint; `/docs/runs.md`, "Penalty-only
run", has its table.

```bash
python train.py --config configs/penalty.yaml --keep-awake
```

```bash
python smoke_test.py spread.peak_penalty.enabled=true spread.hide.enabled=true
```

**Label run.** `configs/labels.yaml` is the penalty-only run's settings on
the label photos, from step 0, with one change to the schedule. On its fixed
ramp the penalty-only run lost bit accuracy (0.964 to 0.895) as the image
loss went to full weight. With `schedule.image_loss.follow_accuracy:
{floor: 0.95, band: 0.01}` the weight is no longer tied to the step count:
after the trigger it goes up by 1/`ramp_steps` per step while the smoothed
bit accuracy is at or above 0.95, down by the same while it is below 0.94,
and holds in between. The peak penalty uses the same weight. The weight is
saved in checkpoints and logged (`schedule/image_loss_scale` in TensorBoard,
`image_scale` in `progress.csv`). `/docs/runs.md`, "Label run", has the
results.

```bash
python train.py --config configs/labels.yaml --max-minutes 120 --keep-awake
```

**Acceptance.** `acceptance.py` answers two questions about a checkpoint: is
the mark read, and can it be seen. On the 256 held-out label images, in eval
mode, with the simulation at full strength and a flattened patch for the
reader, it reports for alpha 0.30, 0.25, 0.20 and 0.15:

- the share of tiles with at most 4 wrong bits among the first 63, which is
  what the BCH code corrects (PASS at 0.80 or more), the mean bit accuracy
  and the share with all 64 bits right;
- the fixed-place measure (PASS below 2.5);
- PSNR in dB, mean and worst tenth of the tiles (no pass line).

A second table reads each tile from several shots: the same marked tile is
photographed 9 times by the simulation, each time with fresh random tilt,
blur, colour, resolution, noise and JPEG, and for K = 1, 3, 5 and 9 the
reader's logits over the first K shots are averaged before the bits are
taken (`--shots`, empty to skip). This is simulated: real frames from one
hand-held scan are less independent than these shots, so they gain less.

It counts bit errors; it does not run the message codec. It writes
`runs/acceptance/<step>/results.md` and `blind.html`: 20 pairs of original
and marked tile side by side, 50 mm wide each in print, left and right
shuffled, 10 with busy content and 10 with large flat areas. Which side is
marked is in `blind_answers.json` and is not printed. Images in
`data/labels/`, if there are any, are measured and shown as a second section.
`blind.html` shows dataset photos with their credit line and is for private
review only.

```bash
python acceptance.py --config configs/labels.yaml
```

Never point it at the checkpoint folder of a run that is still training:
copy the checkpoint elsewhere and pass the copy with `--checkpoint`.

**The pod run and the learning check.** `configs/runpod_spread.yaml` is the
long run with both switches on and paths on a RunPod network volume; see
`RUNPOD.md`. It turns on `early_stop`: 45 minutes into the run, if mean bit
accuracy is below 0.75, the run saves a checkpoint and exits with code 3. It
also turns on the three other stops in the table below: the mark piling up
(`spread.stop_above`, exit code 4), the image loss never switching on
(`image_loss_check`, exit code 5) and 24 hours of training
(`budget.max_hours`, exit code 6).

**Direct reader.** `configs/runpod_direct.yaml` trains with
`train.reader_input: direct`. Dropped for the gate: every test with rotation
ended at chance (`/docs/runs.md`, "Direct mode"). Its check is at 30 minutes
and 0.55 (chance is 0.50).

**TensorBoard.**

```bash
tensorboard --logdir runs
```

Logged: bit accuracy, full-message decode rate (all 64 bits right, and
"within BCH reach": at most 4 errors in the 63-bit codeword), PSNR, the
losses, both schedules, and sample images (original, marked, residual x10,
reader input).

**LPIPS** downloads its AlexNet backbone (about 233 MB) on the first run. See
`THIRD_PARTY.md` for the licence position of those weights.

**GPU memory.** The default batch of 8 peaks at about 4.3 GB with every
simulation step on, which fits an 8 GB card.

## Export for the browser

```bash
python export_bundle.py checkpoints/stage1/step_00140000.pt
```

```bash
python export_bundle.py --untrained
```

Writes `export/<date>-<short git hash>/` with `reader.onnx`, `contract.json`
and `fixtures/` (five portrait camera-like frames, each with its corners, flattened
patch, expected logits and bits). `--untrained` uses random weights and is for
wiring the browser only. With a checkpoint, every setting comes from the
config saved in the checkpoint.

The script runs the ONNX file in ONNX Runtime against PyTorch on the same
inputs and stops before writing `contract.json` if they differ by more than
1e-4. `/docs/model-contract.md` explains `contract.json`.

`locator.onnx` is added once the locator exists; that part has never run.

## The config

Everything is in `configs/default.yaml`; `configs/smoke.yaml`,
`configs/smoke_blur.yaml`, `configs/smoke_resolution.yaml`, `configs/sanity.yaml`,
`configs/long.yaml`, `configs/penalty.yaml`, `configs/labels.yaml`,
`configs/runpod_spread.yaml` and `configs/runpod_direct.yaml` inherit from it.
The parts that need a decision:

| Key | Meaning |
|---|---|
| `demo` | The demo setup: 50 mm printed tile, 1080 x 1920 portrait phone stream, and the measurements. Checked against the range below and copied into `contract.json`. |
| `camera_px_per_tile` | Camera pixels per printed tile, seen straight on: `[154, 340]`, from measurements on a Galaxy S23 Ultra. Every sample draws its own value. One number fixes it. `null` skips the loss-of-resolution step and the starting band uses `band.fallback_step`. |
| `band.grating_step` | `auto`: the highest band whose finest stripes still get `band.min_camera_px_per_cycle` camera pixels per cycle at the low end of the range (see `band.py`). With 154 that is step 8, the top band: 16-51 cycles per tile, 3.01 pixels per cycle. One pixel lower and auto drops to step 6. `/docs/runs.md` has the check. |
| `simulation.frame_px` | The simulated frame: a window onto the surface at 384 pixels per tile, not the camera stream. One number is a square; `[width, height]` gives any shape. |
| `data.train_dir` | The dataset path. |
| `data.center_fraction` | Part of each image the random crop may draw from: this fraction of the width and of the height, centred. 1.0 (the whole image) by default; 0.6 in `labels.yaml`. |
| `data.holdout` | Images kept back from training, picked by a fixed seeded shuffle, for `acceptance.py`. 0 by default; 256 in `labels.yaml`. |
| `data.credit` | A line of text printed below every sheet of dataset photos. `null` by default; the Open Food Facts credit in `labels.yaml`. |
| `train.reader_input` | `flattened`: the reader gets one tile-sized patch flattened with the true homography, random offset, corner jitter. `direct`: no flatten; the reader gets a 384 pixel square of camera frame with the tile at camera size (`simulation.direct.scale: auto`, 154 to 340 pixels), at a random rotation, position and tilt. Not shown to learn: every smoke test with rotation ends at chance (`/docs/runs.md`, "Direct mode"). `simulation.direct.rotation_near_right_angles_deg` and `ramp_with_strength` restrict or ease in the rotation. |
| `spread.peak_penalty`, `spread.hide` | The two switches that keep the mark spread over the tile. Off by default, on in `runpod_spread.yaml`. |
| `early_stop` | The learning check. Off by default, on in the two `runpod_` configs, `penalty.yaml` and `labels.yaml`. |
| `spread.stop_above` | Stop, with a checkpoint and exit code 4, when the spread check's fixed-place measure goes above this: a patch at the same place in every tile. `null` (off) by default; 4.0 in `penalty.yaml`, `labels.yaml` and `runpod_spread.yaml`. |
| `image_loss_check` | Stop, with a checkpoint and exit code 5, if the mean PSNR is still below `min_psnr_db` (14) at `at_step` (15,000): the image loss never switched on. Off by default, on in `runpod_spread.yaml`. |
| `budget` | A limit on the whole run, summed over restarts: stop, with a checkpoint and exit code 6, at `max_hours` of training or at `max_cost_usd` on the cost of the report line (needs `--price-per-hour`). A run stopped this way does not resume until the limit is raised or set to `null`. Both `null` (off) by default; `max_hours: 24` in `runpod_spread.yaml`. It limits what `train.py` counts, not the bill. |
| `schedule.image_loss` | Image-loss weight is 0 until bit accuracy passes `acc_threshold`, then ramps over `ramp_steps`. |
| `schedule.image_loss.follow_accuracy` | `null`: that ramp, which never comes back down. `{floor, band}`: after the trigger the weight moves by 1/`ramp_steps` per step, up while smoothed bit accuracy is at or above `floor`, down while it is below `floor - band`. On in `labels.yaml` (0.95, 0.01). |
| `schedule.sim_strength` | Simulation strength ramps from 0 to `max_strength` over `ramp_steps`. |
| `simulation.*` | One range per effect, each given at strength 1. `simulation.effects` switches single effects off. |
| `model.embedder.message_init`, `model.reader.texture`, `model.reader.pool_norm` | The three defaults that go beyond the literal spec, each with its switch (`dense`, `null`, `batch`). |

A checkpoint remembers its band. If `camera_px_per_tile` changes the derived
band mid-run, resume stops with a message instead of loading mismatched weights.

## Files

| File | Contents |
|---|---|
| `constants.py` | `MESSAGE_BITS`, `TILE_PX` |
| `camera.py` | The camera range, the demo measurements and the simulated frame size, read from the config |
| `band.py` | Starting band, checked against `camera_px_per_tile` |
| `models/` | Embedder, reader, locator slot |
| `simulate.py` | Print-and-camera simulation; returns the frame and the exact homography |
| `geometry.py` | Homographies, the flatten step |
| `losses.py` | LPIPS wrapper |
| `spread.py` | The peak penalty, the random hiding and the spread check (fixed place, strongest spot) |
| `train.py` | Training step, schedules, logging, checkpoints, learning check |
| `checkpoint.py` | Save, find newest, resume |
| `runlog.py` | Run time and cost across restarts, `progress.csv`, `spread.csv` |
| `data.py` | The image loader (upright loading, centre crop, holdout) and the crops preview |
| `fetch_labels.py` | Downloads the label images (Open Food Facts) and writes their attribution file |
| `fetch_dataset.py` | Downloads the earlier training images (Open Images) |
| `acceptance.py` | Acceptance numbers on held-out labels and the blind comparison page |
| `setup_runpod.sh`, `RUNPOD.md` | Preparing and running a RunPod pod |
| `viz.py` | Sample sheets |
| `smoke_test.py` | Smoke test and structure checks |
| `export_bundle.py` | ONNX export, `contract.json` and fixtures for the browser |
| `THIRD_PARTY.md` | Borrowed designs, pretrained weights, datasets, libraries, with licences |

## Not built yet

- The locator and its training stage.
- A Python port of the message codec (CRC-7 and BCH from
  `/docs/message-format.md`). Training and `acceptance.py` only count bit
  errors: "at most 4 wrong of the first 63" stands in for a decoded message.
