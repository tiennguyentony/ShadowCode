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
measure what they claim and pass gradients.

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

**Dataset.** The chosen dataset is Open Images V7, validation split: first
16,000 images, 4.94 GB, images listed as CC BY 2.0 (`THIRD_PARTY.md`).
`fetch_dataset.py` downloads it and can be run again to continue:

```bash
python fetch_dataset.py --out data/open-images-v7-val
```

The dataset path is the config value `data.train_dir`. Any folder of images
works (searched recursively).

```bash
python train.py --config configs/default.yaml data.train_dir=data/open-images-v7-val/images
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
(means over those steps) and the cost so far. The cost needs the machine's
price:

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
sample sheet a run also measures the mark on 16 fixed dataset images (the
spread check): one `[spread]` line and one row in `<log.dir>/spread.csv`,
with the strongest spot as a multiple of the tile average (1 is even, the
long run's corner patch is 20) and the share of the mark's energy in one
96 px square. `spread.stop_above` stops the run when the strongest spot
passes a limit (exit code 4); `log.spread_markdown` keeps the rows as a table
in a Markdown file.

`configs/penalty.yaml` is the long run with the penalty on and the hiding
off, resumed from the step 6,000 checkpoint; `/docs/runs.md`, "Penalty-only
run", has its table.

```bash
python train.py --config configs/penalty.yaml --keep-awake
```

```bash
python smoke_test.py spread.peak_penalty.enabled=true spread.hide.enabled=true
```

**The pod run and the learning check.** `configs/runpod_spread.yaml` is the
long run with both switches on and paths on a RunPod network volume; see
`RUNPOD.md`. It turns on `early_stop`: 45 minutes into the run, if mean bit
accuracy is below 0.75, the run saves a checkpoint and exits with code 3. It
also turns on the two other stops in the table below: the mark piling up
(`spread.stop_above`, exit code 4) and the image loss never switching on
(`image_loss_check`, exit code 5).

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
`configs/long.yaml`, `configs/penalty.yaml`, `configs/runpod_spread.yaml` and
`configs/runpod_direct.yaml` inherit from it.
The parts that need a decision:

| Key | Meaning |
|---|---|
| `demo` | The demo setup: 50 mm printed tile, 1080 x 1920 portrait phone stream, and the measurements. Checked against the range below and copied into `contract.json`. |
| `camera_px_per_tile` | Camera pixels per printed tile, seen straight on: `[154, 340]`, from measurements on a Galaxy S23 Ultra. Every sample draws its own value. One number fixes it. `null` skips the loss-of-resolution step and the starting band uses `band.fallback_step`. |
| `band.grating_step` | `auto`: the highest band whose finest stripes still get `band.min_camera_px_per_cycle` camera pixels per cycle at the low end of the range (see `band.py`). With 154 that is step 8, the top band: 16-51 cycles per tile, 3.01 pixels per cycle. One pixel lower and auto drops to step 6. `/docs/runs.md` has the check. |
| `simulation.frame_px` | The simulated frame: a window onto the surface at 384 pixels per tile, not the camera stream. One number is a square; `[width, height]` gives any shape. |
| `data.train_dir` | The dataset path. |
| `train.reader_input` | `flattened`: the reader gets one tile-sized patch flattened with the true homography, random offset, corner jitter. `direct`: no flatten; the reader gets a 384 pixel square of camera frame with the tile at camera size (`simulation.direct.scale: auto`, 154 to 340 pixels), at a random rotation, position and tilt. Not shown to learn: every smoke test with rotation ends at chance (`/docs/runs.md`, "Direct mode"). `simulation.direct.rotation_near_right_angles_deg` and `ramp_with_strength` restrict or ease in the rotation. |
| `spread.peak_penalty`, `spread.hide` | The two switches that keep the mark spread over the tile. Off by default, on in `runpod_spread.yaml`. |
| `early_stop` | The learning check. Off by default, on in the two `runpod_` configs and `penalty.yaml`. |
| `spread.stop_above` | Stop, with a checkpoint and exit code 4, when the spread check finds the strongest spot above this. `null` (off) by default; 3.0 in `penalty.yaml` and `runpod_spread.yaml`. |
| `image_loss_check` | Stop, with a checkpoint and exit code 5, if the mean PSNR is still below `min_psnr_db` (14) at `at_step` (15,000): the image loss never switched on. Off by default, on in `runpod_spread.yaml`. |
| `schedule.image_loss` | Image-loss weight is 0 until bit accuracy passes `acc_threshold`, then ramps over `ramp_steps`. |
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
| `spread.py` | The peak penalty, the random hiding and the "strongest spot" measure |
| `train.py` | Training step, schedules, logging, checkpoints, learning check |
| `checkpoint.py` | Save, find newest, resume |
| `runlog.py` | Run time and cost across restarts, `progress.csv`, `spread.csv` |
| `fetch_dataset.py` | Downloads the training images |
| `setup_runpod.sh`, `RUNPOD.md` | Preparing and running a RunPod pod |
| `viz.py` | Sample sheets |
| `smoke_test.py` | Smoke test and structure checks |
| `export_bundle.py` | ONNX export, `contract.json` and fixtures for the browser |
| `THIRD_PARTY.md` | Borrowed designs, pretrained weights, datasets, libraries, with licences |

## Not built yet

- The locator and its training stage.
- Evaluation scripts. They need a Python port of the message codec
  (CRC-7 and BCH from `/docs/message-format.md`); training only counts bit errors.
