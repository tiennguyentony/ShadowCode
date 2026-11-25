# Eye-Model Loss and Closer Scanning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a mark that can't be seen on a printed 50 mm tile at arm's length and still reads from a phone at 19–28 cm, by replacing the L2/LPIPS image loss with an S-CIELAB visibility loss steered to a target calibrated on paper.

**Architecture:**
- A new module, `training/visibility.py`, measures how visible the mark is (S-CIELAB at 67 px per degree, with masking).
- `Trainer` gets a second image-loss controller, `follow_visibility`, that scales that loss until the mark is under a target.
- The spread check, `acceptance.py` (now with correlated frames), two print-and-capture scripts, and the new configs all build on the module.
- The pod run is the last step, after a paper calibration, a phone check and a laptop probe.

**Tech Stack:** Python 3, PyTorch (`training/.venv`), the repo's own check functions in `training/smoke_test.py`, YAML configs with `base:` inheritance, TypeScript + Vitest in `web/`.

**Spec:** `docs/superpowers/specs/2026-11-24-eye-model-loss-design.md` (approved 2026-11-24). Read it first; this plan argues from it.

## Global Constraints

- Work on branch `training/eye-model-loss`. Commit there; never merge into or push `master` (the user merges).
- Run Python with `training/.venv/Scripts/python.exe` from `training/`. The system Python has no PyTorch.
- Never open a checkpoint of a run that is still training. Copy it elsewhere first and read the copy.
- Viewing geometry: tile 50 mm (`demo.tile_mm`), viewing distance 500 mm, so **67.1 pixels per degree** for the 384 px tile.
- Camera range for the new run: `camera_px_per_tile: [231, 340]`, `band.max_step: 12` (auto picks step 12: 24 to 76.8 cycles per tile, `message_grid` 192).
- Masking floor: sqrt(0.25 / 1.5) = **0.408** unless the calibration raises it.
- Controller: quantile **0.9** over the batch's per-tile values, EMA **0.98**, up above target, down below **0.9 x target**, rate **4.6e-4**, weight bounds **[0.01, 1000]**.
- Pass lines:
  - decodable (at most 4 wrong of the first 63 bits) at least **0.80** with **3 correlated frames** averaged;
  - at least **95 percent** of held-out tiles at or under the target;
  - fixed place under **2.5**;
  - printed blind test **14 of 20 right or fewer**.
- Reading check at step **80,000**: stop if the 1-frame training bit accuracy (mean of the last 1,000 steps) is below **0.80**.
- Calibration levels (95th-percentile masked ΔE): **0.5, 1, 2, 3, 5, 8**. Calibration gratings at **23 and 74 cycles per tile** (about 4 and 13 cycles per degree). Phone gratings at **24, 38, 51, 77 cycles per tile**.
- Run name `eye_model`, price $0.69 per hour, 140,000 steps from step 0.
- New behaviour is off by default in `configs/default.yaml`; old configs and checkpoints must still load and train as before.

**One change from the spec, flagged for the user:** the spec says the controller weight starts at 1. Here it starts at **0.01** (`init_weight`). The mean masked ΔE of today's mark is several units against a bit loss of about 0.12, so at 1 the first steps would wipe out reading before the controller can react. Starting low lets it climb to the weight that reaches the target.

## Review Focus

- **Old checkpoints load into the new config.** The probe resumes the 140,000-step checkpoint, which has no controller state; the weight must start at `init_weight` and the EMA at the first batch's value (test in Task 2).
- **Bad settings fail at start.** `follow_visibility` with `target: null`, or set together with `follow_accuracy`, must raise at `Trainer` construction, not at the trigger 20,000 steps in (Task 2).
- **No NaN on a zero mark or a flat tile.** ΔE's square root and the Lab cube root must give finite gradients when marked equals clean, on pure white, black and flat-colour tiles (Task 1).
- **A resume restores the controller exactly.** Weight and EMA are saved in checkpoints and restored to the last digit (Task 2).
- **Logs with the new columns still feed `notify.py` and resume.** `progress.csv` and `spread.csv` get columns appended at the end. A run directory with the old header must keep refusing to mix headers as it does today, and `notify.py` must still read a new file (Tasks 3 and 4).

---

### Task 1: The visibility measure

**Files:**
- Create: `training/visibility.py`
- Modify: `training/smoke_test.py` (new `check_visibility()`, called from `run_smoke`)
- Modify: `training/configs/default.yaml` (new `loss.visibility` block, off)

**Interfaces:**
- Produces:
  - `pixels_per_degree(tile_mm: float, distance_mm: float, tile_px: int = TILE_PX) -> float`;
  - `visibility_settings(cfg: dict) -> dict` (merged defaults, checked; keys `enabled`, `distance_mm`, `masking_floor`);
  - `delta_e_map(clean: Tensor, marked: Tensor, ppd: float) -> Tensor` (B,1,H,W);
  - `masking_factor(clean: Tensor, cfg: dict) -> Tensor` (B,1,H,W) in [floor, 1];
  - `tile_visibility(clean: Tensor, marked: Tensor, cfg: dict) -> dict[str, Tensor]` with `"map"` (B,1,H,W) masked ΔE, `"mean"` (B,) and `"p95"` (B,);
  - `batch_statistic(p95: Tensor, quantile: float = 0.9) -> Tensor` (scalar);
  - `opponent(rgb: Tensor) -> Tensor` (B,3,H,W) linear-light opponent channels (used by Task 4).

- [ ] **Step 1: Write `check_visibility()` in `smoke_test.py`.** It needs no trainer. Build tiles at 384 px from a mid-grey and a flat label colour, and assert:
  - `abs(pixels_per_degree(50, 500) - 67.1) < 0.1`;
  - each of the three S-CIELAB filters has DC gain 1 within 1e-4;
  - at equal RGB amplitude (0.02), a luminance cosine at 18 cycles per tile scores a higher mean ΔE than one at 74 cycles per tile, by at least 3x;
  - at the same spacing (23 cycles per tile) and equal opponent-space amplitude, blue-yellow scores lower than luminance;
  - the same residual pasted on a flat tile scores higher than on a busy tile (text from `labelgen`): masking works;
  - `masking_factor` is 1 on perfectly flat colour and never below 0.408 with default settings;
  - marked == clean gives 0 and a finite gradient; also on pure white and pure black tiles;
  - `tile_visibility(...)["mean"].sum().backward()` reaches the residual with a nonzero, finite gradient;
  - `tile_visibility` is the same when the tile is rolled by 100 px. Circular filtering must make the tile edges seamless.
- [ ] **Step 2: Run it and watch it fail.** `.venv/Scripts/python.exe -c "from smoke_test import check_visibility; check_visibility()"` → ImportError on `visibility`.
- [ ] **Step 3: Implement `training/visibility.py`.**
  - sRGB → linear (IEC 61966-2-1) → XYZ (D65 sRGB matrix) → S-CIELAB opponent space (Poirson and Wandell): O1 = 0.279X + 0.72Y − 0.107Z, O2 = −0.449X + 0.29Y − 0.077Z, O3 = 0.086X − 0.59Y + 0.501Z.
  - Spatial filters: sums of Gaussians exp(−r²/s²), each Gaussian normalised to sum 1, with s in degrees times ppd:
    - luminance: weights 1.00327, 0.114416, −0.117686 at spreads 0.05, 0.225, 7.0;
    - red-green: 0.616725, 0.383275 at 0.0685, 0.826;
    - blue-yellow: 0.567885, 0.432115 at 0.0920, 0.6451.

    **Check these constants against Zhang and Wandell (1996) / the reference `scielab` code before relying on them.**
  - Build each kernel on the tile grid with wrapped distances, `torch.fft.fft2` it once per (size, ppd, device), and filter by multiplying spectra.
  - Back through the inverse opponent matrix to XYZ, then CIELAB with white (0.9505, 1.0, 1.089) and the standard linear segment below (6/29)³. ΔE76 = sqrt(dL² + da² + db² + 1e-8).
  - Masking: `sqrt(texture_weights(clean, cfg_with_mask_on) / flat_weight)`, clamped below at `masking_floor` (null means sqrt(busy/flat)).
  - `p95` is `torch.quantile` over each tile's pixels.
  - Add to `default.yaml`, under `loss:`, `visibility: {enabled: false, distance_mm: 500, masking_floor: null}`. `tile_mm` comes from `demo.tile_mm`.
- [ ] **Step 4: Run it and watch it pass.** Same command → prints a "visibility checks passed" line. If the 18 vs 74 cycle ratio is under 3x, recheck the filter spreads before touching the threshold.
- [ ] **Step 5: Commit.** `git add training/visibility.py training/smoke_test.py training/configs/default.yaml` and `git commit -m "feat(training): S-CIELAB visibility measure with masking"`.

### Task 2: The `follow_visibility` controller

**Files:**
- Modify: `training/train.py` (`Trainer.__init__`, new `visibility_follow_settings()`, `update_schedule` → also `update_visibility()`)
- Modify: `training/checkpoint.py` (save and restore `vis_weight`, `vis_ema`)
- Modify: `training/configs/default.yaml` (`schedule.image_loss.follow_visibility: null`)
- Modify: `training/smoke_test.py` (new `check_follow_visibility(trainer)`)

**Interfaces:**
- Consumes: nothing from Task 1. The controller takes a float.
- Produces:
  - `Trainer.vis_weight: float` and `Trainer.vis_ema: float | None`;
  - `Trainer.visibility_follow_settings() -> dict | None` (keys `target, quantile, ema, low, rate, init_weight, min_weight, max_weight`);
  - `Trainer.update_visibility(batch_value: float) -> None`.
  - `image_loss_scale()` is unchanged: it returns the plain 0 to 1 ramp when `follow_accuracy` is null.

- [ ] **Step 1: Write `check_follow_visibility(trainer)`,** modelled on `check_follow_accuracy`. Drive `update_visibility` with made-up values on a copy, with `ema: 0` so each value is its own EMA, target 2, rate 0.1, init 0.01. Assert:
  - nothing moves before the trigger (`image_ramp_start is None`);
  - after it, the weight goes ×1.1 per step while the value is 3 and ÷1.1 per step while it is 1.7 (below 0.9 × 2 = 1.8);
  - it holds at 1.9;
  - it is clamped at 0.01 and 1000;
  - with `ema: 0.98`, the first value sets the EMA (no pull toward 0);
  - a checkpoint saved and loaded restores `vis_weight` and `vis_ema` exactly;
  - loading a checkpoint saved without those keys gives `init_weight` and `None`;
  - `Trainer` raises `ValueError` when `follow_visibility.target` is null, and when `follow_accuracy` and `follow_visibility` are both set.
- [ ] **Step 2: Run it and watch it fail.** `.venv/Scripts/python.exe -c "import smoke_test as s, torch; from config import load_config; from train import Trainer; s.check_follow_visibility(Trainer(load_config('configs/smoke.yaml'), torch.device('cpu')))"` → AttributeError.
- [ ] **Step 3: Implement it.**
  - Defaults when the block is set: `quantile 0.9, ema 0.98, low 0.9, rate 4.6e-4, init_weight 0.01, min_weight 0.01, max_weight 1000`.
  - Call `update_visibility` from `train_step` after `update_schedule`, only when the image loss is on.
  - The settings are validated in `__init__`, next to `follow_settings()`.
  - Document the block in `default.yaml` in the style of `follow_accuracy`.
- [ ] **Step 4: Run it and watch it pass.** Same command → "visibility schedule checks passed".
- [ ] **Step 5: Commit.** `git commit -m "feat(training): follow_visibility controller, saved in checkpoints"`.

### Task 3: Wire the visibility loss into training and the logs

**Files:**
- Modify: `training/train.py` (`train_step`, `evaluate`, the console line, TensorBoard names, the progress row)
- Modify: `training/runlog.py` (`PROGRESS_COLUMNS` + `visibility`, `vis_weight` at the end; `ProgressLog.write` takes them)
- Modify: `training/notify.py` (show the two values in the progress email if present)
- Modify: `training/smoke_test.py` (extend `check_follow_visibility` or add `check_visibility_step(trainer)`)

**Interfaces:**
- Consumes: `tile_visibility`, `batch_statistic`, `visibility_settings` (Task 1); `vis_weight`, `update_visibility` (Task 2).
- Produces:
  - `train_step` metrics `visibility` (the batch statistic, float), `visibility_mean` (float) and `vis_weight` (float), whenever `loss.visibility.enabled`;
  - `evaluate` metrics `visibility`.

- [ ] **Step 1: Write the check.** Run one `train_step` on a smoke trainer with `loss.visibility.enabled=true`, `l2_weight=0`, `lpips_weight=0`, the image loss forced on (`image_ramp_start = 0`, `step` past the ramp) and `follow_visibility` set. Assert:
  - the metrics hold finite `visibility`, `visibility_mean` and `vis_weight`;
  - the embedder's gradient from that step is nonzero, and it changes when `vis_weight` changes;
  - with `loss.visibility.enabled=false` the step's loss equals the pre-change formula, so old configs train as before;
  - a `progress.csv` written with the new columns is read by `notify.py`'s parser without error and shows the two values;
  - `ProgressLog` opened on a directory whose file has the old header behaves as it does today for a changed header.
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Implement it.**
  - The image loss is `ramp × vis_weight × mean(tile_visibility(tiles, out["marked"], cfg)["mean"])`, where `ramp` is `image_loss_scale()`.
  - The peak penalty keeps `ramp × peak weight` and is not multiplied by `vis_weight`.
  - `update_visibility` gets `batch_statistic(p95, quantile)`.
  - Console line: add `seen {visibility:.2f}  w {vis_weight:.3g}`.
  - TensorBoard: `image/visibility`, `schedule/vis_weight`.
  - Progress row: the mean `visibility` over the report window and the last `vis_weight`; columns appended at the end of `PROGRESS_COLUMNS`.
- [ ] **Step 4: Run the check, then the whole smoke test.** `.venv/Scripts/python.exe smoke_test.py` → PASS, every existing check unchanged.
- [ ] **Step 5: Commit.** `git commit -m "feat(training): visibility loss in the training step, logged"`.

### Task 4: Spread check columns

**Files:**
- Modify: `training/spread.py` (`spread_report` gains `visibility_p90`, `visibility_mean`, `mark_cycles`, `colour_share`)
- Modify: `training/runlog.py` (`SPREAD_COLUMNS` appended; `SpreadLog.write` prints them)
- Modify: `training/train.py` (TensorBoard `image/visibility_fixed`, `image/mark_cycles`, `image/colour_share`)
- Modify: `training/smoke_test.py` (`check_spread` extended)

**Interfaces:**
- Consumes: `tile_visibility`, `opponent` (Task 1).
- Produces: `spread_report(...)` returns, in addition:
  - `visibility_p90`: the 90th percentile over images of per-image p95;
  - `visibility_mean`;
  - `mark_cycles`: the residual's energy-weighted mean radial frequency in cycles per tile, DC excluded;
  - `colour_share`: the share of the residual's opponent-space energy in O2 + O3.

  Only when `loss.visibility.enabled`; otherwise NaN.

- [ ] **Step 1: Extend `check_spread`** with synthetic embedders:
  - a residual that is a luminance cosine at 18 cycles per tile gives `mark_cycles` within 1 of 18 and `colour_share` under 0.1;
  - a blue-yellow cosine gives `colour_share` over 0.8;
  - the visibility numbers equal Task 1's functions applied directly.
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Implement it.**
- [ ] **Step 4: Run it and watch it pass.** `.venv/Scripts/python.exe smoke_test.py` → PASS.
- [ ] **Step 5: Commit.** `git commit -m "feat(training): spread check reports visibility, frequency and colour share"`.

### Task 5: Reading check

**Files:**
- Modify: `training/train.py` (config `reading_check: {enabled, at_step, min_bit_acc}`, `EXIT_READING_LOW = 7`, a one-look check modelled on `image_loss_check`, recorded in `run_state.json` as `reading_check`, a stop message)
- Modify: `training/configs/default.yaml` (`reading_check: null`, documented)
- Modify: `training/pod_run.sh`, `training/notify.py` (name exit code 7 wherever exit codes are named)

**Interfaces:**
- Produces: exit code 7 = "reading too weak at the reading check".

- [ ] **Step 1: Write a check** (in `smoke_test.py`, or a 20-step `train.py` run in a temp folder with `reading_check.at_step=10 reading_check.min_bit_acc=0.99`) that the run exits with 7, saves a checkpoint, writes `reading_check.result: failed`, and refuses to resume while the check is on, as `image_loss_check` does.
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Implement it.**
- [ ] **Step 4: Run it and watch it pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat(training): reading check, exit code 7"`.

### Task 6: Correlated frames and the new acceptance pass lines

**Files:**
- Modify: `training/simulate.py` (new `simulate_scan`)
- Modify: `training/acceptance.py` (`read_scans`, pass lines, report)
- Modify: `training/smoke_test.py` (`check_multi_shot` extended or new `check_scan()`)

**Interfaces:**
- Produces:
  - `simulate_scan(surface: Tensor, cfg: dict, strength: float, frames: int) -> list[SimResult]`. Pose (homography), blur kernel, colour shift and camera resolution are drawn once per sample and shared by all frames; sensor noise and JPEG quality are drawn per frame.
  - `read_scans(trainer, tiles, bits, alpha, frames, seed, chunk=8) -> tuple[Tensor, Tensor, Tensor]`, the same shapes as `read_shots`. Corner jitter is fresh per frame.
  - `MIN_DECODED` now applies to `vote(read_scans(..., frames=3))`.
  - New `MIN_UNDER_TARGET = 0.95`.

- [ ] **Step 1: Write `check_scan()`.** On a smoke trainer:
  - all frames of a scan share the homography exactly;
  - with noise and JPEG switched off, every frame of a scan is identical;
  - with them on, the frames differ;
  - `simulate_scan(..., frames=1)` draws from the same distributions as `simulate`: compare the mean blur, colour and resolution draws over 256 samples within sampling error.
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Implement it.**
  - Refactor `simulate` so both functions share the draw and apply steps; don't duplicate them.
  - In `acceptance.py`:
    - the main table's reading column becomes "decodable, 3 correlated frames" next to "1 frame";
    - the independent multi-shot table stays as information;
    - a visibility column: the share of tiles with p95 ≤ the config's `follow_visibility.target`, with flat tiles (flat share ≥ 0.5 from `content_scores`) reported separately;
    - the PASS line text in `results.md` names the three pass lines.

    With no target in the config, the visibility column shows the p95 distribution and no pass.
- [ ] **Step 4: Check it against the fresh run's numbers.** `.venv/Scripts/python.exe acceptance.py --config configs/runpod_labels_generated_fresh.yaml --checkpoint kept/labels_generated_fresh/20261124-a05a3bd/checkpoint/step_00140000.pt --alphas 0.30 --out runs/acceptance_fresh_scan`. Expected:
  - the 1-frame figure equals today's 0.855;
  - the 3-correlated-frame figure falls between 0.855 and the independent 0.977.
- [ ] **Step 5: Commit.** `git commit -m "feat(training): correlated-frame scans and the new acceptance pass lines"`.

### Task 7: Calibration sheet

**Files:**
- Create: `training/calibration_sheet.py`
- Modify: `training/smoke_test.py` (`check_calibration_levels()`: the scaled tiles hit their levels)

**Interfaces:**
- Consumes: `tile_visibility` (Task 1); `content_scores`, `tiles` and HTML helpers from `acceptance.py`; `labelgen.ACCEPTANCE_SEEDS`.
- Produces:
  - the CLI `python calibration_sheet.py --config <cfg> --checkpoint <pt> --out runs/calibration`, which writes `calibration.html`, `calibration_key.json` and `pairs/*.png`;
  - `python calibration_sheet.py --out runs/calibration --seen 3,7,12`, which prints the target and any masking-floor change by the spec's rules.

- [ ] **Step 1: Write `check_calibration_levels()`.** For a flat tile and a busy tile, with a stand-in residual, `scale_to_level(clean, residual, level, cfg) -> Tensor` returns a marked tile whose p95 masked ΔE is within 5 percent of the level for every level in (0.5, 1, 2, 3, 5, 8). It bisects on a factor multiplying the residual, clamped to [0, 1] after adding.
- [ ] **Step 2: Run it and watch it fail.**
- [ ] **Step 3: Implement it.** Contents of the sheet:
  - three flat-colour labels (the flattest by `content_scores`) and two busy ones, each at all six levels, marked by the 140,000-step model at alpha 0.30 and then scaled;
  - gratings, luminance and blue-yellow, at 23 and 74 cycles per tile, on mid-grey, at all six levels;
  - each beside its clean original, left and right shuffled (seed 0), 50 mm wide, A4 pages, numbered pairs, as `blind.html` is laid out.

  `--seen` takes the numbers of the pairs the user saw a difference in. It prints:
  - the target: the highest level at which no flat-colour pair was seen;
  - the masking floor, raised if a busy pair at or below that target was seen;
  - the target lowered if a 74-cycle grating at or below it was seen;
  - each rule that fired.
- [ ] **Step 4: Run it and watch it pass, then make the sheet.** `.venv/Scripts/python.exe calibration_sheet.py --config configs/runpod_labels_generated_fresh.yaml --checkpoint kept/labels_generated_fresh/20261124-a05a3bd/checkpoint/step_00140000.pt --out runs/calibration`. Expected: `runs/calibration/calibration.html` opens in a browser with 66 numbered pairs at 50 mm.
- [ ] **Step 5: Commit** (not the `runs/` output, which is git-ignored). `git commit -m "feat(training): calibration sheet for the visibility target"`.

### Task 8: Phone colour-detail check

**Files:**
- Create: `training/phone_gratings.py`
- Modify: `web/src/ui/measure.ts`, `web/measure/index.html` (a "Save frame" button)
- Create: `web/src/ui/__tests__/frameName.test.ts`; the helper lives in a new `web/src/ui/frameName.ts`

**Interfaces:**
- Produces:
  - `python phone_gratings.py make --out runs/phone_gratings` writes `gratings.html`: one page with 12 patches (luminance, red-green, blue-yellow × 24, 38, 51, 77 cycles per tile), each one 50 mm tile, plus a flat grey reference, with four solid black corner squares around the grid;
  - `python phone_gratings.py measure --frame <png> --px-per-tile <n>` finds the four corner squares, flattens the grid, measures each patch's amplitude at its frequency in its opponent channel, runs the same page through `simulate` at the same px per tile with tilt off, and prints a table of phone, simulation and their ratio per patch, with a verdict line: "colour detail matches" or "the phone keeps under half of the simulation's detail at …";
  - `frameFileName(width: number, height: number, when: Date) -> string` gives `frame-1080x1920-20261125-143000.png`.

- [ ] **Step 1: Write the failing tests.**
  - Vitest: `frameFileName(1080, 1920, new Date(Date.UTC(2026, 10, 25, 14, 30, 0)))` equals `"frame-1080x1920-20261125-143000.png"` (UTC).
  - Python: `phone_gratings.py measure` run on a frame made by simulating the page itself reports a ratio of 1.0 ± 0.05 for every patch.
- [ ] **Step 2: Run them and watch them fail.** `cd web; npx vitest run src/ui/__tests__/frameName.test.ts` and the Python check.
- [ ] **Step 3: Implement it.** The button is enabled only while frozen and downloads the frozen canvas's pixels, which are the stream's own (`canvas.toBlob` → anchor download).
- [ ] **Step 4: Run them and watch them pass.** `cd web; npm test` → all pass.
- [ ] **Step 5: Commit.** `git commit -m "feat: phone grating check and save-frame button on the measure page"`.

### Task 9: Configs, contract and docs

**Files:**
- Create: `training/configs/probe_eye.yaml`, `training/configs/runpod_eye.yaml`
- Modify: `training/export_bundle.py` (a `reading` block in the contract; bump `CONTRACT_VERSION` to 2)
- Modify: `docs/model-contract.md` (the `reading` block: average logits over the last 3 frames of a scan; scan distance 19 to 28 cm)
- Modify: `training/RUNPOD.md` (new section "The eye-model run": calibration, phone check, probe, pod commands, pass lines)
- Modify: `docs/runs.md` (status row and a section "Eye-model run", design link)

**Interfaces:**
- `runpod_eye.yaml`: `base: runpod_labels_generated_fresh.yaml`, plus:
  - `camera_px_per_tile: [231, 340]`;
  - `demo.measured`: only the 20 cm row;
  - `band.max_step: 12`;
  - `loss: {l2_weight: 0, lpips_weight: 0, visibility: {enabled: true}}`;
  - `schedule.image_loss: {follow_accuracy: null, follow_visibility: {target: null}}`, where `target` is filled in from Task 10;
  - `reading_check: {enabled: true, at_step: 80000, min_bit_acc: 0.80}`;
  - log and checkpoint dirs `/workspace/runs/eye_model/...`;
  - `checkpoint.keep` from Step 3.
- `probe_eye.yaml`: `base: runpod_labels_generated_fresh.yaml`, the **old** range and band, the same loss and controller, plus:
  - `train.total_steps: 143000`;
  - local dirs `runs/probe_eye/logs` and `runs/probe_eye/checkpoints`;
  - `early_stop` off, `image_loss_check` off, `reading_check` off, `budget` null;
  - `data.num_workers: 4`.

- [ ] **Step 1: Check the new config resolves the band.** A failing check in `smoke_test.py`: loading `runpod_eye.yaml` with `target=1` gives `resolve_band(...).grating_step == 12`, cycles (24, 76.8), `message_grid` 192, and `check_demo` passes.
- [ ] **Step 2: Write the configs, the contract block and the docs.** Then run the check and watch it pass.
- [ ] **Step 3: Time a step and measure the checkpoint.** `.venv/Scripts/python.exe train.py --config configs/runpod_eye.yaml --fresh --max-minutes 3 schedule.image_loss.follow_visibility.target=1 log.dir=<tmp>/logs checkpoint.dir=<tmp>/ckpt data.num_workers=4 train.total_steps=40 checkpoint.every=40`. Expected:
  - it prints band step 12 (24 to 76.8), range 231 to 340, the visibility settings and the target;
  - it trains 40 steps and saves.

  Record the seconds per step and the checkpoint size in `docs/runs.md`. Set `checkpoint.keep` so keep × size + 0.3 GB (the kept bundle) + 2.7 GB (already on the volume) stays under 9 GB of the 10 GB volume.
- [ ] **Step 4: Export a bundle from that 40-step checkpoint into a temp folder.** `contract.json` shows `camera.trained_px_per_tile [231, 340]`, `starting_band` 24 to 76.8 and `reading.frames_averaged: 3`. Delete the temp folder.
- [ ] **Step 5: Commit.** `git commit -m "feat(training): eye-model configs, contract reading block, run docs"`.

### Task 10: Calibrate the target (needs the user)

- [ ] **Step 1:** Ask the user to print `training/runs/calibration/calibration.html` at 100 percent on the printer the demo will use, look at each pair from arm's length, and send the numbers of the pairs where they see a difference.
- [ ] **Step 2:** `.venv/Scripts/python.exe calibration_sheet.py --out runs/calibration --seen <their numbers>`. Expected: a target and possibly a raised masking floor.
- [ ] **Step 3:** If every level was seen, stop and bring it back to the user (the spec: the run waits). Otherwise:
  - write the target into `runpod_eye.yaml` and `probe_eye.yaml` (and `loss.visibility.masking_floor` if it changed);
  - record the answers and the result in `docs/runs.md`;
  - commit: `git commit -m "feat(training): visibility target from the printed calibration"`.

### Task 11: Phone check (needs the user)

- [ ] **Step 1:**
  - Make the page: `.venv/Scripts/python.exe phone_gratings.py make --out runs/phone_gratings`.
  - Ask the user to print `gratings.html` at 100 percent and, on the S23 Ultra, open the measure page.
  - At about 20 cm and about 28 cm they freeze a frame and save it, and measure px per cm on a ruler as they did before.
- [ ] **Step 2:** `.venv/Scripts/python.exe phone_gratings.py measure --frame <file> --px-per-tile <px_per_cm × 5>` for each frame. Record both tables in `docs/runs.md`.
- [ ] **Step 3:** If the verdict is "under half" for blue-yellow or red-green at any frequency the phone should resolve, stop. Bring the table to the user with a proposed chroma blur for the simulation, matched to the measured loss, before going on. Otherwise commit the record.

### Task 12: Laptop probe

- [ ] **Step 1:** Copy `kept/labels_generated_fresh/20261124-a05a3bd/checkpoint/step_00140000.pt` to `runs/probe_eye/checkpoints/` (a copy, never the kept file).
- [ ] **Step 2:** `.venv/Scripts/python.exe train.py --config configs/probe_eye.yaml --max-minutes 30`, run in the background. It resumes at 140,000 and trains about 3,000 steps.
- [ ] **Step 3:** Read the spread rows and look at the sample sheets at 140,000, 142,000 and the end. Pass:
  - `mark_cycles` or `colour_share` rises;
  - `visibility_p90` falls toward the target;
  - the sheets show the grid fading without a new patch.
- [ ] **Step 4:** `.venv/Scripts/python.exe acceptance.py --config configs/probe_eye.yaml --checkpoint runs/probe_eye/checkpoints/<newest>.pt --alphas 0.30 --out runs/acceptance_probe_eye`. Pass: decodable with 3 correlated frames above 0.5.
- [ ] **Step 5:**
  - Record it in `docs/runs.md` and commit.
  - If it fails, stop and bring the numbers and sheets to the user. The spec says back to the design before any pod money is spent.

### Task 13: The pod run and acceptance (needs the user to start the pod)

- [ ] **Step 1:** Push nothing. Tell the user the pod needs this branch's code. Commands are in `training/RUNPOD.md`, "The eye-model run" (Task 9):
  - `git fetch` and `git checkout training/eye-model-loss` on the pod (the branch must be pushed by the user);
  - `bash pod_run.sh eye_model configs/runpod_eye.yaml --price-per-hour 0.69 --max-minutes 1500 data.num_workers=8`.
- [ ] **Step 2:** While it runs, read the progress emails and spread rows the user forwards. Look at the sheets at 30,000, 60,000 and 100,000, not only the numbers.
- [ ] **Step 3:** At the end:
  - `python keep_result.py --run eye_model --pod root@<ip> --port <port> --key $HOME\.ssh\runpod_key`, until `CHECK PASSED`;
  - then `acceptance.py --config configs/runpod_eye.yaml --checkpoint kept/eye_model/<id>/checkpoint/step_00140000.pt --out runs/acceptance_eye`.
- [ ] **Step 4:** Ask the user to print `runs/acceptance_eye/00140000/blind.html` and pick the marked side of each pair at arm's length. Compare with `blind_answers.json`. Pass at 14 of 20 right or fewer.
- [ ] **Step 5:** Write the results into `docs/runs.md` (status table and the "Eye-model run" section) and commit.
