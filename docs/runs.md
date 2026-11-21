# Training runs

What the training runs are set up for, what was checked before them, and what
has been run. Kept by whoever works in `/training`. Newest settings first.

## Status, 2026-11-21

| Run | State |
|---|---|
| Long run (main reader, flattened, laptop) | **Not started.** Waiting for a go-ahead. Settings below. |
| Sanity run | **Not started.** No checkpoint or run folder exists under `/training` on the laptop, and nothing was training on its GPU when the settings below were applied. No run has used an older camera range. |
| Direct reader (RunPod) | Not started. See "Direct mode" below before starting it. |
| Smoke tests | Run on 2026-11-21 with the new settings; results below. |

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

Not matched to the phone yet. `simulation.direct.scale` is 0.8 to 1.25, so
the direct reader sees a tile at about 307 to 480 pixels in its 384 pixel
crop. On the phone a tile is 154 to 340 pixels. The camera range is applied
to direct mode as blur only, not as size. A direct run started now answers
"can it learn without the locator", not "does it work on the phone".

Matching it to the phone is agreed and comes next, after the sanity run has
started. The pod is not to be launched before that is done and its cost per
step is reported.

## contract.json

New bundles carry a `camera` block: tile size, stream size and orientation,
the trained range, the measurements and the starting band.
`docs/model-contract.md` explains it. Older bundles were left as they are.
