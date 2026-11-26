# Training runs on RunPod

Two runs are set up for a pod. Each is started with `pod_run.sh`, which stops
the pod when the run ends, and has a name: its folder on the volume is
`/workspace/runs/<name>`, and `notify.py`, `keep_result.py`, `finish_run.sh`
and `watchdog.sh` take the same name.

| Run | Config | Name |
|---|---|---|
| Embedder and reader on generated label art (next) | `configs/runpod_labels_generated.yaml` | `labels_generated` |
| Its 10-minute benchmark | the same | `labels_generated_bench` |
| The spread run on photos | `configs/runpod_spread.yaml` | `spread` |

RunPod facts in this guide were read from docs.runpod.io on 2026-11-21 and
2026-11-22. Prices change; check the console.

## The generated-label run: checklist

`configs/runpod_labels_generated.yaml` is `configs/labels_generated.yaml`
(tiles drawn by `labelgen.py`, both spread options on, 140,000 steps) with
its paths on the volume, no GPU-name check, the four automatic stops of the
spread run (45-minute learning check, image-loss check at step 15,000,
fixed-place limit 4, 24 hours of training), and **no `data.num_workers`**:
the tiles are drawn on the CPU while the GPU trains, so the number of loader
processes comes from timing the pod's own CPU (step 6).

There is no dataset to download. **Nothing in this run has been run on a
pod yet**, including the stop: step 7 is its first live test.

**In the RunPod console, before anything is billed**

1. **Billing** page: make sure auto-pay is **not** enabled (the docs describe
   enabling it there, with a threshold and an amount). Load a small balance:
   enough for setup and the benchmark, about one hour at the price of the pod
   you will pick. You top up for the long run in step 10, once the benchmark
   has given an estimate. Under **Notifications**, turn on **Low balance
   alert** (it is sent by email). Why: when the balance reaches $0, RunPod
   stops every pod. A pod on a network volume keeps its data, but the volume
   is still billed, and the docs say a volume that cannot be paid for may
   eventually be deleted. The account also has a default spend limit of $80
   per hour, which does not limit a long run's total.
2. **Storage > Network Volumes > New**: 10 GB, in a data center that has the
   GPUs below. The run needs about 2 GB: three checkpoints of 0.25 GB, the
   kept results of the benchmark and of the run (about 0.27 GB each), the
   LPIPS weights (0.23 GB) and the code.
3. **Deploy a pod from that volume** (a network volume can only be attached
   at deploy time):
   - Template **Runpod PyTorch**; volume mounted at `/workspace`.
   - **One GPU with 24 GB.** Among the 24 GB types RunPod lists are the
     RTX 4090, RTX 3090, RTX A5000, L4 and A30. Which is fastest for this run,
     and the cheapest per step, has not been measured.
   - **At least 8 vCPUs.** RunPod's docs say vCPUs and RAM come from the host
     machine, not from the GPU type, and do not say whether the GPU list shows
     them. If the deploy page shows a vCPU count, pick 8 or more. Step 6
     prints the count the pod really has.
   - **On-demand.** RunPod's docs say spot (interruptible) pods have been
     retired; if an interruptible option is offered anyway, do not pick it.
   - A public IP with TCP port 22 and your SSH key (for `notify.py` and
     `keep_result.py`, section 9 and 10).

   Write down the price per hour and, from **Connect**, the IP and port.

**On the laptop, once:** `python notify.py --test` in `training`, and wait
for the message (section 9).

**On the pod** (web terminal or ssh)

4. Code: `cd /workspace && git clone https://github.com/tiennguyentony/ShadowCode.git`,
   then `cd /workspace/ShadowCode && git checkout master && git log -1 --oneline`.
   The commit has to have `pod_run.sh` in `training` (section 3).
5. `tmux new -s labels`, then `cd /workspace/ShadowCode/training`.
6. Setup: `CONFIG=configs/runpod_labels_generated.yaml bash setup_runpod.sh`.
   It must end with `ready - nothing has been started` and say that
   `runpodctl` was found. Instead of a download it times `labelgen.py` with 1,
   4 and 8 loader processes (and the pod's vCPU count, if higher) and prints
   `fastest: N workers, ... tiles/s`, and the two commands below with that N
   filled in. **If the vCPU line shows fewer than 8, stop here**: terminate
   the pod and deploy another.
7. **The 10-minute benchmark**, the line setup printed, with your price:

   ```bash
   bash pod_run.sh --bench labels_generated_bench configs/runpod_labels_generated.yaml --max-minutes 10 --price-per-hour 0.00 data.num_workers=N
   ```

   It trains for 10 minutes from step 0 in `/workspace/runs/labels_generated_bench`
   (not the real run's folder), then prints the report: steps per second,
   GPU utilisation, whether the CPU held the GPU up, and the hours and GPU cost
   of 140,000 steps at that speed. Then it exports and keeps the result, waits
   three minutes, and **stops the pod through the same path the long run
   uses**. Read the report on the screen during those three minutes; it is
   also saved as `/workspace/runs/labels_generated_bench/bench_report.txt`.
8. **Check in the console that the pod has stopped**, about five minutes
   after the report. This is the live test of the stop. If it has not
   stopped, stop it by hand, and do not leave the long run unattended: the
   `runpodctl` lines at the end of `/workspace/runs/labels_generated_bench/console.log`
   say why.
9. Decide from the report. If `CPU bottleneck` says YES, a pod with more
   vCPUs would be faster. If the image loss was off for the whole benchmark,
   the report says so: its estimate is then too low, because later steps do
   more work. If the estimate is over 24 hours, the run stops at 24 hours
   (it says near which step).
10. Top up the balance to the estimate plus the margin you want. Start the
    pod again (**Pods > Start**; the IP and port may change). Stopping wiped
    the container disk, so run step 5 and 6 again.
11. **The run**, the second line setup printed, with your price:

    ```bash
    bash pod_run.sh labels_generated configs/runpod_labels_generated.yaml --price-per-hour 0.00 data.num_workers=N
    ```

    The first lines have to show `launched pod_run.sh labels_generated`,
    `device=cuda`, the `spread:` line with both options on, and `budget: the
    run stops, with a checkpoint, at 24 h of training`. `pod_run.sh` starts
    the watchdog itself (`watchdog: watching ...` in `console.log`): it stops
    the pod if no progress row has come for an hour. Leave tmux with `Ctrl-b`
    then `d`.
12. On the laptop, in a terminal of its own, in `training`:
    `python notify.py --no-laptop --run labels_generated --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>`
    (leave out `--no-laptop` if a laptop run is going too).
13. When the `ENDED` message arrives, the pod stops itself a few minutes
    later. Check in the console that it has.
14. Start the pod again and, on the laptop, in `training`:
    `python keep_result.py --run labels_generated --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>`
    and wait for `CHECK PASSED` (section 10). It copies to
    `training/kept/labels_generated/<bundle id>/`.
15. Stop the pod. **Delete the pod and the volume only after `CHECK PASSED`**
    (section 7). The volume is billed until it is deleted.

## The generated-label fix: 2,000-step check, then the run

The generated-label run stopped itself at step 38,000 with the mark piled up
in the tile corner (`/docs/runs.md`, "Generated-label pod run: the stop and
the fix"). `configs/runpod_labels_generated_fix.yaml` is the same run with the
peak penalty at weight 16 instead of 4 and the texture mask on. It is checked
first for 2,000 steps, resumed from the old run's step 34,000 (the least
piled-up checkpoint left), under the run name `labels_generated_fix`.

**Keep the old run's volume** until this has passed: its
`checkpoints/step_00034000.pt` is the start.

**On the pod** (start it again: **Pods > Start**; the IP and port may change)

1. `tmux new -s fix`, then
   `cd /workspace/ShadowCode && git fetch origin && git checkout training/spread-texture-fix && git pull && git log -1 --oneline`.
2. `cd training && CONFIG=configs/runpod_labels_generated_fix.yaml bash setup_runpod.sh`
   (stopping wiped the container disk). Note the `fastest: N workers` line;
   the old run used 8.
3. Put the start checkpoint into the new run's folder:

   ```bash
   mkdir -p /workspace/runs/labels_generated_fix/checkpoints
   cp /workspace/runs/labels_generated/checkpoints/step_00034000.pt /workspace/runs/labels_generated_fix/checkpoints/
   ```

4. **The check**, 2,000 steps (34,000 to 36,000), about 8 minutes of training:

   ```bash
   bash pod_run.sh labels_generated_fix configs/runpod_labels_generated_fix.yaml --price-per-hour 0.69 --max-minutes 15 train.total_steps=36000 data.num_workers=8
   ```

   The first lines have to show
   `spread: peak penalty on (weight 16, ...); ...; texture mask on (...)` and
   `resumed from /workspace/runs/labels_generated_fix/checkpoints/step_00034000.pt at step 34000`.
   `--max-minutes 15` is only a guard: 2,000 steps take about 7.5 minutes
   at 15,990 steps per hour.

**On the laptop**, in `training`, one email at the end only:

```powershell
python notify.py --no-laptop --run labels_generated_fix --every-steps 0 --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>
```

**Pass lines**, read from the step 36,000 row of the ENDED email (the row at
34,000 is the start):

| Line | Pass |
|---|---|
| Fixed place | under 2 (start 3.20) |
| Bit accuracy, mean of the last 1,000 training steps | 0.94 or more |
| PSNR | above the start (23.48 dB on the fixed images at 34,000) |
| Energy on the busier half | above 60 percent (start 49) |

With the texture mask the mark may move onto detail and the PSNR may fall
while the mark gets harder to see; read the PSNR line together with the
sample sheet.

**If it passes**, the same name resumes from 36,000 to 140,000, with the
normal reports (one every 10,000 steps):

```bash
bash pod_run.sh labels_generated_fix configs/runpod_labels_generated_fix.yaml --price-per-hour 0.69 data.num_workers=8
```

```powershell
python notify.py --no-laptop --run labels_generated_fix --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>
```

104,000 steps: about 6.5 hours and $4.49 of GPU time at 15,990 steps per
hour and $0.69 per hour (estimate).

**If it fails** (fixed place still 2 or more at 36,000), do not resume: the
mark is too set in the old checkpoints. Start the same config from step 0
under a new name; the image loss only switches on near step 16,000, so its
first useful look is the spread row at about 20,000 (1.25 hours, $0.86), and
the whole run is about 8.8 hours and $6.04.

## The generated-label run, v2: 2,000-step check, then the run

The fix's check (above) ran on 2026-11-23 and failed on one line: fixed place
2.09 at 36,000 (line: under 2). The image-loss weight fell from 0.29 to 0.20
during it, so the mark did not fade (`/docs/runs.md`, "The fix's 2,000-step
check, and v2"). `configs/runpod_labels_generated_v2.yaml`: peak penalty 24,
the image-loss weight rises from a bit accuracy of 0.945 (was 0.95) and three
times as fast, spread check and sheets every 2,000 steps. Run name
`labels_generated_v2`, resumed from the fix check's step 36,000.

**Keep `/workspace/runs/labels_generated_fix/`** until v2 has passed: its
`checkpoints/step_00036000.pt` is the start, and it is only on the pod. Copy
it to the laptop first (in the repository root):

```powershell
scp -P <port> -i $HOME\.ssh\runpod_key root@<pod ip>:/workspace/runs/labels_generated_fix/checkpoints/step_00036000.pt training\runs\pod\labels_generated_fix\
```

**On the pod**

1. `tmux new -s v2`, then
   `cd /workspace/ShadowCode && git fetch origin && git checkout training/labels-generated-v2 && git pull && git log -1 --oneline`.
2. `cd training && CONFIG=configs/runpod_labels_generated_v2.yaml bash setup_runpod.sh`
   (stopping wiped the container disk). The fix used 8 workers.
3. The start checkpoint:

   ```bash
   mkdir -p /workspace/runs/labels_generated_v2/checkpoints
   cp /workspace/runs/labels_generated_fix/checkpoints/step_00036000.pt /workspace/runs/labels_generated_v2/checkpoints/
   ```

4. **The check**, 36,000 to 38,000:

   ```bash
   bash pod_run.sh labels_generated_v2 configs/runpod_labels_generated_v2.yaml --price-per-hour 0.69 --max-minutes 15 train.total_steps=38000 data.num_workers=8
   ```

   The first lines have to show `peak penalty on (weight 24, ...)`,
   `texture mask on`, `up at or above 0.945, down below 0.94, by 1/5000 per step`
   and `resumed from /workspace/runs/labels_generated_v2/checkpoints/step_00036000.pt at step 36000`.

**On the laptop**, in `training`, one email at the end only:

```powershell
python notify.py --no-laptop --run labels_generated_v2 --every-steps 0 --pod root@<pod ip> --port <port> --key $HOME\.ssh\runpod_key
```

**Pass lines**, the step 38,000 row of the ENDED email against the 36,000
row (the start, measured at resume):

| Line | Pass |
|---|---|
| Fixed place | under 2 (2.09 at the fix's 36,000) |
| Bit accuracy, mean of the last 1,000 training steps | 0.94 or more |
| PSNR on the fixed images | not below the 36,000 row (23.51 dB at the fix's 36,000) |
| Energy on the busier half | not below the 36,000 row (67 percent) |

**If it passes**, start the pod again (the IP and port may change), repeat
steps 1 and 2, and resume the same name from 38,000 to 140,000:

```bash
bash pod_run.sh labels_generated_v2 configs/runpod_labels_generated_v2.yaml --price-per-hour 0.69 --max-minutes 480 data.num_workers=8
```

```powershell
python notify.py --no-laptop --run labels_generated_v2 --pod root@<pod ip> --port <port> --key $HOME\.ssh\runpod_key
```

102,000 steps: about 6.5 hours of training and $4.50 of GPU time at about
15,900 steps per hour and $0.69 per hour (estimate). `--max-minutes 480`
caps it at 8 hours ($5.52).

**If it fails**, one change, by which line failed:

| Failed | Next |
|---|---|
| Fixed place 2 or more, but below 2.09 and the rest pass | `spread.peak_penalty.weight` 24 -> 32, resumed from v2's 38,000 |
| Bit accuracy below 0.94, the rest pass | `follow_accuracy` back to floor 0.95, band 0.01, resumed from 36,000 again |
| PSNR or busier half fell, bit accuracy 0.94 or more | a minimum image-loss weight (a new schedule option, code change), resumed from 36,000 again |
| Fixed place rose above 2.09, or two or more lines failed | resuming does not work: v2 from step 0 under a new name, about 8.8 hours and $6 |

## The generated-label run, v2 from step 0

The v2 full run, resumed from 38,000, still had the first run's lower-right
patch in its sample sheets (38,000 and 40,000), and fixed place went back up
from 1.79 to 1.93 (`/docs/runs.md`, "v2 from step 0"). Every resumed run
carries that patch, so v2 starts again from step 0:
`configs/runpod_labels_generated_fresh.yaml`, the v2 settings with the spread
stop at 2.5 instead of 4. Run name `labels_generated_fresh`.

**1. Stop the v2 run, keep the pod up.** In the pod's tmux session (`tmux
attach -t v2` if you are not in it):

- Ctrl-c once. train.py saves a checkpoint and exits; the bundle is exported
  and kept (a minute or two).
- When it prints `stopping the pod in 180 s`, press Ctrl-b then d (detach),
  and within those 180 s:

  ```bash
  tmux kill-session -t v2
  pkill -f watchdog.sh
  pgrep -af "train.py|pod_run.sh|watchdog.sh" || echo "nothing left running"
  ```

  The watchdog is only ended by pod_run.sh after the 180 s, so end it by
  hand: left running, it would stop the pod an hour later. If you miss the
  180 s the pod stops itself; start it again in the console (the IP and port
  may change) and run the setup line in step 2.

On the laptop, stop the notify.py that watches `labels_generated_v2` (Ctrl-c
in its window) once its ENDED email has come.

**2. Start from step 0.**

```bash
tmux new -s fresh
cd /workspace/ShadowCode && git pull && git log -1 --oneline
cd training
bash pod_run.sh labels_generated_fresh configs/runpod_labels_generated_fresh.yaml --price-per-hour 0.69 --max-minutes 600 data.num_workers=8
```

Only if the pod was stopped and started again, run
`CONFIG=configs/runpod_labels_generated_fresh.yaml bash setup_runpod.sh`
before `pod_run.sh` (stopping wipes the container disk).

The first lines have to show `starting from step 0`, `peak penalty on
(weight 24, ...)`, `texture mask on`, `image-loss check: ... at step 24000`
and `the run stops if the fixed-place measure goes above 2.5`.

**If a run shows `at step 15000` instead** (started from commit `0aabc73`,
before the config moved the check): it will stop itself at step
15,000, because the image loss only switches on near 16,000 and the PSNR is
about 12.8 dB until then. Stop it as in step 1 (tmux session `fresh`, Ctrl-c,
detach, kill the session and the watchdog within the 180 s), then:

```bash
tmux new -s fresh
cd /workspace/ShadowCode && git pull && git log -1 --oneline
cd training
bash pod_run.sh labels_generated_fresh configs/runpod_labels_generated_fresh.yaml --price-per-hour 0.69 --max-minutes 600 data.num_workers=8
```

It resumes from the checkpoint Ctrl-c saved and checks at 24,000. If it
already stopped at 15,000 (exit code 5), it resumes only with
`image_loss_check.enabled=false` added to that line.

**On the laptop**, in `training`, a report every 10,000 steps, the end and
alerts at once:

```powershell
python notify.py --no-laptop --run labels_generated_fresh --pod root@<pod ip> --port <port> --key $HOME\.ssh\runpod_key
```

140,000 steps: about 9.3 hours and $6.40 at about 15,000 steps per hour and
$0.69 per hour (estimate). `--max-minutes 600` caps it at 10 hours ($6.90).

**What to look at, and when.** Read the sample sheet in each email, not only
the numbers: a patch shows as a block of dots in the same place on every
flat tile (third column).

| Step | About | What should be true |
|---|---|---|
| 10,000 | 0.7 h, $0.46 | bit accuracy rising (the learning check at 45 minutes needs 0.75) |
| 20,000 | 1.3 h, $0.92 | the image loss is on (it switched on at 16,304 in the first run) and PSNR is rising; fixed place near 1 |
| 30,000 | 2.0 h, $1.38 | the first run's patch formed at 26,000 to 28,000: fixed place still under 2, no block on the flat tiles. This is the email that says whether starting fresh worked. |

The run stops itself, with a checkpoint, if fixed place goes above 2.5 at
any spread check (exit code 4).

## The eye-model run

The fresh run left the mark plainly visible (a dot grid, PSNR 29 dB). This run
pushes it under a visibility target measured the way the eye sees a print
(S-CIELAB, masked), and scans from 19 to 28 cm instead of 19 to 42 cm so the
mark can sit in finer spacings the eye is weak at. Why and how:
`/docs/superpowers/specs/2026-11-24-eye-model-loss-design.md`. Run name
`eye_model`, `configs/runpod_eye.yaml`, from step 0, 140,000 steps, $0.69 per
hour. The band is new (step 12, 24 to 76.8 cycles per tile, `message_grid`
192), so it cannot resume from the earlier runs' checkpoints. A to C are on
the laptop and cost nothing; the pod is started only if they pass.

Commands run from `training/` with `.venv/Scripts/python.exe` (written
`python` below). The 140,000-step checkpoint is
`kept/labels_generated_fresh/20261124-a05a3bd/checkpoint/step_00140000.pt`;
never edit it in place.

**A. The calibration sheet: your own eye sets the target.**

```powershell
python calibration_sheet.py --config configs/runpod_labels_generated_fresh.yaml --checkpoint kept/labels_generated_fresh/20261124-a05a3bd/checkpoint/step_00140000.pt --out runs/calibration
```

Print `runs/calibration/calibration.html` at 100 percent, on the printer the
demo will use: each grey grating square is 50 mm wide (check one with a
ruler), each label 75 mm on its long side. The labels are whole labels,
marked 50 mm tile by tile as the app marks artwork, each scaled so its most
visible 50 mm window sits at the pair's level. Hold the sheet at arm's length
(about 50 cm). Every pair is a picture and its clean original, left and right
shuffled; write down the numbers of the pairs where you see any difference.
Then:

```powershell
python calibration_sheet.py --out runs/calibration --seen 3,7,12
```

(use your numbers). It prints the target, the 95th-percentile masked delta E:
the highest level with no flat-colour pair seen, lowered if a fine grating was
seen. `runs/calibration/calibration_result.json` has it too. Exit code 1 and
"the run waits" mean even the lowest level showed a difference: stop and go
back to the design. If it also prints a new masking floor (a busy-label pair
was seen at or below the target), rebuild the sheet with
`loss.visibility.masking_floor=<floor>` appended to the first command, look at
the busy rows again, and add the same override to every command below. The
rebuilt sheet's `calibration_key.json` keeps the override, so `--seen` on it
measures under the same floor; `--seen` itself takes no override.

**B. The phone's fine colour detail.** The eye model pushes the mark into fine
colour, where phone cameras are weakest. Check the phone against the
simulation before spending anything:

```powershell
python phone_gratings.py make --out runs/phone_gratings
```

Print `runs/phone_gratings/gratings.html` at 100 percent. On the phone, open
the web app's measure page (`npm run dev` in `web`, then the `/measure/` page
on the phone), hold the sheet upright with all four black squares in view at
21 to 22 cm and again at about 28 cm (nearer than 21 cm the corner squares
fall off the 1080 px stream), press Freeze and then Save frame, and measure
px per cm on a ruler as before. Copy each saved PNG to the laptop:

```powershell
python phone_gratings.py measure --frame <frame.png> --px-per-tile <px per cm x 5>
```

The simulation compared with is `configs/runpod_eye.yaml`'s by default.
"colour detail matches" is a pass. If the phone keeps under half of the
simulation's grating somewhere, the simulation needs a matching chroma blur
before the pod run: stop and go back to the design.

**C. The laptop probe.** The new loss on a copy of the 140,000-step
checkpoint, with the old range and band (it loads there), about 3,000 steps
(about 25 minutes on the RTX 4070):

```powershell
New-Item -ItemType Directory -Force runs/probe_eye/checkpoints
Copy-Item kept/labels_generated_fresh/20261124-a05a3bd/checkpoint/step_00140000.pt runs/probe_eye/checkpoints/
python train.py --config configs/probe_eye.yaml schedule.image_loss.follow_visibility.target=<target>
```

The baseline is the first spread row: on resume, before its first step,
train.py writes a row for step 140,000 to the probe's new
`runs/probe_eye/logs/spread.csv`, measured under the probe config, so with
the visibility columns (`visibility_p90`, `mark_cycles`, `colour_share`) that
the kept run never had. After it comes a row, and a sample sheet, every 500
steps (`log.sheet_every: 500`): 140,500 to 143,000.

**What the probe can tell.** It tests the direction the new loss pushes the
mark, not where it settles: the weight starts at 0.01 and climbs at most by
1.00046 per step, so 3,000 steps move it only about 4 times (to about 0.04).
The mark will not reach the target here.

Watch bit accuracy in the first few hundred steps (the `step ...` lines and
the first progress rows): L2 and LPIPS are off from the first step, so a fast
fall there means the new loss is breaking the reading, not shaping it.

Pass: against the 140,000 row, the later spread rows show the mark's mean
frequency (`mark_cycles`) or its colour share (`colour_share`) rising and
`visibility_p90` falling, or at least not rising; decodable with 3 correlated
frames stays above 0.5 (copy the newest probe checkpoint elsewhere, then
`python acceptance.py --config configs/probe_eye.yaml --checkpoint <the copy>
schedule.image_loss.follow_visibility.target=<target>`); and the sample
sheets show the grid fading, not a new patch. If it fails, go back to the
design before any pod money is spent.

**D. The pod run.** Push the branch from the laptop first (`git push origin
training/eye-model-loss`): the pod checks it out from GitHub. Set up the pod
as in sections 1 to 4, then look at the volume before starting:

```bash
df -h /workspace
du -sh /workspace/runs/*
```

The run keeps 8 checkpoints of 0.5 GB (4.0 GB) plus a 0.3 GB bundle. If
`/workspace` has less than about 5 GB free, delete the old runs' checkpoint
folders, `/workspace/runs/labels_generated_fresh/checkpoints` and
`/workspace/runs/labels_generated_v2/checkpoints`, but only once each run is
kept on the laptop and `keep_result.py` printed `CHECK PASSED` for it
(section 10). Then, in tmux:

```bash
tmux new -s eye
cd /workspace/ShadowCode && git fetch origin && git checkout training/eye-model-loss && git pull && git log -1 --oneline
cd training
bash pod_run.sh eye_model configs/runpod_eye.yaml --price-per-hour 0.69 --max-minutes 900 data.num_workers=8 schedule.image_loss.follow_visibility.target=<target>
```

(plus `loss.visibility.masking_floor=<floor>` if the sheet gave one). The
commit `git log` prints has to be the branch's newest (compare with `git log
-1 --oneline` on the laptop). Without the target train.py refuses to start. The first lines have to show
`grating_step 12`, `message_grid 192`, `camera: 231 to 340 px per printed
tile`, `visibility loss on: ...` and `visibility controller: target <your target> ...` (check the target here), `starting fresh`, `reading check: ... below 0.8 at step 80000` and
`the run stops if the fixed-place measure goes above 2.5`.

**The controller weight starts at 0.01, not 1.** The design says 1;
`init_weight` is left at its default 0.01. At 1 the first steps after the
image loss switches on would wipe out the reading before the controller
reacts. The weight then climbs by `rate` (4.6e-4) per step while the smoothed
reading is above the target, so it can move tenfold in about 5,000 steps.

**The controller's reading.** Each step's 8 tiles give one p95 masked delta
E each; the controller keeps the last 256 of them (32 steps) and reads the
**96th percentile** of that window (`follow_visibility.quantile: 0.96`,
`window: 256`), smoothed. Held at the target, that leaves about 96 percent
of tiles at or under it, against the pass line's 95. (Until 2026-11-24 it
read the 0.9 quantile of each batch of 8, which settles near 86 to 89
percent; the final review changed it.) The startup line `visibility
controller: target ... (the 96th percentile ... over a window of the last 256
tiles ...)` confirms it.

**Two visibility numbers.** `visibility` in the progress rows and emails (and
`seen` in the step lines) is the controller's reading above, on training
tiles, averaged over the report window: it should come down to the target
and stay between 0.9 x target and the target. `visibility_p90` in the spread
rows is the 90th percentile over the 256 fixed images of their p95, measured
at that step: a lower percentile, so it sits below the progress figure. Judge
the controller by the first, the trend by the second.

Time and cost: the design guessed about 10 hours and $7 at the fresh run's
pace; the bigger message layer may slow it. The laptop's 40-step timing is in
`/docs/runs.md` ("Eye-model run"); the pod's figure will differ, so read the
steps per second off the pod's first report and redo the sum. `--max-minutes
900` caps it at 15 hours ($10.35); a time stop only costs a restart.

On the laptop, as for the fresh run:

```powershell
python notify.py --no-laptop --run eye_model --pod root@<pod ip> --port <port> --key $HOME\.ssh\runpod_key
```

**Stops, on their own.** Learning check at 45 minutes (bit accuracy at least
0.75; the finer band may learn slower), spread stop above 2.5 (exit code 4),
image-loss check at 24,000 (exit code 5), **reading check at step 80,000**:
stop (exit code 7) if the 1-frame training bit accuracy, the mean of the last
1,000 steps, is below 0.80, since that run cannot pass even with 3 frames
averaged; and 24 hours of training. Look at the sample sheets at 30,000,
60,000 and 100,000, not only at the numbers.

**Pass lines.** Copy the final checkpoint to the laptop, then:

```powershell
python acceptance.py --config configs/runpod_eye.yaml --checkpoint <copy of step_00140000.pt> schedule.image_loss.follow_visibility.target=<target>
```

The run passes when all of these hold:

- decodable (at most 4 wrong of the first 63 bits) at least 0.80 with 3
  correlated frames averaged;
- at least 95 percent of held-out tiles at or under the target;
- fixed place under 2.5;
- printed blind test 14 of 20 right or fewer: print
  `runs/acceptance/<step>/blind.html` at 100 percent, pick the marked side of
  each pair at arm's length, and compare with `blind_answers.json`. 15 or
  more right is unlikely by chance (about 2 percent): it can be seen.

Then keep the result (section 10) as for any run.

## The spread run

The pod trains `configs/runpod_spread.yaml`: the long run's settings
(flattened reader, phone camera range, top band, 140,000 steps) with the two
options that keep the mark spread over the whole tile both on. The laptop's
long run puts the code into one bright patch at the tile corner
(`/docs/runs.md`, "Corner patch"); this run is the attempt to fix that.

**No training run has used the two options yet.** They were checked on the
CPU only: they run and gradients flow. The run stops itself, with a
checkpoint, in four cases: bit accuracy under 0.75 after 45 minutes; the
mark piling up at the same place in every tile (fixed-place measure above 4
at any sample sheet); PSNR still below
14 dB at step 15,000, which means the image loss never switched on; and 24
hours of training, the only one of the four that limits time or cost. The
sample sheets, one every 2,000 steps, show the mark itself; section 6 says
how to copy one to the laptop and what to look for.

**This guide has not been run on a pod.** `setup_runpod.sh` was run only on
the laptop against an ordinary folder standing in for the volume.

The direct reader (`configs/runpod_direct.yaml`) is dropped for the gate: it
never left chance with rotation (`/docs/runs.md`, "Direct mode"). The config
is still there; section 8 says how to run it.

## Checklist: the whole run, in order

Each step names the section that has the details and the reasons.

**On the laptop, once**

1. Make a Gmail app password and put it in
   `~/.shadowcode/gmail_app_password.txt` (section 9, six steps).
2. In `training`, send the test message and wait for it to arrive:
   `python notify.py --test`

**In the RunPod console**

3. Billing: turn auto-pay off, load only what the run should cost, set a low
   balance alert. The balance is the only hard limit on the bill (section 5).
4. Create a 20 GB network volume (section 1).
5. Deploy a pod **from that volume**: Runpod PyTorch template, one GPU with
   12 GB or more, 8 or more vCPUs, on-demand, volume at `/workspace`, a
   public IP with TCP port 22, your SSH key (section 2). Write down the
   price per hour and, from **Connect**, the IP and the port.

**On the pod** (web terminal or ssh)

6. Get the code and check the commit is the newest one on GitHub (section 3):
   `cd /workspace && git clone https://github.com/tiennguyentony/ShadowCode.git`
   then `cd /workspace/ShadowCode && git checkout master && git log -1 --oneline`
7. Set up; it downloads the dataset the first time (section 4):
   `cd /workspace/ShadowCode/training && bash setup_runpod.sh`
   It has to end with `ready - nothing has been started`, say that
   `runpodctl` was found, and list the stops, the 24 hours among them.
8. Open tmux and go to the folder (section 5):
   `tmux new -s spread`, then `cd /workspace/ShadowCode/training`
9. Start the run with the **stop-the-pod command** of section 5, with
   your price in place of `0.00`:
   `bash pod_run.sh spread configs/runpod_spread.yaml --price-per-hour 0.00`
10. Read the first lines before you leave. They have to show `device=cuda`,
    `spread: peak penalty on ...; hiding on ...`, and
    `budget: the run stops, with a checkpoint, at 24 h of training`.
11. Leave tmux with `Ctrl-b` then `d`. The watchdog of section 5 was started
    by `pod_run.sh`.

**On the laptop**

12. In a terminal of its own, in `training`, start the watcher (section 9):
    `python notify.py --run spread --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>`
    It prints one `sent:` line per run within a minute or two.

**While it runs**

13. A report arrives every 10,000 steps for each run (`notify.every_steps`),
    with the spread rows of the 2,000-step checks since the last one. Section 5, "What you
    will see", says what the numbers mean and section 6 what a good and a
    bad sample sheet look like. The run can stop itself at 45 minutes (not
    learning), at step 15,000 (image loss never on), at any sample sheet
    (mark piled up) or at 24 hours. Each stop sends an `ENDED` message.

**When the `ENDED` message arrives**

14. The pod stops itself about three minutes later. Check in the console
    that it has.
15. Start the pod again (or deploy any pod on the same volume) and, on the
    laptop, in `training`, copy and check the result (section 10):
    `python keep_result.py --run spread --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>`
    Wait for `CHECK PASSED`.
16. Stop the pod. Fill in GPU, price and start time in `/docs/runs.md`,
    "Spread run (RunPod)", and rebuild its table with the command there.
17. Delete the pod and the volume only after `CHECK PASSED` (section 7).
    Until the volume is deleted it is still billed.

Not every step has been done on a real pod: see the note at the top, and
the "Tested" paragraphs of sections 5, 9 and 10.

## What is billed

| Thing | Billed | Stops when |
|---|---|---|
| GPU pod | per second while running, at the hourly price shown on the pod | you stop or terminate the pod |
| Container disk | $0.10 per GB per month while the pod runs; not charged while stopped | you stop or terminate the pod |
| Network volume | $0.07 per GB per month (under 1 TB), always, pod or no pod | you delete the volume |

A 20 GB network volume is about $1.40 a month. If the account balance reaches
$0, RunPod stops the pods itself; a pod with a network volume keeps its data.

## 1. Network volume

Create it first: **Storage > Network Volumes > New**.

- Size: **20 GB**. The dataset is 4.94 GB, each checkpoint is roughly 0.25 GB
  (three are kept), the LPIPS weights are 0.23 GB. Use 30 GB if you fetch the
  whole dataset split (12.88 GB).
- Data center: one that has the GPU you want. The volume decides which GPUs
  the pod can use.
- Network volumes are Secure Cloud only, and a volume can only be attached
  when the pod is deployed, not afterwards.

## 2. Pod template

Deploy the pod **from the network volume** (so that it is attached), with:

- Template: the official **Runpod PyTorch** template. The image tag RunPod's
  docs use as their example is
  `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`. Any image
  with Python 3.10 or newer, PyTorch 2.2 or newer and CUDA works;
  `setup_runpod.sh` checks.
- GPU: one NVIDIA GPU with 12 GB or more. The flattened run at batch 8 peaks
  at about 4.3 GB on the laptop. The two options work on tensors the run
  already holds, so they should add little; that was not measured on a GPU.
- Volume mount path: `/workspace` (the default). The config assumes it.
- Container disk: the template default. It only has to hold the pip packages.
- On-demand. RunPod's docs (2026-11-22) say spot (interruptible) pods have
  been retired. A run restarted for any reason resumes from its newest
  checkpoint and loses at most 2,000 steps.
- For `scp` or `rsync`, the pod needs a public IP with TCP port 22 exposed.
  RunPod's proxied "basic SSH" does not support them. `runpodctl send`
  (section 6) needs neither.

Note the price per hour the pod shows. You pass it to `train.py`.

## 3. Code

Connect (web terminal, or the SSH command the console shows) and put the code
on the volume:

```bash
cd /workspace && git clone https://github.com/tiennguyentony/ShadowCode.git
```

```bash
cd /workspace/ShadowCode && git checkout master
```

If the code is already on the volume from an earlier pod, update it instead:

```bash
cd /workspace/ShadowCode && git pull
```

The branch has to be pushed first. Without git (private repository, no token
on the pod), pack the committed `training` folder on the laptop and copy it:

```bash
git archive --format=tar.gz -o training.tar.gz HEAD training
```

```bash
scp -P <port> training.tar.gz root@<pod ip>:/workspace/
```

and on the pod:

```bash
mkdir -p /workspace/ShadowCode && tar -xzf /workspace/training.tar.gz -C /workspace/ShadowCode
```

## 4. Setup

Run this after every pod start. A new container has lost its pip packages;
everything else is on the volume.

```bash
cd /workspace/ShadowCode/training && bash setup_runpod.sh
```

For another config, name it: `CONFIG=configs/runpod_labels_generated.yaml bash setup_runpod.sh`.
It does four things and starts nothing:

1. Checks that `/workspace` exists and is writable, and prints its size.
2. Installs `requirements.txt` (PyTorch is already in the template) and tmux.
3. Checks the GPU with `nvidia-smi` and by running a convolution on it.
4. Fetches the dataset to `/workspace/datasets/open-images-v7-val` if it is
   not complete there, and the LPIPS weights to `/workspace/cache/torch`.
   A config with `data.source: generated_labels` has no dataset: instead,
   `labelgen.py --bench 300 --workers 1 4 8` (and the vCPU count, from
   `RUNPOD_CPU_COUNT` or `nproc`, if above 8) times the label generator,
   prints the fastest worker count and its tiles per second, and saves the
   timing to `/workspace/cache/labelgen_bench.txt`. The LPIPS weights are
   fetched either way.

At the end it prints the `pod_run.sh` commands for the config, with the
fastest worker count filled in for generated labels.

The dataset is **Open Images V7, validation split**, first 16,000 images by
image ID, **4.94 GB**, images listed as **CC BY 2.0** (`THIRD_PARTY.md`). The
download time on a pod has not been measured; expect minutes, and the script
prints the rate as it goes. The first run on a new volume downloads it; later
runs find it complete and skip it. To fetch the whole split (41,620 images,
12.88 GB):

```bash
DATASET_COUNT=all ALLOW_LARGE=1 bash setup_runpod.sh
```

## 5. Run

Start a tmux session, so the run survives a dropped connection:

```bash
tmux new -s spread
```

Inside it, with your pod's price in place of `0.00`:

```bash
cd /workspace/ShadowCode/training && bash pod_run.sh spread configs/runpod_spread.yaml --price-per-hour 0.00
```

**`pod_run.sh` is the stop-the-pod command.** It takes the run's name and
config, then any `train.py` options and overrides, and puts everything under
`/workspace/runs/<name>/` (it passes `log.dir` and `checkpoint.dir` to
`train.py` itself). When `train.py` exits, for any reason (finished, stopped
by one of its four stops, `Ctrl-c`, or crashed), it:

1. writes `train.py exited with code N` to `console.log`;
2. with `--bench` only, prints the benchmark report (`bench_report.py`);
3. runs `finish_run.sh`, which exports the bundle and puts everything worth
   keeping into `/workspace/runs/<name>/kept/<bundle id>/` with checksums
   (section 10), while the pip packages the export needs are still there;
4. waits three minutes, so the laptop's watcher (section 9) can read why the
   run ended before the pod goes away;
5. asks RunPod to stop the pod (`runpodctl pod stop "$RUNPOD_POD_ID"`), again
   every minute until that works.

Nothing in steps 2 to 4 can prevent step 5. Only a mistake found before
anything starts (no price, a price of 0, a config that does not load, no
`data.num_workers` for the generated-label config) ends it without stopping
the pod, with `Nothing was started and the pod was not stopped.` A crash in
the first seconds stops the pod too, about three minutes later; the reason
is then in `console.log`, on the volume. To keep the pod after all, close the
tmux session during the three-minute wait (`tmux kill-session -t spread`).

RunPod's docs say every pod comes with `runpodctl` and a pod-scoped API key,
and sets `RUNPOD_POD_ID`; `setup_runpod.sh` checks both. If either is
missing, `pod_run.sh` says so before the wait and cannot stop the pod. The
second form it tries, `runpodctl stop pod`, is the older syntax, which the
docs now list as legacy. Check in the console afterwards that the pod really
is stopped.

**If the run hangs** it never exits, and `pod_run.sh` never reaches the stop.
For that it starts `watchdog.sh <name>` in the background: it stops the pod
when `progress.csv` has had no new row for an hour (a row comes every 1,000
steps). The hour counts from when the watchdog starts, so an old
`progress.csv` from an earlier pod session does not trip it. `pod_run.sh`
ends it just before its own stop. For a run started without `pod_run.sh`,
start one by hand from the training folder:

```bash
tmux new -d -s watchdog 'bash watchdog.sh spread'
```

Tested on the laptop with a stand-in for `runpodctl` that fails twice before
it works, and a stand-in for `train.py` that trains nothing: exit codes 0, 1
and 6, the benchmark report, a crash before any log was written, every
argument mistake above, and the watchdog with no rows, with rows that stop
coming, and with an old `progress.csv`. Not run on a pod, and not with the
real `train.py`. The `tmux` lines were not run either (the laptop has no
tmux).

**What neither covers:** the container restarting, or the pod losing its
network. Nothing inside the pod can act then. The one limit that does not
depend on the pod is the account balance: RunPod stops every pod when it
reaches $0, and a pod on a network volume keeps its data. So keep auto-pay
off and load only what this run should cost, and set a low balance alert
(**Billing > Notifications**). Do not leave the balance at $0 for long:
the volume is still billed, and RunPod's billing page says a volume whose
charges cannot be covered may eventually be deleted.

**Leave the pod running** when the run ends (it keeps billing until you stop it):

```bash
mkdir -p /workspace/runs/spread && export TORCH_HOME=/workspace/cache/torch && python3 train.py --config configs/runpod_spread.yaml --price-per-hour 0.00 2>&1 | tee -ia /workspace/runs/spread/console.log
```

`notify.py` cannot tell whether this one is still running (it looks for the
`log.dir` that `pod_run.sh` passes); give it `--pod-match runpod_spread.yaml`.

tmux:

| Do | Keys or command |
|---|---|
| Leave the session, run continues | `Ctrl-b` then `d` |
| Come back | `tmux attach -t spread` |
| List sessions | `tmux ls` |
| Scroll back | `Ctrl-b` then `[`, arrows or PgUp, `q` to leave |
| Stop the run, with a checkpoint | `Ctrl-c` once, in the session |
| Close the session | `exit` in it, or `tmux kill-session -t spread` |

With `pod_run.sh`, `Ctrl-c` also stops the pod, three minutes after
`train.py` has saved its checkpoint and the result has been kept.

From any other shell on the pod:

```bash
tail -f /workspace/runs/spread/console.log
```

### What you will see

At the start, one line confirms both options are on:

```
spread: peak penalty on (weight 4, strongest 2 percent of 48 px windows, with the image loss); hiding on (up to 50 percent of the reader's patch blanked, training steps only)
```

Then one line every 1,000 steps, also written to `logs/progress.csv`, and one
line for every sample sheet. The numbers in the examples below are made up to
show the format:

```
[report] step    3000  bit_acc 0.871  psnr 12.3 dB  (mean of last 1000 steps)  0.21 h  cost $0.14
sample sheet: /workspace/runs/spread/logs/samples/step_00004000.png  (strongest spot is 1.6 times the tile average)
```

- `bit_acc` and `psnr` are means over the last 1,000 training steps. Chance
  is 0.500. The reader's patch has regions blanked in those steps, so expect
  `bit_acc` to run below the laptop's numbers at the same step.
- `strongest spot` on the sheet line is the mark's energy in its strongest
  48 pixel windows, divided by the average over the tile, on one training
  batch. 1 is perfectly even. It is high for the corner patch (20 in the
  laptop's long run) and also for a mark that follows the image content
  (about 4), so it is only a number to look at.
- After each sheet line comes a `[spread]` line, measured on 256 fixed
  images; its rows are also in `logs/spread.csv`. Its first number, `fixed
  place`, is the one that matters: it is high only for a patch at the same
  place in every tile. A mark that follows the image reads 1.4 to 2.7, the
  laptop's corner patch 18 to 20 (`/docs/runs.md`, "The fixed-place baseline
  and the limit of 4").
- `cost` is `--price-per-hour` times the time `train.py` has been running on
  this run, summed over restarts. It is an estimate of the GPU bill only. It
  leaves out setup and download time, time the pod idles after the run, and
  storage. The RunPod billing page is the real figure.
- The PSNR stays near 12 dB while the image loss is off, which it is until
  bit accuracy passes 0.95. On the laptop that took about 4,000 steps. With
  regions blanked it may take longer. If the PSNR is still below 14 dB at
  step 15,000, the image loss never switched on, and the run stops itself
  (see below).

The 45-minute check prints one of:

```
learning check at step 9000, 45 minutes in: mean bit accuracy over the last 1000 steps is 0.612 (chance is 0.500, required 0.75) -> NOT LEARNING, stopping the run
learning check at step 9000, 45 minutes in: mean bit accuracy over the last 1000 steps is 0.931 (chance is 0.500, required 0.75) -> passed, the run continues
```

A stopped run saves a checkpoint and exits with code 3. It then refuses to
resume until you add `early_stop.enabled=false` (continue anyway) or `--fresh`
(start over; old checkpoints are moved to a `replaced-...` folder). Passing the
check means only that the reader is learning. It says nothing about the
corner patch.

The thresholds are in `configs/runpod_spread.yaml` under `early_stop`.

The run stops itself in two more cases. Both save a checkpoint first, and
with the stop-the-pod command the pod stops too.

**The mark piles up at the same place in every tile**
(`spread.stop_above: 4.0`, exit code 4). Checked at every sample sheet, on
the `fixed place` number of the `[spread]` line:

```
Stopped: at step 12000 the fixed-place measure is 4.62, above the limit of 4 (spread.stop_above): the mark is piling up at the same place in every tile. Strongest spot 6.10, 31.0 percent of the mark's energy in one 96 px square. The same command would resume and stop again at the next sample sheet unless the limit is changed.
```

That is the result the run exists to find out: the two options did not
prevent a fixed patch. A mark that follows the image content does not trip
this; it is accepted. To go on regardless, add `spread.stop_above=null` to
the command.

**The image loss never switched on** (`image_loss_check`, exit code 5). One
look at step 15,000, at the mean PSNR of the last 1,000 steps:

```
image-loss check at step 15000: mean PSNR over the last 1000 steps is 12.4 dB (required 14) -> image loss never switched on, stopping the run
image-loss check at step 15000: mean PSNR over the last 1000 steps is 21.7 dB (required 14) -> passed, the run continues
```

A run stopped this way refuses to resume until you add
`image_loss_check.enabled=false` or `--fresh`. It means the reader never
reached 0.95 bit accuracy with regions of its patch blanked, so the mark is
still at full strength. The 45-minute check does not catch that: 0.75 is
enough to pass it.

**Reading too weak** (`reading_check`, exit code 7). One look at `at_step`
(80,000 for an invisibility run), at the mean bit accuracy of the last
`log.report_every` steps: below `min_bit_acc`, the run stops with a checkpoint.
It refuses to resume until you add `reading_check.enabled=false` or `--fresh`.

**The hours limit** (`budget.max_hours: 24`, exit code 6). The other stops
judge the training; this one only counts. Checked at every step, on the
hours of the report line: the time `train.py` has spent on this run, summed
over restarts.

```
Stopped: the run reached its limit at step 131000: 24.00 h of training, the limit is 24 h (budget.max_hours). cost $16.56.
```

It saves a checkpoint first, and a run stopped this way refuses to resume at
the same limit. To go on, raise it or switch it off on the command line,
after the price: `budget.max_hours=36` or `budget.max_hours=null`.
`budget.max_cost_usd` is the same on the cost of the report line and needs
`--price-per-hour`; it is off. Both limit what `train.py` counts. Setup, the
download, a pod idling after the run and storage are not in it, so the
account balance is still the only limit on the bill (above). A pod started
from a commit before this limit existed (`bf9c7a0` or older) does not have
it.

### After a restart

Checkpoints (every 2,000 steps), TensorBoard files, sample sheets,
`progress.csv` and `run_state.json` are all under `/workspace/runs/spread`, on
the volume. After a pod restart or a new pod on the same volume: run section
4, then the same command from section 5. It prints `resumed from ... at step
N` and carries the time and cost on. If the newest checkpoint file is
damaged, the one before it is used.

## 6. Copy a sample sheet to the laptop

Sheets are PNG files, about 3 MB each, one every 2,000 steps:

```bash
ls /workspace/runs/spread/logs/samples/
```

**With `runpodctl`** (no public IP needed; `runpodctl` is on every pod and
has to be installed on the laptop once, see RunPod's installation guide). On
the pod, in a shell outside the tmux session:

```bash
runpodctl send /workspace/runs/spread/logs/samples/step_00010000.png
```

It prints a one-time code and the command to run on the other computer. On
the laptop, in the folder where the file should land, run that command:

```bash
runpodctl receive <the code it printed>
```

**With `scp`** (needs the public IP and port from section 2). On the laptop;
the IP and port are in the pod's **Connect** panel, and `-i` names the SSH key
registered with RunPod:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 root@<pod ip>:/workspace/runs/spread/logs/samples/step_00010000.png .
```

All sheets at once:

```bash
scp -P <port> -i ~/.ssh/id_ed25519 -r root@<pod ip>:/workspace/runs/spread/logs/samples ./spread-samples
```

Copying does not disturb the run.

**What to look for.** Each row is original, marked, difference times 10.

- Before the image loss is on (PSNR near 12 dB) the mark covers the whole
  tile at full strength in every run. Those sheets say nothing yet.
- The laptop run's patch showed between steps 6,000 and 10,000, two to six
  thousand steps after its image loss switched on. Judge this run the same
  distance after its PSNR starts to rise.
- Good: no bright patch at the same place in every row, and `fixed place`
  on the `[spread]` line stays under about 2.
- Accepted: the mark sits on the busy parts of each image and leaves the
  flat parts nearly clean, differently in every row. The laptop's
  penalty-only run did this from step 10,000 (`fixed place` 1.59).
- Bad: the difference column goes grey with a bright patch at the corners in
  every row, as in the laptop's `runs/long/samples/step_00020000.png`
  (`fixed place` 18 to 20). The run stops itself at 4.

The images are dataset images under CC BY 2.0: credit the authors from
`ATTRIBUTION.csv` before publishing a sheet.

## 7. Stop the pod so billing ends

Stopping the run (`Ctrl-c`) does **not** stop billing. The pod does.

1. **Stop the run** if it is still going: `Ctrl-c` once in the tmux session,
   and wait for `saved ...step_XXXXXXXX.pt`.
2. **Keep the result**: run `keep_result.py` on the laptop and wait for
   `CHECK PASSED` (section 10). Stopping the pod loses nothing, so this can
   also be done after step 3, on a pod started again. Anything else you want,
   for example every sample sheet, is a plain copy (needs the public-IP SSH,
   see section 2):

   ```bash
   scp -P <port> -i ~/.ssh/id_ed25519 -r root@<pod ip>:/workspace/runs/spread/logs ./runpod-spread-logs
   ```

3. **Stop or terminate the pod.** In the console: **Pods > your pod > Stop**,
   or **Terminate**. From a shell on the pod:

   ```bash
   runpodctl pod stop "$RUNPOD_POD_ID"
   ```

   ```bash
   runpodctl pod delete "$RUNPOD_POD_ID"
   ```

   Either ends the GPU charge. Stop keeps the pod in your list for a restart
   and wipes its container disk (the pip packages); terminate (`delete`)
   removes the pod and everything not on a network volume. With a network
   volume mounted at `/workspace`, nothing of the run is lost either way.
4. **Confirm in the console** that no pod is listed as running.
5. **The network volume is still billed** ($0.07 per GB per month) until you
   delete it: **Storage > Network Volumes > Delete**, or
   `runpodctl network-volume delete <volume-id>`. That erases the dataset and
   every checkpoint on it for good. **Delete the volume only after
   `keep_result.py` has printed `CHECK PASSED`** (section 10). Billing has
   ended completely only when there is no pod and no volume.

## 8. The direct reader

Dropped for the gate, and not recommended: every test with rotation ended at
chance (`/docs/runs.md`, "Direct mode"). The config still runs. Use the steps
above with `direct` in place of `spread` in the paths and the session name,
`configs/runpod_direct.yaml` as the config, and

```bash
CONFIG=configs/runpod_direct.yaml bash setup_runpod.sh
```

for the setup, and `bash pod_run.sh direct configs/runpod_direct.yaml ...` to
run it. Its check is at 30 minutes and 0.55.

## 9. Email reports

`notify.py` runs **on the laptop**, not on the pod. It watches the laptop's
run and the pod's run and sends email through Gmail:

| When | What |
|---|---|
| Every `notify.every_steps` training steps (10,000, `configs/default.yaml`), each run | the newest row, **every spread row since the last email**, hours and cost so far, an estimate of what is left, and the newest sample sheet attached |
| A run ends | at once, whatever the step: finished, one of the automatic stops (with its exit code), an interrupt, or a crash, with the rows since the last email, the newest sample sheet and the last console lines |
| The pod does not answer for 5 minutes | at once, with the last row seen |
| No new progress row for 30 minutes | at once, for either run |

Only the email is less often. The spread check, the sample sheets, the
checkpoints and the automatic stop on a piled-up mark stay at every 2,000
steps on the pod (they are `train.py`'s), so a pile-up still stops the run
within 2,000 steps and its ENDED message carries the rows that show it.
`--every-steps N` overrides the interval for one watcher; `--every-steps 0`
sends no progress reports at all, only the end and the alerts (used for the
2,000-step check run). `--config <file>` reads `notify.every_steps` from
another config.

The password never goes to the pod and is never in the repository.

**It needs full SSH to the pod**: a public IP with TCP port 22 exposed, and
your SSH key registered with RunPod (section 2). The proxied "basic SSH"
cannot run a command. The pod's address and port are in its **Connect**
panel and change with every new pod.

### Set up, once

1. In your Google account, **2-Step Verification** has to be on
   (myaccount.google.com, Security). App passwords do not exist without it.
   They are also unavailable if 2-Step Verification is set up for security
   keys only, on a work or school account, or with Advanced Protection.
2. Open **https://myaccount.google.com/apppasswords** and sign in again if
   asked.
3. Give it a name, for example `ShadowCode notifier`, and create it.
4. Google shows 16 letters. Copy them. They are not shown again.
5. Put them in a file of their own, one line, **outside the repository**:

   ```powershell
   notepad $HOME\.shadowcode\gmail_app_password.txt
   ```

   Paste, save, close. The spaces between the groups do not matter. The
   folder already holds `notify.json` with the address the mail goes from
   and to.
6. Send the test message, from `training`:

   ```powershell
   python notify.py --test
   ```

   It prints `sent: test message` and the message arrives with the laptop
   run's newest sample sheet attached. Do not rely on the watcher before
   this has worked.

To take the password back: the same Google page, **Remove**. Google also
revokes every app password when you change your account password; make a
new one then.

**Why it cannot be committed.** The file is in `~/.shadowcode`, outside the
repository, so git cannot add it. `podlink.py` refuses to read the folder if
it is ever moved inside a git repository. The repository's `.gitignore` also
ignores `.shadowcode/`, `*app_password*` and `notify.json`, in case a copy is
dropped in by mistake. The password is never printed, logged or written
anywhere else.

### Run it

In a terminal of its own, from `training`, with the pod's address and the
run's name (the one given to `pod_run.sh`):

```powershell
python notify.py --run labels_generated --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>
```

`--no-pod` watches the laptop run only, `--no-laptop` the pod run only.
`Ctrl-c` stops it; the same command carries on from what it has already sent
(`runs/notify_state.json`, kept per run name, so a new run starts with
nothing sent). The first look sends each run's newest row, so you know it is
working (not with `--every-steps 0`). It recognises the pod run by the `log.dir` that `pod_run.sh` passes
to `train.py`; for a run started another way, add `--pod-match <text in its
command line>`.

It also copies the pod's `progress.csv` and `spread.csv` to
`training/runs/pod/<run>/` at every look, and each sheet it emails to
`training/runs/pod/<run>/samples/`. It refuses a mirror folder that holds a
laptop run (the laptop's own `runs/labels_generated`, for example). The table
command in `/docs/runs.md`, "Spread run (RunPod)", works on that copy.

### What it cannot do

- **It only runs while the laptop is awake** and its terminal is open. No
  mail is not good news: a report should come every 10,000 steps (about 38
  minutes on the RTX PRO 4000 pod). If they stop, look.
- **It reads the laptop run without touching it**: only `progress.csv`,
  `spread.csv`, `console.log` and finished sample sheets. It never opens
  that run's checkpoints or `run_state.json`, which the run replaces and
  deletes; on Windows that fails while another program has them open.
- **Why a pod run ended** is read from the `train.py exited with code` line
  of the stop-the-pod command (section 5). A pod that was started with an
  older command stops within seconds, and you get "cannot be reached" five
  minutes later instead of the reason.
- A key with a passphrase needs `ssh-agent`: the watcher never asks for
  input. If a new pod reuses an address, ssh refuses the changed host key;
  delete `~/.shadowcode/known_hosts` then.

Tested on the laptop: reports, the stops, a crash, an unreachable pod and a
stalled run, against the laptop run's real logs and against a stand-in for
ssh; `--run` with a benchmark folder written by `pod_run.sh` (the `ENDED`
message names the `--max-minutes` stop). Not tested: a real pod.

## 10. Keep the result

**The pod and the volume may only be deleted after this has printed
`CHECK PASSED`.** Until then the only copy of the trained model is on the
volume.

One checkpoint is **243 MB** (243,175,661 bytes on the laptop, for the same
models): embedder, reader, the optimizer's state and the schedule state. The
run keeps the newest three on the volume.

When the run has ended, on the laptop, from `training`, with the run's name:

```powershell
python keep_result.py --run labels_generated --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>
```

It does three things and changes nothing on the pod outside
`/workspace/runs/<run>/kept` and `/workspace/runs/<run>/export`:

1. **Stage**, on the pod: `finish_run.sh` exports the bundle from the newest
   checkpoint and writes `/workspace/runs/<run>/kept/<bundle id>/` with a
   checksum for every file. If `pod_run.sh` already did this (section 5), it
   is found and not done twice.
2. **Copy** that folder with `scp` to `training/kept/<run>/<bundle id>/`
   (git-ignored), about 270 MB.
3. **Check**: every file on the laptop is hashed again (SHA-256) and compared
   with the pod's list; nothing may be missing or extra; the list itself must
   be the pod's; the bundle must have been exported from this checkpoint.

| It prints | Exit code | Means |
|---|---|---|
| `CHECK PASSED` | 0 | The copy is complete and identical. The pod and volume may be deleted. |
| `CHECK FAILED` | 1 | Do not delete anything. Run the command again; it copies afresh. |
| `COPY GOOD, BUNDLE MISSING` | 2 | Checkpoint, config and commit are safe here, but the export failed on the pod. Export on the laptop with the command it prints, then delete. |

What is in `training/kept/<run>/<bundle id>/`:

| | |
|---|---|
| `checkpoint/step_XXXXXXXX.pt` | the final checkpoint |
| `bundle/` | `reader.onnx`, `contract.json`, `fixtures/`, as `export_bundle.py` writes them |
| `config/` | the config files, and `resolved.json`: the config stored in the checkpoint, with every override the run was started with |
| `COMMIT` | the commit of the code on the pod, and any local changes there |
| `run/` | `progress.csv`, `spread.csv`, `run_state.json`, `console.log`, the last sample sheets |
| `environment.txt` | Python, PyTorch and package versions, the GPU |
| `SHA256SUMS`, `VERIFIED.txt` | the pod's checksums, and when the copy last passed |

The bundle id is the date of the export and the short commit, for example
`20261122-bf9c7a0`. To check a copy again later, without a pod:

```powershell
python keep_result.py --check kept/<run>/<bundle id>
```

**The pod has to be running** for the first command, with full SSH (section
2). If it has stopped itself:

- Start it again in the console. If step 1 was already done by the
  stop-the-pod command, nothing needs installing: go straight to the command.
  Otherwise run section 4 first; the export needs the pip packages.
- If the console says there are no GPUs, deploy a new pod from the same
  volume. Any GPU will do; the copy needs none.
- Stop the pod again when `CHECK PASSED` is printed. This command does not.

Tested on the laptop against a stand-in for ssh, with the cancelled long
run's step 26,000 checkpoint: staged, exported, copied and checked in 23
seconds; again with `--run`, on a copy of the laptop's generated-label step
6,056 checkpoint (19 seconds, `CHECK PASSED`); a flipped byte, a missing file, an extra file, an unreachable pod
and a failed export are each reported. It is the first export made from a
trained checkpoint. Not tested: a real pod, and the export under the pod's
PyTorch version.

### If the models will be served behind an API

`bundle/` holds only the **reader**, as ONNX. That is what the browser needs.
A server that also embeds needs more, and `kept/` has most of it:

- **The embedder.** It is not exported. Its weights are in the checkpoint
  (`models.embedder`); to run it you need the checkpoint and the model code.
- **The code at the commit in `COMMIT`**: `models/`, `band.py`,
  `constants.py`, `config.py`, `geometry.py`. The checkpoint is weights
  only; it cannot be loaded without the classes that built it. The
  repository has it as long as that commit is pushed and never rewritten;
  a `git archive` of it next to the checkpoint removes the doubt.
- **`config/resolved.json`**: the model is built from it (`build_models`),
  not from `default.yaml` as it may read by then.
- **`environment.txt`**: the PyTorch version the weights were trained and
  checked with.
- **The flatten step.** The reader takes an upright patch of exactly one
  tile. `contract.json` states the step and `geometry.py` is the reference.
  Something has to find the four corners first; the locator is not built.
- **The message code.** The models carry 64 raw bits. Error correction and
  the checksum are outside them (`/docs/message-format.md`, the codec in
  `/web`).
- **`bundle/fixtures/`**: frames with the logits the reader must give. Any
  server build can be checked against them.
- Not needed to serve: the optimizer state (two thirds of the checkpoint),
  the dataset, the LPIPS weights. Needed only to train on: all three.
- Licences: `THIRD_PARTY.md`. The dataset images are CC BY 2.0 and are in
  the sample sheets under `run/samples/`, not in the models' files.
