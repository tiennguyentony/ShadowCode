# Running the spread run on RunPod

The pod trains `configs/runpod_spread.yaml`: the long run's settings
(flattened reader, phone camera range, top band, 140,000 steps) with the two
options that keep the mark spread over the whole tile both on. The laptop's
long run puts the code into one bright patch at the tile corner
(`/docs/runs.md`, "Corner patch"); this run is the attempt to fix that.

**No training run has used the two options yet.** They were checked on the
CPU only: they run and gradients flow. The run stops itself, with a
checkpoint, in three cases: bit accuracy under 0.75 after 45 minutes; the
mark piling up (strongest spot above 3 at any sample sheet); PSNR still below
14 dB at step 15,000, which means the image loss never switched on. The
sample sheets, one every 2,000 steps, show the mark itself; section 6 says
how to copy one to the laptop and what to look for.

**This guide has not been run on a pod.** The RunPod facts in it (mount path,
what is billed, `runpodctl` commands, file transfer) were read from
docs.runpod.io on 2026-11-21. `setup_runpod.sh` was run only on the laptop
against an ordinary folder standing in for the volume. Prices change; check
the console.

The direct reader (`configs/runpod_direct.yaml`) is dropped for the gate: it
never left chance with rotation (`/docs/runs.md`, "Direct mode"). The config
is still there; section 8 says how to run it.

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
- On-demand, not spot (interruptible), unless you accept restarts. A restarted run
  resumes from its newest checkpoint and loses at most 2,000 steps.
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
cd /workspace/ShadowCode && git checkout training/pipeline
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

It does four things and starts nothing:

1. Checks that `/workspace` exists and is writable, and prints its size.
2. Installs `requirements.txt` (PyTorch is already in the template) and tmux.
3. Checks the GPU with `nvidia-smi` and by running a convolution on it.
4. Fetches the dataset to `/workspace/datasets/open-images-v7-val` if it is
   not complete there, and the LPIPS weights to `/workspace/cache/torch`.

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

Inside it:

```bash
cd /workspace/ShadowCode/training && export TORCH_HOME=/workspace/cache/torch && mkdir -p /workspace/runs/spread
```

Then one of the two commands below, with your pod's price in place of `0.00`.

**Stop the pod when the run ends** (finished, stopped by one of its three
checks, or crashed). This is the one to use if you will not be watching:

```bash
python3 train.py --config configs/runpod_spread.yaml --price-per-hour 0.00 2>&1 | tee -ia /workspace/runs/spread/console.log; runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
```

`setup_runpod.sh` says at the end whether `RUNPOD_POD_ID` and `runpodctl` are
present. If they are not, this command cannot stop the pod. The second form
after `||` is the older `runpodctl` syntax, for images that ship an older
version. Check in the console afterwards that the pod really is stopped.

**Leave the pod running** when the run ends (it keeps billing until you stop it):

```bash
python3 train.py --config configs/runpod_spread.yaml --price-per-hour 0.00 2>&1 | tee -ia /workspace/runs/spread/console.log
```

tmux:

| Do | Keys or command |
|---|---|
| Leave the session, run continues | `Ctrl-b` then `d` |
| Come back | `tmux attach -t spread` |
| List sessions | `tmux ls` |
| Scroll back | `Ctrl-b` then `[`, arrows or PgUp, `q` to leave |
| Stop the run, with a checkpoint | `Ctrl-c` once, in the session |
| Close the session | `exit` in it, or `tmux kill-session -t spread` |

With the stop-the-pod command, `Ctrl-c` also stops the pod, because
`runpodctl` runs as soon as `train.py` exits.

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
- `strongest spot` is the number for the corner patch: the mark's energy in
  its strongest 48 pixel windows, divided by the average over the tile. 1 is
  perfectly even. Measured on the laptop's checkpoints: 1.0 at step 4,000
  and 1.1 at step 6,000 (even), 3.3 at step 7,001 (starting to pile up),
  20 at steps 20,000 and 22,000 (the corner patch).
- After each sheet line comes a `[spread]` line: the same two measures on 16
  fixed images, which is the steadier number. Its rows are also in
  `logs/spread.csv`.
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

**The mark piles up** (`spread.stop_above: 3.0`, exit code 4). Checked at
every sample sheet, on the 16 fixed images of the `[spread]` line:

```
Stopped: at step 12000 the strongest spot is 3.41 times the tile average, above the limit of 3 (spread.stop_above). 19.0 percent of the mark's energy is in one 96 px square. The same command would resume and stop again at the next sample sheet unless the limit is changed.
```

That is the result the run exists to find out: the two options did not keep
the mark even. To go on regardless, add `spread.stop_above=null` to the
command.

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
- Good: the difference column shows a fine texture over the whole tile and no
  bright corner, and `strongest spot` stays below about 2 well after the PSNR
  has risen past 20 dB.
- Bad: the difference column goes grey with a bright patch at the corners, as
  in the laptop's `runs/long/samples/step_00020000.png`, and `strongest spot`
  climbs: the laptop run was at 3.3 by 20 dB and at 20 by 26 dB.
- The number on the sheet line is from one training batch of 8, so it moves
  a little from sheet to sheet. The trend is what counts.

The images are dataset images under CC BY 2.0: credit the authors from
`ATTRIBUTION.csv` before publishing a sheet.

## 7. Stop the pod so billing ends

Stopping the run (`Ctrl-c`) does **not** stop billing. The pod does.

1. **Stop the run** if it is still going: `Ctrl-c` once in the tmux session,
   and wait for `saved ...step_XXXXXXXX.pt`.
2. **Copy what you want to keep** to the laptop (needs the public-IP SSH, see
   section 2):

   ```bash
   scp -P <port> -i ~/.ssh/id_ed25519 -r root@<pod ip>:/workspace/runs/spread ./runpod-spread
   ```

3. **Stop or terminate the pod.** In the console: **Pods > your pod > Stop**,
   or **Terminate**. From a shell on the pod:

   ```bash
   runpodctl pod stop "$RUNPOD_POD_ID"
   ```

   ```bash
   runpodctl pod delete "$RUNPOD_POD_ID"
   ```

   Either ends the GPU charge. Stop keeps the pod in your list for a restart;
   terminate (`delete`) removes it. With a network volume nothing is lost
   either way: everything under `/workspace` stays on the volume.
4. **Confirm in the console** that no pod is listed as running.
5. **The network volume is still billed** ($0.07 per GB per month) until you
   delete it: **Storage > Network Volumes > Delete**, or
   `runpodctl network-volume delete <volume-id>`. That erases the dataset and
   every checkpoint on it for good, so do step 2 first. Billing has ended
   completely only when there is no pod and no volume.

## 8. The direct reader

Dropped for the gate, and not recommended: every test with rotation ended at
chance (`/docs/runs.md`, "Direct mode"). The config still runs. Use the steps
above with `direct` in place of `spread` in the paths and the session name,
`configs/runpod_direct.yaml` as the config, and

```bash
CONFIG=configs/runpod_direct.yaml bash setup_runpod.sh
```

for the setup. Its check is at 30 minutes and 0.55.
