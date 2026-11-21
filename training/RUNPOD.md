# Running the direct reader on RunPod

The pod trains the direct reader (`configs/runpod_direct.yaml`: no locator, no
flatten) while the laptop trains the main reader.

**Direct mode has not yet been shown to learn.** Nothing in this repository
has trained it above chance. The run checks itself 30 minutes in and stops if
bit accuracy is still near 50 percent, so a failed attempt costs about half an
hour of GPU time, provided the pod is stopped afterwards (see "Run").

**This guide has not been run on a pod.** The RunPod facts in it (mount path,
what is billed, `runpodctl` commands) were read from docs.runpod.io on
2026-11-21. `setup_runpod.sh` was run only on the laptop against an ordinary
folder standing in for the volume. Prices change; check the console.

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
  at about 4.3 GB on the laptop; direct mode peaked at 3.6 GB in a short test without LPIPS.
- Volume mount path: `/workspace` (the default). The config assumes it.
- Container disk: the template default. It only has to hold the pip packages.
- On-demand, not spot (interruptible), unless you accept restarts. A restarted run
  resumes from its newest checkpoint and loses at most 2,000 steps.
- For `scp` or `rsync`, the pod needs a public IP with TCP port 22 exposed.
  RunPod's proxied "basic SSH" does not support them.

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
tmux new -s direct
```

Inside it:

```bash
cd /workspace/ShadowCode/training && export TORCH_HOME=/workspace/cache/torch && mkdir -p /workspace/runs/direct
```

Then one of the two commands below, with your pod's price in place of `0.00`.

**Stop the pod when the run ends** (finished, stopped by the 30-minute check,
or crashed). This is the one to use if you will not be watching:

```bash
python3 train.py --config configs/runpod_direct.yaml --price-per-hour 0.00 2>&1 | tee -ia /workspace/runs/direct/console.log; runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
```

`setup_runpod.sh` says at the end whether `RUNPOD_POD_ID` and `runpodctl` are
present. If they are not, this command cannot stop the pod. The second form
after `||` is the older `runpodctl` syntax, for images that ship an older
version. Check in the console afterwards that the pod really is stopped.

**Leave the pod running** when the run ends (it keeps billing until you stop it):

```bash
python3 train.py --config configs/runpod_direct.yaml --price-per-hour 0.00 2>&1 | tee -ia /workspace/runs/direct/console.log
```

tmux:

| Do | Keys or command |
|---|---|
| Leave the session, run continues | `Ctrl-b` then `d` |
| Come back | `tmux attach -t direct` |
| List sessions | `tmux ls` |
| Scroll back | `Ctrl-b` then `[`, arrows or PgUp, `q` to leave |
| Stop the run, with a checkpoint | `Ctrl-c` once, in the session |
| Close the session | `exit` in it, or `tmux kill-session -t direct` |

With the stop-the-pod command, `Ctrl-c` also stops the pod, because
`runpodctl` runs as soon as `train.py` exits.

From any other shell on the pod:

```bash
tail -f /workspace/runs/direct/logs/progress.csv
```

### What you will see

One line every 1,000 steps, also written to `progress.csv`. The numbers in
the examples below are made up to show the format:

```
[report] step    3000  bit_acc 0.502  psnr 19.8 dB  (mean of last 1000 steps)  0.21 h  cost $0.14
```

- `bit_acc` and `psnr` are means over the last 1,000 training steps. Chance is 0.500.
- `cost` is `--price-per-hour` times the time `train.py` has been running on
  this run, summed over restarts. It is an estimate of the GPU bill only. It
  leaves out setup and download time, time the pod idles after the run, and
  storage. The RunPod billing page is the real figure.
- The PSNR stays low while the image loss is off, which it is until bit
  accuracy passes 0.95.

The 30-minute check prints one of:

```
learning check at step 9000, 30 minutes in: mean bit accuracy over the last 1000 steps is 0.501 (chance is 0.500, required 0.55) -> NOT LEARNING, stopping the run
learning check at step 9000, 30 minutes in: mean bit accuracy over the last 1000 steps is 0.731 (chance is 0.500, required 0.55) -> passed, the run continues
```

A stopped run saves a checkpoint and exits with code 3. It then refuses to
resume until you add `early_stop.enabled=false` (continue anyway) or `--fresh`
(start over; old checkpoints are moved to a `replaced-...` folder). Passing the
check means only that accuracy left chance. It does not mean direct mode works.

The thresholds are in `configs/runpod_direct.yaml` under `early_stop`.

### After a restart

Checkpoints (every 2,000 steps), TensorBoard files, `progress.csv` and
`run_state.json` are all under `/workspace/runs/direct`, on the volume. After
a pod restart or a new pod on the same volume: run section 4, then the same
command from section 5. It prints `resumed from ... at step N` and carries the
time and cost on. If the newest checkpoint file is damaged, the one before it
is used.

## 6. Stop the pod so billing ends

Stopping the run (`Ctrl-c`) does **not** stop billing. The pod does.

1. **Stop the run** if it is still going: `Ctrl-c` once in the tmux session,
   and wait for `saved ...step_XXXXXXXX.pt`.
2. **Copy what you want to keep** to the laptop (needs the public-IP SSH, see
   section 2):

   ```bash
   scp -P <port> -r root@<pod ip>:/workspace/runs/direct ./runpod-direct
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
