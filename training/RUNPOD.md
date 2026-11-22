# Running the spread run on RunPod

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

**This guide has not been run on a pod.** The RunPod facts in it (mount path,
what is billed, `runpodctl` commands, file transfer) were read from
docs.runpod.io on 2026-11-21. `setup_runpod.sh` was run only on the laptop
against an ordinary folder standing in for the volume. Prices change; check
the console.

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
   then `cd /workspace/ShadowCode && git checkout training/pipeline && git log -1 --oneline`
7. Set up; it downloads the dataset the first time (section 4):
   `cd /workspace/ShadowCode/training && bash setup_runpod.sh`
   It has to end with `ready - nothing has been started`, say that
   `runpodctl` was found, and list the stops, the 24 hours among them.
8. Open tmux and set the folder (section 5):
   `tmux new -s spread`, then
   `cd /workspace/ShadowCode/training && export TORCH_HOME=/workspace/cache/torch && mkdir -p /workspace/runs/spread`
9. Start the run with the long **stop-the-pod command** of section 5, with
   your price in place of `0.00`.
10. Read the first lines before you leave. They have to show `device=cuda`,
    `spread: peak penalty on ...; hiding on ...`, and
    `budget: the run stops, with a checkpoint, at 24 h of training`.
11. Leave tmux with `Ctrl-b` then `d`, and start the watchdog of section 5
    (the `tmux new -d -s watchdog ...` line).

**On the laptop**

12. In a terminal of its own, in `training`, start the watcher (section 9):
    `python notify.py --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>`
    It prints one `sent:` line per run within a minute or two.

**While it runs**

13. A report arrives every 2,000 steps for each run. Section 5, "What you
    will see", says what the numbers mean and section 6 what a good and a
    bad sample sheet look like. The run can stop itself at 45 minutes (not
    learning), at step 15,000 (image loss never on), at any sample sheet
    (mark piled up) or at 24 hours. Each stop sends an `ENDED` message.

**When the `ENDED` message arrives**

14. The pod stops itself about three minutes later. Check in the console
    that it has.
15. Start the pod again (or deploy any pod on the same volume) and, on the
    laptop, in `training`, copy and check the result (section 10):
    `python keep_result.py --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>`
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

**Stop the pod when the run ends** (finished, stopped by one of its four
stops, or crashed). This is the one to use if you will not be watching:

```bash
python3 train.py --config configs/runpod_spread.yaml --price-per-hour 0.00 2>&1 | tee -ia /workspace/runs/spread/console.log; echo "train.py exited with code ${PIPESTATUS[0]} at $(date -u)" | tee -a /workspace/runs/spread/console.log; bash finish_run.sh 2>&1 | tee -a /workspace/runs/spread/console.log; sleep 180; until runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"; do sleep 60; done
```

When `train.py` exits, for any reason, this does four things in order:

1. writes its exit code to `console.log`;
2. runs `finish_run.sh`, which exports the bundle and puts everything worth
   keeping into `/workspace/runs/spread/kept/<bundle id>/` with checksums
   (section 10), while the pip packages the export needs are still there;
3. waits three minutes, so the laptop's watcher (section 9) can read why the
   run ended before the pod goes away;
4. asks RunPod to stop the pod, again every minute until that works.

Nothing in steps 2 and 3 can prevent step 4. If `finish_run.sh` is not on the
pod (a checkout from before it existed) or fails, the line says so and goes
on. A crash in the first seconds stops the pod too, about three minutes
later; the reason is then in `console.log`, on the volume.

`setup_runpod.sh` says at the end whether `RUNPOD_POD_ID` and `runpodctl` are
present. If they are not, this command cannot stop the pod. The second form
after `||` is the older `runpodctl` syntax, for images that ship an older
version. Check in the console afterwards that the pod really is stopped.

**If the run hangs** it never exits, and the command above never reaches the
stop. Start this once, from any shell on the pod, after the run has started.
It stops the pod when `progress.csv` has had no new row for an hour (a row
comes every 1,000 steps, about 8 minutes on the laptop):

```bash
tmux new -d -s watchdog 's=$(date +%s); f=/workspace/runs/spread/logs/progress.csv; while sleep 300; do m=$(stat -c %Y "$f" 2>/dev/null || echo 0); [ "$m" -lt "$s" ] && m=$s; if [ $(( $(date +%s) - m )) -gt 3600 ]; then echo "no report line for an hour at $(date -u): stopping the pod" >> /workspace/runs/spread/console.log; until runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"; do sleep 60; done; fi; done'
```

The hour counts from when the watchdog starts, so an old `progress.csv` from
an earlier pod session does not trip it. `tmux kill-session -t watchdog`
ends it. It has to be started again after every pod start.

Both commands were tested on the laptop with a stand-in for `runpodctl`
(exit codes 4 and 1, a stop that fails twice before it works, a run that
stops writing rows). Neither has stopped a real pod yet, and the `tmux`
wrapper of the second was not run.

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

   Either ends the GPU charge. Stop keeps the pod in your list for a restart;
   terminate (`delete`) removes it. With a network volume nothing is lost
   either way: everything under `/workspace` stays on the volume.
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

for the setup. Its check is at 30 minutes and 0.55.

## 9. Email reports

`notify.py` runs **on the laptop**, not on the pod. It watches the laptop's
run and the pod's run and sends email through Gmail:

| When | What |
|---|---|
| Every sample sheet (2,000 steps), each run | the report row, the last rows, hours and cost so far, an estimate of what is left, and the sample sheet attached |
| A run ends | at once: finished, one of the automatic stops (with its exit code), an interrupt, or a crash, with the last console lines |
| The pod does not answer for 5 minutes | at once, with the last row seen |
| No new progress row for 30 minutes | at once, for either run |

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

In a terminal of its own, from `training`, with the pod's address:

```powershell
python notify.py --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>
```

`--no-pod` watches the laptop run only. `Ctrl-c` stops it; the same command
carries on from what it has already sent (`runs/notify_state.json`). The
first look sends each run's newest row, so you know it is working.

It also copies the pod's `progress.csv` and `spread.csv` to
`training/runs/spread/` at every look, and each sheet it emails to
`training/runs/spread/samples/`. The table command in `/docs/runs.md`,
"Spread run (RunPod)", works on that copy as it is.

### What it cannot do

- **It only runs while the laptop is awake** and its terminal is open. No
  mail is not good news: a report should come every 2,000 steps. If they
  stop, look.
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
ssh. Not tested: a real pod.

## 10. Keep the result

**The pod and the volume may only be deleted after this has printed
`CHECK PASSED`.** Until then the only copy of the trained model is on the
volume.

One checkpoint is **243 MB** (243,175,661 bytes on the laptop, for the same
models): embedder, reader, the optimizer's state and the schedule state. The
run keeps the newest three on the volume.

When the run has ended, on the laptop, from `training`:

```powershell
python keep_result.py --pod root@<pod ip> --port <port> --key $HOME\.ssh\<key>
```

It does three things and changes nothing on the pod outside
`/workspace/runs/spread/kept` and `/workspace/runs/spread/export`:

1. **Stage**, on the pod: `finish_run.sh` exports the bundle from the newest
   checkpoint and writes `/workspace/runs/spread/kept/<bundle id>/` with a
   checksum for every file. If the stop-the-pod command already did this
   (section 5), it is found and not done twice.
2. **Copy** that folder with `scp` to `training/kept/<bundle id>/`
   (git-ignored), about 270 MB.
3. **Check**: every file on the laptop is hashed again (SHA-256) and compared
   with the pod's list; nothing may be missing or extra; the list itself must
   be the pod's; the bundle must have been exported from this checkpoint.

| It prints | Exit code | Means |
|---|---|---|
| `CHECK PASSED` | 0 | The copy is complete and identical. The pod and volume may be deleted. |
| `CHECK FAILED` | 1 | Do not delete anything. Run the command again; it copies afresh. |
| `COPY GOOD, BUNDLE MISSING` | 2 | Checkpoint, config and commit are safe here, but the export failed on the pod. Export on the laptop with the command it prints, then delete. |

What is in `training/kept/<bundle id>/`:

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
python keep_result.py --check kept/<bundle id>
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
seconds; a flipped byte, a missing file, an extra file, an unreachable pod
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
