"""Email reports for the training runs, sent from the laptop. Nothing runs on the pod.

    python notify.py --test                           one test email, then exit
    python notify.py --pod root@<pod ip> --port <port> --key ~/.ssh/<key>
    python notify.py --no-pod                         the laptop run only

It watches two runs and sends, through Gmail:

* a report for every spread row (one per sample sheet, every 2,000 steps),
  with the sample sheet attached;
* at once: a run that ended (finished, one of the automatic stops, a crash,
  an interrupt), a pod that cannot be reached for 5 minutes, and 30 minutes
  with no new progress row.

The laptop run is only read, and only through files it appends to
(progress.csv, spread.csv, console.log) and finished sample sheets. It never
opens that run's checkpoints or run_state.json: the run replaces and deletes
those, and on Windows that fails while another program has them open.

The pod is read over ssh once a minute (public IP and TCP port 22; RUNPOD.md
section 2). Its small log files are mirrored to runs/spread/ here, so the
table command in docs/runs.md works on the mirror, and its sheets are copied
as they are emailed.

Address, password and pod details are outside the repository, in
~/.shadowcode/ (podlink.py). RUNPOD.md, "Email reports", has the setup.

Standard library only. It never imports torch.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import smtplib
import ssl
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from podlink import NO_WINDOW, PASSWORD_FILE, SETTINGS_FILE, Pod, last_line, load_settings, pod_from, require_outside_repository

HERE = Path(__file__).resolve().parent
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
UNREACHABLE_AFTER_S = 5 * 60

# What the last lines of the console say about why train.py is no longer
# running. Checked from the start of the newest session; the last match wins.
SESSION_START = re.compile(r"^(device=|launched )")
EXIT_LINE = re.compile(r"train\.py exited with code (\d+)")
ENDINGS = [
    (re.compile(r"^Stopped: at step \d+ the fixed-place measure"), "automatic stop: the mark piled up at a fixed place (exit code 4)"),
    (re.compile(r"NOT LEARNING|^Stopped: bit accuracy is below"), "automatic stop: the learning check failed (exit code 3)"),
    (re.compile(r"image loss never switched on"), "automatic stop: the image loss never switched on (exit code 5)"),
    (re.compile(r"^Stopped:"), "automatic stop"),
    (re.compile(r"^finished \d+ steps"), "finished"),
    (re.compile(r"^time limit reached"), "stopped by --max-minutes, checkpoint saved"),
    (re.compile(r"^stopped by SIG"), "interrupted (Ctrl-C or SIGTERM), checkpoint saved"),
    (re.compile(r"^Traceback \(most recent call last\)"), "crash"),
]
EXIT_CODES = {
    0: "finished",
    3: "automatic stop: the learning check failed (exit code 3)",
    4: "automatic stop: the mark piled up at a fixed place (exit code 4)",
    5: "automatic stop: the image loss never switched on (exit code 5)",
    6: "automatic stop: the limit on hours or cost was reached (exit code 6)",
    130: "interrupted (Ctrl-C or SIGTERM), checkpoint saved",
}

# Sent to the pod on stdin. Prints its small log files between marker lines.
POD_SNAPSHOT = r"""
cd "$1" 2>/dev/null || { echo "@@NORUN"; exit 0; }
for f in logs/progress.csv logs/spread.csv logs/run_state.json; do echo "@@FILE $f"; cat "$f" 2>/dev/null; echo; done
echo "@@FILE console_tail"; tail -n 120 console.log 2>/dev/null; echo
n=0; for c in /proc/[0-9]*/cmdline; do tr '\0' ' ' < "$c" 2>/dev/null | grep -q "[t]rain\.py.*$2" && n=$((n+1)); done
echo "@@ALIVE $n"
echo "@@END"
"""


def log(text: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {text}", flush=True)


# -- reading a run ----------------------------------------------------------------


def read_rows(path: Path) -> list[dict]:
    """Rows of a CSV the run appends to. A last line still being written is dropped."""
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if None not in row.values() and row.get("step", "").isdigit()]


def read_tail(path: Path, lines: int = 120) -> list[str]:
    if not path.is_file():
        return []
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 64_000))
        text = f.read().decode("utf-8", errors="replace")
    return text.splitlines()[-lines:]


def laptop_alive(match: str) -> bool | None:
    """Is a train.py with ``match`` in its command line running here? None: could not tell."""
    if os.name == "nt":
        cmd = [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' } | ForEach-Object { $_.CommandLine }",
        ]
    else:
        cmd = ["ps", "-eo", "args"]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=30, creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    text = done.stdout.decode("utf-8", errors="replace")
    return any("train.py" in line and match in line for line in text.splitlines())


class LaptopRun:
    """The run on this machine, read from its own log folder."""

    def __init__(self, name: str, directory: Path, match: str):
        self.name, self.directory, self.match = name, directory, match
        self.key = "laptop"

    def view(self) -> dict:
        return {
            "reachable": True,
            "started": self.directory.is_dir(),
            "progress": read_rows(self.directory / "progress.csv"),
            "spread": read_rows(self.directory / "spread.csv"),
            "console": read_tail(self.directory / "console.log"),
            "alive": laptop_alive(self.match),
        }

    def sheet(self, step: int) -> Path | None:
        path = self.directory / "samples" / f"step_{step:08d}.png"
        return path if path.is_file() else None


class PodRun:
    """The run on the pod: one ssh call per look, mirrored into a local folder."""

    def __init__(self, name: str, pod: Pod, remote: str, mirror: Path, match: str):
        self.name, self.pod, self.remote, self.mirror, self.match = name, pod, remote.rstrip("/"), mirror, match
        self.key = "pod"

    def view(self) -> dict:
        try:
            done = self.pod.run_script(POD_SNAPSHOT, self.remote, self.match, timeout=90)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {"reachable": False, "why": f"{type(error).__name__}: {error}"}
        out = done.stdout.decode("utf-8", errors="replace")
        if done.returncode != 0 or ("@@END" not in out and "@@NORUN" not in out):
            return {"reachable": False, "why": last_line(done.stderr)}
        if "@@NORUN" in out:
            return {"reachable": True, "started": False, "progress": [], "spread": [], "console": [], "alive": False}
        files: dict[str, list[str]] = {}
        current, alive = None, None
        for line in out.splitlines():
            if line.startswith("@@FILE "):
                current = line[7:].strip()
                files[current] = []
            elif line.startswith("@@ALIVE "):
                alive, current = int(line[8:].strip() or 0) > 0, None
            elif line.startswith("@@END"):
                current = None
            elif current is not None:
                files[current].append(line)
        self.mirror.mkdir(parents=True, exist_ok=True)
        for remote_name, local_name in (("logs/progress.csv", "progress.csv"), ("logs/spread.csv", "spread.csv"),
                                        ("logs/run_state.json", "run_state.json"), ("console_tail", "console_tail.log")):
            body = "\n".join(files.get(remote_name, [])).strip("\n")
            if body:
                temporary = self.mirror / (local_name + ".tmp")
                temporary.write_text(body + "\n", encoding="utf-8", newline="\n")
                os.replace(temporary, self.mirror / local_name)
        return {
            "reachable": True,
            "started": True,
            "progress": read_rows(self.mirror / "progress.csv"),
            "spread": read_rows(self.mirror / "spread.csv"),
            "console": [line for line in files.get("console_tail", []) if line.strip()],
            "alive": alive,
        }

    def sheet(self, step: int) -> Path | None:
        name = f"step_{step:08d}.png"
        local = self.mirror / "samples" / name
        if local.is_file():
            return local
        try:
            return self.pod.copy_from(f"{self.remote}/logs/samples/{name}", self.mirror / "samples", timeout=300)
        except (OSError, subprocess.TimeoutExpired) as error:
            log(f"{self.name}: could not copy {name} ({error})")
            return None


# -- what the console says ----------------------------------------------------------


def ending(console: list[str]) -> tuple[str, str | None]:
    """(why train.py is not running, the line that says so) from the newest session in the console."""
    start = max((i for i, line in enumerate(console) if SESSION_START.search(line)), default=0)
    session = console[start:]
    for line in reversed(session):
        found = EXIT_LINE.search(line)
        if found:
            code = int(found.group(1))
            said = next((s for s in reversed(session) if any(p.search(s) for p, _ in ENDINGS)), None)
            if code in EXIT_CODES:
                return EXIT_CODES[code], said
            # A code this file does not know: train.py's own stops all say "Stopped:" first.
            stopped_itself = said is not None and said.startswith("Stopped:")
            return (f"automatic stop (exit code {code})" if stopped_itself else f"crash (exit code {code})"), said
    for line in reversed(session):
        for pattern, reason in ENDINGS:
            if pattern.search(line):
                return reason, line
    return "stopped without a final message: a crash, or the process was killed", None


# -- email --------------------------------------------------------------------------


class Mailer:
    def __init__(self, settings: dict, dry_run: Path | None):
        self.sender = settings.get("gmail_address")
        self.to = settings.get("to") or self.sender
        self.dry_run = dry_run
        self.count = 0
        if not self.sender:
            raise SystemExit(f"stopped: {SETTINGS_FILE} has no \"gmail_address\". RUNPOD.md, \"Email reports\", has the setup.")

    def password(self) -> str:
        require_outside_repository(PASSWORD_FILE)
        if not PASSWORD_FILE.is_file():
            raise SystemExit(f"stopped: no app password at {PASSWORD_FILE}. RUNPOD.md, \"Email reports\", says how to make one.")
        secret = "".join(PASSWORD_FILE.read_text(encoding="utf-8-sig").split())
        if len(secret) != 16 or not secret.isalpha():
            log(f"warning: {PASSWORD_FILE} does not hold 16 letters; a Gmail app password is 16 letters")
        return secret

    def send(self, subject: str, body: str, attachment: Path | None = None) -> bool:
        """True once the message is handed to Gmail (or written out, in a dry run)."""
        message = EmailMessage()
        message["From"], message["To"], message["Subject"] = self.sender, self.to, f"[ShadowCode] {subject}"
        message.set_content(body)
        if attachment is not None:
            message.add_attachment(attachment.read_bytes(), maintype="image", subtype="png", filename=attachment.name)
        try:
            if self.dry_run is not None:
                self.dry_run.mkdir(parents=True, exist_ok=True)
                self.count = len(list(self.dry_run.glob("*.eml"))) + 1
                (self.dry_run / f"{self.count:03d}.eml").write_bytes(bytes(message))
            else:
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=60, context=ssl.create_default_context()) as smtp:
                    smtp.login(self.sender, self.password())
                    smtp.send_message(message)
        except smtplib.SMTPAuthenticationError:
            log("NOT SENT: Gmail refused the login. Check the address in notify.json and make a new app password.")
            return False
        except (OSError, smtplib.SMTPException) as error:
            log(f"NOT SENT ({type(error).__name__}: {error}); it will be tried again: {subject}")
            return False
        log(f"sent: {subject}" + (f"  [{attachment.name}]" if attachment else ""))
        return True


def number(text: str | None, digits: int) -> str:
    try:
        return f"{float(text):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def pace(progress: list[dict], total_steps: int) -> list[str]:
    """Hours, cost and what is left, from progress.csv. Estimates, and marked as such."""
    if not progress:
        return []
    last = progress[-1]
    lines = [f"Training time so far: {number(last['elapsed_hours'], 2)} h"]
    if last.get("cost_usd"):
        lines[0] += f", cost so far ${number(last['cost_usd'], 2)} (price per hour times training time)"
    recent = progress[-6:]
    hours = float(recent[-1]["elapsed_hours"]) - float(recent[0]["elapsed_hours"])
    steps = int(recent[-1]["step"]) - int(recent[0]["step"])
    if hours > 0 and steps > 0:
        left = (total_steps - int(last["step"])) / (steps / hours)
        lines.append(f"Pace: {steps / hours:,.0f} steps per hour. Estimate: {left:.1f} h to step {total_steps:,} at this pace.")
        if last.get("cost_usd") and float(last["elapsed_hours"]) > 0:
            per_hour = float(last["cost_usd"]) / float(last["elapsed_hours"])
            lines.append(f"Estimate: ${float(last['cost_usd']) + left * per_hour:.2f} for the whole run at this pace.")
    return lines


def table(spread: list[dict], count: int = 6) -> list[str]:
    lines = ["step | fixed place | strongest spot | 96 px square | busier half | PSNR dB | bit accuracy"]
    for row in spread[-count:]:
        lines.append(
            f"{int(row['step']):,} | {row['fixed_place']} | {row['strongest_spot']} | "
            f"{float(row['square_share']) * 100:.1f}% | {float(row['busy_share']) * 100:.0f}% | "
            f"{row['psnr_db'] or row['psnr_fixed_db'] + ' (fixed images)'} | {row['bit_acc'] or 'n/a'}"
        )
    return lines


def report_text(name: str, row: dict, view: dict, total_steps: int, note: str = "") -> tuple[str, str]:
    step = int(row["step"])
    subject = (
        f"{name} step {step:,}: fixed place {row['fixed_place']}, "
        f"bit {number(row['bit_acc'], 3)}, PSNR {number(row['psnr_db'] or row['psnr_fixed_db'], 1)} dB"
    )
    body = [
        f"{name}, step {step:,} of {total_steps:,}" + (f"  ({note})" if note else ""),
        f"Row written {row['local_time']} (the clock of the machine the run is on)",
        "",
        f"Fixed place: {row['fixed_place']}   (the run stops itself above its limit, 4 in both run configs)",
        f"Strongest spot: {row['strongest_spot']}",
        f"Energy in one 96 px square: {float(row['square_share']) * 100:.1f} percent",
        f"Energy on the busier half: {float(row['busy_share']) * 100:.0f} percent",
        f"PSNR: {number(row['psnr_db'], 2)} dB (mean of the last 1,000 training steps); {row['psnr_fixed_db']} dB on the fixed images",
        f"Bit accuracy: {number(row['bit_acc'], 4)} (mean of the last 1,000 training steps)",
        "",
        *pace(view["progress"], total_steps),
        "",
        "Latest rows:",
        *table(view["spread"]),
    ]
    return subject, "\n".join(body)


def situation(name: str, view: dict, total_steps: int, console_lines: int = 30) -> str:
    """The last rows and console lines, for the body of an alert."""
    body = []
    if view.get("spread"):
        body += ["Latest spread rows:", *table(view["spread"], 4), ""]
    if view.get("progress"):
        last = view["progress"][-1]
        body += [f"Last progress row: step {int(last['step']):,} at {last['utc_time']}, bit accuracy {last['bit_acc']}, PSNR {last['psnr_db']} dB",
                 *pace(view["progress"], total_steps), ""]
    if view.get("console"):
        body += [f"Last {min(console_lines, len(view['console']))} console lines of {name}:", *view["console"][-console_lines:]]
    return "\n".join(body)


# -- deciding what to send ------------------------------------------------------------


def utc_seconds(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def check(run, st: dict, mailer: Mailer, args) -> None:
    """One look at one run. ``st`` is that run's remembered state; it is changed in place."""
    now = time.time()
    view = run.view()
    st.setdefault("first_look", now)

    if not view["reachable"]:
        st.setdefault("unreachable_since", now)
        gone = now - st["unreachable_since"]
        if not st.get("ended") and not st.get("unreachable_told") and gone >= UNREACHABLE_AFTER_S:
            last = st.get("last_row")
            body = (
                f"{run.name} has not answered over ssh for {gone / 60:.0f} minutes ({view['why']}).\n\n"
                "Either the pod has stopped or been restarted, or its address changed, or the network is down. "
                "Look at the pod in the RunPod console: a pod that is still running is still billed.\n\n"
                + (f"Last row seen: step {last['step']:,} at {last['utc_time']}." if last else "No row was seen before this.")
            )
            st["unreachable_told"] = mailer.send(f"ALERT {run.name}: cannot be reached", body)
    else:
        if st.pop("unreachable_since", None) and st.pop("unreachable_told", None):
            mailer.send(f"{run.name}: reachable again", situation(run.name, view, args.total_steps, 10))
        if not view["started"]:
            return

        # Reports: one per spread row. The first look sends only the newest row.
        rows = view["spread"]
        if rows and "reported_step" not in st:
            subject, body = report_text(run.name, rows[-1], view, args.total_steps, "the watcher has just started")
            if mailer.send(subject, body, run.sheet(int(rows[-1]["step"]))):
                st["reported_step"] = int(rows[-1]["step"])
        for row in rows if "reported_step" in st else []:
            step = int(row["step"])
            if step <= st["reported_step"]:
                continue
            sheet = run.sheet(step)
            subject, body = report_text(run.name, row, view, args.total_steps)
            if sheet is None:
                body += "\n\nThe sample sheet for this step could not be read, so it is not attached."
            if not mailer.send(subject, body, sheet):
                break
            st["reported_step"] = step
        if not rows:
            st.setdefault("reported_step", -1)

        # Ended: train.py is not running. Two looks in a row, or at once when the console gives an exit code.
        alive = view["alive"]
        if alive:
            st["was_alive"], st["not_alive_looks"] = True, 0
            if st.pop("ended", None):
                log(f"{run.name}: running again")
        elif alive is False:
            st["not_alive_looks"] = st.get("not_alive_looks", 0) + 1
            known = st.get("was_alive") or view["progress"] or view["console"]
            exit_line = any(EXIT_LINE.search(line) for line in view["console"][-5:])
            if known and not st.get("ended") and (st["not_alive_looks"] >= 2 or exit_line):
                reason, said = ending(view["console"])
                body = f"{run.name}: train.py is no longer running.\nWhy: {reason}\n" + (f"It said: {said}\n" if said else "")
                if run.key == "pod":
                    body += (
                        "\nIf the run was started with the stop-the-pod command, the pod stops itself now. Check in the "
                        "RunPod console that it has. To keep the result: RUNPOD.md, \"Keep the result\".\n"
                    )
                if not st.get("was_alive"):
                    body += "\nIt was already stopped when the watcher first looked.\n"
                if mailer.send(f"ENDED {run.name}: {reason}", body + "\n" + situation(run.name, view, args.total_steps)):
                    st["ended"] = {"reason": reason, "at": now}

    # Stalled: no new progress row for too long. Counted from the newest row, or from the first look.
    if st.get("ended") or not (st.get("was_alive") or st.get("last_row")):
        if view["reachable"] and view.get("progress"):
            last = view["progress"][-1]
            st["last_row"] = {"step": int(last["step"]), "utc_time": last["utc_time"]}
        return
    if view["reachable"] and view.get("progress"):
        last = view["progress"][-1]
        row = {"step": int(last["step"]), "utc_time": last["utc_time"]}
        if st.get("last_row") != row and st.pop("stall_told", None):
            log(f"{run.name}: rows are arriving again")
        st["last_row"] = row
    since = utc_seconds(st["last_row"]["utc_time"]) if st.get("last_row") else st["first_look"]
    quiet = now - since
    if quiet >= args.stall_minutes * 60 and not st.get("stall_told"):
        last = st.get("last_row")
        body = (
            f"{run.name}: no new progress row for {quiet / 60:.0f} minutes "
            f"(a row comes every 1,000 steps). "
            + (f"The last one is step {last['step']:,} at {last['utc_time']}.\n\n" if last else "No row has been written yet.\n\n")
            + ("The run is hung or very slow, or the pod is gone. A pod that is still running is still billed.\n\n" if run.key == "pod"
               else "The run is hung or very slow, or the laptop was asleep.\n\n")
            + (situation(run.name, view, args.total_steps) if view["reachable"] else "The pod could not be reached for this look.")
        )
        st["stall_told"] = mailer.send(f"ALERT {run.name}: no new row for {quiet / 60:.0f} minutes", body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="send one test email and exit")
    parser.add_argument("--pod", help="root@<pod ip> (default: \"pod\" in notify.json)")
    parser.add_argument("--port", type=int, help="the pod's public SSH port")
    parser.add_argument("--key", help="private key registered with RunPod, e.g. ~/.ssh/id_ed25519")
    parser.add_argument("--no-pod", action="store_true", help="watch the laptop run only")
    parser.add_argument("--no-laptop", action="store_true", help="watch the pod run only")
    parser.add_argument("--laptop-run", default="runs/penalty", help="log folder of the laptop run")
    parser.add_argument("--laptop-match", default="penalty.yaml", help="text in the laptop run's command line")
    parser.add_argument("--pod-run", default="/workspace/runs/spread", help="run folder on the pod")
    parser.add_argument("--pod-match", default="runpod_spread.yaml", help="text in the pod run's command line, after train.py")
    parser.add_argument("--pod-mirror", default="runs/spread", help="local folder the pod's logs and sheets are copied to")
    parser.add_argument("--total-steps", type=int, default=140000)
    parser.add_argument("--interval", type=float, default=60.0, help="seconds between looks")
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--state", default="runs/notify_state.json", help="what has been sent already")
    parser.add_argument("--dry-run", metavar="DIR", help="write the messages to DIR as .eml files instead of sending")
    parser.add_argument("--once", action="store_true", help="look once and exit")
    args = parser.parse_args()

    settings = load_settings()
    mailer = Mailer(settings, Path(args.dry_run) if args.dry_run else None)
    laptop = LaptopRun("penalty run (laptop)", HERE / args.laptop_run, args.laptop_match)

    if args.test:
        rows = read_rows(laptop.directory / "spread.csv")
        sheet = laptop.sheet(int(rows[-1]["step"])) if rows else None
        body = (
            "This is the test message from training/notify.py. If you can read it, the address and the app "
            "password work.\n\n"
            + (f"Attached: the laptop run's newest sample sheet ({sheet.name}), to test attachments.\n" if sheet else
               "No sample sheet was found to attach.\n")
        )
        return 0 if mailer.send("test message", body, sheet) else 1

    runs: list = [] if args.no_laptop else [laptop]
    pod = None if args.no_pod else pod_from(args, settings)
    if pod is not None:
        runs.append(PodRun("spread run (pod)", pod, args.pod_run, HERE / args.pod_mirror, args.pod_match))
    elif not args.no_pod:
        raise SystemExit("stopped: no pod given. Pass --pod root@<pod ip> --port <port>, or --no-pod for the laptop run only.")
    if mailer.dry_run is None:
        mailer.password()  # fail now, not at the first report

    state_path = HERE / args.state
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    log(f"watching: {', '.join(r.name for r in runs)}; reports to {mailer.to}; a look every {args.interval:g} s")
    try:
        while True:
            for run in runs:
                try:
                    check(run, state.setdefault(run.key, {}), mailer, args)
                except Exception as error:  # noqa: BLE001 - one bad look must not end the watcher
                    log(f"{run.name}: this look failed ({type(error).__name__}: {error})")
            temporary = state_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
            os.replace(temporary, state_path)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log("stopped by Ctrl-C; start it again with the same command and it carries on from what it has sent")
        return 0


if __name__ == "__main__":
    sys.exit(main())
