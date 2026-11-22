"""Laptop side: settings kept outside the repository, and ssh/scp to the pod.

Used by notify.py and keep_result.py. Standard library only; it never imports
torch, so it cannot touch the GPU a training run is using.

Everything private lives in one folder in the user's home, not in the
repository, so git cannot add it:

    ~/.shadowcode/notify.json              addresses and, optionally, the pod's ssh details
    ~/.shadowcode/gmail_app_password.txt   the Gmail app password, one line
    ~/.shadowcode/known_hosts              host keys of the pods, kept apart from ~/.ssh

The folder can be moved with the environment variable SHADOWCODE_HOME. Code
here refuses to read it if it sits inside a git repository.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

SETTINGS_DIR = Path(os.environ.get("SHADOWCODE_HOME") or Path.home() / ".shadowcode")
SETTINGS_FILE = SETTINGS_DIR / "notify.json"
PASSWORD_FILE = SETTINGS_DIR / "gmail_app_password.txt"
KNOWN_HOSTS = SETTINGS_DIR / "known_hosts"

# No window flashes up on Windows each time ssh or PowerShell is started.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def repository_holding(path: Path) -> Path | None:
    """The git work tree ``path`` lies in, or None."""
    path = path.resolve()
    for folder in (path, *path.parents):
        if (folder / ".git").exists():
            return folder
    return None


def require_outside_repository(path: Path) -> None:
    repo = repository_holding(path)
    if repo is not None:
        raise SystemExit(
            f"stopped: {path} is inside the git repository {repo}. Private settings must stay outside "
            f"every repository so they cannot be committed. Move them to {Path.home() / '.shadowcode'}."
        )


def load_settings() -> dict:
    require_outside_repository(SETTINGS_DIR)
    if not SETTINGS_FILE.is_file():
        return {}
    return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))


def command(name: str) -> list[str]:
    """``ssh`` or ``scp``; PODLINK_SSH / PODLINK_SCP name another program (used by the tests)."""
    override = os.environ.get(f"PODLINK_{name.upper()}")
    return shlex.split(override) if override else [name]


@dataclass
class Pod:
    """One pod, reached over full SSH (public IP, TCP port 22 exposed; RUNPOD.md section 2)."""

    host: str            # root@<pod ip>
    port: int = 22
    key: str | None = None

    def _options(self) -> list[str]:
        SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
        options = [
            "-o", "BatchMode=yes",                      # never wait for a password or a yes/no
            "-o", "ConnectTimeout=15",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "StrictHostKeyChecking=accept-new",   # a new pod is accepted, a changed key is refused
            "-o", f"UserKnownHostsFile={KNOWN_HOSTS.as_posix()}",
        ]
        if self.key:
            options += ["-i", str(Path(self.key).expanduser())]
        return options

    def run_script(self, script: str, *args: str, timeout: float = 120) -> subprocess.CompletedProcess:
        """Run a bash script on the pod. It is sent on stdin, so nothing has to be installed there."""
        cmd = [*command("ssh"), "-p", str(self.port), *self._options(), self.host, "bash", "-s", "--", *args]
        return subprocess.run(
            cmd,
            input=script.replace("\r\n", "\n").encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            creationflags=NO_WINDOW,
        )

    def copy_from(self, remote: str, into: Path, recursive: bool = False, timeout: float = 3600) -> Path:
        """Copy a file or folder from the pod into the local folder ``into``. Returns the local path."""
        into.mkdir(parents=True, exist_ok=True)
        cmd = [*command("scp"), "-P", str(self.port), *self._options()]
        if recursive:
            cmd.append("-r")
        # The destination is "." with the folder as working directory: a Windows
        # path with a drive letter would be read by scp as "host:path".
        cmd += [f"{self.host}:{remote}", "."]
        done = subprocess.run(cmd, cwd=into, capture_output=True, timeout=timeout, creationflags=NO_WINDOW)
        if done.returncode != 0:
            raise OSError(f"scp of {remote} failed: {last_line(done.stderr)}")
        return into / remote.rstrip("/").rsplit("/", 1)[-1]


def last_line(raw: bytes) -> str:
    lines = raw.decode("utf-8", errors="replace").strip().splitlines()
    return lines[-1] if lines else "(no message)"


def pod_from(args, settings: dict) -> Pod | None:
    """The pod named on the command line, else the one in notify.json, else None."""
    saved = settings.get("pod") or {}
    host = getattr(args, "pod", None) or saved.get("host")
    if not host:
        return None
    return Pod(
        host=host,
        port=int(getattr(args, "port", None) or saved.get("port") or 22),
        key=getattr(args, "key", None) or saved.get("key"),
    )
