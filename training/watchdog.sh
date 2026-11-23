#!/usr/bin/env bash
# Stop the pod when a run has written no new progress row for an hour: a hung
# run never exits, so pod_run.sh never reaches its stop.
#
#     bash watchdog.sh labels_generated
#
# pod_run.sh starts this by itself, in the background, and ends it just before
# it stops the pod. To start one by hand (a run started without pod_run.sh),
# from the training folder:
#
#     tmux new -d -s watchdog 'bash watchdog.sh labels_generated'
#
# It watches <VOLUME>/runs/<run name>/logs/progress.csv. A row comes every
# 1,000 steps. The hour counts from when the watchdog starts, so an old
# progress.csv from an earlier pod session does not trip it.
#
# Settings, as environment variables: VOLUME (/workspace), STALL_SECONDS
# (3600), CHECK_EVERY (300), STOP_RETRY (60).
set -u
NAME="${1:-}"
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "usage: bash watchdog.sh <run name>" >&2; exit 2; }
RUN="${VOLUME:-/workspace}/runs/$NAME"
STALL_SECONDS="${STALL_SECONDS:-3600}"
CHECK_EVERY="${CHECK_EVERY:-300}"
STOP_RETRY="${STOP_RETRY:-60}"
f="$RUN/logs/progress.csv"
mkdir -p "$RUN"

s=$(date +%s)
echo "watchdog: watching $f; the pod is stopped after $STALL_SECONDS s without a new row" >> "$RUN/console.log"
# Sleep in the background and wait for it, so that a kill (pod_run.sh ends it) also ends the sleep.
nap=""
trap '[ -n "$nap" ] && kill "$nap" 2>/dev/null; exit 0' TERM INT
while :; do
    sleep "$CHECK_EVERY" &
    nap=$!
    wait "$nap"
    m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    [ "$m" -lt "$s" ] && m=$s
    if [ $(( $(date +%s) - m )) -gt "$STALL_SECONDS" ]; then
        echo "watchdog: no report line for $STALL_SECONDS s at $(date -u): stopping the pod" >> "$RUN/console.log"
        until runpodctl pod stop "${RUNPOD_POD_ID:-}" >> "$RUN/console.log" 2>&1 \
            || runpodctl stop pod "${RUNPOD_POD_ID:-}" >> "$RUN/console.log" 2>&1; do
            sleep "$STOP_RETRY"
        done
        echo "watchdog: runpodctl accepted the stop at $(date -u)" >> "$RUN/console.log"
        exit 0
    fi
done
