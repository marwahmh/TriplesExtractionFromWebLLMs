#!/usr/bin/env bash
# Run MINE to completion, restarting it whenever it wedges.
#
#   ./run_mine_watchdog.sh llama33-70b            # both splits, until nothing is left
#   ./run_mine_watchdog.sh llama33-70b val        # one split
#   STALL_MIN=10 MAX_ROUNDS=12 ./run_mine_watchdog.sh llama33-70b
#
# WHY THIS EXISTS. The judge occasionally produces a generation that trickles tokens indefinitely.
# httpx's timeout is a READ timeout, so it never fires on a slow-but-alive stream, and the worker
# holding that request is lost for good. Observed three times on 2026-08-25: the server stays
# healthy and answers new requests in under a second while MINE sits at 0% CPU forever.
#
# There is no total-request deadline available in the client, so instead of preventing the wedge we
# detect it and restart. That is safe and cheap because score_article writes each article's log as
# it finishes and MINE_FORCE=0 skips what is already on disk, so every round resumes exactly where
# the last one stopped. Each round completed roughly 115 scorings before wedging.
#
# Stops when a round leaves the log count unchanged AND the run exited on its own (nothing left to
# do), or after MAX_ROUNDS. Free: the judge is on the cluster.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/opt/anaconda3/bin/python
JUDGE="${1:-llama33-70b}"
SPLIT="${2:-}"
STALL_MIN="${STALL_MIN:-10}"          # minutes with no new log before we call it wedged
MAX_ROUNDS="${MAX_ROUNDS:-12}"
WINDOW_MIN="${WINDOW_MIN:-10}"                 # throughput window
MIN_LOGS_PER_WINDOW="${MIN_LOGS_PER_WINDOW:-14}"  # healthy is ~2 scorings/min = 40+ logs/10min
POLL=60

LOG="$HERE/runs/mine_watchdog_$(date +%Y%m%d-%H%M%S).log"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

n_logs() { find "$HERE/results" -name 'mine_judgments*.json' 2>/dev/null | wc -l | tr -d ' '; }
mine_running() { pgrep -f "10_mine_benchmark" >/dev/null 2>&1; }

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8001/v1}"
export VLLM_TIMEOUT_S="${VLLM_TIMEOUT_S:-300}"
export MINE_FORCE=0                   # resume: never re-judge what is already on disk
export MINE_WORKERS="${MINE_WORKERS:-12}"   # see the note in nb10: wedged workers block the queue

cd "$HERE"
say "watchdog: judge=$JUDGE split=${SPLIT:-both} stall=${STALL_MIN}min max_rounds=$MAX_ROUNDS"
say "starting from $(n_logs) judgment log(s) on disk"

for round in $(seq 1 "$MAX_ROUNDS"); do
  start_logs=$(n_logs)
  say "--- round $round: launching MINE (resume from $start_logs logs) ---"
  ./run_mine.sh "$JUDGE" ${SPLIT:+$SPLIT} >>"$LOG" 2>&1 &
  runner=$!

  last_change=$SECONDS
  last_count=$start_logs
  win_start=$SECONDS          # throughput window
  win_count=$start_logs
  while kill -0 "$runner" 2>/dev/null; do
    sleep "$POLL"
    now=$(n_logs)
    # THROUGHPUT FLOOR. Silence is not the only failure: when only SOME workers wedge, the run keeps
    # writing the occasional log and a silence-only detector never fires, so it crawls at a fraction
    # of capacity forever. Measured 2026-08-25: llama answers a MINE-shaped call in ~8.6s and the
    # machine was idle, which predicts ~0.4 min/scoring across 4 workers, yet throughput was 2.7 --
    # about four workers' worth of work being done by one. Restarting clears the wedged workers.
    if [ $((SECONDS - win_start)) -ge $((WINDOW_MIN * 60)) ]; then
      gained=$((now - win_count))
      if [ "$gained" -lt "$MIN_LOGS_PER_WINDOW" ]; then
        say "  degraded: only $gained log(s) in ${WINDOW_MIN} min (floor $MIN_LOGS_PER_WINDOW) — workers likely wedged, restarting"
        pkill -f "10_mine_benchmark" 2>/dev/null; sleep 5; kill "$runner" 2>/dev/null; break
      fi
      win_start=$SECONDS; win_count=$now
    fi
    if [ "$now" -ne "$last_count" ]; then
      last_count=$now
      last_change=$SECONDS
    elif [ $((SECONDS - last_change)) -ge $((STALL_MIN * 60)) ]; then
      say "  wedged: no new log for ${STALL_MIN} min (at $now) — restarting"
      pkill -f "10_mine_benchmark" 2>/dev/null
      sleep 5
      kill "$runner" 2>/dev/null
      break
    fi
  done
  wait "$runner" 2>/dev/null
  pkill -f "10_mine_benchmark" 2>/dev/null; sleep 3

  end_logs=$(n_logs)
  say "  round $round: $start_logs -> $end_logs logs (+$((end_logs - start_logs)))"

  # A round that added nothing and was not killed for wedging means there is nothing left to judge.
  if [ "$end_logs" -eq "$start_logs" ] && [ $((SECONDS - last_change)) -lt $((STALL_MIN * 60)) ]; then
    say "  no work left — stopping"
    break
  fi
done

say "finished. $(n_logs) judgment log(s) on disk."
say "reports:"
ls -1 "$HERE"/results/*/_global/evaluation/mine_report_"$JUDGE"_*.json 2>/dev/null \
  | sed 's|.*/results/|  |' | tee -a "$LOG" || say "  none"
say "log: $LOG"
