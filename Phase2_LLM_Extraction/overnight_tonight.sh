#!/usr/bin/env bash
# OVERNIGHT 2026-08-25. Start once, go to sleep.
#
#   1. stop MINE, so qwen 6e gets the machine (they both hold e5; 8 GB is not enough for two,
#      and that pairing already OOM-killed gemini's 6e on 2026-08-24)
#   2. wait for 6e
#   3. when 6e's graphs exist, run MINE on validation, then on test
#
# THE GATE IS THE GRAPHS, NOT THE MANIFEST. On 2026-08-24 gemini's 6e wrote all 80 graphs and was
# then killed before closing its manifest; a script that waited for the manifest waited forever and
# cost a night. This waits for the 80 consolidated graphs OR a closed manifest, whichever comes
# first, and gives up if 6e dies without producing them.
#
# Everything here is FREE: qwen and llama are on the cluster. No API spend, nothing irreversible.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/opt/anaconda3/bin/python
LOG="$HERE/runs/overnight_$(date +%Y%m%d-%H%M%S).log"
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

GRAPH='nary_graph_dedup_main_test_r1_extended_consolidated.json'
n_graphs() { ls "$HERE"/results/qwen3-32b-think/art*/e2e/"$GRAPH" 2>/dev/null | wc -l | tr -d ' '; }
manifest_closed() {
  "$PY" - <<'EOF'
import json, sys
from pathlib import Path
try:
    d = json.loads(Path("results/qwen3-32b-think/_runs/e2e__dedup__main_test_r1.json").read_text())
    sys.exit(0 if d["stages"].get("6e", {}).get("finished") else 1)
except Exception:
    sys.exit(1)
EOF
}
e2e_running() { pgrep -f "6e_e2e_relation_canonicalization" >/dev/null 2>&1; }

cd "$HERE"

# ── 1. stop MINE ─────────────────────────────────────────────────────────────────────────────
if pgrep -f "10_mine_benchmark" >/dev/null 2>&1; then
  say "stopping MINE so 6e gets the cores and the RAM"
  pkill -f "10_mine_benchmark" 2>/dev/null
  sleep 3
fi
say "6e running: $(e2e_running && echo yes || echo NO)   consolidated graphs: $(n_graphs)/80"

# ── 2. wait for 6e ───────────────────────────────────────────────────────────────────────────
say "waiting for qwen 6e (checks every 5 min, gives up after 7h or if it dies)"
deadline=$((SECONDS + 25200))
while [ $SECONDS -lt $deadline ]; do
  if [ "$(n_graphs)" -ge 80 ] || manifest_closed; then
    say "6e DONE — $(n_graphs)/80 consolidated graphs on disk"; break
  fi
  if ! e2e_running; then
    say "6e is NOT running and only $(n_graphs)/80 graphs exist — it died. Stopping here."
    say "  morning: check the newest runs/*__main_test_r1/6e.log, then restart 6e."
    say "  (the malformed-record guard is in place, so this would be a NEW failure worth reading)"
    exit 1
  fi
  sleep 300
done
if [ "$(n_graphs)" -lt 80 ] && ! manifest_closed; then
  say "gave up waiting; $(n_graphs)/80 graphs. Not starting MINE."; exit 1
fi

# ── 3. MINE, validation then test ────────────────────────────────────────────────────────────
# Timeout 600s not the usual 3600: a MINE judge call takes ~17s, so 600 still has huge margin but
# a hung request surfaces in 10 minutes instead of an hour. Both stalls on 2026-08-24 were requests
# blocking against the 1-hour default.
export VLLM_BASE_URL=http://localhost:8001/v1
export VLLM_TIMEOUT_S=600

say "MINE validation (3 models, 4 variants, judge llama33-70b)"
./run_mine.sh llama33-70b val >>"$LOG" 2>&1 \
  && say "  OK  MINE val" || say "  FAILED MINE val — see $LOG"

say "MINE test (3 models, 2 variants)"
./run_mine.sh llama33-70b test >>"$LOG" 2>&1 \
  && say "  OK  MINE test" || say "  FAILED MINE test — see $LOG"

# ── summary for the morning ──────────────────────────────────────────────────────────────────
say "done. State:"
( cd notebooks && "$PY" -m run_manifest index >/dev/null 2>&1 )
grep -E "main_test_r1" results/RUNS.md | tee -a "$LOG"
say "MINE reports:"
ls -1 results/*/_global/evaluation/mine_report_llama33-70b_*.json 2>/dev/null | sed 's|.*/results/|  |' | tee -a "$LOG" || say "  none"
say "full log: $LOG"
