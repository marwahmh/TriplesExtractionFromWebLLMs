#!/usr/bin/env bash
# OVERNIGHT B — finish the qwen3-32b-think test run (6d -> 6e). Start NOW, own terminal.
#
# It BLOCKS until gemini's 6e closes its manifest, then starts. That wait is not politeness: 6d and
# 6e each load e5-base-v2 AND the ms-marco cross-encoder in-process, and this machine has 8 GB of
# RAM with swap already near its limit. Two such stages at once risks an OOM that would take both
# runs down — including the gemini 6e you have paid for.
#
# FREE: qwen is served on the cluster. No API spend.
#
# Grounding is deliberately NOT here. It waits for the judge benchmark (overnight C) to say which
# judge to use; see run_grounding.sh, which is ready to go once that is decided.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/opt/anaconda3/bin/python
LOG="$HERE/runs/overnight_qwen_$(date +%Y%m%d-%H%M%S).log"
export PYDEVD_DISABLE_FILE_VALIDATION=1 VLLM_TIMEOUT_S=3600

say() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }

gemini_6e_done() {
  "$PY" - <<'EOF'
import json, sys
from pathlib import Path
p = Path("results/gemini-2.5-flash/_runs/e2e__dedup__main_test_r1.json")
try:
    d = json.loads(p.read_text())
    sys.exit(0 if d["stages"].get("6e", {}).get("finished") else 1)
except Exception:
    sys.exit(1)
EOF
}

cd "$HERE"

say "waiting for gemini 6e to close (checked every 5 min) — qwen 6d/6e must not share RAM with it"
while ! gemini_6e_done; do sleep 300; done
say "gemini 6e CLOSED. Starting qwen 6d -> 6e on the test split."

# The vLLM tunnel must still be up. run_e2e.sh preflights it and refuses to start if not.
MODEL=qwen3-32b-think SPLIT=test REPEAT=1 EXPERIMENT=main ./run_e2e.sh 6d 6e >>"$LOG" 2>&1 \
  && say "OK  qwen 6d+6e finished" \
  || say "FAILED qwen 6d+6e — see $LOG (resume with fresh:false, do NOT plain re-run: fresh=true reprocesses all 80)"

say "final state:"
( cd notebooks && "$PY" -m run_manifest index >/dev/null 2>&1 )
grep -E "main_test_r1" results/RUNS.md | tee -a "$LOG"
n=$(ls results/qwen3-32b-think/art*/e2e/nary_graph_dedup_main_test_r1_extended_consolidated.json 2>/dev/null | wc -l | tr -d ' ')
say "qwen consolidated graphs: $n/80"
