#!/usr/bin/env bash
# Ground the TEST-split graphs of every model that has them, with ONE judge.
#
#   ./run_grounding.sh qwen3-32b-think          # the current benchmark winner
#   ./run_grounding.sh deepseek-v4-flash        # if the hosted benchmark says it is better
#   ./run_grounding.sh gpt-oss-120b
#   ./run_grounding.sh <judge> llama33-70b      # restrict to one model's graphs
#
# ONE judge for every model is what makes the arms comparable — so run this once, after the judge
# benchmark (overnight_C_judges.sh) has said which judge to trust. Re-running with a different
# judge writes to a different summary file and does not overwrite the first.
#
# Note on self-judging: qwen3-32b-think and llama33-70b are both REPORTED pipeline models, so
# either of them judging its own graphs is a self-report. deepseek-v4-flash and gpt-oss-120b are
# outside the reported set and have no such problem — which is the reason the benchmark was
# extended to them.
#
# nb9b loads no local embedding or reranker model, so this is light on RAM and safe to run
# alongside a canonicalization stage. Cost depends on the judge: the two vLLM judges are free on
# the cluster; the two hosted ones bill Fireworks.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/opt/anaconda3/bin/python
NB="$HERE/notebooks/0_shared/9b_grounding_fulltext.ipynb"

JUDGE="${1:-}"
ONLY="${2:-}"
if [ -z "$JUDGE" ]; then
  echo "usage: $0 <judge-backend> [model]"
  echo "  judges: qwen3-32b-think | llama33-70b | deepseek-v4-flash | gpt-oss-120b"
  exit 2
fi

GRAPH=nary_graph_dedup_main_test_r1_extended_consolidated.json
export PYDEVD_DISABLE_FILE_VALIDATION=1 VLLM_TIMEOUT_S=3600

for MODEL in gemini-2.5-flash llama33-70b qwen3-32b-think; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$MODEL" ] && continue
  n=$(ls "$HERE"/results/"$MODEL"/art*/e2e/"$GRAPH" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$n" -eq 0 ]; then
    echo "[$(date +%H:%M)] SKIP $MODEL — no consolidated test graphs yet"
    continue
  fi
  if [ "$MODEL" = "$JUDGE" ]; then
    echo "[$(date +%H:%M)] NOTE $MODEL is judging its OWN graphs — this is a self-report; say so wherever the number appears"
  fi
  OUT="$HERE/runs/$(date +%Y%m%d-%H%M%S)__grounding_${MODEL}_by_${JUDGE}"
  mkdir -p "$OUT"
  echo "[$(date +%H:%M)] grounding $MODEL ($n graphs), judge $JUDGE"
  GROUND_MODEL="$MODEL" GROUND_BACKEND="$JUDGE" GROUND_TAG=main_test_r1 \
  GROUND_GRAPH="$GRAPH" GROUND_TRACK=e2e \
  bash -c "cd '$HERE/notebooks/0_shared' && '$PY' -m jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
      --output-dir '$OUT' --output 9b.ipynb '$NB'" >"$OUT/9b.log" 2>&1 \
    && echo "[$(date +%H:%M)]   OK  $MODEL" \
    || echo "[$(date +%H:%M)]   FAILED $MODEL — see $OUT/9b.log"
done

echo
echo "grounding summaries on disk:"
ls -1 "$HERE"/results/*/_global/grounding_summary*main_test_r1* 2>/dev/null | sed 's|.*/results/|  |' || echo "  none"
