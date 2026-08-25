#!/usr/bin/env bash
# MINE recall benchmark over every finished graph set, with ONE judge.
#
#   ./run_mine.sh llama33-70b                    # both splits, e2e consolidated
#   ./run_mine.sh llama33-70b --dry-run
#   ./run_mine.sh llama33-70b test               # test split only
#   ./run_mine.sh llama33-70b val                # validation only (adds the staged arm)
#   MINE_VARIANTS=e2e_consolidated,staged_consolidated ./run_mine.sh llama33-70b val
#
# ONE judge across every arm, for the reason nb10 states itself: a per-model judge would confound
# "this model extracts less" with "this judge is stricter".
#
# FORCE=1 IS SET DELIBERATELY. nb10 resumes by skipping any article that already wrote a judgment
# log — whoever judged it. Existing logs on this tree were written by earlier llama and qwen runs,
# so without FORCE most articles would keep their OLD verdicts while the report claimed the current
# judge. nb10 detects that ("!! MIXED JUDGES on disk") but the honest fix is to re-judge, which is
# what FORCE does. It costs a full pass; it is free on the cluster judges.
#
# SELF-JUDGING: llama33-70b and qwen3-32b-think are both REPORTED models, so either one judging its
# own tree is a self-report — the script says so when it happens. gpt-oss-120b and deepseek-v4-flash
# are outside the reported set and have no such problem, but they bill Fireworks.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/opt/anaconda3/bin/python
NB="$HERE/notebooks/evaluation/10_mine_benchmark.ipynb"

JUDGE="${1:-}"
WHICH="${2:-both}"
if [ -z "$JUDGE" ]; then
  echo "usage: $0 <judge-backend> [test|val|both] [--dry-run]"
  echo "  judges: llama33-70b | qwen3-32b-think | gpt-oss-120b | deepseek-v4-flash"
  exit 2
fi
DRY=""
for a in "$@"; do [ "$a" = "--dry-run" ] && DRY=1; done
[ "$WHICH" = "--dry-run" ] && WHICH=both

export PYDEVD_DISABLE_FILE_VALIDATION=1
# Respect a caller-supplied timeout. This used to be pinned to 3600, which silently undid the
# shorter value the overnight wrapper set and left wedged requests blocking for a full hour.
# A MINE judge call takes ~17s, so anything above ~300 is already generous.
export VLLM_TIMEOUT_S="${VLLM_TIMEOUT_S:-3600}"
export MINE_BACKEND="$JUDGE"
# FORCE=1 re-judges everything. That is right when SWITCHING judges, because nb10 resumes by
# skipping any article that already has a log whoever judged it, so a resumed run would silently
# mix verdicts from two judges under one report. It is wrong when RESUMING an interrupted run of
# the SAME judge: score_article writes each article's log as it finishes, so MINE_FORCE=0 picks up
# exactly where it stopped and re-judges nothing.
#   MINE_FORCE=0 ./run_mine.sh llama33-70b val    # resume the same judge after a crash
export MINE_FORCE="${MINE_FORCE:-1}"

# Which judge pruned the *_grounded graphs. A grounded-variant score is "recall AFTER that judge
# removed what it found ungrounded", so the pruning judge is part of the number's identity and must
# be reported with it. Read from the graphs themselves rather than assumed.
pruning_judge() {
  "$PY" - <<'EOF' 2>/dev/null || echo unknown
import json, glob
from collections import Counter
c = Counter()
for f in glob.glob('results/*/art*/e2e/nary_graph_grounded_pruned.json'):
    try: c[(json.load(open(f)).get('grounding_config') or {}).get('backend')] += 1
    except Exception: pass
print(", ".join(f"{k} ({v})" for k, v in c.most_common()) or "unknown")
EOF
}

run_split() {   # $1 tag   $2 default variants
  local TAG="$1" VARS="${MINE_VARIANTS:-$2}"
  local present=()
  for M in gemini-2.5-flash llama33-70b qwen3-32b-think; do
    n=$(ls "$HERE/results/$M/"art*/e2e/"nary_graph_dedup_${TAG}_extended_consolidated.json" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -gt 0 ]; then
      present+=("$M")
      [ "$M" = "$JUDGE" ] && echo "  NOTE  $M is judging its OWN graphs on $TAG — self-report"
    else
      echo "  SKIP  $M / $TAG — no consolidated graphs"
    fi
  done
  if [ ${#present[@]} -eq 0 ]; then echo "  nothing to score for $TAG"; return; fi
  local MODELS; MODELS=$(IFS=,; echo "${present[*]}")
  echo "[$(date +%H:%M)] MINE $TAG — models: $MODELS | judge $JUDGE | variants $VARS"
  if [ -n "$DRY" ]; then echo "  (dry run — nothing judged)"; return; fi
  local OUT="$HERE/runs/$(date +%Y%m%d-%H%M%S)__mine_${TAG}_by_${JUDGE}"
  mkdir -p "$OUT"
  MINE_MODELS="$MODELS" MINE_TAG="$TAG" MINE_VARIANTS="$VARS" \
  bash -c "cd '$HERE/notebooks/evaluation' && '$PY' -m jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
      --output-dir '$OUT' --output 10.ipynb '$NB'" >"$OUT/10.log" 2>&1 \
    && echo "[$(date +%H:%M)]   OK  $TAG" \
    || echo "[$(date +%H:%M)]   FAILED $TAG — see $OUT/10.log"
}

# Validation has both tracks (the staged arm exists only there); test is e2e only.
echo "  grounded variants were pruned by: $(pruning_judge)"
echo
if [ "$WHICH" = "test" ] || [ "$WHICH" = "both" ]; then
  run_split main_test_r1 "e2e_consolidated,e2e_grounded"
fi
if [ "$WHICH" = "val" ] || [ "$WHICH" = "both" ]; then
  run_split main_val_r1  "e2e_consolidated,e2e_grounded,staged_consolidated,staged_grounded"
fi

echo
echo "MINE reports on disk:"
ls -1 "$HERE"/results/*/_global/evaluation/mine_report_"$JUDGE"_*.json 2>/dev/null | sed 's|.*/results/|  |' || echo "  none"
