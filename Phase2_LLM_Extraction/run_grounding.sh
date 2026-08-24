#!/usr/bin/env bash
# Ground EVERY finished graph set with ONE judge — both splits, both tracks, every model.
#
#   ./run_grounding.sh gpt-oss-120b              # the benchmark winner (91.3% recall, 92.6% acc)
#   ./run_grounding.sh gpt-oss-120b --dry-run    # list what would run, judge nothing
#   ./run_grounding.sh deepseek-v4-flash         # cheaper external judge
#   ./run_grounding.sh qwen3-32b-think           # free, but self-judges its own graphs
#
# ONE judge across every arm is the point: a grounding rate is a statement about a (graph, judge)
# pair, so arms judged by different judges are not comparable. The judge benchmark (nb9d, 500
# human-labelled items) is what picks it.
#
# WHAT IT SWEEPS — each combination is skipped when its consolidated graphs do not exist yet:
#   test  / e2e     gemini, llama, qwen        80 articles each
#   val   / e2e     gemini, llama, qwen        20 articles each
#   val   / staged  gemini, llama, qwen        20 articles each
#
# OVERWRITE NOTE. The per-article outputs (nary_graph_grounded_fulltext.json / _pruned.json) are
# NOT judge-tagged, so this REPLACES whatever judged them last. The qwen3-32b-think versions were
# archived to results_archive/grounded_by_qwen3-32b-think_20260824/ on 2026-08-24. The _global
# summaries ARE tagged (grounding_summary_<judge>_<track>_<tag>.json), so every judge's headline
# numbers coexist and can be compared.
#
# COST. vLLM judges are free on the cluster. Hosted judges bill Fireworks: gpt-oss-120b was
# measured at ~178 input / ~191 output tokens per assertion, so the full ~27k-assertion sweep is
# roughly $4.45 at $0.22/$0.66 per 1M — less with prompt caching.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/opt/anaconda3/bin/python
NB="$HERE/notebooks/0_shared/9b_grounding_fulltext.ipynb"

JUDGE="${1:-}"
DRY="${2:-}"
if [ -z "$JUDGE" ]; then
  echo "usage: $0 <judge-backend> [--dry-run]"
  echo "  judges: gpt-oss-120b | deepseek-v4-flash | qwen3-32b-think | llama33-70b"
  exit 2
fi

MODELS="gemini-2.5-flash llama33-70b qwen3-32b-think"
# label | track dir | run tag
COMBOS="test_e2e|e2e|main_test_r1
val_e2e|e2e|main_val_r1
val_staged|two_staged/3_nary_graph|main_val_r1"

ok=0; skip=0; fail=0
for MODEL in $MODELS; do
  while IFS='|' read -r LABEL TRACK TAG; do
    [ -z "$LABEL" ] && continue
    GRAPH="nary_graph_dedup_${TAG}_extended_consolidated.json"
    n=$(ls "$HERE/results/$MODEL/"art*/"$TRACK/$GRAPH" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -eq 0 ]; then
      echo "  SKIP  $MODEL / $LABEL — no consolidated graphs"
      skip=$((skip+1)); continue
    fi
    if [ "$MODEL" = "$JUDGE" ]; then
      echo "  NOTE  $MODEL / $LABEL — judge is the model that WROTE these graphs (self-report)"
    fi
    if [ "$DRY" = "--dry-run" ]; then
      echo "  WOULD $MODEL / $LABEL — $n articles, track $TRACK, tag $TAG"
      ok=$((ok+1)); continue
    fi
    OUT="$HERE/runs/$(date +%Y%m%d-%H%M%S)__ground_${MODEL}_${LABEL}_by_${JUDGE}"
    mkdir -p "$OUT"
    echo "[$(date +%H:%M)] $MODEL / $LABEL — $n articles, judge $JUDGE"
    GROUND_MODEL="$MODEL" GROUND_BACKEND="$JUDGE" GROUND_TAG="$TAG" \
    GROUND_GRAPH="$GRAPH" GROUND_TRACK="$TRACK" \
    bash -c "cd '$HERE/notebooks/0_shared' && '$PY' -m jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
        --output-dir '$OUT' --output 9b.ipynb '$NB'" >"$OUT/9b.log" 2>&1 \
      && { echo "[$(date +%H:%M)]   OK"; ok=$((ok+1)); } \
      || { echo "[$(date +%H:%M)]   FAILED — see $OUT/9b.log"; fail=$((fail+1)); }
  done <<< "$COMBOS"
done

echo
echo "done: $ok ran, $skip skipped, $fail failed"
echo
echo "grounding summaries by $JUDGE:"
ls -1 "$HERE"/results/*/_global/grounding_summary_"$JUDGE"_* 2>/dev/null | sed 's|.*/results/|  |' || echo "  none"
