#!/usr/bin/env bash
# OVERNIGHT C — add deepseek-v4-flash and gpt-oss-120b to the judge benchmark.
# Start this NOW, in its own terminal. Independent of everything else.
#
# WHY THESE TWO. The benchmark currently ranks qwen3-32b-think against llama33-70b, and both are
# reported pipeline models — so whichever judges the grounding, one model ends up marking its own
# homework, and using a different judge per model would make the arms incomparable. deepseek and
# gpt-oss are OUTSIDE the reported set: if either matches qwen on the benchmark, it can judge all
# three models' graphs with no self-report and no comparability caveat at all.
#
# SAFE: nb9d has REBUILD=False, so the 500 human-labelled items and the gold key are only READ.
# Section 2b skips any judge whose verdict file already exists, so the qwen/llama verdicts already
# scored cannot be overwritten.
#
# COST: hosted on Fireworks ($FIREWORKS_API_KEY), NOT your Gemini balance and NOT the cluster.
# One qwen run over the 500 items cost ~150k tokens; at deepseek's $0.14/$0.28 per 1M that is a few
# cents per run. Three runs each, both judges: well under a dollar.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/opt/anaconda3/bin/python
NB="$HERE/notebooks/0_shared/9d_judge_benchmark.ipynb"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/runs/${STAMP}__judge_bench_hosted"
mkdir -p "$OUT"

export PYDEVD_DISABLE_FILE_VALIDATION=1
# Three repeats each: the benchmark's whole point is that one run is not a measurement. qwen varied
# across a 2.0-point band on identical items, and llama was bit-identical — the spread IS a result.
# ONE JUDGE PER INVOCATION, deliberately. Section 2b writes verdicts only AFTER its whole task
# pool finishes, so putting both judges in one run makes it all-or-nothing: killing it at 80%
# loses everything. deepseek is cheap and non-thinking, so it goes first and banks its result
# before gpt-oss — which reasons unconditionally and has a recorded history of long calls and 504s
# — is allowed to start. Section 2b also skips any judge already on disk, so re-running is free.
run_judge() {   # $1 backend
  echo "[$(date +%H:%M)] judge benchmark: $1 x3 over 500 items"
  BENCH_JUDGES="[{\"backend\":\"$1\",\"repeats\":3}]" \
  bash -c "cd '$HERE/notebooks/0_shared' && '$PY' -m jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
      --output-dir '$OUT' --output '9d_$1.ipynb' '$NB'" >"$OUT/9d_$1.log" 2>&1 \
    && echo "[$(date +%H:%M)]   OK  $1" \
    || echo "[$(date +%H:%M)]   FAILED $1 — see $OUT/9d_$1.log (the other judge is unaffected)"
}

run_judge deepseek-v4-flash
run_judge gpt-oss-120b

# Score every judge on disk, old and new together, so the morning view is one table.
echo
echo "=== all judges scored against the human key ==="
( cd "$HERE/notebooks" && "$PY" - <<'EOF'
import sys, statistics as st
from pathlib import Path
sys.path.insert(0, '.')
import judge_benchmark as jb
out = Path('/Users/marwah/Documents/Thesis/TriplesExtractionFromWebLLMs/Phase2_LLM_Extraction/results/gemini-2.5-flash/_global')
res = jb.compare(out)
if not isinstance(res, list):
    print("no scores returned"); raise SystemExit
groups = {}
for r in res:
    groups.setdefault(r['tag'].rsplit('_run', 1)[0], []).append(r)
print(f"\n{'judge':<46}{'runs':>5}{'recall':>9}{'FPR':>7}{'prec':>7}{'acc':>7}")
for name, g in sorted(groups.items(), key=lambda kv: -st.mean(r['recall'] for r in kv[1])):
    m = lambda k: st.mean(r[k] for r in g) * 100
    print(f"  {name:<44}{len(g):>5}{m('recall'):>8.1f}%{m('fpr'):>6.1f}%{m('precision'):>6.1f}%{m('accuracy'):>6.1f}%")
print("\nper-operator recall (share of corrupted facts caught):")
ops = ['number_edit','object_swap','subject_swap','hedge_flip','property_swap','direction_flip']
hdr = "  " + f"{'operator':<16}" + "".join(f"{n.split('_p')[0][:14]:>15}" for n in groups)
print(hdr)
for op in ops:
    row = f"  {op:<16}"
    for name, g in groups.items():
        vals = [r['by_op'][op]['caught'] / r['by_op'][op]['n'] * 100 for r in g if op in r.get('by_op', {})]
        row += f"{st.mean(vals):>14.1f}%" if vals else f"{'-':>15}"
    print(row)
EOF
) 2>&1 | tee -a "$OUT/9d.log"
echo
echo "[$(date +%H:%M)] done. Full log: $OUT/9d.log"
