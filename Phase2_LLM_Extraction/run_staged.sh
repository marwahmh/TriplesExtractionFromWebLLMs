#!/usr/bin/env bash
# Run the STAGED track in order, stopping at the first failure. The twin of run_e2e.sh, so both
# tracks can be run under the SAME experiment name and compared.
#
#   ./run_staged.sh                    # the binary chain: 2 -> 4 -> 6 -> 7
#   ./run_staged.sh 2 4 6b 7b          # the reified chain (nb6b/nb7b) instead
#   ./run_staged.sh 4 6 7              # a subset, in the order given
#   EXPERIMENT=no-rerank ./run_staged.sh       # a named experiment (no notebook edits)
#   REPEAT=2 ./run_staged.sh           # repeat the SAME arm under its own tag (main -> main_r2)
#   SPLIT=val ./run_staged.sh         # the validation split (SPLIT=test for the reported set)
#   MODEL=qwen3-32b-think ./run_staged.sh   # same experiment, another model's tree + backend
#   PHASE2_OVERWRITE=1 ./run_staged.sh # redo a run in place, over its own existing outputs
#   INPLACE=1 ./run_staged.sh          # also write outputs back into the source notebooks
#   SKIP_SERVER_CHECK=1 ./run_staged.sh        # skip the vLLM preflight (e.g. Gemini backend)
#   E2E_PYTHON=/path/to/python ./run_staged.sh # override the interpreter
#
# Order matters and is NOT checked: nb4 needs nb2's entities, nb6/nb6b need nb4's typed entities,
# nb7/nb7b need those relations. Each notebook asserts on its own missing input, naming the stage
# that produces it.
#
# Every notebook reads ONE config file (configs/<EXPERIMENT>.json) via notebooks/experiment_config.py,
# so the stages cannot disagree about the results tree or the run tags. EXPERIMENT reaches them as
# $PHASE2_EXPERIMENT.
#
# Each stage records itself in results/<model>/_runs/<run_id>.json (notebooks/run_manifest.py) and
# REFUSES to start when that record says it already finished under the same tag — so a re-run cannot
# quietly replace outputs that are already reported. REPEAT=<n> for another sample of the same arm,
# a new experiment file for a different configuration, PHASE2_OVERWRITE=1 to redo one in place.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NBDIR="$HERE/notebooks/two_staged"
PY="${E2E_PYTHON:-/opt/anaconda3/bin/python}"
BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
EXPERIMENT="${EXPERIMENT:-${PHASE2_EXPERIMENT:-main}}"
export PHASE2_EXPERIMENT="$EXPERIMENT"      # the notebooks' config cell reads this
# MODEL runs this experiment against another model's tree/backend, so an ablation does not need a
# config file per (experiment, model).
[ -n "${MODEL:-}" ] && export PHASE2_MODEL="$MODEL"
# SPLIT picks the article set AND tags the run with it (main -> main_val / main_test), so a
# validation number can never land on top of a test number of the same experiment.
[ -n "${SPLIT:-}" ] && export PHASE2_SPLIT="$SPLIT"
# REPEAT is the same arm sampled again: same config, own run tag (main -> main_r2), own manifest.
[ -n "${REPEAT:-}" ] && export PHASE2_REPEAT="$REPEAT"
MIN_CTX="${MIN_CTX:-40960}"        # the consolidation prompts are the largest in the pipeline
export PYDEVD_DISABLE_FILE_VALIDATION=1
export VLLM_TIMEOUT_S="${VLLM_TIMEOUT_S:-3600}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUNDIR="$HERE/runs/${STAMP}__staged__${EXPERIMENT}${SPLIT:+_$SPLIT}${REPEAT:+_r$REPEAT}"

nb_for() {
  case "$1" in
    2)  echo "2_oie_extraction.ipynb" ;;
    4)  echo "4_type_extension.ipynb" ;;
    6)  echo "6_relation_extraction.ipynb" ;;
    6b) echo "6b_event_reification.ipynb" ;;
    7)  echo "7_property_extension.ipynb" ;;
    7b) echo "7b_nary_property_extension.ipynb" ;;
    *)  echo "" ;;
  esac
}

if [ $# -gt 0 ]; then ORDER=("$@"); else ORDER=(2 4 6 7); fi
for k in "${ORDER[@]}"; do
  [ -n "$(nb_for "$k")" ] || { echo "unknown notebook key: $k  (use 2 / 4 / 6 / 6b / 7 / 7b)"; exit 2; }
done

# ── interpreter ────────────────────────────────────────────────────────────────────────────
"$PY" -c 'import nbconvert, nbclient' 2>/dev/null || {
  echo "FAIL  $PY has no nbconvert."
  echo "      install it there:  $PY -m pip install nbconvert nbclient ipykernel"
  echo "      or point at another interpreter:  E2E_PYTHON=... ./run_staged.sh"
  exit 1
}

echo "── config ────────────────────────────────────────────────────────────────────"
( cd "$HERE/notebooks" && "$PY" -m experiment_config check "$EXPERIMENT" ) || {
  echo
  echo "  FAIL  experiment '$EXPERIMENT' is inconsistent (see above)."
  echo "        list what exists:  (cd $HERE/notebooks && $PY -m experiment_config list)"
  exit 1
}
echo "  running: ${ORDER[*]}"

# ── vLLM server ────────────────────────────────────────────────────────────────────────────
if [ "${SKIP_SERVER_CHECK:-0}" != "1" ]; then
  echo
  echo "── vLLM server ───────────────────────────────────────────────────────────────"
  if ! MODELS_JSON="$(curl -fsS --max-time 5 "${BASE_URL}/models" 2>/dev/null)"; then
    echo "  FAIL  no server answering at $BASE_URL"
    echo "        bring the tunnel up first:  $HERE/cluster/labia-tunnel.sh"
    echo "        (or SKIP_SERVER_CHECK=1 if this run uses the Gemini backend)"
    exit 1
  fi
  "$PY" - "$MODELS_JSON" "$MIN_CTX" <<'PYEOF'
import json, sys
d = (json.loads(sys.argv[1]).get("data") or [{}])[0]
ctx, min_ctx = d.get("max_model_len"), int(sys.argv[2])
print(f"  served model : {d.get('id')}")
print(f"  context      : {ctx}")
if ctx and ctx < min_ctx:
    print(f"\n  WARN  context {ctx} < {min_ctx}. The consolidation calls are the largest in the")
    print( "        pipeline and will fail on length; the build stages will still complete.")
    print( "        To fix:  YARN=1 MAXLEN=65536 sbatch cluster/serve_vllm.sbatch")
PYEOF
fi

# ── run ────────────────────────────────────────────────────────────────────────────────────
mkdir -p "$RUNDIR"
# Snapshot what this run IS, next to what it printed. The manifest under results/ is the primary
# record; this copy makes the log directory self-describing when it is read on its own.
( cd "$HERE/notebooks" && "$PY" -m experiment_config show "$EXPERIMENT" ) > "$RUNDIR/config.json"
{
  echo "experiment   $EXPERIMENT${SPLIT:+   split $SPLIT}${REPEAT:+   repeat $REPEAT}"
  echo "notebooks    ${ORDER[*]}"
  echo "started      $(date -Iseconds)"
  echo "host         $(hostname)"
  echo "python       $PY"
  echo "commit       $(git -C "$HERE" rev-parse HEAD 2>/dev/null || echo '(not a git checkout)')"
  echo "branch       $(git -C "$HERE" rev-parse --abbrev-ref HEAD 2>/dev/null)"
  echo "dirty        $(test -n "$(git -C "$HERE" status --porcelain 2>/dev/null)" && echo yes || echo no)"
  echo "backend url  ${VLLM_BASE_URL:-(gemini)}"
} > "$RUNDIR/provenance.txt"
echo
echo "── running ${ORDER[*]} ───────────────────────────────────────────────────────"
echo "   logs -> $RUNDIR"
echo
for k in "${ORDER[@]}"; do
  nb="$(nb_for "$k")"
  echo "=============================================================================="
  echo "  $k   $nb   ($(date +%H:%M:%S))"
  echo "=============================================================================="
  args=(--to notebook --execute
        --ExecutePreprocessor.timeout=-1
        --ExecutePreprocessor.kernel_name=python3)
  if [ "${INPLACE:-0}" = "1" ]; then
    args+=(--inplace "$NBDIR/$nb")
  else
    args+=(--output-dir "$RUNDIR" --output "$nb" "$NBDIR/$nb")
  fi
  # cd into the notebook dir so ROOT resolution (walk up for results/ + dataset/) behaves as in Jupyter
  ( cd "$NBDIR" && "$PY" -m jupyter nbconvert "${args[@]}" ) 2>&1 | tee "$RUNDIR/$k.log"
  echo "  $k done ($(date +%H:%M:%S))"
  echo
done

# ── summary ────────────────────────────────────────────────────────────────────────────────
echo "── result ────────────────────────────────────────────────────────────────────"
"$PY" - "$HERE" "$EXPERIMENT" <<'PYEOF'
import json, sys
from pathlib import Path

here, name = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(here / "notebooks"))
import experiment_config as xcfg

cfg = xcfg.resolve(name)
model, variant, tag = cfg["results_model"], cfg.get("variant") or "", cfg.get("run_tag") or ""
osfx = f"_{tag}" if tag else ""                     # binary side: run tag only
nsfx = (f"_{variant}" if variant else "") + osfx     # n-ary side: extraction arm + run tag
G = here / "results" / model / "_global"

def n_of(path, key):
    try:
        return len(json.loads(path.read_text())[key])
    except Exception:
        return None

print(f"  experiment {name!r}   tree {model}   binary tag {osfx!r}   n-ary tag {nsfx!r}")
print(f"  minted TYPES              : {n_of(G / f'type_vocabulary{osfx}_consolidated.json', 'types')}")
print(f"  minted PROPERTIES (binary): {n_of(G / f'relation_vocabulary{osfx}_consolidated.json', 'properties')}")
print(f"  minted PROPERTIES (n-ary) : {n_of(G / f'nary_relation_vocabulary{nsfx}_consolidated.json', 'properties')}")

for stage, tagd in (("oie", osfx), ("types", osfx), ("relations", osfx), ("extraction", nsfx)):
    m = G / f"metrics_{stage}{tagd}.json"
    print(f"  metrics {stage:<10}: {m.name if m.exists() else '(not saved - run that stage metrics cell)'}")

# The run's own record: config + commit + outputs + token spend, per stage. A stage missing here
# ran without closing its manifest (its metrics cell did not execute), which is worth knowing NOW
# rather than when the number is being written up.
import run_manifest
rid = run_manifest.run_id_for(cfg, track="two_staged")
rec = run_manifest.load(here / "results" / model, rid)
if rec is None:
    print(f"  manifest          : NOT WRITTEN for run id {rid} — this run left no provenance.")
else:
    print(f"  manifest          : results/{model}/_runs/{rid}.json")
    for s, v in rec.get("stages", {}).items():
        tu = (v.get("token_usage") or {}).get("total_tokens")
        state = v.get("finished") or "OPEN (metrics cell did not run)"
        print(f"    {s:<3} {state}  ·  {len(v.get('outputs') or [])} file(s)"
              + (f"  ·  {int(tu):,} tokens" if tu else ""))
run_manifest.index()
PYEOF
echo
echo "  logs + executed notebooks: $RUNDIR"
echo "  every run in this tree   : (cd $HERE/notebooks && $PY -m run_manifest list)  ·  results/RUNS.md"
