#!/usr/bin/env bash
# Run the e2e track (6c -> 6d -> 6e) in order, stopping at the first failure.
#
#   ./run_e2e.sh                 # all three, in order, under the `main` experiment
#   ./run_e2e.sh 6d 6e           # a subset, in the order given
#   EXPERIMENT=no-rerank ./run_e2e.sh         # run a named experiment (no notebook edits)
#   REPEAT=2 ./run_e2e.sh        # repeat the SAME arm under its own tag (main -> main_r2)
#   SPLIT=val ./run_e2e.sh         # the validation split (SPLIT=test for the reported set)
#   MODEL=qwen3-32b-think ./run_e2e.sh   # same experiment, another model's tree + backend
#   PHASE2_OVERWRITE=1 ./run_e2e.sh           # redo a run in place, over its own existing outputs
#   INPLACE=1 ./run_e2e.sh       # also write outputs back into the source notebooks
#   SKIP_SERVER_CHECK=1 ./run_e2e.sh          # skip the vLLM preflight (e.g. Gemini backend)
#   E2E_PYTHON=/path/to/python ./run_e2e.sh   # override the interpreter
#
# Executed copies and per-notebook logs land in runs/<timestamp>__<experiment>/, together with the
# resolved config and the commit they ran from — so the log directory says which run it is without
# being read. The run's own record of itself is results/<model>/_runs/<run_id>.json, written by the
# notebooks (see notebooks/run_manifest.py). The source notebooks stay clean unless INPLACE=1.
#
# A stage REFUSES to start when the manifest says it already finished under the same tag. That is
# the guard against a re-run silently landing on top of numbers that are already written up: use
# REPEAT for another sample of the same arm, a new experiment file for a different configuration,
# or PHASE2_OVERWRITE=1 when replacing the run in place is what you mean.
#
# The three notebooks read ONE config file (configs/<EXPERIMENT>.json) via notebooks/experiment_config.py,
# so they can no longer disagree about the results tree, the variant or the run tags — this script
# just resolves the experiment, prints it, and refuses to start if the config is internally
# inconsistent. EXPERIMENT is passed to the notebooks as $PHASE2_EXPERIMENT.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NBDIR="$HERE/notebooks/e2e"
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
MIN_CTX="${MIN_CTX:-40960}"        # 6d/6e consolidation prompts measure ~25-40k tokens on this corpus
# debugpy prints a "frozen modules" warning per kernel start on py3.12; pure noise.
export PYDEVD_DISABLE_FILE_VALIDATION=1
# One completion can take far longer than the openai client's 600s default on a contended GPU.
export VLLM_TIMEOUT_S="${VLLM_TIMEOUT_S:-3600}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUNDIR="$HERE/runs/${STAMP}__${EXPERIMENT}${SPLIT:+_$SPLIT}${REPEAT:+_r$REPEAT}"

nb_for() {
  case "$1" in
    6c) echo "6c_e2e_reification.ipynb" ;;
    6d) echo "6d_e2e_type_canonicalization.ipynb" ;;
    6e) echo "6e_e2e_relation_canonicalization.ipynb" ;;
    *)  echo "" ;;
  esac
}

if [ $# -gt 0 ]; then ORDER=("$@"); else ORDER=(6c 6d 6e); fi
for k in "${ORDER[@]}"; do
  [ -n "$(nb_for "$k")" ] || { echo "unknown notebook key: $k  (use 6c / 6d / 6e)"; exit 2; }
done

# ── interpreter ────────────────────────────────────────────────────────────────────────────
"$PY" -c 'import nbconvert, nbclient' 2>/dev/null || {
  echo "FAIL  $PY has no nbconvert."
  echo "      install it there:  $PY -m pip install nbconvert nbclient ipykernel"
  echo "      or point at another interpreter:  E2E_PYTHON=... ./run_e2e.sh"
  exit 1
}

echo "── config ────────────────────────────────────────────────────────────────────"
# One source of truth: configs/<EXPERIMENT>.json. `check` reports the settings that would silently
# waste a run (an unset results tree, a relation stage pointed at a type run this chain never
# produces, parallelism that is invalid for the chosen candidate pool) and exits non-zero on them.
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
    print(f"\n  WARN  context {ctx} < {min_ctx}. The 6d/6e consolidation calls are the largest in")
    print( "        the pipeline (~25-40k tokens on this corpus) and will fail on length. The build")
    print( "        stages will still complete. To fix, re-serve with more context:")
    print( "            YARN=1 MAXLEN=65536 sbatch cluster/serve_vllm.sbatch")
    print( "        Continuing in 5s — Ctrl-C to stop.")
PYEOF
  "$PY" -c "
import json,sys,time
d=(json.loads(sys.argv[1]).get('data') or [{}])[0]
if (d.get('max_model_len') or 10**9) < int(sys.argv[2]): time.sleep(5)
" "$MODELS_JSON" "$MIN_CTX"
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
import glob, json, sys
from pathlib import Path

here, name = Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(here / "notebooks"))
import experiment_config as xcfg

cfg = xcfg.resolve(name)
model, variant, tag = cfg["results_model"], cfg.get("variant") or "", cfg.get("run_tag") or ""
sfx = (f"_{variant}" if variant else "") + (f"_{tag}" if tag else "")
G = here / "results" / model / "_global/e2e"

def n_of(path, key):
    try:
        return len(json.loads(path.read_text())[key])
    except Exception:
        return None

t = n_of(G / f"type_vocabulary{sfx}_consolidated.json", "types")
p = n_of(G / f"nary_relation_vocabulary{sfx}_consolidated.json", "properties")
print(f"  experiment {name!r}   tree {model}   tag {sfx}")
print(f"  minted TYPES      : {t if t is not None else 'not written'}")
print(f"  minted PROPERTIES : {p if p is not None else 'not written'}")

files = sorted(glob.glob(str(here / "results" / model / f"art*/e2e/nary_graph{sfx}_extended_consolidated.json")))
tot = sc = 0
for f in files:
    for r in json.loads(Path(f).read_text())["relations"]:
        tot += 1
        sc += r.get("property_source") == "schema.org"
if tot:
    print(f"  relations         : {tot} over {len(files)} article(s)")
    print(f"  schema.org reuse  : {100*sc/tot:.1f}%  ({sc} schema.org / {tot-sc} minted)")
else:
    print("  relations         : no consolidated graphs found (6e consolidation may not have run)")

# The full metrics each stage saved — everything the write-up needs, in one place. The extraction
# metrics are tagged by VARIANT alone (6c writes before any run tag exists); the two canonicalization
# stages by the full suffix.
vsfx = f"_{variant}" if variant else ""
for stage, tag in (("extraction", vsfx), ("types", sfx), ("relations", sfx)):
    m = G / f"metrics_{stage}{tag}.json"
    print(f"  metrics {stage:<10}: {m.name if m.exists() else '(not saved - run that stage metrics cell)'}")

# The run's own record: config + commit + outputs + token spend, per stage. A stage missing here
# ran without closing its manifest (its metrics cell did not execute), which is worth knowing NOW
# rather than when the number is being written up.
import run_manifest
rid = run_manifest.run_id_for(cfg, track="e2e")
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
