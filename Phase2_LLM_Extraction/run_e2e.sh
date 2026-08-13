#!/usr/bin/env bash
# Run the e2e track (6c -> 6d -> 6e) in order, stopping at the first failure.
#
#   ./run_e2e.sh                 # all three, in order
#   ./run_e2e.sh 6d 6e           # a subset, in the order given
#   INPLACE=1 ./run_e2e.sh       # also write outputs back into the source notebooks
#   SKIP_SERVER_CHECK=1 ./run_e2e.sh          # skip the vLLM preflight (e.g. Gemini backend)
#   E2E_PYTHON=/path/to/python ./run_e2e.sh   # override the interpreter
#
# Executed copies and per-notebook logs land in runs/<timestamp>/, so the source notebooks stay
# clean unless INPLACE=1. The notebooks themselves are the source of truth for every setting; this
# script only checks that the three agree with each other before spending any GPU time.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NBDIR="$HERE/notebooks/e2e"
PY="${E2E_PYTHON:-/opt/anaconda3/bin/python}"
BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
MIN_CTX="${MIN_CTX:-40960}"        # 6d/6e consolidation prompts measure ~25-40k tokens on this corpus
# debugpy prints a "frozen modules" warning per kernel start on py3.12; pure noise.
export PYDEVD_DISABLE_FILE_VALIDATION=1
# One completion can take far longer than the openai client's 600s default on a contended GPU.
export VLLM_TIMEOUT_S="${VLLM_TIMEOUT_S:-3600}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUNDIR="$HERE/runs/$STAMP"

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
# Reads each notebook's config cell and refuses to start on the two mismatches that have silently
# wasted runs before: the three notebooks pointing at DIFFERENT results trees, and 6e's INPUT_TAG
# not naming 6d's RUN_TAG (6e then reads a run that does not exist, or the wrong one).
"$PY" - "$NBDIR" "${ORDER[@]}" <<'PYEOF'
import json, re, sys
from pathlib import Path

nbdir, keys = Path(sys.argv[1]), sys.argv[2:]
files = {"6c": "6c_e2e_reification.ipynb",
         "6d": "6d_e2e_type_canonicalization.ipynb",
         "6e": "6e_e2e_relation_canonicalization.ipynb"}

def cfg(key):
    src = "".join(c for cell in json.loads((nbdir / files[key]).read_text())["cells"]
                  if cell["cell_type"] == "code" for c in cell["source"])
    def g(name):
        m = re.search(rf'^\s*{name}\s*=\s*(None|"[^"]*")', src, re.M)
        if not m:
            return None
        v = m.group(1)
        return None if v == "None" else v.strip('"')
    return {"model": g("RESULTS_MODEL"), "backend": g("BACKEND") or "= RESULTS_MODEL",
            "variant": g("VARIANT"), "run_tag": g("RUN_TAG"), "input_tag": g("INPUT_TAG")}

C = {k: cfg(k) for k in files}
print(f"  {'nb':<4}{'results tree':<20}{'variant':<10}{'RUN_TAG':<12}{'INPUT_TAG':<12}")
for k in files:
    c = C[k]
    mark = " <- running" if k in keys else ""
    print(f"  {k:<4}{str(c['model']):<20}{str(c['variant']):<10}"
          f"{str(c['run_tag'] or '-'):<12}{str(c['input_tag'] or '-'):<12}{mark}")

errs = []
models = {C[k]["model"] for k in keys}
if len(models) > 1:
    errs.append(f"RESULTS_MODEL differs across the notebooks being run: {sorted(models)} — "
                "they would read and write different results/ trees.")
variants = {C[k]["variant"] for k in keys if C[k]["variant"] is not None}
if len(variants) > 1:
    errs.append(f"VARIANT differs: {sorted(variants)} — 6d and 6e must canonicalize the same graph.")
if "6e" in keys and C["6e"]["input_tag"] is not None and C["6e"]["input_tag"] != C["6d"]["run_tag"]:
    errs.append(f"6e INPUT_TAG={C['6e']['input_tag']!r} != 6d RUN_TAG={C['6d']['run_tag']!r} — "
                "6e would look for a 6d run that this chain does not produce.")
if errs:
    print()
    for e in errs:
        print(f"  FAIL  {e}")
    sys.exit(1)
print("\n  consistent.")
PYEOF

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
"$PY" - "$HERE" "$NBDIR" <<'PYEOF'
import json, re, glob
from pathlib import Path
import sys

here, nbdir = Path(sys.argv[1]), Path(sys.argv[2])
src = "".join(c for cell in json.loads((nbdir / "6e_e2e_relation_canonicalization.ipynb").read_text())["cells"]
              if cell["cell_type"] == "code" for c in cell["source"])
def g(n):
    m = re.search(rf'^\s*{n}\s*=\s*"([^"]*)"', src, re.M)
    return m.group(1) if m else ""
model, variant, tag = g("RESULTS_MODEL"), g("VARIANT"), g("RUN_TAG")
sfx = (f"_{variant}" if variant else "") + (f"_{tag}" if tag else "")
G = here / "results" / model / "_global/e2e"

def n_of(path, key):
    try:
        return len(json.loads(path.read_text())[key])
    except Exception:
        return None

t = n_of(G / f"type_vocabulary{sfx}_consolidated.json", "types")
p = n_of(G / f"nary_relation_vocabulary{sfx}_consolidated.json", "properties")
print(f"  tree {model}   tag {sfx}")
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
PYEOF
echo
echo "  logs + executed notebooks: $RUNDIR"
