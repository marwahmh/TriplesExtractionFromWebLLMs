#!/bin/bash
# Score every extracted assertion with the entailment verifier on a Lab-IA GPU.
# Submit from the login node:  sbatch nli_verify.sbatch
#
# Unlike serve_vllm.sbatch this is a BATCH job, not a server: it runs, writes
# results/<model>/<id>/<track>/nli_verdicts.json, and exits. No tunnel needed. Copy the verdicts
# back to your Mac and run sections 5-7 of 9c locally.
#
#SBATCH --job-name=nli
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --exclude=n[51-55,101-102]   # keep to n[1-5]: A6000. The V100s would also work here (this
                                     # is a 0.4B encoder in fp32) but the A6000s are less contended.
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail

# compute nodes reach the internet only through the lab proxy
export http_proxy=http://webproxy.lab-ia.fr:8080/
export https_proxy=$http_proxy
export HF_HOME=$HOME/hf_cache        # BeeGFS: shared across nodes, download the model once

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vllm                  # already has torch 2.6.0+cu124 and transformers 4.51.3

# Where the synced project lives on the cluster. Override at submit time if you put it elsewhere:
#   ROOT=$HOME/somewhere/Phase2_LLM_Extraction sbatch nli_verify.sbatch
ROOT=${ROOT:-$HOME/Phase2_LLM_Extraction}
RESULTS_MODEL=${RESULTS_MODEL:-gemini-2.5-flash}
TRACK=${TRACK:-e2e}
BATCH=${BATCH:-64}                   # A6000 48GB handles 64-128 at 512 tokens; halves itself on OOM
ARTICLES=${ARTICLES:-}               # e.g. ARTICLES="art1 art5"; empty = every article with a log

echo "=================================================="
echo " node    : $(hostname)"
echo " root    : $ROOT"
echo " results : $RESULTS_MODEL   track: $TRACK"
echo " batch   : $BATCH   articles: ${ARTICLES:-<all>}"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# $ARTICLES is intentionally unquoted so an empty value expands to nothing
python "$ROOT/cluster/nli_verify.py" \
  --root "$ROOT" \
  --results-model "$RESULTS_MODEL" \
  --track "$TRACK" \
  --batch-size "$BATCH" \
  ${ARTICLES:+--articles $ARTICLES}
