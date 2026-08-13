#!/bin/bash
# Gate: does an open-weight verbaliser produce hypotheses the entailment model can judge?
# Run this ONCE before trusting any agreement number.   sbatch verbalise_check.sbatch
#SBATCH --job-name=vcheck
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --exclude=n[51-55,101-102]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
export http_proxy=http://webproxy.lab-ia.fr:8080/
export https_proxy=$http_proxy
export HF_HOME=$HOME/hf_cache
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vllm
ROOT=${ROOT:-$HOME/Phase2_LLM_Extraction}
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python "$ROOT/cluster/verbalise_check.py" --root "$ROOT" \
  --results-model "${RESULTS_MODEL:-gemini-2.5-flash}" \
  --articles ${ARTICLES:-art1 art5} --n "${N:-60}"
