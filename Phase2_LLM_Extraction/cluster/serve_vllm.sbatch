#!/bin/bash
# Serve an open-weight LLM on a Lab-IA A6000 node, OpenAI-compatible, on port 8000.
# Submit from the login node:  sbatch serve_vllm.sbatch
#
#SBATCH --job-name=vllm
#SBATCH --partition=all
#SBATCH --qos=default
#SBATCH --exclude=n[51-55,101-102]   # keep to n[1-5]: A6000 (cc 8.6, bf16). V100s are cc 7.0.
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8            # the default of 2 makes weight loading painfully slow
#SBATCH --mem=64G
#SBATCH --time=48:00:00              # default QoS allows 2 days
#SBATCH --output=%x-%j.log

set -euo pipefail

# compute nodes reach the internet only through the lab proxy
export http_proxy=http://webproxy.lab-ia.fr:8080/
export https_proxy=$http_proxy
export HF_HOME=$HOME/hf_cache        # BeeGFS: shared across nodes, download weights once

# xformers is uninstallable here (needs glibc 2.28; cluster is 2.17) and was omitted from the
# env. It is only an alternative attention backend — pin FlashAttention so nothing reaches for it.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate vllm

# The system gcc is 4.8.5 (CentOS 7) and has no <stdatomic.h>, so Triton cannot JIT-compile its
# CUDA utils. Point CC/CXX at the conda toolchain instead.
CONDA_GCC=$(ls "$CONDA_PREFIX"/bin/*-linux-gnu-gcc 2>/dev/null | head -1)
if [ -n "$CONDA_GCC" ]; then
    export CC="$CONDA_GCC"
    export CXX="${CONDA_GCC%gcc}g++"
    echo "compiler: $CC"
else
    echo "WARNING: no conda gcc found — Triton will fall back to /usr/bin/gcc and fail"
fi

MODEL=${MODEL:-Qwen/Qwen3-14B}
SERVED=${SERVED:-qwen3-14b}
# Measured need: ~2.6k prompt + up to 24k output (the e2e graph is genuinely huge — see the
# sizing note in notebooks/llm_backend.py). 32768 is the floor for the structured arm; the
# thinking arm needs more still. Big models leave little KV cache room after their weights, so a
# 70B at 4-bit cannot serve this context on one card — give it 2 GPUs or lower MAXLEN.
MAXLEN=${MAXLEN:-32768}
# Tensor parallelism. Must match the GPU count requested (override at submit time:
#   TP=2 sbatch --gres=gpu:2 --cpus-per-task=16 --mem=96G serve_vllm.sbatch
# A 70B at 4-bit needs TP=2 here: its weights leave too little room for a 32k KV cache on one card.
TP=${TP:-1}

# Qwen3's native context is 32768, which is enough for the structured arm (~2.6k prompt + ~24k
# JSON) but NOT for the thinking arm: reasoning adds another 10-17k tokens on these articles.
# YARN=1 enables YaRN rope scaling so MAXLEN can exceed 32768 (Qwen3 supports up to 131072).
# The JSON below deliberately contains no spaces so it survives unquoted word-splitting.
YARN=${YARN:-0}
if [ "$YARN" = "1" ]; then
    ROPE='--rope-scaling {"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
else
    ROPE=""
fi

# Reasoning parser — OFF by default, deliberately. It splits <think>...</think> out of the reply
# into a separate `reasoning_content` field, but llm_backend._split_thinking already strips those
# tags client-side, so it adds nothing and costs two failure modes:
#   * models without think tokens (Llama, Gemma) 400 on EVERY request
#   * vLLM's deepseek_r1 parser assumes generation STARTS inside a think block, so any reply
#     without a </think> — i.e. every non-thinking reply — is swallowed whole into
#     reasoning_content, leaving `content` null
# Whether the model actually reasons is a client-side setting (enable_thinking, via BACKENDS).
# THINK=1 opts back in if you ever want server-side splitting.
THINK=${THINK:-0}

if [ "$THINK" = "0" ]; then
    REASONING=""
else
    # The cluster is glibc 2.17 (CentOS 7), which caps torch at 2.6.0 and vLLM at 0.8.5.
    # vLLM renamed the reasoning flags after 0.8, so pick whichever this build accepts.
    HELP=$(vllm serve --help 2>&1 || true)
    if echo "$HELP" | grep -q -- "--reasoning-parser" && echo "$HELP" | grep -q "qwen3"; then
        REASONING="--reasoning-parser qwen3"
    elif echo "$HELP" | grep -q -- "--enable-reasoning"; then
        REASONING="--enable-reasoning --reasoning-parser deepseek_r1"   # same <think> tags
    else
        REASONING=""
        echo "WARNING: no reasoning parser available; <think> blocks will land in the JSON"
    fi
fi
echo "reasoning flags: ${REASONING:-<none>}   (THINK=$THINK)"

echo "=================================================="
echo " node   : $(hostname)"
echo " model  : $MODEL  (served as '$SERVED')"
echo " ctx    : $MAXLEN   tensor-parallel: $TP"
echo " tunnel : ssh -N -L 8000:$(hostname):8000 slurm"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --host 0.0.0.0 is required so the login node can relay to this port
# Eager mode skips torch.compile + CUDA graph capture. It removes a whole class of build failures,
# which is why it was the default while the server was being brought up — but it costs 1.3-1.8x on
# SINGLE-request decode, because a 64-layer forward pass at batch 1 is kernel-launch-bound, and it
# also disables async output processing (vLLM warns about this at startup). Now that the server is
# known good, compile by default. EAGER=1 puts the safe path back.
if [ "${EAGER:-0}" = "1" ]; then EAGER_FLAG="--enforce-eager"; else EAGER_FLAG=""; fi

# N-gram speculative decoding. The model drafts the next few tokens by looking for a repeat of the
# last few in its own context, then verifies them all in ONE forward pass. It is a large win here
# and a small one in general, because this pipeline's output is unusually self-similar:
#   * "evidence" spans are verbatim sentences copied out of the prompt
#   * the JSON keys (id/type/confidence/evidence/relation_phrase) repeat thousands of times
# Drafting is free at batch 1 — the GPU is bandwidth-bound, not compute-bound, so the verify pass
# costs the same as the single-token pass it replaces. With greedy decoding (temperature 0, which
# is what llm_backend sends) the accepted tokens are the ones argmax would have produced anyway,
# so this does NOT change the extraction — it is not a quality/speed trade.
# SPEC=0 disables it if a build turns out not to accept the flag.
SPEC=${SPEC:-1}
SPEC_FLAG=""
if [ "$SPEC" = "1" ]; then
    if vllm serve --help 2>&1 | grep -q -- "--speculative-config"; then
        SPEC_FLAG='--speculative-config {"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'
    else
        echo "WARNING: this build has no --speculative-config; running without speculation"
    fi
fi
echo "eager: ${EAGER_FLAG:-<off, cuda graphs on>}   spec: ${SPEC_FLAG:-<none>}"

# $REASONING, $EAGER_FLAG and $SPEC_FLAG are intentionally unquoted so empty values expand to
# nothing (bash 4.2). The SPEC JSON deliberately contains no spaces so it survives word-splitting.
vllm serve "$MODEL" \
  --served-model-name "$SERVED" \
  --max-model-len "$MAXLEN" \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.92 \
  $ROPE \
  $REASONING \
  $EAGER_FLAG \
  $SPEC_FLAG \
  --host 0.0.0.0 \
  --port 8000
