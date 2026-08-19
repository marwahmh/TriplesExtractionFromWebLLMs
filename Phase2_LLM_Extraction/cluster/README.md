# Running the pipeline on open-weight LLMs (Lab-IA cluster)

This directory holds everything needed to run the Phase 2 extraction notebooks against an
**open-weight model served by vLLM on the Lab-IA GPU cluster**, instead of the Gemini API.

The notebooks stay on your Mac and are edited and run exactly as before. The only thing that
changes is where `gemini_json` sends its request: instead of Google, it calls `localhost:8000`,
which an SSH tunnel forwards to a GPU node at Lab-IA.

```
Mac (Jupyter)  →  localhost:8000  →[ ssh tunnel ]→  slurm  →  n3:8000 (vLLM + Qwen3-14B)
```

Swapping back to Gemini is deleting one notebook cell.

---

## Contents

| File | Runs on | Purpose |
|---|---|---|
| `serve_vllm.sbatch` | cluster | Slurm job that serves a model on an A6000, OpenAI-compatible, port 8000 |
| `labia-tunnel.sh` | Mac | Finds the node running the job and forwards `localhost:8000` to it |
| `../notebooks/llm_backend.py` | Mac | Drop-in replacement for `gemini_json` |

---

## Part 0 — SSH access (assumed already working)

You reach the cluster with `ssh slurm` through three hops
(`gateway-lab` → `via.lisn.upsaclay.fr` → `slurm.lab-ia.fr`), configured in `~/.ssh/config`.

Optional but recommended — add connection multiplexing to the `slurm` block so the three-hop chain
is negotiated once instead of on every command:

```
Host slurm
    # ... your existing settings ...
    ControlMaster auto
    ControlPath ~/.ssh/cm-%r@%h:%p
    ControlPersist 10m
```

**You never SSH into a compute node.** The tunnel forwards *through* the login node to the node's
port, so compute-node SSH access is not required.

---

## Part 1 — One-time cluster setup

Do this once. It takes 30–45 minutes, mostly downloads. Everything lands in `$HOME`
(`/mnt/beegfs/home/sulaiman`), which is shared BeeGFS storage visible from every compute node — so
one install serves all future jobs.

### 1.1 Log in and set the proxy

```bash
ssh slurm
export http_proxy=http://webproxy.lab-ia.fr:8080/
export https_proxy=$http_proxy
```

Compute and login nodes reach the internet **only** through this proxy.

### 1.2 Install Miniforge

The cluster has no conda, no pip, and only Python 3.6.8 system-wide.

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
conda create -y -n vllm python=3.11
conda activate vllm
```

> BeeGFS is slow with the many-small-files pattern conda produces. 10–20 minutes is normal.

### 1.3 Install vLLM, excluding `xformers`

vLLM 0.8.5.post1 hard-pins `xformers==0.0.29.post2`, which has no wheel for this cluster's glibc
(see [Why the versions are pinned](#why-the-versions-are-pinned)). It is only an alternative
attention backend and is unused on Ampere GPUs, so it is dropped from the dependency set.

```bash
cd $HOME
pip download --no-deps --only-binary=:all: "vllm==0.8.5.post1" -d $HOME/vllm_whl

python - <<'PY'
import glob, zipfile, pathlib
whl = glob.glob(str(pathlib.Path.home() / "vllm_whl" / "vllm-*.whl"))[0]
z = zipfile.ZipFile(whl)
meta = z.read(next(n for n in z.namelist() if n.endswith("METADATA"))).decode()
reqs = []
for line in meta.splitlines():
    if not line.startswith("Requires-Dist:"):
        continue
    r = line.split(":", 1)[1].strip()
    if r.split(";")[0].strip().startswith("xformers"):   # the one we can't have
        continue
    if "extra ==" in r:                                  # optional extras
        continue
    reqs.append(r)                                       # keep the environment marker intact
pathlib.Path("vllm_reqs.txt").write_text("\n".join(reqs) + "\n")
print(f"{len(reqs)} requirements written, xformers excluded")
PY

pip install --only-binary=:all: -r vllm_reqs.txt
pip install --only-binary=:all: --no-deps "vllm==0.8.5.post1"
```

Reading the pins from the wheel metadata (rather than transcribing them) means nothing is missed,
and keeping the `; python_version …` markers lets pip resolve conditional pins correctly.

### 1.4 Pin `transformers` back to the 4.51 line

vLLM 0.8.5 declares `transformers>=4.51.1` with **no upper bound**, so pip installs a later major
version whose internals vLLM cannot use.

```bash
pip install --only-binary=:all: "transformers>=4.51.1,<4.52"
python -c "import transformers, tokenizers, huggingface_hub as h; print(transformers.__version__, tokenizers.__version__, h.__version__)"
```

Expect roughly `4.51.3  0.21.4  0.36.2`. This also pulls `tokenizers` and `huggingface-hub` back
down, which is correct — they were dragged forward too.

### 1.5 Install a modern compiler

The system gcc is 4.8.5 and has no `<stdatomic.h>` (needs gcc ≥ 4.9). Triton JIT-compiles a small C
shim at runtime and cannot build it with that compiler.

```bash
conda install -y -c conda-forge gcc_linux-64 gxx_linux-64
ls $CONDA_PREFIX/bin/*-linux-gnu-gcc
```

That must print a path. `serve_vllm.sbatch` globs for it and exports it as `CC`/`CXX`.

### 1.6 Download model weights

Do this on the **login node**, not inside a GPU job — otherwise you pay GPU allocation time to
download 28GB.

```bash
export HF_HOME=$HOME/hf_cache
echo 'export HF_HOME=$HOME/hf_cache' >> ~/.bashrc
nohup huggingface-cli download Qwen/Qwen3-14B > ~/hf_download.log 2>&1 &
```

`nohup … &` detaches it so a dropped SSH connection can't kill it. Watch with
`du -sh $HF_HOME` until it reaches ~28GB.

### 1.7 Copy the job script

From your **Mac**:

```bash
cd /Users/marwah/Documents/Thesis/TriplesExtractionFromWebLLMs/Phase2_LLM_Extraction
scp cluster/serve_vllm.sbatch slurm:~/
```

Repeat this whenever `serve_vllm.sbatch` changes.

---

## Part 2 — Every session

### 2.1 Start the server (cluster)

```bash
ssh slurm
source $HOME/miniforge3/etc/profile.d/conda.sh && conda activate vllm
sbatch serve_vllm.sbatch
squeue -u $USER
```

The conda environment does **not** persist across SSH sessions — you must re-activate it every
time. If `conda: command not found`, you skipped the `source` line.

Note the job ID, then follow the log (it doesn't exist until the job starts running):

```bash
until [ -f vllm-<jobid>.log ]; do sleep 10; done; tail -f vllm-<jobid>.log
```

Wait for these lines, in order:

```
compiler: /mnt/beegfs/home/sulaiman/miniforge3/envs/vllm/bin/x86_64-conda-linux-gnu-gcc
reasoning flags: --enable-reasoning --reasoning-parser deepseek_r1
NVIDIA RTX A6000, 49140 MiB
Loading safetensors checkpoint shards: 100%
INFO:     Application startup complete.
```

`Ctrl-C` stops the tail; it does not stop the job. If `squeue` shows `PD`, all five A6000 nodes are
busy — check with `sinfo -N -n n1,n2,n3,n4,n5 -O "NodeHost:8,StateLong:10,GresUsed:22"`.

### 2.2 Open the tunnel (Mac, its own terminal)

```bash
cd /Users/marwah/Documents/Thesis/TriplesExtractionFromWebLLMs/Phase2_LLM_Extraction
./cluster/labia-tunnel.sh
```

**Leave this terminal open.** Closing it drops the connection. The script resolves which node got
the job, so you never hardcode `n3`.

### 2.3 Verify (Mac, another terminal)

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool
```

Seeing `"id": "qwen3-14b"` means the whole chain works.

### 2.4 Run the notebooks

See [Part 3](#part-3--the-notebook-cell). Nothing else changes.

### 2.5 Shut down when finished

```bash
ssh slurm 'scancel -n vllm'
```

The job holds the GPU for up to 48h otherwise. The A6000s are heavily contended — release them.

---

## Part 3 — The notebook cell

Every notebook takes its model selection in the config cell:

```python
RESULTS_MODEL = "qwen3-14b"    # whose results to read/write: results/<RESULTS_MODEL>/
BACKEND       = RESULTS_MODEL  # which model runs this notebook's LLM calls
```

These are **separate on purpose**. Pipeline notebooks (2, 4, 6, 6b, 6c, 6d, 6e, 7, 7b) leave
`BACKEND = RESULTS_MODEL` — the model that produced the graphs also processes them. Evaluation
notebooks (9, 9b, 10) can set `BACKEND = "gemini-2.5-flash"` while pointing `RESULTS_MODEL` at
another model, so **one consistent judge scores every model's output**. Read-only notebooks
(8 visualization, 3 schema matching) have `RESULTS_MODEL` alone and no backend at all.

| Value | Runs on | JSON | Thinking |
|---|---|---|---|
| `gemini-2.5-flash` | Gemini API (the notebook's own function) | soft + `json_repair` | on |
| `qwen3-14b`, `qwen3-32b` | vLLM | grammar-enforced | off |
| `gemma3-12b`, `gemma3-27b`, `llama33-70b` | vLLM | grammar-enforced | off |
| `qwen3-14b-think`, `qwen3-32b-think` | vLLM | soft + `json_repair` | **on** |

The registry is `BACKENDS` at the bottom of `../notebooks/llm_backend.py`; add a row to add a model.
Each row's `model` must equal the server's `--served-model-name` (`SERVED=` in the sbatch). Results
go to `results/<BACKEND>/` — keyed by backend name, not model, so the structured and thinking
configs of one model never share grown `_global/` vocabularies.

Wiring is two lines at the **bottom of the cell that defines `gemini_json`** (the one with
`client = genai.Client(...)`). Do **not** put them in a separate cell — see the warning below.

```python
# --- select the backend ---------------------------------------------------------------------
import sys, pathlib, importlib
for _p in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents):
    if (_p / "llm_backend.py").exists():
        sys.path.insert(0, str(_p)); break
import llm_backend
importlib.reload(llm_backend)   # plain `import` is a no-op once cached; edits would go unnoticed

llm_backend.activate(BACKEND, ROOT, globals())
```

`activate` sets `gemini_json`, `GEMINI_MODEL`, `RESULTS_DIR`, `MAX_OUTPUT_TOKENS` and
`THINKING_BUDGET`, then prints what it did. With `BACKEND="gemini"` it leaves the notebook's own
function untouched and only repoints `RESULTS_DIR`, so the Gemini path never routes through this
module. Per-run tweaks go through the same call:
`llm_backend.activate(BACKEND, ROOT, globals(), temperature=0.6)`.

Also add `openai` to the cell's existing `%pip install` line.

**`RESULTS_DIR` is the important part.** Every model gets its own subtree:

```
results/
├── gemini-2.5-flash/     ← RESULTS_DIR default, set in the config cell
│   ├── _global/          ← vocabularies grown by 6d/6e, per model
│   ├── art1/ art2/ ...
├── qwen3-14b/            ← created on first open-weight run
└── qwen3-32b/
```

Without this, an open-model run overwrites the Gemini graphs *and* — worse — grows the same
`_global/e2e/*_vocabulary.json` that 6d/6e use to measure schema.org reuse-vs-mint, silently
contaminating the numbers across models. Use the identical block in 6d, 6e, 8 and 10 and the whole
chain follows; no other path edits are needed anywhere.

The config cell sets `RESULTS_DIR = ROOT / "results" / GEMINI_MODEL`, so the Gemini path is
self-consistent too — comment out the redirect block and outputs go to `results/gemini-2.5-flash/`.

> **Why it must live in that cell, not a separate one.** Both define `gemini_json`, and whichever
> ran *most recently* wins — cell position is irrelevant, execution order is everything. A separate
> cell is silently undone by re-running the Gemini cell, or by "Run All" if it sits above. The
> symptom is `404 models/qwen3-14b is not found` from Google: `gemini_json` reverted to Gemini while
> `GEMINI_MODEL` (set elsewhere) kept the open-model name. Putting the redirect last in the same
> cell makes that impossible.

The upward search for `llm_backend.py` replaces a plain `sys.path.insert(0, "..")`, which only works
when the kernel's working directory happens to be `notebooks/<track>/`. If Jupyter was started at
the repo root, the bare `".."` import fails and `gemini_json` is never rebound.

Run the notebook top to bottom as usual. Everything downstream works unchanged because
`llm_backend` reproduces `gemini_json`'s contract exactly:

- same signature `(system_prompt, user_message, model=None, temperature=0)`
- same return value — a parsed JSON object
- same side effects — `last_latency_s`, and `LAST_USAGE` with
  `prompt_tokens / output_tokens / thoughts_tokens / total_tokens`
- same lenient parsing — strict → trailing-comma fix → `json_repair` → raise with context

**To go back to Gemini, delete the cell.**

---

## Part 4 — Running a different model

### 4.1 What fits on one A6000 (48GB)

| Model | Precision | VRAM | Notes |
|---|---|---|---|
| Qwen3-8B | bf16 | ~16GB | fastest iteration |
| Gemma 3 12B | bf16 | ~24GB | no thinking mode — useful as a non-reasoning control |
| **Qwen3-14B** | bf16 | ~28GB | **current default** |
| Qwen3-32B | AWQ 4-bit | ~18GB | declare the quantization in the write-up |
| Gemma 3 27B | AWQ 4-bit | ~16GB | |
| Llama 3.3 70B | AWQ 4-bit | ~40GB | tight; works only because `--max-model-len` is small |
| Mistral Small 24B | bf16 | ~48GB | too tight for one card — use 4-bit |

bf16 above ~28GB needs two GPUs (`--gres=gpu:2 --tensor-parallel-size 2`), which is hard to get
given contention. Prefer 4-bit on one card.

### 4.2 How to launch one

No file edits — override with environment variables:

```bash
MODEL=Qwen/Qwen3-32B-AWQ SERVED=qwen3-32b sbatch serve_vllm.sbatch
```

Then in the notebook: `llm_backend.configure(model="qwen3-32b")`.

Download the weights first on the login node (`huggingface-cli download <repo-id>`), or the job
spends its GPU time downloading.

### 4.3 Before you pick one

- **Verify the exact HuggingFace repo id.** A wrong one fails at download. AWQ variants in
  particular are often community-published under names you have to check.
- **Models released after ~April 2025 will probably not work.** vLLM is frozen at 0.8.5 and
  transformers at 4.51 (see below), so anything needing newer support — gpt-oss, GLM-4.5, Kimi K2 —
  cannot be served here. Qwen3, Qwen2.5, Gemma 3, Llama 3.x, and Mistral Small are all fine.
- **Quantization is auto-detected** from the model config; you don't normally need
  `--quantization`. AWQ Marlin kernels need compute capability ≥ 8.0, and the A6000s are 8.6.

### 4.4 Two GPUs, if you can get them

Edit `serve_vllm.sbatch`: `--gres=gpu:2`, and add `--tensor-parallel-size 2` to the `vllm serve`
command. Both GPUs must be on one node, which `--nodes=1` already enforces.

---

## Part 5 — Structured output vs. thinking

**On vLLM 0.8.5 these are mutually exclusive.** A JSON grammar constrains the very first generated
token, so the model can never emit `<think>` — reasoning is suppressed regardless of what
`enable_thinking` says. Newer vLLM applies the grammar only after `</think>`; this build cannot, and
it cannot be upgraded.

That gives two usable configurations:

| `configure(...)` | JSON reliability | Reasoning | Use for |
|---|---|---|---|
| `structured=True, thinking=False` | grammar-enforced, ~100% | none | the main runs |
| `structured=False, thinking=True` | prompt + `json_repair` salvage | real `<think>` blocks | the reasoning ablation |

The second is the fairer comparison to the Gemini baseline: `response_mime_type="application/json"`
was never a hard grammar either, which is precisely why the notebooks carry `_loads_lenient` and
`json_repair`. Report parse-failure rates from that arm — the number is a result, not a nuisance.

Two further caveats:

- **`THINKING_BUDGET` loses resolution.** Gemini's `-1` / `0` / `N` token budget has no equivalent;
  Qwen3 is on/off only. Positive caps are ignored.
- **Thinking is expensive.** On a one-sentence toy input, 472 of 494 output tokens were reasoning.
  On full articles expect several thousand tokens per call and a large slowdown.

### The thinking arm needs a bigger context than 32768

Reasoning tokens count against the context window. Gemini's own numbers on these articles show what
the e2e call actually costs:

| Article | prompt | thinking | JSON | total |
|---|---|---|---|---|
| art1 | 2,679 | 13,776 | 18,846 | **35,301** |
| art2 | 3,297 | 17,255 | 24,066 | **44,618** |
| art5 | 2,614 | 10,471 | 15,132 | 28,217 |

Qwen3's native context is 32,768 — fine for the structured arm (no reasoning), too small for the
thinking arm on the larger articles. Serve that arm with YaRN scaling:

```bash
MODEL=Qwen/Qwen3-32B-AWQ SERVED=qwen3-32b YARN=1 MAXLEN=65536 EAGER=0 sbatch serve_vllm.sbatch
```

At 65,536 the KV cache costs ~16 GB for Qwen3-32B, against ~25 GB free after its 4-bit weights —
comfortable on one A6000. Note YaRN is active for *every* request on that server, including the
structured arm, and can slightly affect short-context behaviour; if you want both arms strictly
comparable, run the structured arm on its own server at the native 32768.

### Server-side reasoning parser: leave it off

vLLM can split `<think>…</think>` out of the reply into a separate `reasoning_content` field
(`THINK=1`). Don't — `llm_backend._split_thinking` strips the tags client-side anyway, and the
parser only adds failure modes:

- models with no think tokens (Llama, Gemma) return **400 on every request**
- the `deepseek_r1` parser assumes generation *starts* inside a think block and runs to `</think>`;
  any reply without one — i.e. every non-thinking reply — is swallowed whole into
  `reasoning_content`, leaving `content` null

`THINK` is unrelated to whether the model reasons. That is `enable_thinking`, set per backend in
`BACKENDS`. `_split_thinking` handles all three shapes (parser on, parser over-claiming, parser
off), so nothing downstream depends on the server's choice.

---

## Why the versions are pinned

Everything traces to one fact: **the cluster runs CentOS 7 with glibc 2.17**, on both login and
compute nodes. Verify with `ldd --version`.

| Constraint | Consequence |
|---|---|
| glibc 2.17 | PyTorch ≥ 2.7 ships `manylinux_2_28` wheels → **torch capped at 2.6.0** |
| torch 2.6.0 | **vLLM capped at 0.8.5.post1** (0.9+ requires torch 2.7) |
| glibc 2.17 | `xformers` ≥ 0.0.28 also `manylinux_2_28` → must be excluded |
| vLLM 0.8.5 | reasoning parsers are `{deepseek_r1, granite}`; no `qwen3` parser |
| vLLM 0.8.5 | last release with Qwen3 support — do not downgrade further |
| gcc 4.8.5 | no `<stdatomic.h>` → Triton JIT fails → conda gcc required |
| transformers unbounded pin | must be manually held at `<4.52` |

The A6000s themselves are fine — compute capability 8.6, driver 550.144.03, full bf16 and
FlashAttention-2 support. The software stack, not the hardware, is the limitation.

Worth raising with the cluster administrators: glibc 2.17 in 2026 locks Lab-IA out of essentially
all current ML tooling.

---

## Cluster reference

**Hardware**

| Nodes | GPUs | VRAM | Compute capability | Usable |
|---|---|---|---|---|
| `n[1-5]` | 2× RTX A6000 | 48GB | 8.6 | ✅ bf16, FA2, AWQ Marlin |
| `n[51-55]` | 3× Tesla V100 | 32GB | 7.0 | ⚠️ no bf16, no FA2 |
| `n[101-102]` | 4× Tesla V100 (NVLink) | 32GB | 7.0 | ⚠️ same |

`serve_vllm.sbatch` excludes the V100 nodes. They are more numerous (shorter queues) but Volta lacks
bf16 entirely, and Gemma in particular has a history of fp16 overflow. If you ever use them,
validate the output against an A6000 run before reporting it.

**Partitions and QoS** — use `--partition=all --qos=default`:

| QoS | Priority | MaxWall | MaxJobs | MaxGPUs |
|---|---|---|---|---|
| **default** (partition `all`) | **1000** | **48h** | **6** | **6** |
| preempt | 900 | 24h | 6 | 6 |
| pcie | **0** | 24h | 1 | 4 |
| nvlink | **0** | 24h | 1 | 4 |

The `pcie` and `nvlink` partitions name the better hardware but their QoS sit at **priority 0** —
you would queue behind everything. `all` + `default` is strictly better on every axis.

**Storage** — `$HOME` is `/mnt/beegfs/home/sulaiman`, shared across all nodes, no enforced quota.
Model weights live in `$HF_HOME` (`~/hf_cache`).

**Network** — outbound internet only via `http://webproxy.lab-ia.fr:8080/`. Both the login node and
compute nodes can reach HuggingFace and PyPI through it.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `conda: command not found` | fresh SSH session | `source $HOME/miniforge3/etc/profile.d/conda.sh` |
| `ModuleNotFoundError` for something you installed | environment not activated | `conda activate vllm` |
| `Connection refused` on port 8000 | tunnel terminal closed, or job ended | re-run `labia-tunnel.sh`; check `ssh slurm 'squeue -u $USER'` |
| Tunnel says "No RUNNING job" | job still `PD`, or it crashed | `tail vllm-*.log` on the login node |
| Notebook hangs on first call | server still loading weights | wait for `Application startup complete` |
| `No matching distribution found for torch==2.x` | glibc 2.17 ceiling | do not chase it — the pins above are the answer |
| `xformers … which is not installed` | expected | harmless; `VLLM_ATTENTION_BACKEND=FLASH_ATTN` covers it |
| `fatal error: stdatomic.h` | system gcc 4.8.5 | step 1.5, and confirm the `compiler:` line in the log |
| `libcuda.so.1: cannot open shared object file` **on the login node** | login node has no GPU | expected, ignore |
| `content: null`, JSON in `reasoning_content` | `deepseek_r1` parser + grammar | already handled by `llm_backend` |
| `400 ... reasoning parser could not locate think start/end tokens` | reasoning parser enabled for a model with no `<think>` tokens (Llama, Gemma) | relaunch with `THINK=0` — the sbatch now infers this from the model name |
| `404 models/qwen3-14b is not found` (from Google) | Gemini cell ran after the backend cell and took `gemini_json` back | re-run the backend cell, then re-run the failed cell |
| `AssertionError` in `xgrammar_decoding.py`, engine shuts down, job exits | n-gram speculation + grammar-constrained JSON (vLLM 0.8.5) | relaunch the STRUCTURED arms with `SPEC=0`; thinking arms are unaffected |
| Endless output / repetition | `temperature=0` with thinking on | `thinking=False`, or raise temperature to 0.6 and record it |
| Job pending for hours | A6000s saturated | `sinfo -N -n n1,n2,n3,n4,n5 -O "NodeHost:8,GresUsed:22"` |
| Startup stalls ~30s on a proxy DNS error | flaky proxy resolution | harmless; add `export HF_HUB_OFFLINE=1` to the sbatch once weights are cached |

### Job script knobs

| Variable | Default | Effect |
|---|---|---|
| `MODEL` | `Qwen/Qwen3-14B` | HuggingFace repo id |
| `SERVED` | `qwen3-14b` | name the API exposes; must match the `model` in `BACKENDS` |
| `MAXLEN` | `32768` | `--max-model-len`; the e2e graph needs ~24k output + ~2.6k prompt |
| `TP` | `1` | tensor parallelism; must equal the GPU count requested |
| `THINK` | `0` | server-side reasoning parser. Leave off — `llm_backend` strips `<think>` tags itself, and the parser only adds failure modes. Whether the model *reasons* is a client-side setting in `BACKENDS` |
| `YARN` | `0` | `1` enables YaRN rope scaling so `MAXLEN` can exceed a model's native context. Needed for the **thinking arm** on Qwen3 (native 32768) — see below |
| `EAGER` | `0` | `1` forces `--enforce-eager` (skips `torch.compile` + CUDA graphs). Eager costs 1.3-1.8x on single-request decode and disables async output processing; it was the default only while the server was being brought up |
| `SPEC` | `1` | n-gram speculative decoding. Big win on this workload — `evidence` spans are verbatim prompt text and the JSON keys repeat endlessly, so the drafter hits often. Exact under greedy decoding, so it does not change the extraction. `0` disables |

Example: `MODEL=Qwen/Qwen3-8B SERVED=qwen3-8b EAGER=0 sbatch serve_vllm.sbatch`

### Llama 3.3 70B (needs 2 GPUs)

At 4-bit its ~40GB of weights leave only ~4GB for KV cache on one 48GB card — about 13k tokens,
well short of the 32k this pipeline needs. Two cards give ~48GB of KV instead.

```bash
# login node — pre-download ~40GB
huggingface-cli download lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit

MODEL=lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit SERVED=llama33-70b TP=2 EAGER=0 SPEC=0 \
  sbatch --gres=gpu:2 --cpus-per-task=16 --mem=96G serve_vllm.sbatch
```

Two caveats. Both A6000s must be free **on the same node**, which is the hard part given contention
— watch with `sinfo -N -n n1,n2,n3,n4,n5 -O "NodeHost:8,GresUsed:22"`. And these are PCIe cards, not
NVLink, so tensor-parallel communication costs more than on the `nvlink` partition's V100s; expect
noticeably fewer tokens/s than a single-GPU model.
