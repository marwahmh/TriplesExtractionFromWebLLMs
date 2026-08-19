"""Open-weight backend for the extraction notebooks — a drop-in for `gemini_json`.

Talks to a vLLM server over its OpenAI-compatible API instead of the Gemini API, keeping the
same call signature, the same return value (parsed JSON), and the same side effects
(`last_latency_s`, `LAST_USAGE`) the notebooks already read.

Usage — pick a backend at the top of the notebook, then activate it at the END of the cell that
defines the notebook's own Gemini `gemini_json`:

    BACKEND = "qwen3-14b"                         # config cell; see BACKENDS below for the choices

    import llm_backend                            # last lines of the Gemini client cell
    llm_backend.activate(BACKEND, ROOT, globals())

`activate` sets `gemini_json`, `GEMINI_MODEL`, `RESULTS_DIR`, `MAX_OUTPUT_TOKENS` and
`THINKING_BUDGET` in the notebook. BACKEND="gemini" leaves the notebook's own Gemini function
alone and only repoints RESULTS_DIR, so switching providers is a one-word edit and nothing
downstream changes.
"""

import __main__
import json
import os
import re
import time

import httpx
from openai import OpenAI

try:
    from json_repair import repair_json           # salvages malformed LLM JSON, as in the notebooks
except Exception:
    repair_json = None

try:
    import nb_parallel                            # thread-local usage slot (see llm_json below)
except Exception as _exc:                         # only bites a parallel batch, so say so out loud
    print(f"  [llm_backend] nb_parallel unavailable ({_exc}); per-thread token accounting is off — "
          f"run batch cells with WORKERS=1")
    nb_parallel = None


BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
# Read timeout for ONE completion. The openai client defaults to 600s, which is far too short
# here: a thinking-arm extraction emits a 30k-token graph plus 10-17k reasoning tokens, and a
# contended single-GPU server QUEUES the parallel article calls, so one request's wall time is
# several times its own generation time. Connect stays short so a dead tunnel still fails fast.
TIMEOUT_S = float(os.environ.get("VLLM_TIMEOUT_S", 3600))
MODEL = "qwen3-14b"                               # matches --served-model-name in the sbatch script
THINKING = True                                   # Qwen3: on/off only, no token budget
STRUCTURED = True                                 # grammar-constrained JSON (see note)
MAX_OUTPUT_TOKENS = None                          # no cap — see sizing note below
TEMPERATURE = None                                # if set, overrides the caller's temperature
TOP_P = None                                      # Qwen3 recommends 0.95 whenever temperature > 0
SEED = None                                       # sampling RNG seed; only meaningful when temperature > 0

KIND = "vllm"                                     # "vllm" (self-hosted) or "hosted" (commercial API)
API_KEY = "EMPTY"                                 # vLLM ignores this; hosted endpoints need a real key
PARAMS = {}                                       # extra request params for hosted backends

# KIND exists because three things in a request are vLLM-specific and are errors elsewhere:
#   * chat_template_kwargs.enable_thinking — a vLLM hook into the chat template. Hosted providers
#     either ignore it or 400. Their equivalents differ per model and go through PARAMS
#     (gpt-oss: reasoning_effort; DeepSeek: a separate model id for thinking mode).
#   * api_key="EMPTY" — fine against a private tunnel, a 401 against anyone who bills.
#   * the STRUCTURED/THINKING exclusion below is a property of vLLM 0.8.5, not of the models.
#     Hosted endpoints coordinate the two, so a hosted backend may set both.

# SEED pins the sampler's RNG, which matters only for the THINKING arm — that one runs at
# temperature 0.6 (greedy sends Qwen3 into repetition loops), so without a seed no run is
# re-runnable. The structured arm is greedy and has no RNG to pin, so a seed there does nothing.
#
# What a seed does NOT buy is a deterministic run. vLLM batches requests continuously, batch shape
# decides the reduction order inside the matmuls, and that moves logits in the low bits — enough to
# flip argmax wherever the top two tokens are near-tied, after which a 20k-token graph diverges.
# So the same seed reproduces a run only against the SAME server config: same vLLM build, model
# revision, tensor-parallel size, eager-vs-CUDA-graphs, speculative decoding on/off, and the same
# concurrency. Treat it as "this run can be repeated", not "this output is determined".

# SIZING — None matches the notebooks' own Gemini setting and lets vLLM use everything left of
# --max-model-len after the prompt. The real cap is therefore the server's context window, which
# must be large: the e2e extraction legitimately produces a huge graph. Gemini's numbers for the
# same prompts (results/gemini-2.5-flash/_global/e2e/token_usage_extraction_dedup.json):
#     art1  18.8k JSON tokens (+13.8k thinking)   art2  24.1k (+17.3k)
#     art4   4.0k             (+8.6k)             art5  15.1k (+10.5k)
# So serve with --max-model-len 32768 (MAXLEN=32768) at minimum for the structured arm.
#
# TEMPERATURE/TOP_P exist for a different failure: greedy decoding can send Qwen3 into repetition
# loops. The symptom is running out of context with far MORE tokens than the sizes above. Leave
# them at None (temperature 0, matching the Gemini baseline) unless you actually see that.

# NOTE — on vLLM 0.8.5 these two are mutually exclusive. A JSON grammar constrains the very first
# token, so the model can never emit <think> and no reasoning happens, whatever THINKING says.
#   STRUCTURED=True,  THINKING=False -> grammar-enforced JSON, no reasoning  (most reliable)
#   STRUCTURED=False, THINKING=True  -> real reasoning, JSON via prompt + json_repair (like Gemini)
# Newer vLLM coordinates the two so the grammar applies only after </think>; this build does not.

last_latency_s = None
LAST_USAGE = None

def _timeout():
    return httpx.Timeout(TIMEOUT_S, connect=10.0)

_client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=_timeout())


def _reset_sampling():
    """Clear the sampling globals back to 'defer to the caller'.

    configure() only ever *sets* a param — passing None means "leave it alone", so nothing can put
    TEMPERATURE/TOP_P/SEED back to None once a thinking backend has set them. Without this, a
    notebook that activates qwen3-32b-think and then a second backend in the same kernel runs the
    second one at Qwen3's temperature 0.6 / top_p 0.95 / seed 1234, silently and invisibly.
    activate() calls this first so each activation starts from a clean slate.
    """
    global TEMPERATURE, TOP_P, SEED
    TEMPERATURE = TOP_P = SEED = None


def _resolve_key(env_var, backend):
    """Read a hosted backend's API key, failing loudly and early if it isn't set."""
    key = os.environ.get(env_var)
    if not key:
        raise RuntimeError(
            f"{backend}: ${env_var} is not set. Export it before starting Jupyter — a key set in a "
            f"notebook cell after the kernel started is fine, but one set in a shell profile after "
            f"the kernel started is not visible to it.")
    return key


def configure(model=None, thinking=None, structured=None, max_output_tokens=None,
              temperature=None, top_p=None, seed=None, base_url=None, timeout=None,
              kind=None, api_key=None, params=None):
    """Set the backend globals. Call once per notebook before any extraction cell."""
    global MODEL, THINKING, STRUCTURED, MAX_OUTPUT_TOKENS, TEMPERATURE, TOP_P, SEED, BASE_URL, _client
    global TIMEOUT_S, KIND, API_KEY, PARAMS
    if kind is not None:
        KIND = kind
    if api_key is not None:
        API_KEY = api_key
    if params is not None:
        PARAMS = dict(params)
    if model is not None:
        MODEL = model
    if thinking is not None:
        THINKING = thinking
    if structured is not None:
        STRUCTURED = structured
    if max_output_tokens is not None:
        MAX_OUTPUT_TOKENS = max_output_tokens
    if temperature is not None:
        TEMPERATURE = temperature
    if top_p is not None:
        TOP_P = top_p
    if seed is not None:
        SEED = seed
    if timeout is not None:
        TIMEOUT_S = float(timeout)
    if base_url is not None:
        BASE_URL = base_url
    if base_url is not None or timeout is not None or api_key is not None:
        _client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=_timeout())
    if STRUCTURED and THINKING and KIND == "vllm":
        print("  [llm_backend] STRUCTURED=True suppresses reasoning on vLLM 0.8.5 — "
              "set structured=False to actually get <think> blocks")
    return {"model": MODEL, "thinking": THINKING, "structured": STRUCTURED,
            "max_output_tokens": MAX_OUTPUT_TOKENS, "temperature": TEMPERATURE,
            "top_p": TOP_P, "seed": SEED, "base_url": BASE_URL, "kind": KIND, "params": PARAMS}


def _loads_lenient(text, finish_reason=None):
    """Same escalation the notebooks use: strict -> trailing-comma fix -> json_repair -> raise."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    for cand in (t, re.sub(r",(\s*[}\]])", r"\1", t)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    if repair_json is not None:
        try:
            obj = json.loads(repair_json(t))
            print(f"    [json_repair] salvaged a malformed response ({len(t)} chars)")
            return obj
        except Exception:
            pass
    try:
        json.loads(t)
    except json.JSONDecodeError as e:
        pos = getattr(e, "pos", 0)
        ctx = t[max(0, pos - 160): pos + 80]
        truncated = finish_reason == "length"
        raise ValueError(
            f"vLLM JSON unparseable (finish_reason={finish_reason}; {len(t)} chars; "
            f"error at char {pos}). "
            + ("Looks TRUNCATED (hit max_tokens) — raise MAX_OUTPUT_TOKENS or split the call. "
               if truncated else
               "NOT truncated (finished normally) — malformed token mid-JSON. ")
            + f"json error: {e}\n...context around the error (>>> marks the spot):\n"
            + f"{ctx[:160]!r} >>> {ctx[160:]!r}") from None


_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.S)


def _split_thinking(message):
    """Return (answer_text, reasoning_text), whoever did the splitting.

    Three cases, all seen on this setup:
      * server parser worked      -> content=JSON, reasoning_content=CoT
      * server parser over-claimed -> content=None, reasoning_content=the whole JSON. vLLM's
        deepseek_r1 parser assumes generation starts inside a think block and runs to `</think>`;
        when a JSON grammar suppresses those tags there is no `</think>`, so it swallows everything.
      * no server parser          -> content='<think>...</think>{json}', reasoning_content empty

    Hosted providers split the CoT out too but do not agree on the field name: vLLM and DeepSeek
    call it `reasoning_content`, OpenRouter and several others call it `reasoning`. Take whichever
    is present — an unread CoT would otherwise be silently dropped from the token ledger.
    """
    content = message.content
    reasoning = (getattr(message, "reasoning_content", None)
                 or getattr(message, "reasoning", None) or "")

    if content is None:
        content, reasoning = reasoning, ""

    m = _THINK_RE.search(content)
    if m:
        reasoning = reasoning or m.group(1)
        content = content[m.end():]

    return content, reasoning


def _thinking_kwargs(kind, thinking, params):
    """Per-kind request pieces for reasoning control. See the KIND note at the top of the file."""
    if kind == "vllm":
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}}}
    return dict(params or {})       # hosted: whatever the provider calls it, straight from the spec


def llm_json(system_prompt, user_message, model=None, temperature=0):
    """Drop-in replacement for the notebooks' `gemini_json`."""
    global last_latency_s, LAST_USAGE

    kwargs = dict(
        model=model or MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_message}],
        temperature=TEMPERATURE if TEMPERATURE is not None else temperature,
        **_thinking_kwargs(KIND, THINKING, PARAMS),
    )
    if TOP_P is not None:
        kwargs["top_p"] = TOP_P
    if SEED is not None:
        kwargs["seed"] = SEED
    if STRUCTURED:
        kwargs["response_format"] = {"type": "json_object"}
    if MAX_OUTPUT_TOKENS is not None:
        kwargs["max_tokens"] = MAX_OUTPUT_TOKENS

    t0 = time.perf_counter()
    resp = _client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0        # local: the global below is shared across threads
    last_latency_s = elapsed

    choice, usage = resp.choices[0], resp.usage
    answer, reasoning = _split_thinking(choice.message)

    if choice.finish_reason == "length":
        print(f"    [llm_backend] TRUNCATED at {usage.completion_tokens} output tokens after "
              f"{elapsed:.0f}s — raise the server's --max-model-len (MAXLEN=...), or if the "
              f"count is far above the sizing note, it is a repetition loop (try temperature=0.6)")

    details = getattr(usage, "completion_tokens_details", None)
    thoughts = getattr(details, "reasoning_tokens", None)
    if thoughts is None:
        # this vLLM build doesn't report reasoning tokens; approximate from the CoT text
        thoughts = len(reasoning) // 4 or None

    LAST_USAGE = {"prompt_tokens": usage.prompt_tokens,
                  "output_tokens": usage.completion_tokens,
                  "thoughts_tokens": thoughts,
                  "total_tokens": usage.total_tokens}

    # The notebooks read bare `last_latency_s` / `LAST_USAGE` from their own namespace, which under
    # Jupyter is __main__ — not this module. Setting them here only would leave the notebook's
    # copies at None: a crash on `f"{last_latency_s:.1f}s"`, and silently lost token accounting.
    __main__.last_latency_s = last_latency_s
    __main__.LAST_USAGE = LAST_USAGE
    # …and a per-thread copy, because those globals hold whichever call finished LAST. A batch cell
    # running articles in parallel must read its own call's counts: nb_parallel.last_usage().
    # The same record feeds the ledger behind nb_parallel.usage_report().
    if nb_parallel is not None:
        nb_parallel.record_usage(LAST_USAGE, seconds=elapsed)

    return _loads_lenient(answer, choice.finish_reason)


# ---------------------------------------------------------------------------------------------
# Backend registry — one switch per notebook instead of commenting blocks in and out.
#
#   kind      "gemini" keeps the notebook's own API function; "vllm" and "hosted" swap in llm_json
#   model     for vllm this must equal the server's --served-model-name (SERVED=... in the sbatch);
#             for hosted it is the provider's model id, which is NOT the Hugging Face repo name
#   key_env   hosted only: the environment variable holding the API key
#   params    hosted only: extra request params, e.g. {"reasoning_effort": "high"}
# The key IS the results folder name: results/<BACKEND>/. Keyed by backend rather than by model so
# two configs of the same weights (structured vs thinking) never share grown _global/ vocabularies —
# and so a notebook can resolve `ROOT / "results" / NAME` without importing anything.
#   url       optional; only needed when two servers are tunnelled at once, e.g. after
#             `PORT=8001 ./cluster/labia-tunnel.sh n5` -> url="http://localhost:8001/v1"
# ---------------------------------------------------------------------------------------------
BACKENDS = {
    "gemini-2.5-flash": dict(kind="gemini", model="gemini-2.5-flash"),

    # structured arm — grammar-enforced JSON, no reasoning (vLLM 0.8.5 cannot do both)
    "qwen3-14b":        dict(kind="vllm", model="qwen3-14b",   thinking=False, structured=True),
    "qwen3-32b":        dict(kind="vllm", model="qwen3-32b",   thinking=False, structured=True),
    "gemma3-12b":       dict(kind="vllm", model="gemma3-12b",  thinking=False, structured=True),
    "gemma3-27b":       dict(kind="vllm", model="gemma3-27b",  thinking=False, structured=True),
    "llama33-70b":      dict(kind="vllm", model="llama33-70b", thinking=False, structured=True),

    # thinking arm — soft JSON + reasoning, the closer match to the Gemini baseline
    "qwen3-14b-think":  dict(kind="vllm", model="qwen3-14b",   thinking=True,  structured=False),
    "qwen3-32b-think":  dict(kind="vllm", model="qwen3-32b",   thinking=True,  structured=False),

    # ---- hosted open-weight endpoints -------------------------------------------------------
    # No cluster, no tunnel, no MAXLEN sizing: these are 128k+ context and are billed per token.
    # NOTE these are NOT rows in the structured-vs-thinking table. That split is a vLLM 0.8.5
    # limitation; both of these do grammar-constrained JSON *and* reasoning in one call, so they
    # are a third condition and should be reported as one. See REPORTING_PLAN.md.
    #
    # gpt-oss-120b reasons unconditionally (the harmony format has no off switch); the only control
    # is reasoning_effort low|medium|high. "high" is the setting that matches what the Gemini
    # baseline was allowed to spend.
    "gpt-oss-120b": dict(kind="hosted", model="accounts/fireworks/models/gpt-oss-120b",
                         url="https://api.fireworks.ai/inference/v1",
                         key_env="FIREWORKS_API_KEY",
                         thinking=True, structured=True,
                         params={"reasoning_effort": "high"}),

    # DeepSeek V4 Flash, served first-party (MIT weights, ~284B/13B active, 1M context). Going
    # direct rather than through an aggregator means the manifest's model id actually pins who
    # served it and at what precision. The legacy deepseek-chat / deepseek-reasoner ids were
    # retired 2026-07-24; thinking mode is a SEPARATE model id — confirm the current one at
    # api-docs.deepseek.com before adding a "-think" row here.
    "deepseek-v4-flash": dict(kind="hosted", model="deepseek-v4-flash",
                              url="https://api.deepseek.com/v1",
                              key_env="DEEPSEEK_API_KEY",
                              thinking=False, structured=True),
}


def make_llm(name, base_url=None, seed=None, temperature=None, top_p=None,
             max_output_tokens=None, timeout=None, gemini_fn=None, verify=True):
    """A `gemini_json`-shaped callable BOUND to one backend, without touching module state.

    `activate` configures the module: model, base_url and client are globals, so a second call
    redirects everything already using the first. That is fine when a notebook runs one backend at a
    time and wrong when it runs two at once — the second judge's server would silently answer the
    first judge's calls, and the numbers would look perfectly plausible. This returns a closure
    instead, so N of them can be in flight on N threads.

    Gemini has no per-call client to build here; pass the notebook's own `gemini_json` as gemini_fn.
    """
    spec = dict(BACKENDS[name])
    kind = spec.pop("kind")
    model = spec.pop("model")
    if kind == "gemini":
        if gemini_fn is None:
            raise ValueError("make_llm(...) for a Gemini backend needs gemini_fn=<the notebook's gemini_json>")
        return gemini_fn

    url = base_url or spec.pop("url", None) or BASE_URL
    thinking = spec.get("thinking", THINKING)
    structured = spec.get("structured", STRUCTURED)
    params = spec.get("params")
    if thinking and kind == "vllm":               # greedy decoding sends Qwen3 into repetition loops
        temperature = 0.6 if temperature is None else temperature
        top_p = 0.95 if top_p is None else top_p
        seed = 1234 if seed is None else seed
    key = _resolve_key(spec["key_env"], name) if kind == "hosted" else "EMPTY"
    client = OpenAI(base_url=url, api_key=key,
                    timeout=httpx.Timeout(float(timeout or TIMEOUT_S), connect=10.0))

    if verify:
        # Ask the server what it serves, now, before any work is queued. A dead tunnel or a node
        # that exited is otherwise invisible until the run finishes: every call fails, run_judge
        # retries and gives up, and the batch still reports "judged N/N" with every verdict null.
        # Failing here costs ten seconds; not failing here costs the whole run.
        try:
            served = [m.id for m in client.models.list().data]
        except Exception as exc:
            where = ("no vLLM server answering at {url} — check the tunnel and that the job is "
                     "still running" if kind == "vllm" else
                     "{url} did not answer — check the key in ${env} and that the account is "
                     "funded").format(url=url, env=spec.get("key_env"))
            raise RuntimeError(f"{name}: {where} ({type(exc).__name__}).") from None
        if model not in served:
            if kind == "vllm":
                raise RuntimeError(
                    f"{name}: {url} is serving {served}, not {model!r}. The tunnel is pointing at "
                    f"the wrong node, or the job was started with a different SERVED= name.")
            # A hosted catalogue is long and churns as models are added and retired, so a miss here
            # is a warning: the id may be valid and merely unlisted. A wrong id still 404s on the
            # first real call, with the provider's own message.
            print(f"  [make_llm:{name}] {model!r} is not in {url}'s catalogue ({len(served)} "
                  f"models). It may be unlisted, or renamed — check the provider's model page.")

    def call(system_prompt, user_message, model=model, temperature=temperature):
        kwargs = dict(model=model,
                      messages=[{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_message}],
                      temperature=0 if temperature is None else temperature,
                      **_thinking_kwargs(kind, thinking, params))
        if top_p is not None:
            kwargs["top_p"] = top_p
        if seed is not None:
            kwargs["seed"] = seed
        if structured:
            kwargs["response_format"] = {"type": "json_object"}
        if max_output_tokens is not None:
            kwargs["max_tokens"] = max_output_tokens

        t0 = time.perf_counter()
        resp = client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0
        choice, usage = resp.choices[0], resp.usage
        answer, reasoning = _split_thinking(choice.message)
        if choice.finish_reason == "length":
            print(f"    [make_llm:{name}] TRUNCATED at {usage.completion_tokens} output tokens")
        details = getattr(usage, "completion_tokens_details", None)
        thoughts = getattr(details, "reasoning_tokens", None) or (len(reasoning) // 4 or None)
        if nb_parallel is not None:                # thread-local ledger: safe under a shared pool
            nb_parallel.record_usage({"prompt_tokens": usage.prompt_tokens,
                                      "output_tokens": usage.completion_tokens,
                                      "thoughts_tokens": thoughts,
                                      "total_tokens": usage.total_tokens},
                                     seconds=elapsed, tag=name)
        return _loads_lenient(answer, choice.finish_reason)

    call.backend = name
    call.info = {"backend": name, "kind": kind, "model": model, "base_url": url,
                 "thinking": thinking, "structured": structured, "params": params,
                 "temperature": temperature, "top_p": top_p, "seed": seed}
    return call


def activate(name, root, ns, results=None, **overrides):
    """Point a notebook at BACKENDS[name]. Call as the LAST statement of the Gemini client cell.

    `ns` is the notebook's globals(). Writing through it is what makes the swap stick: the cell
    above defines its own `gemini_json`, and whichever binding happens last wins.

    `results` names the results/ subfolder to read and write, and defaults to `name`. Pass it
    explicitly when the model doing the work is NOT the model whose outputs are being processed —
    e.g. grounding qwen3-14b's graphs with Gemini as the judge:
        activate("gemini-2.5-flash", ROOT, globals(), results="qwen3-14b")

    `overrides` go to configure() for vllm backends (e.g. temperature=0.6, max_output_tokens=8192).

    A THINKING backend gets Qwen3's own recommended sampling (temperature 0.6, top_p 0.95, seed
    1234) unless the caller overrides it. This is not a preference: greedy decoding is what sends
    Qwen3 into repetition loops in thinking mode, and the symptom — a call that runs out of context
    having emitted far MORE tokens than the sizing note predicts — reads as "the model is slow", so
    it can burn a whole run before anyone notices. The default lives here rather than in each
    notebook because it is a property of the BACKEND, and a notebook that forgot to repeat it was
    silently running the arm wrong. The structured arm stays greedy, matching the Gemini baseline.
    """
    if name not in BACKENDS:
        raise KeyError(f"unknown BACKEND {name!r}; choose one of: {', '.join(BACKENDS)}")
    spec = dict(BACKENDS[name])
    kind = spec.pop("kind")
    model = spec.pop("model")
    results = results or name
    if "url" in spec:
        spec["base_url"] = spec.pop("url")
    _reset_sampling()                                 # see the note there: activations must not stack

    ns["GEMINI_MODEL"] = model
    ns["RESULTS_DIR"] = root / "results" / results
    note = "" if results == name else f"   (reading/writing another model's tree)"

    if kind == "gemini":
        # Leave the notebook's own gemini_json/MAX_OUTPUT_TOKENS/THINKING_BUDGET untouched — they
        # were just (re)defined above in this same cell.
        print(f"backend: GEMINI {model}  ->  results/{results}/{note}")
        return {"backend": name, "kind": kind, "model": model, "results_dir": ns["RESULTS_DIR"]}

    # The Qwen3 sampling defaults are Qwen3's own recommendation and are deliberately NOT applied
    # to hosted backends: those models have their own, and a hosted run is compared against the
    # Gemini baseline, which ran greedy.
    if spec.get("thinking") and kind == "vllm":       # see the docstring: greedy + thinking loops
        for k, v in (("temperature", 0.6), ("top_p", 0.95), ("seed", 1234)):
            overrides.setdefault(k, v)                # an explicit override still wins

    # Both are passed on every activate, never left to persist: a notebook that switches backends
    # mid-kernel would otherwise send the previous backend's key or reasoning params to the new one.
    key_env = spec.pop("key_env", None)
    if kind == "hosted" and "api_key" not in overrides:
        overrides["api_key"] = _resolve_key(key_env, name)
    overrides.setdefault("api_key", "EMPTY")
    spec.setdefault("params", {})

    configure(model=model, kind=kind, **spec, **overrides)
    ns["gemini_json"] = llm_json
    ns["MAX_OUTPUT_TOKENS"] = MAX_OUTPUT_TOKENS
    ns["THINKING_BUDGET"] = -1 if THINKING else 0     # Qwen3 is on/off; no token budget exists
    # Print the sampling params too: a thinking run left greedy silently produces repetition loops
    # that look like "the model is slow", so the arm's temperature belongs where it can be seen.
    print(f"backend: {'vLLM' if kind == 'vllm' else 'HOSTED'} {BASE_URL} model={model} "
          f"thinking={THINKING} structured={STRUCTURED} max_tokens={MAX_OUTPUT_TOKENS} "
          f"temperature={'caller' if TEMPERATURE is None else TEMPERATURE} "
          f"top_p={TOP_P} seed={SEED} timeout={TIMEOUT_S:.0f}s"
          + (f" params={PARAMS}" if PARAMS else "")
          + f"  ->  results/{results}/{note}")
    return {"backend": name, "kind": kind, "model": model, "thinking": THINKING,
            "structured": STRUCTURED, "results_dir": ns["RESULTS_DIR"]}
