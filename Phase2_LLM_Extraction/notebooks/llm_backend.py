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
MODEL = "qwen3-14b"                               # matches --served-model-name in the sbatch script
THINKING = True                                   # Qwen3: on/off only, no token budget
STRUCTURED = True                                 # grammar-constrained JSON (see note)
MAX_OUTPUT_TOKENS = None                          # no cap — see sizing note below
TEMPERATURE = None                                # if set, overrides the caller's temperature
TOP_P = None                                      # Qwen3 recommends 0.95 whenever temperature > 0

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

_client = OpenAI(base_url=BASE_URL, api_key="EMPTY")


def configure(model=None, thinking=None, structured=None, max_output_tokens=None,
              temperature=None, top_p=None, base_url=None):
    """Set the backend globals. Call once per notebook before any extraction cell."""
    global MODEL, THINKING, STRUCTURED, MAX_OUTPUT_TOKENS, TEMPERATURE, TOP_P, BASE_URL, _client
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
    if base_url is not None:
        BASE_URL = base_url
        _client = OpenAI(base_url=BASE_URL, api_key="EMPTY")
    if STRUCTURED and THINKING:
        print("  [llm_backend] STRUCTURED=True suppresses reasoning on vLLM 0.8.5 — "
              "set structured=False to actually get <think> blocks")
    return {"model": MODEL, "thinking": THINKING, "structured": STRUCTURED,
            "max_output_tokens": MAX_OUTPUT_TOKENS, "temperature": TEMPERATURE,
            "base_url": BASE_URL}


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
    """
    content = message.content
    reasoning = getattr(message, "reasoning_content", None) or ""

    if content is None:
        content, reasoning = reasoning, ""

    m = _THINK_RE.search(content)
    if m:
        reasoning = reasoning or m.group(1)
        content = content[m.end():]

    return content, reasoning


def llm_json(system_prompt, user_message, model=None, temperature=0):
    """Drop-in replacement for the notebooks' `gemini_json`."""
    global last_latency_s, LAST_USAGE

    kwargs = dict(
        model=model or MODEL,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_message}],
        temperature=TEMPERATURE if TEMPERATURE is not None else temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": THINKING}},
    )
    if TOP_P is not None:
        kwargs["top_p"] = TOP_P
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
#   kind      "gemini" keeps the notebook's own API function; "vllm" swaps in llm_json
#   model     for vllm this must equal the server's --served-model-name (SERVED=... in the sbatch)
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
}


def activate(name, root, ns, results=None, **overrides):
    """Point a notebook at BACKENDS[name]. Call as the LAST statement of the Gemini client cell.

    `ns` is the notebook's globals(). Writing through it is what makes the swap stick: the cell
    above defines its own `gemini_json`, and whichever binding happens last wins.

    `results` names the results/ subfolder to read and write, and defaults to `name`. Pass it
    explicitly when the model doing the work is NOT the model whose outputs are being processed —
    e.g. grounding qwen3-14b's graphs with Gemini as the judge:
        activate("gemini-2.5-flash", ROOT, globals(), results="qwen3-14b")

    `overrides` go to configure() for vllm backends (e.g. temperature=0.6, max_output_tokens=8192).
    """
    if name not in BACKENDS:
        raise KeyError(f"unknown BACKEND {name!r}; choose one of: {', '.join(BACKENDS)}")
    spec = dict(BACKENDS[name])
    kind = spec.pop("kind")
    model = spec.pop("model")
    results = results or name
    if "url" in spec:
        spec["base_url"] = spec.pop("url")

    ns["GEMINI_MODEL"] = model
    ns["RESULTS_DIR"] = root / "results" / results
    note = "" if results == name else f"   (reading/writing another model's tree)"

    if kind == "gemini":
        # Leave the notebook's own gemini_json/MAX_OUTPUT_TOKENS/THINKING_BUDGET untouched — they
        # were just (re)defined above in this same cell.
        print(f"backend: GEMINI {model}  ->  results/{results}/{note}")
        return {"backend": name, "kind": kind, "model": model, "results_dir": ns["RESULTS_DIR"]}

    configure(model=model, **spec, **overrides)
    ns["gemini_json"] = llm_json
    ns["MAX_OUTPUT_TOKENS"] = MAX_OUTPUT_TOKENS
    ns["THINKING_BUDGET"] = -1 if THINKING else 0     # Qwen3 is on/off; no token budget exists
    print(f"backend: vLLM {BASE_URL} model={model} "
          f"thinking={THINKING} structured={STRUCTURED} max_tokens={MAX_OUTPUT_TOKENS}"
          f"  ->  results/{results}/{note}")
    return {"backend": name, "kind": kind, "model": model, "thinking": THINKING,
            "structured": STRUCTURED, "results_dir": ns["RESULTS_DIR"]}
