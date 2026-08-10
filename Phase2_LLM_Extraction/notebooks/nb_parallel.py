"""Batch-run bookkeeping for the notebooks: a thread pool, and one token/time ledger.

Two jobs. (1) Run the per-article batches in parallel where the articles are independent.
(2) Keep one ledger of every LLM call — prompt / output / THINKING tokens and latency — so any
notebook can end its batch cell with `usage_report(...)` and get the same honest account of what
the run cost. Both the notebooks' own Gemini clients and `llm_backend.llm_json` (vLLM) feed it.

PART 1 — PARALLELISM

The batch cells of 6c (e2e extraction), 9 / 9b (grounding) and 10 (MINE-1) all do the same thing:
loop over articles, make a handful of LLM calls each, write one file per article. That work is
I/O-bound — nearly all the wall-clock is spent waiting on the API — and the articles are
independent, so a thread pool turns an N-article run into roughly an N/WORKERS-article one.

Only steps whose articles are INDEPENDENT belong here. The canonicalization notebooks (4, 7, 7b,
6d, 6e) grow one shared vocabulary article by article, and article k's minting decisions depend on
what article k-1 added — running those in parallel would change the result, not just the speed.

What this module adds over a bare `ThreadPoolExecutor.map`:

  * per-task stdout capture. The notebooks print progress as they go; run those prints concurrently
    and the lines interleave into noise. Each task gets a buffer bound to its own thread, and the
    whole block is printed in one piece, by the MAIN thread, when that task finishes. That also
    side-steps ipykernel's shaky handling of writes coming from background threads.
  * `quiet()` — a thread-local `redirect_stdout`. `contextlib.redirect_stdout` swaps `sys.stdout`
    for the whole process, so one article silencing its judge would swallow every other article's
    output for the duration.
  * a thread-local `LAST_USAGE`. The extraction notebook reads token counts off a module global
    right after each call; with several calls in flight that global belongs to whichever thread
    wrote last. `record_usage` / `last_usage` keep one slot per thread.
  * failures are returned, not raised — one bad article never kills the batch.

Usage:

    import nb_parallel
    res  = nb_parallel.pmap(extract_e2e, ARTICLE_IDS, workers=4)
    runs = [(r.item, r.value) for r in res if r.ok]

`workers=1` runs the plain sequential loop with live printing — the exact old behaviour, which is
what you want while debugging a single article.

Picking WORKERS: the ceiling is the provider's rate limit, not the CPU. Gemini 2.5 Flash on a paid
tier tolerates 4-8 comfortably; on the free tier (10 RPM) stay at 1-2. A local vLLM server batches
requests itself and is happiest with several in flight — 4-8 there too, though a 70B on two A6000s
will queue rather than speed up much beyond that.

PART 2 — TOKEN AND TIME ACCOUNTING

    nb_parallel.reset_usage()                       # start of the batch cell
    with nb_parallel.usage_scope(aid):              # (pmap does this for you, per item)
        process_article(aid)
    nb_parallel.usage_report("nb4 · type extension", wall_seconds=..., write=...)

Thinking tokens are tracked as their own column: the API bills them as output but reports them
outside the output count, and on a reasoning model they are routinely the larger half — a total
that hides them understates the run. `last_usage()` returns the calling THREAD's most recent call,
which is what a parallel batch must read; the module-level LAST_USAGE globals belong to whichever
call finished last.
"""

import contextlib
import io
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

DEFAULT_WORKERS = 4

LAST_WALL_SECONDS = None      # wall time of the most recent pmap() — pass it to usage_report()


# --------------------------------------------------------------------------------------------
# Thread-aware stdout: one `print` destination per thread, installed once over the real stream.
# --------------------------------------------------------------------------------------------
class _ThreadedStdout:
    """A `sys.stdout` stand-in that routes each thread's writes to that thread's own buffer.

    Threads with no buffer bound (the main thread, normally) write straight through to the real
    stream, so nothing changes outside a `capture()` block."""

    def __init__(self, real):
        self._real = real
        self._local = threading.local()

    def _target(self):
        return getattr(self._local, "buf", None) or self._real

    def write(self, s):
        return self._target().write(s)

    def flush(self):
        target = self._target()
        if hasattr(target, "flush"):
            return target.flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def isatty(self):
        return False

    def __getattr__(self, name):          # everything else (encoding, fileno, …) is the real stream's
        return getattr(self.__dict__["_real"], name)


_proxy: Optional[_ThreadedStdout] = None


def install():
    """Put the thread-aware stdout in place (idempotent). Call once, near the client cell."""
    global _proxy
    current = sys.stdout
    # Re-running the client cell reloads this module, which drops `_proxy` while `sys.stdout` still
    # holds the proxy the previous version installed. Unwrap those instead of stacking on them.
    while type(current).__name__ == "_ThreadedStdout" and current is not _proxy:
        current = current.__dict__["_real"]
    if _proxy is None:
        _proxy = _ThreadedStdout(current)
    elif current is not _proxy:
        _proxy._real = current            # kernel restart / stream swap: re-wrap whatever is current
    sys.stdout = _proxy
    return _proxy


@contextlib.contextmanager
def capture():
    """Send THIS thread's prints to a fresh buffer for the duration; yields the buffer."""
    proxy = install()
    previous = getattr(proxy._local, "buf", None)
    buf = io.StringIO()
    proxy._local.buf = buf
    try:
        yield buf
    finally:
        proxy._local.buf = previous


def quiet(fn, *args, **kwargs):
    """Call `fn` with its output swallowed — for THIS thread only, unlike `redirect_stdout`."""
    with capture():
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------------------------
# Thread-local token usage (see the module docstring).
# --------------------------------------------------------------------------------------------
_usage = threading.local()

# Every recorded call also lands in a shared ledger, so any notebook can end its batch cell with
# `usage_report(...)` and get one honest account of what the run cost: calls, prompt / output /
# THINKING tokens, and wall time. Thinking tokens are billed as output but reported separately by
# the API — on a reasoning model they are routinely the larger half, so a total that hides them
# understates the run. Notebooks that never see a reasoning model just report zero.
USAGE_KEYS = ("prompt_tokens", "output_tokens", "thoughts_tokens", "total_tokens")

_ledger = []
_ledger_lock = threading.Lock()


def record_usage(usage, seconds=None, tag=None):
    """Record one call: per-thread (for `last_usage`) and in the shared ledger. Returns `usage`."""
    _usage.value = usage
    entry = {k: (usage or {}).get(k) for k in USAGE_KEYS}
    entry["seconds"] = seconds
    entry["tag"] = tag if tag is not None else getattr(_usage, "tag", None)
    with _ledger_lock:
        _ledger.append(entry)
    return usage


def record_gemini_usage(resp, seconds=None, tag=None):
    """Pull `usage_metadata` off a google-genai response, record it, and return it as a dict.

    None-safe: a response without usage metadata records a row of Nones rather than raising, so
    token accounting can never be the thing that breaks an extraction run."""
    try:
        um = resp.usage_metadata
        usage = {"prompt_tokens": um.prompt_token_count,
                 "output_tokens": um.candidates_token_count,
                 "thoughts_tokens": getattr(um, "thoughts_token_count", None),
                 "total_tokens": um.total_token_count}
    except Exception:                                     # noqa: BLE001
        usage = {k: None for k in USAGE_KEYS}
    return record_usage(usage, seconds=seconds, tag=tag)


def last_usage():
    """Usage of the most recent LLM call made by THIS thread (None if it has made none)."""
    return getattr(_usage, "value", None)


@contextlib.contextmanager
def usage_scope(tag):
    """Tag every call this thread makes inside the block, so `usage_report` can break totals down
    per article. `pmap` opens one of these per item automatically."""
    previous = getattr(_usage, "tag", None)
    _usage.tag = tag
    try:
        yield
    finally:
        _usage.tag = previous


def reset_usage():
    """Clear the ledger — call at the start of a batch so the report covers only this run."""
    with _ledger_lock:
        _ledger.clear()


def usage_calls():
    """A copy of the ledger: one row per LLM call."""
    with _ledger_lock:
        return list(_ledger)


def usage_totals():
    """Summed ledger: {calls, prompt_tokens, output_tokens, thoughts_tokens, total_tokens, seconds}."""
    rows = usage_calls()
    out = {"calls": len(rows)}
    out.update({k: sum(r[k] or 0 for r in rows) for k in USAGE_KEYS})
    out["seconds"] = round(sum(r["seconds"] or 0 for r in rows), 1)
    return out


def _by_tag(rows):
    per = {}
    for r in rows:
        t = r["tag"] if r["tag"] is not None else "-"
        d = per.setdefault(t, {"calls": 0, "seconds": 0.0, **{k: 0 for k in USAGE_KEYS}})
        d["calls"] += 1
        d["seconds"] += r["seconds"] or 0
        for k in USAGE_KEYS:
            d[k] += r[k] or 0
    for d in per.values():
        d["seconds"] = round(d["seconds"], 1)
    return per


def usage_report(label="", write=None, wall_seconds=None, per_tag=True, **meta):
    """Print (and optionally write) what the LLM calls in the ledger cost.

    `wall_seconds` is the batch's real elapsed time; without it the report shows summed call time,
    which for a parallel run is larger than the clock on the wall. Returns the report dict."""
    rows = usage_calls()
    totals = usage_totals()
    per = _by_tag(rows) if per_tag else {}
    tagged = [t for t in per if t != "-"]

    print(f"\n{'=' * 72}\nLLM USAGE" + (f" — {label}" if label else ""))
    if per_tag and tagged:
        print(f'{"":14}{"calls":>6}{"prompt":>11}{"output":>11}{"thinking":>10}{"total":>11}{"llm_s":>9}')
        for t in sorted(per):
            d = per[t]
            print(f'{str(t)[:13]:14}{d["calls"]:6}{d["prompt_tokens"]:11,}{d["output_tokens"]:11,}'
                  f'{d["thoughts_tokens"]:10,}{d["total_tokens"]:11,}{d["seconds"]:9.1f}')
        print("-" * 72)
    print(f'{"TOTAL":14}{totals["calls"]:6}{totals["prompt_tokens"]:11,}{totals["output_tokens"]:11,}'
          f'{totals["thoughts_tokens"]:10,}{totals["total_tokens"]:11,}{totals["seconds"]:9.1f}')
    if wall_seconds is not None:
        # only meaningful once the wall clock is long enough to divide by (a parallel batch)
        speedup = (f"  ({totals['seconds'] / wall_seconds:.1f}x overlap)"
                   if wall_seconds >= 1 and totals["seconds"] > wall_seconds * 1.05 else "")
        print(f"wall clock: {wall_seconds:.1f}s   (summed LLM time {totals['seconds']:.1f}s){speedup}")
    if not totals["thoughts_tokens"]:
        print("thinking tokens: 0 — this backend reported none (thinking off, or not reported)")

    report = {"label": label, "totals": totals, "wall_seconds": wall_seconds,
              "per_tag": per, "calls": rows, **meta}
    if write is not None:
        from pathlib import Path as _Path
        _Path(write).parent.mkdir(parents=True, exist_ok=True)
        _Path(write).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        print(f"wrote {write}")
    return report


# --------------------------------------------------------------------------------------------
# The runner.
# --------------------------------------------------------------------------------------------
@dataclass
class Result:
    item: Any
    value: Any = None
    error: Optional[BaseException] = None
    output: str = ""
    seconds: float = 0.0

    @property
    def ok(self):
        return self.error is None


def _indent(text, prefix="   "):
    return "\n".join(prefix + line for line in text.rstrip("\n").split("\n"))


def pmap(fn, items, workers=DEFAULT_WORKERS, label=str, echo=True, header=True):
    """Run `fn(item)` over `items` on a thread pool; return a Result per item, in input order.

    Output from each task is captured and printed as one block when that task finishes, so the
    articles' progress lines never interleave. Exceptions are caught and attached to the Result
    (`.ok` is False) instead of aborting the batch. `workers=1` skips the pool entirely and runs
    the plain sequential loop, printing live.
    """
    items = list(items)
    n = len(items)
    if header and n:
        how = "sequentially" if (workers or 1) <= 1 or n == 1 else f"on {min(workers, n)} threads"
        print(f"{n} item(s), running {how}\n")

    t_start = time.perf_counter()
    results = [None] * n

    if (workers or 1) <= 1 or n <= 1:                       # sequential: live output, old behaviour
        for i, item in enumerate(items):
            t0 = time.perf_counter()
            if echo:
                print(f"[{i + 1}/{n}] {label(item)}")
            try:
                with usage_scope(str(item)):
                    value, error = fn(item), None
            except Exception as exc:                        # noqa: BLE001 - one item must not kill the run
                value, error = None, exc
                if echo:
                    print(f"   FAILED: {type(exc).__name__}: {exc}")
            results[i] = Result(item, value, error, "", time.perf_counter() - t0)
    else:
        install()
        lock = threading.Lock()

        def _run(i, item):
            t0 = time.perf_counter()
            with capture() as buf:
                try:
                    with usage_scope(str(item)):
                        value, error = fn(item), None
                except Exception as exc:                    # noqa: BLE001
                    value, error = None, exc
            return i, Result(item, value, error, buf.getvalue(), time.perf_counter() - t0)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nbpar") as pool:
            futures = [pool.submit(_run, i, item) for i, item in enumerate(items)]
            for done, future in enumerate(as_completed(futures), start=1):
                i, result = future.result()
                results[i] = result
                if echo:
                    with lock:                              # printed from the MAIN thread, one block
                        print(f"[{done}/{n}] {label(result.item)}  ({result.seconds:.1f}s)")
                        if result.output.strip():
                            print(_indent(result.output))
                        if result.error is not None:
                            print(f"   FAILED: {type(result.error).__name__}: {result.error}")

    global LAST_WALL_SECONDS
    LAST_WALL_SECONDS = time.perf_counter() - t_start
    failed = [r for r in results if not r.ok]
    if header and n:
        print(f"\ndone — {n - len(failed)}/{n} ok in {LAST_WALL_SECONDS:.1f}s"
              + (f"; FAILED: {', '.join(str(label(r.item)) for r in failed)}" if failed else ""))
    return results


def values(results):
    """The values of the successful results, dropping failures."""
    return [r.value for r in results if r.ok]
