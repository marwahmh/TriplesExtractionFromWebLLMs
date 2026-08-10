"""Run the per-article notebook batches in parallel.

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
"""

import contextlib
import io
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

DEFAULT_WORKERS = 4


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


def record_usage(usage):
    """Record the usage of the call THIS thread just made. Returns it, for use inline."""
    _usage.value = usage
    return usage


def last_usage():
    """Usage of the most recent LLM call made by THIS thread (None if it has made none)."""
    return getattr(_usage, "value", None)


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

    failed = [r for r in results if not r.ok]
    if header and n:
        print(f"\ndone — {n - len(failed)}/{n} ok in {time.perf_counter() - t_start:.1f}s"
              + (f"; FAILED: {', '.join(str(label(r.item)) for r in failed)}" if failed else ""))
    return results


def values(results):
    """The values of the successful results, dropping failures."""
    return [r.value for r in results if r.ok]
