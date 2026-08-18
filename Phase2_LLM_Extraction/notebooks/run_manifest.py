"""One JSON record per run, written by the run itself — so a reported number stays traceable.

A stage of this pipeline writes files whose names carry a `run_tag` and nothing else. That was
enough while the tag was being typed by hand for one comparison at a time, and it stops being enough
the moment the numbers go into a thesis: `nary_graph_dedup_rr_orig2_mixedCons_extended.json` records
*what was varied*, not which config produced it, on which corpus, from which commit, at what token
cost. The notebooks that made it have since been edited. Nothing on disk can answer "what was this?"

This module makes each stage leave that answer behind:

    results/<model>/_runs/<run_id>.json

with, per stage, the fully resolved config the run actually read, the git commit and dirty flag at
start, the article ORDER (the vocabulary stages are incremental, so order is part of the run), every
output file it wrote, the token usage it spent, and the stage metrics dict.

It is wired in, not called by hand:

    experiment_config.load()  ->  begin(cfg, section)      opens the stage, guards against clobber
    run_metrics.save()        ->  finish(metrics=...)      closes it with outputs + metrics

so a notebook gets provenance by doing what it already does. `begin` REFUSES a stage that a manifest
says already finished under the same run id — that is the guard that stops a re-run from quietly
overwriting the run whose numbers are already written up. `PHASE2_OVERWRITE=1`, or `"overwrite":
true` in the config, is the deliberate way past it.

CLI:

    python -m run_manifest list [model]        every run on disk, one line each
    python -m run_manifest show <run_id>       the whole record
    python -m run_manifest diff <a> <b>        which config keys differ between two runs
    python -m run_manifest index               regenerate results/RUNS.md
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["begin", "finish", "note", "path_for", "run_id_for", "load", "all_runs", "index",
           "OVERWRITE_ENV"]

OVERWRITE_ENV = "PHASE2_OVERWRITE"
DISABLE_ENV = "PHASE2_NO_MANIFEST"      # escape hatch for a scratch run that should leave no record

# Which track a notebook section belongs to. The run id is per TRACK, because the staged and e2e
# chains write different files under the same experiment and must not share one record.
_TRACK_OF = {"2": "two_staged", "4": "two_staged", "6": "two_staged", "6b": "two_staged",
             "7": "two_staged", "7b": "two_staged",
             "6c": "e2e", "6d": "e2e", "6e": "e2e"}

# The stage currently open in this kernel: {"path", "run_id", "section", "results_dir", "started",
# "started_exact"}. `finish` prefers this over guessing, and falls back to the manifest on disk when
# the kernel was restarted between the build cell and the metrics cell.
_ACTIVE: dict | None = None

# Files that are not a stage's output even when they were touched while it ran.
_IGNORE_PARTS = ("_runs", "__pycache__", ".DS_Store", "_CONTENTS.txt", "README.md", "RUNS.md")


# --------------------------------------------------------------------------------------------
# identity + locations
# --------------------------------------------------------------------------------------------
def run_id_for(cfg: dict, section: str | None = None, track: str | None = None) -> str:
    """`<track>__<variant>__<run_tag>` — the unit a manifest describes.

    One id spans every stage of one chain: 6c/6d/6e of the same experiment write into ONE record,
    which is what makes "how much did this run cost end to end" answerable. The variant is in the id
    because two extraction arms are two different runs even under one experiment name.
    """
    track = track or _TRACK_OF.get(section or "", "e2e")
    return f'{track}__{cfg.get("variant") or "novariant"}__{cfg.get("run_tag") or "untagged"}'


def runs_dir(results_dir) -> Path:
    return Path(results_dir) / "_runs"


def path_for(results_dir, run_id: str) -> Path:
    return runs_dir(results_dir) / f"{run_id}.json"


def load(results_dir, run_id: str) -> dict | None:
    p = path_for(results_dir, run_id)
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _git(root: Path) -> dict:
    """Commit / branch / dirty at this moment. A run made from uncommitted notebooks is still a
    valid run — it just has to SAY so, which `dirty` does."""
    def _run(*args):
        try:
            return subprocess.run(args, cwd=root, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    commit = _run("git", "rev-parse", "HEAD")
    if not commit:
        return {"commit": None, "branch": None, "dirty": None}
    return {"commit": commit,
            "short": commit[:8],
            "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD") or None,
            "dirty": bool(_run("git", "status", "--porcelain"))}


def _env(cfg: dict) -> dict:
    return {"host": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "backend": cfg.get("backend"),
            "gemini_model": cfg.get("gemini_model"),
            "match_model": cfg.get("match_model"),
            "vllm_base_url": os.environ.get("VLLM_BASE_URL")}


# --------------------------------------------------------------------------------------------
# begin / finish
# --------------------------------------------------------------------------------------------
def begin(cfg: dict, section: str, results_dir, root: Path | None = None,
          overwrite: bool | None = None, verbose: bool = True) -> dict | None:
    """Open `section` in this run's manifest. Returns the record (or None when disabled).

    Raises RuntimeError when the same section already FINISHED under this run id and overwrite was
    not asked for — the run about to start would land on top of files someone may already be citing.
    """
    global _ACTIVE
    if os.environ.get(DISABLE_ENV) == "1":
        _ACTIVE = None
        return None

    results_dir = Path(results_dir)
    track = _TRACK_OF.get(section, "e2e")
    rid = run_id_for(cfg, section, track)
    p = path_for(results_dir, rid)
    rec = load(results_dir, rid) or {
        "run_id": rid,
        "experiment": cfg.get("experiment"),
        "description": cfg.get("description"),
        "track": track,
        "results_model": cfg.get("results_model"),
        "variant": cfg.get("variant"),
        "run_tag": cfg.get("run_tag"),
        "input_tag": cfg.get("input_tag"),
        "created": _now(),
        "stages": {},
    }

    if overwrite is None:
        overwrite = os.environ.get(OVERWRITE_ENV) == "1" or bool(cfg.get("overwrite"))
    prev = rec["stages"].get(section, {})
    done = prev.get("finished")

    # A stage that finished having processed NOTHING is not a result to protect. When every article
    # failed — the llama server returning 500 on all four requests, say — nb_parallel swallows the
    # per-article errors, the metrics cell still runs, and the stage closes with articles=0. The
    # guard would then block every legitimate retry of a run that never happened. Protecting real
    # numbers is the point; there are none here, so take it over and say so.
    if done and not overwrite and (prev.get("metrics") or {}).get("summary", {}).get("articles") == 0:
        print(f"  note  stage {section!r} is recorded as finished on {done} but processed 0 "
              f"article(s) — an empty run, so it is not protected. Taking it over.")
        done = None

    if done and not overwrite:
        raise RuntimeError(
            f"\n  stage {section!r} already finished under run id {rid!r} on {done}.\n"
            f"  Its outputs are on disk and may already be reported; starting now would overwrite\n"
            f"  them in place, leaving the manifest describing one run and the files another.\n"
            f"    manifest : {p}\n"
            f"  Pick ONE:\n"
            f"    * a repeat of the same arm   ->  REPEAT=2 (or 3) in the environment, or\n"
            f"                                     \"repeat\": 2 in configs/{cfg.get('experiment')}.json\n"
            f"    * a different configuration  ->  a new experiment file with its own run_tag\n"
            f"    * really redo this one       ->  {OVERWRITE_ENV}=1 in the environment")

    _guard_extraction(rec, section, cfg, results_dir, rid, overwrite)
    _guard_concurrent(rec, section, rid, p, overwrite)

    # An overwrite may replace a stage, but it must not ERASE the record of what it replaced. The
    # write below overwrites `prev` wholesale, and when `prev` finished, that record is the only
    # account of a run whose files are still on disk — its outputs, token spend and metrics. This
    # bit was learned from a one-article repair run that reused an existing run_tag: it took over
    # 6b/7b and left the manifest claiming those stages covered a single article, with the real
    # 20-article spend gone. Keep the superseded record instead, oldest first.
    attempts = list(prev.get("attempts") or [])
    if prev.get("finished"):
        attempts.append({k: v for k, v in prev.items() if k != "attempts"})
        print(f"  note  the finished record for stage {section!r} is kept as attempt "
              f"#{len(attempts)} (started {prev.get('started', '?')[:19]}, "
              f"{len(prev.get('outputs') or [])} output file(s)) — "
              f"python -m run_manifest show {rid}")

    rec["stages"][section] = {
        "started": _now(),
        "owner": {"pid": os.getpid(), "host": socket.gethostname(), "at": _now()},
        "finished": None,
        "git": _git(root or _root_of(results_dir)),
        "env": _env(cfg),
        "config": cfg,                    # the WHOLE resolved config, not a summary of it
        "articles": cfg.get("selected_articles"),
        "outputs": [],
        "token_usage": None,
        "metrics": None,
        **({"attempts": attempts} if attempts else {}),
    }
    rec["updated"] = _now()
    _write(p, rec)
    # Two starts, deliberately: `started` is backdated a second so a file written in the same clock
    # tick as this call still counts as ours (mtime granularity), while `started_exact` is the real
    # instant, used to decide which stage a file belongs to. Using the backdated one for BOTH is how
    # a repeat that begins immediately after the previous one ends absorbs its outputs.
    _now_t = time.time()
    _ACTIVE = {"path": p, "run_id": rid, "section": section, "results_dir": results_dir,
               "started": _now_t - 1.0, "started_exact": _now_t}
    if verbose:
        g = rec["stages"][section]["git"]
        print(f'  run id  {rid}   stage {section}   commit {g.get("short") or "?"}'
              f'{" (dirty)" if g.get("dirty") else ""}')
        print(f'  manifest {p}')
    return rec


# 6c names its output by VARIANT alone — `e2e/nary_graph_dedup.json` carries no run tag, because
# every later stage has to find it by variant. That is fine until two runs of the same variant cover
# the same articles: the second silently replaces the graphs the first one's numbers were computed
# from, and every downstream tag then points at an extraction that no longer exists.
#
# 6b, the staged track's twin, is NOT in this list: it writes `nary_graph<_variant>_<run_tag>.json`,
# so its repeats already separate themselves and guarding it would refuse a legitimate re-run.
_EXTRACTION_SECTIONS = ("6c",)


def _guard_extraction(rec: dict, section: str, cfg: dict, results_dir: Path, rid: str,
                      overwrite: bool) -> None:
    """Refuse an extraction that would overwrite ANOTHER run's graphs for the same variant.

    Article-aware on purpose: running 6c for the test split after the validation split is the normal
    case and must pass — the two cover disjoint articles and write into different `art<N>/` folders.
    Only an OVERLAP is a real clobber.
    """
    if section not in _EXTRACTION_SECTIONS or overwrite:
        return
    mine = set(cfg.get("selected_articles") or [])
    variant = cfg.get("variant")
    for p in sorted(runs_dir(results_dir).glob("*.json")):
        if p.stem == rid:
            continue
        try:
            other = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        st = other.get("stages", {}).get(section)
        if not st or not st.get("finished") or other.get("variant") != variant:
            continue
        theirs = set(st.get("articles") or other.get("stages", {}).get(section, {})
                     .get("config", {}).get("selected_articles") or [])
        clash = (mine & theirs) if (mine and theirs) else {"(unknown — one run's article list is unset)"}
        if not clash:
            continue                       # disjoint article sets: different split, no collision
        raise RuntimeError(
            f"\n  extraction stage {section!r} for variant {variant!r} already ran under run id "
            f"{other['run_id']!r}\n  and covered {len(clash)} of the same article(s): "
            f"{sorted(clash)[:6]}{' …' if len(clash) > 6 else ''}\n"
            f"  Its graphs carry NO run tag (nary_graph_{variant}.json), so this run would replace\n"
            f"  them in place — and every downstream tag that reads them would then describe an\n"
            f"  extraction that no longer exists.\n"
            f"    manifest : {p}\n"
            f"  Pick ONE:\n"
            f"    * canonicalization repeat  ->  run 6d/6e only (REPEAT=<n> ./run_e2e.sh 6d 6e);\n"
            f"                                   they tag their own outputs and reuse this extraction\n"
            f"    * a NEW extraction sample  ->  give it its own variant in the config\n"
            f'                                   ("variant": "{variant}2"), which is what variant is for\n'
            f"    * really re-extract        ->  {OVERWRITE_ENV}=1")


def _alive(pid: int) -> bool:
    """Is that process still running on THIS machine? signal 0 checks existence without touching it."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:          # exists, owned by someone else
        return True
    except (OSError, TypeError):
        return False


def _guard_concurrent(rec: dict, section: str, rid: str, path: Path, overwrite: bool) -> None:
    """Refuse a stage another LIVE process already has open on the same run id.

    The finished-stage guard cannot see this case: neither run has finished, so nothing on disk says
    the files are spoken for. Two jobs on two GPUs writing one track and tag would interleave two
    vocabularies into the same files and the damage would be invisible — both manifests would look
    reasonable.

    A crashed run must NOT lock the tag forever, and this pipeline crashes often enough to matter, so
    the lock is only honoured while the owning process is actually alive. Same pid means the config
    cell was simply re-run in one kernel, which is normal and allowed.
    """
    st = rec.get("stages", {}).get(section) or {}
    owner = st.get("owner")
    if not owner or st.get("finished") or overwrite:
        return
    same_host = owner.get("host") == socket.gethostname()
    if same_host and owner.get("pid") == os.getpid():
        return                                    # this very process, re-running its config cell
    if same_host and not _alive(owner.get("pid")):
        print(f"  note  stage {section!r} was left open by pid {owner.get('pid')} "
              f"(started {owner.get('at')}), which is no longer running — taking it over.")
        return
    where = "this machine" if same_host else f"host {owner.get('host')!r}"
    raise RuntimeError(
        f"\n  stage {section!r} of run id {rid!r} is ALREADY OPEN in another process:\n"
        f"    pid {owner.get('pid')} on {where}, since {owner.get('at')}\n"
        f"  Both would write the same *_{rec.get('variant')}_{rec.get('run_tag')}* files and\n"
        f"  interleave two vocabularies into them — with nothing on disk to show it afterwards.\n"
        f"    manifest : {path}\n"
        f"  To run two jobs at once, give this one its own identity — any ONE of:\n"
        f"    * a different model tree   MODEL=<other>\n"
        f"    * a different split        SPLIT=test\n"
        f"    * a different repeat       REPEAT=2\n"
        f"    * a different experiment   EXPERIMENT=<other>\n"
        f"  (or {OVERWRITE_ENV}=1 if the other process is dead and you know it)")


def finish(metrics: dict | None = None, section: str | None = None, results_dir=None,
           articles=None, verbose: bool = True, osfx: str | None = None,
           track: str | None = None) -> Path | None:
    """Close the open stage: record what it wrote, what it spent, and its metrics.

    Called for you by `run_metrics.save`, so the metrics cell every notebook already ends with is
    what closes the record. Safe to call twice (it re-scans and rewrites).
    """
    act = _ACTIVE
    if act is None:
        if os.environ.get(DISABLE_ENV) == "1":
            return None
        act = _recover(results_dir, section, metrics)
        if act is None:
            if verbose:
                print("  (no open run manifest — provenance for this stage was not recorded; "
                      "re-run the config cell at the top to open one)")
            return None

    rec = load(act["results_dir"], act["run_id"])
    if rec is None:
        return None
    st = rec["stages"].setdefault(act["section"], {"started": None})
    # A file another stage FINISHED with before this one started is not this stage's output. Without
    # this, stages that run back to back — the repeats of an arm, or a whole chain inside one clock
    # second, as the tests do — credit each later stage with everything written before it.
    # Checked across every manifest in the tree, not just this one, because consecutive repeats are
    # separate manifests; and only against stages that finished BEFORE this one began, so a file
    # this run legitimately overwrites still counts as its own.
    claimed = _claimed_before(act["results_dir"], act.get("started_exact") or act["started"],
                              rec["run_id"], act["section"])
    outs = [o for o in _outputs_since(act["results_dir"], act["started"], osfx=osfx, track=track)
            if o["path"] not in claimed]
    st["finished"] = _now()
    st.pop("owner", None)                  # released: the finished-stage guard takes over from here
    st["outputs"] = outs
    st["token_usage"] = _token_usage(act["results_dir"], outs)
    if articles is not None:
        st["articles"] = list(articles)
    elif metrics and isinstance(metrics.get("summary"), dict):
        st.setdefault("articles", None)
    if metrics is not None:
        st["metrics"] = metrics
    rec["updated"] = _now()
    _write(act["path"], rec)
    if verbose:
        tu = st["token_usage"] or {}
        tot = tu.get("total_tokens")
        print(f'  manifest {act["path"].name}: stage {act["section"]} closed — '
              f'{len(outs)} output file(s)' + (f', {tot:,} tokens' if tot else ''))
    return act["path"]


def note(**kv) -> None:
    """Attach anything else worth keeping to the open stage (a prompt name, a manual observation)."""
    if _ACTIVE is None:
        return
    rec = load(_ACTIVE["results_dir"], _ACTIVE["run_id"])
    if rec is None:
        return
    rec["stages"].setdefault(_ACTIVE["section"], {}).setdefault("notes", {}).update(kv)
    rec["updated"] = _now()
    _write(_ACTIVE["path"], rec)


# --------------------------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------------------------
def _write(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str))


def _root_of(results_dir: Path) -> Path:
    """`<Phase2_LLM_Extraction>` — results_dir is `<root>/results/<model>`."""
    return Path(results_dir).resolve().parent.parent


def _recover(results_dir, section, metrics) -> dict | None:
    """Re-attach to a manifest when the kernel was restarted between build and metrics.

    Picks the most recently updated manifest of this track that has `section` OPEN. Deliberately
    conservative: no open stage, no guess — a wrong attribution is worse than a missing one.
    """
    if results_dir is None or section is None:
        return None
    results_dir = Path(results_dir)
    best = None
    for p in sorted(runs_dir(results_dir).glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        st = rec.get("stages", {}).get(section)
        if not st or st.get("finished"):
            continue
        if metrics and metrics.get("experiment") and rec.get("experiment") \
                and metrics["experiment"] != rec["experiment"]:
            continue
        if best is None or rec.get("updated", "") > best[0].get("updated", ""):
            best = (rec, p)
    if best is None:
        return None
    rec, p = best
    started = st_started = rec["stages"][section].get("started")
    try:
        started = datetime.fromisoformat(st_started).timestamp()
    except (TypeError, ValueError):
        started = 0.0
    return {"path": p, "run_id": rec["run_id"], "section": section,
            "results_dir": results_dir, "started": started, "started_exact": started}


def _claimed_before(results_dir: Path, started: float, run_id: str, section: str) -> set:
    """Paths already claimed by a stage that finished before `started` — in ANY manifest here."""
    claimed = set()
    for p in sorted(runs_dir(results_dir).glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for s, v in rec.get("stages", {}).items():
            if rec.get("run_id") == run_id and s == section:
                continue                       # this very stage
            fin = v.get("finished")
            if not fin:
                continue
            try:
                if datetime.fromisoformat(fin).timestamp() > started:
                    continue                   # finished after we began: could be our own overwrite
            except (TypeError, ValueError):
                pass
            claimed.update(o["path"] for o in (v.get("outputs") or []))
    return claimed


# Which directories each track writes into. `two_staged` and `binary` share `_global`, which is the
# PARENT of e2e's `_global/e2e` — so membership has to exclude the nested one explicitly, or a staged
# run would claim every e2e file as its own.
_TRACK_DIRS = {"e2e": ("e2e",), "two_staged": ("two_staged",), "binary": ("two_staged",)}


def _in_track(rel: Path, track: str) -> bool:
    parts = rel.parts
    if track == "e2e":
        return "e2e" in parts
    return "two_staged" in parts or (parts[0] == "_global" and "e2e" not in parts)


def _outputs_since(results_dir: Path, since: float, osfx: str | None = None,
                   track: str | None = None) -> list[dict]:
    """Every file under this results tree modified since the stage started, and belonging to it.

    Mtime alone is not enough once two runs share a model tree — a staged job on one GPU and an e2e
    job on another would each claim the other's files, and the token totals would silently be the
    sum of both. So a file must ALSO sit in this track's directories and carry this run's suffix.

    Both filters are needed. The suffix alone would let a staged run claim `_global/e2e/...` files
    of the same tag; the track alone would let two runs of different tags in one track collide.
    When `osfx`/`track` are not given the filters are skipped and the old behaviour applies, which
    is what a direct call (outside run_metrics.save) still gets.
    """
    out = []
    for p in sorted(Path(results_dir).rglob("*")):
        if not p.is_file() or any(part in _IGNORE_PARTS for part in p.parts) \
                or p.name in _IGNORE_PARTS:
            continue
        rel_p = p.relative_to(results_dir)
        if track and not _in_track(rel_p, track):
            continue
        if osfx and osfx not in p.name:
            continue
        try:
            stat = p.stat()
        except OSError:
            continue
        if stat.st_mtime >= since:
            out.append({"path": str(p.relative_to(results_dir)),
                        "bytes": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(
                            timespec="seconds")})
    return out


def _token_usage(results_dir: Path, outputs: list[dict]) -> dict | None:
    """Summed totals of every `token_usage*.json` this stage wrote — the run's cost, in one number."""
    tot: dict = {}
    files = []
    complete = True
    for o in outputs:
        name = Path(o["path"]).name
        if not (name.startswith("token_usage") and name.endswith(".json")):
            continue
        try:
            d = json.loads((Path(results_dir) / o["path"]).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        t = d.get("totals") if isinstance(d, dict) else None
        if not isinstance(t, dict):
            continue
        files.append(o["path"])
        # A ledger that says it covers only some of the articles makes this total a floor. Record
        # that here so nothing downstream has to re-open the ledger to find out.
        if (d.get("coverage") or {}).get("complete") is False:
            complete = False
        for k, v in t.items():
            if isinstance(v, (int, float)):
                tot[k] = round(tot.get(k, 0) + v, 2)
    if not files:
        return None
    tot["files"] = files
    if not complete:
        tot["complete"] = False
    return tot


# --------------------------------------------------------------------------------------------
# reading runs back
# --------------------------------------------------------------------------------------------
def _results_root(start: Path | None = None) -> Path:
    start = (start or Path(__file__).resolve()).resolve()
    for p in (start, *start.parents):
        if (p / "results").is_dir() and (p / "dataset").is_dir():
            return p / "results"
    return Path("results")


def all_runs(model: str | None = None, results_root: Path | None = None) -> list[dict]:
    """Every manifest on disk, newest first, each with `_model` and `_path` attached."""
    root = Path(results_root) if results_root else _results_root()
    runs = []
    for mdir in sorted(root.iterdir()) if root.is_dir() else []:
        if not mdir.is_dir() or (model and mdir.name != model):
            continue
        for p in sorted((mdir / "_runs").glob("*.json")) if (mdir / "_runs").is_dir() else []:
            try:
                rec = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            rec["_model"], rec["_path"] = mdir.name, p
            runs.append(rec)
    return sorted(runs, key=lambda r: r.get("updated") or r.get("created") or "", reverse=True)


def _summary_row(rec: dict) -> dict:
    """The few numbers worth seeing without opening the file."""
    stages = rec.get("stages", {})
    done = [s for s, v in stages.items() if v.get("finished")]
    open_ = [s for s, v in stages.items() if not v.get("finished")]
    toks = sum((v.get("token_usage") or {}).get("total_tokens", 0) or 0 for v in stages.values())
    # A ledger that a stage only partly recorded must not be summed into a run total silently: the
    # periodic-consolidation cells used to write their cost file after the first chunk, so some
    # validation runs hold five articles' spend for a twenty-article stage. Carry the fact out.
    partial = sorted(s for s, v in stages.items()
                     if (v.get("token_usage") or {}).get("complete") is False)
    git = next((v.get("git", {}) for v in stages.values() if v.get("git")), {})
    arts = next((v.get("articles") for v in reversed(list(stages.values())) if v.get("articles")),
                None)
    rel = (stages.get("6e") or stages.get("7b") or {}).get("metrics") or {}
    typ = (stages.get("6d") or stages.get("4") or {}).get("metrics") or {}
    s = rel.get("summary", {}) if isinstance(rel, dict) else {}
    t = typ.get("summary", {}) if isinstance(typ, dict) else {}
    return {"run_id": rec.get("run_id"), "model": rec.get("_model"),
            "experiment": rec.get("experiment"), "updated": (rec.get("updated") or "")[:16],
            "stages": ",".join(sorted(done)) or "-",
            "open": ",".join(sorted(open_)),
            "articles": len(arts) if arts else s.get("articles") or t.get("articles"),
            "tokens": toks or None, "tokens_partial": partial,
            "commit": git.get("short"), "dirty": git.get("dirty"),
            "types": t.get("vocab_consolidated"), "props": s.get("vocab_consolidated"),
            "schema_pct": s.get("triple_schema_org_pct")}


def index(results_root: Path | None = None, write: bool = True) -> str:
    """The markdown table of every run in the tree; written to `results/RUNS.md`."""
    root = Path(results_root) if results_root else _results_root()
    rows = [_summary_row(r) for r in all_runs(results_root=root)]
    cols = [("run_id", "run id"), ("model", "model"), ("experiment", "experiment"),
            ("stages", "stages done"), ("open", "open"), ("articles", "arts"),
            ("types", "types"), ("props", "props"), ("schema_pct", "schema %"),
            ("tokens", "tokens"), ("commit", "commit"), ("updated", "updated")]
    out = ["# Runs in this tree",
           "",
           "Generated by `python -m run_manifest index` from the manifests in "
           "`results/<model>/_runs/`. One row per run; `types` / `props` are the "
           "POST-consolidation vocabulary sizes, `schema %` the share of triples on a schema.org "
           "property. A row with something in `open` did not finish that stage. A `*` on `tokens` "
           "means at least one stage recorded only part of its spend — the stage's ledger in "
           "`results/<model>/_global/` says which articles it covers, and the total is a floor, "
           "not the run's cost.",
           ""]
    if not rows:
        out += ["_No runs yet._", ""]
    else:
        out.append("| " + " | ".join(h for _, h in cols) + " |")
        out.append("|" + "|".join("---" for _ in cols) + "|")
        for r in rows:
            cells = []
            for k, _ in cols:
                v = r.get(k)
                if k == "commit" and v:
                    v = f'`{v}`' + ("+dirty" if r.get("dirty") else "")
                if k == "tokens" and v:
                    v = f"{int(v):,}" + ("*" if r.get("tokens_partial") else "")
                cells.append("" if v in (None, "") else str(v))
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    md = "\n".join(out)
    if write:
        (root / "RUNS.md").write_text(md)
    return md


def _main(argv):
    cmd = argv[1] if len(argv) > 1 else "list"
    arg = argv[2] if len(argv) > 2 else None
    if cmd == "list":
        rows = [_summary_row(r) for r in all_runs(model=arg)]
        if not rows:
            print("  no runs recorded yet.")
            return 0
        print(f'  {"run id":<34} {"model":<18} {"stages":<10} {"arts":>4} {"types":>5} '
              f'{"props":>5} {"tokens":>10}  {"commit":<10} updated')
        for r in rows:
            toks = (format(int(r["tokens"]), ",") + ("*" if r.get("tokens_partial") else "")
                    if r["tokens"] else "")
            commit = (r["commit"] or "") + ("+d" if r["dirty"] else "")
            print(f'  {str(r["run_id"]):<34} {str(r["model"]):<18} {r["stages"]:<10} '
                  f'{r["articles"] or "":>4} {r["types"] or "":>5} {r["props"] or "":>5} '
                  f'{toks:>10}  {commit:<10} {r["updated"]}'
                  + (f'   OPEN: {r["open"]}' if r["open"] else ""))
        return 0
    if cmd == "show":
        for r in all_runs():
            if r.get("run_id") == arg or Path(str(r.get("_path"))).stem == arg:
                r.pop("_path", None)
                print(json.dumps(r, indent=2, ensure_ascii=False, default=str))
                return 0
        print(f"  no run {arg!r}. Try: python -m run_manifest list")
        return 1
    if cmd == "diff":
        b = argv[3] if len(argv) > 3 else None
        runs = {r["run_id"]: r for r in all_runs()}

        def pick(k):                       # accept a full run id or any unique fragment of one
            return next((v for i, v in runs.items() if k and (i == k or k in i)), None)

        ra, rb = pick(arg), pick(b)
        if not ra or not rb:
            print("  usage: python -m run_manifest diff <run_id> <run_id>")
            return 1
        def cfgs(r):
            return next((s.get("config") or {} for s in r["stages"].values() if s.get("config")), {})
        ca, cb = cfgs(ra), cfgs(rb)
        def flat(d, pre=""):
            out = {}
            for k, v in d.items():
                out.update(flat(v, f"{pre}{k}.")) if isinstance(v, dict) else out.update(
                    {f"{pre}{k}": v})
            return out
        fa, fb = flat(ca), flat(cb)
        keys = sorted(set(fa) | set(fb))
        print(f'  {"key":<34} {ra["run_id"]:<28} {rb["run_id"]}')
        n = 0
        for k in keys:
            if fa.get(k) != fb.get(k):
                n += 1
                print(f"  {k:<34} {str(fa.get(k)):<28} {fb.get(k)}")
        print(f"  {n} differing key(s)." if n else "  configs are identical.")
        return 0
    if cmd == "index":
        print(index())
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
