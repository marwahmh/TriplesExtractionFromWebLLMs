"""One config file per experiment — the notebooks pick an experiment BY NAME.

Every knob that decides what a run produces (which model tree, which articles, retrieval /
rerank / consolidation settings, the output tag) lives in `configs/<experiment>.json`, not in
a notebook cell. A notebook's config cell shrinks to:

    EXPERIMENT = "main"                       # <- the only line you edit
    import experiment_config as xcfg
    CFG = xcfg.load(EXPERIMENT, globals(), "6d")

`load` writes the section's keys into the notebook's namespace as UPPER_CASE globals
(`inline_vocab_candidates` -> `INLINE_VOCAB_CANDIDATES`), plus the shared ones every notebook
needs (`RESULTS_MODEL`, `BACKEND`, `VARIANT`, `RUN_TAG`, `INPUT_TAG`, `SELECTED_ARTICLES`, ...),
and prints a one-block banner of what it set. Anything the notebook assigns AFTER that call
still wins, so a one-off tweak is still a one-line edit — it is just no longer the way a
reproducible run is defined.

Why this exists: a run is defined by ~25 knobs spread over three notebooks that must agree with
each other (6d's RUN_TAG is 6e's INPUT_TAG; all three need the same VARIANT and results tree).
Reconstructing a finished run from the notebooks was already impossible — they had moved on —
which is what `notebooks/e2e/CURRENT_CONFIG.txt` was hand-written to work around. A config file
is that snapshot, kept honest by being the thing the run actually reads.

Files:
    configs/_base.json          every default, one place
    configs/<name>.json         one experiment: "extends": "_base" + only what it changes
Selection order: the `name` argument -> $PHASE2_EXPERIMENT -> "main". (There is deliberately no
`default` alias any more: it resolved to the same settings as `main` under a DIFFERENT tag, so a run
started without naming an experiment silently produced a second, differently-named copy of the
reported configuration.)

The output tag names itself: `run_tag` defaults to the EXPERIMENT NAME, so an experiment cannot
inherit another one's tag by forgetting to set it, and a repeat of the same arm is `REPEAT=2` in the
environment (-> `main_r2`) rather than a hand-invented name. `load` also opens this run's manifest
(see run_manifest.py), which is what makes a finished run traceable to a config and a commit — and
what refuses to start a stage that already finished under the same tag.

Also used outside notebooks:
    python -m experiment_config list                 # every experiment, one line each
    python -m experiment_config show main            # the fully-resolved config
    python -m experiment_config check main           # cross-notebook consistency + tag collisions
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = ["load", "resolve", "config_dir", "available", "check", "describe", "article_key",
           "section_keys", "results_dir", "slug", "SECTIONS", "REPEAT_ENV"]

# A repeat of an ARM, not a new configuration: same config file, own outputs, own manifest.
# Sampling is not deterministic at temperature 0 (see notebooks/e2e/CURRENT_CONFIG.txt §7 — 69 vs 91
# raw mints under identical settings), so a reported arm is run three times and given as mean+range.
REPEAT_ENV = "PHASE2_REPEAT"

# Which notebook consumes which section. A section name is the notebook's number/letter, so the
# call in the notebook reads `xcfg.load(EXPERIMENT, globals(), "6d")`. Both tracks are listed
# because the staged notebooks take the same knobs as their e2e twins — that is exactly what
# makes the two tracks comparable under one experiment name.
SECTIONS = {
    "2":  "two_staged · entity extraction",
    "4":  "two_staged · type canonicalization",
    "6":  "two_staged · relation extraction",
    "6b": "two_staged · event reification",
    "7":  "two_staged · binary property canonicalization",
    "7b": "two_staged · n-ary property canonicalization",
    "6c": "e2e · single-call extraction",
    "6d": "e2e · type canonicalization",
    "6e": "e2e · relation canonicalization",
}

# Shared keys go into every notebook's namespace, whatever section it asked for.
SHARED_KEYS = ("results_model", "backend", "variant", "run_tag", "input_tag", "repeat", "split",
               "selected_articles", "match_model", "gemini_model")

# `"split": "validation" | "test"` resolves `selected_articles` from dataset/splits.json and tags the
# run with it. The split has to be part of the run's IDENTITY, not just its inputs: a validation
# number and a test number of the same experiment are different results, and one silently overwriting
# the other is the mistake this whole tagging scheme exists to prevent.
SPLITS = {"validation": "val", "test": "test"}
SPLIT_ENV = "PHASE2_SPLIT"


def article_key(aid: str):
    """Sort key that orders article ids NUMERICALLY: art2 before art10, not after it.

    Every canonicalization stage is incremental — what article k mints depends on what k-1 added —
    so the order the corpus is walked in is part of the run. Plain `sorted()` gives the
    lexicographic order (art1, art10, art100, art11, art2, ...), which is neither the corpus order
    nor stable as the corpus grows past a power of ten. `art<N>` is the row's position in
    dataset/articles.csv, so numeric order IS corpus order.
    """
    import re as _re
    m = _re.match(r"^([^\d]*)(\d+)$", str(aid))
    return (m.group(1), int(m.group(2))) if m else (str(aid), 0)


def _split_articles(split: str) -> list[str]:
    """Article ids of one split, re-derived from dataset/splits.json by URL.

    Loaded from the dataset script rather than copied here, because `art<N>` is only the row position
    in articles.csv: editing the CSV renumbers the files, and a hard-coded list would then quietly
    name different articles than the split was defined over.
    """
    import importlib.util
    p = project_root() / "dataset" / "select_splits.py"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} is missing — a config asked for split {split!r}, but the corpus has never been "
            f"split. Run:  python dataset/select_splits.py")
    spec = importlib.util.spec_from_file_location("select_splits", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not mod.SPLITS.exists():
        raise FileNotFoundError(
            f"{mod.SPLITS} is missing — run:  python dataset/select_splits.py")
    return mod.resolve(split)


def slug(name: str) -> str:
    """`no-rerank` -> `no_rerank`: an experiment name as a filename-safe output tag."""
    import re as _re
    return _re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")


def project_root() -> Path:
    """`<Phase2_LLM_Extraction>` — the directory holding configs/, results/ and dataset/."""
    return config_dir().parent


def results_dir(cfg: dict) -> Path:
    """The tree this config reads and writes: `<root>/results/<results_model>`."""
    return project_root() / "results" / str(cfg.get("results_model") or "")


def config_dir() -> Path:
    """`<Phase2_LLM_Extraction>/configs`, found by walking up from this file."""
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "configs").is_dir():
            return p / "configs"
    return here.parent.parent / "configs"


def available() -> list[str]:
    """Experiment names, excluding the `_base` defaults file."""
    return sorted(p.stem for p in config_dir().glob("*.json") if not p.stem.startswith("_"))


def _read(name: str) -> dict:
    p = config_dir() / f"{name}.json"
    if not p.exists():
        raise FileNotFoundError(
            f"no config {name!r} in {config_dir()}. Available: {', '.join(available()) or '(none)'}")
    return json.loads(p.read_text())


def _merge(base: dict, over: dict) -> dict:
    """Recursive dict merge — a section in the experiment file overrides only the keys it names."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve(name: str | None = None) -> dict:
    """The fully-merged config: `_base` (and any `extends` chain) under the named experiment.

    Follows `extends` up to a root, so an experiment can be defined as a delta on another
    experiment rather than on the defaults."""
    name = name or os.environ.get("PHASE2_EXPERIMENT") or "main"
    chain, seen = [], set()
    while name and name not in seen:
        seen.add(name)
        cfg = _read(name)
        chain.append(cfg)
        name = cfg.get("extends")
    merged: dict = {}
    for cfg in reversed(chain):            # root first, so the named experiment wins
        merged = _merge(merged, cfg)
    merged.pop("extends", None)

    # $MODEL runs an experiment against ANOTHER model without a config file per (experiment, model).
    # An ablation is a claim about a mechanism, and a mechanism claim is worth more when it is shown
    # on three models than on one — but 7 ablations x 3 models is 21 files that differ in one key.
    # Applied before the backend default, so the backend follows unless a config named one.
    model = os.environ.get("PHASE2_MODEL") or os.environ.get("MODEL")
    if model:
        merged["results_model"] = model

    # Defaults that are expressed as "same as another key" rather than as a literal, so an
    # experiment only has to state them when they differ.
    merged.setdefault("backend", None)
    if merged["backend"] is None:
        merged["backend"] = merged["results_model"]

    # The output tag NAMES ITSELF. Leaving `run_tag` unset used to inherit `_base`'s literal — every
    # experiment that forgot it wrote on top of the same files. Unset now means "call it after the
    # experiment", which is unique by construction because config filenames are.
    if not merged.get("run_tag") and merged.get("run_tag") != "":
        merged["run_tag"] = slug(merged.get("experiment") or "")

    # The split decides WHICH articles and is part of the tag: main -> main_val / main_test.
    split = merged.get("split") or os.environ.get(SPLIT_ENV) or os.environ.get("SPLIT")
    if split:
        split = {"val": "validation", "validation": "validation",
                 "test": "test"}.get(str(split).lower(), str(split))
        if split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}, not {split!r}")
        if merged.get("selected_articles"):
            # An explicit list is a deliberate override (a re-run of three failed articles, say);
            # it wins, but the tag still says which split this run belongs to.
            pass
        else:
            merged["selected_articles"] = _split_articles(split)
        if merged.get("run_tag"):
            merged["run_tag"] = f'{merged["run_tag"]}_{SPLITS[split]}'
    merged["split"] = split

    # $PHASE2_ARTICLES repairs ONE article inside an existing run: it replaces the article list but
    # leaves the TAG alone, so the stage writes the same *_<variant>_<run_tag>* filenames the rest of
    # the run already uses. Without it, redoing a single failed article means either a new tag (wrong
    # filenames) or reusing another experiment's tag (a collision the check rejects) — both of which
    # came up when art52 died in gemini's 6c and art12 in qwen's nb6b.
    only = os.environ.get("PHASE2_ARTICLES")
    if only:
        merged["selected_articles"] = [a.strip() for a in only.split(",") if a.strip()]
        merged["articles_overridden"] = True

    # A repeat is the same arm run again, so it takes the same config and its own tag: main -> main_r2.
    rep = merged.get("repeat")
    if rep in (None, ""):
        rep = os.environ.get(REPEAT_ENV) or os.environ.get("REPEAT")
    try:
        rep = int(rep) if rep not in (None, "") else None
    except (TypeError, ValueError):
        rep = None
    merged["repeat"] = rep
    if rep and merged.get("run_tag"):
        merged["run_tag"] = f'{merged["run_tag"]}_r{rep}'

    # input_tag follows run_tag (repeat included) unless the config names a DIFFERENT upstream run
    # on purpose — which is how a relation-stage-only variation is run on an existing type stage.
    merged.setdefault("input_tag", None)
    if merged["input_tag"] is None:
        merged["input_tag"] = merged.get("run_tag")
    return merged


def section_keys(cfg: dict, section: str) -> dict:
    """The shared keys plus `section`'s own, as one flat dict (section wins on a clash)."""
    flat = {k: cfg[k] for k in SHARED_KEYS if k in cfg}
    if section is not None:
        if section not in cfg:
            raise KeyError(f"config has no section {section!r}; it has: "
                           f"{', '.join(k for k in cfg if isinstance(cfg[k], dict))}")
        flat.update(cfg[section])
    return flat


def load(name: str | None = None, ns: dict | None = None, section: str | None = None,
         verbose: bool = True, manifest: bool = True) -> dict:
    """Resolve `name`, write `section`'s settings into `ns` as UPPER_CASE globals, OPEN the manifest.

    Returns the flat dict that was applied, so a notebook can also read a value without going
    through its global (`CFG["top_k_match"]`). `ns` is the notebook's `globals()`; pass None to
    resolve without touching any namespace — which also means no manifest, so `show` / `check` and
    any offline inspection leave no trace of a run that did not happen.

    With `ns`, this also opens `results/<model>/_runs/<run_id>.json` and RAISES if this stage already
    finished under the same run id. That refusal is the point: it is the difference between "the
    numbers in the table came from this commit and this config" and "the numbers came from whatever
    ran last". Get past it with REPEAT=2 (another repeat of the arm), a new experiment file, or
    PHASE2_OVERWRITE=1 when redoing a run in place is what you actually mean.
    """
    cfg = resolve(name)
    flat = section_keys(cfg, section)
    if ns is not None:
        for k, v in flat.items():
            ns[k.upper()] = v
        ns["EXPERIMENT"] = cfg.get("experiment", name)
        ns["EXPERIMENT_CONFIG"] = cfg
    if verbose:
        print(_banner(cfg, section, flat))

    if ns is not None and manifest and section is not None:
        import sys as _sys                        # imported late: the CLI paths never need it
        if str(Path(__file__).resolve().parent) not in _sys.path:
            _sys.path.insert(0, str(Path(__file__).resolve().parent))
        import run_manifest
        rec = run_manifest.begin(cfg, section, results_dir(cfg), root=project_root(),
                                 verbose=verbose)
        if rec is not None:
            ns["RUN_ID"] = rec["run_id"]
            ns["RUN_MANIFEST"] = run_manifest.path_for(results_dir(cfg), rec["run_id"])
    return flat


def _banner(cfg: dict, section: str | None, flat: dict) -> str:
    title = f'experiment {cfg.get("experiment", "?")!r}'
    if section:
        title += f'  ·  section {section} ({SECTIONS.get(section, "?")})'
    lines = [title, "-" * max(len(title), 60)]
    if cfg.get("description"):
        lines.append(f'  {cfg["description"]}')
        lines.append("-" * max(len(title), 60))
    shared = [k for k in SHARED_KEYS if k in flat]
    own = [k for k in flat if k not in shared]

    def _rows(keys):
        for k in keys:
            v = flat[k]
            if isinstance(v, list) and len(v) > 6:          # keep a 100-article list to one line
                v = f"[{len(v)} articles: {v[0]} … {v[-1]}]"
            lines.append(f"  {k.upper():<28} {v!r}")

    _rows(shared)
    if shared and own:
        lines.append("  " + "·" * 28)                        # shared keys above, this section's below
    _rows(own)
    return "\n".join(lines)


def describe(name: str | None = None) -> str:
    """The whole config as pretty JSON — what to paste into a thesis appendix."""
    return json.dumps(resolve(name), indent=2)


def check(name: str | None = None) -> tuple[list[str], list[str]]:
    """-> (errors, notes). An error would silently waste the run; a note is a legitimate but
    unusual choice worth seeing in the banner before the GPU time is spent.

    These are the mistakes that have actually cost runs here: the notebooks in one chain pointing
    at different results trees or VARIANTs, and 6e reading a 6d run the chain never produces.
    Since all three now read ONE config file, the first two are impossible by construction and
    what remains is checking that the settings are mutually coherent.
    """
    cfg = resolve(name)
    errs, notes = [], []
    if not cfg.get("results_model"):
        errs.append("results_model is unset — nothing knows which results/ tree to use.")

    # Two experiments writing one tag in one tree is the failure that cannot be seen on disk
    # afterwards: the files carry the tag, and the tag no longer identifies a configuration.
    mine = (cfg.get("results_model"), cfg.get("variant"), cfg.get("run_tag"))
    if cfg.get("run_tag"):
        for other in available():
            if other == (cfg.get("experiment") or name):
                continue
            try:
                o = resolve(other)
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            if (o.get("results_model"), o.get("variant"), o.get("run_tag")) == mine:
                errs.append(f"run_tag {cfg['run_tag']!r} in tree {mine[0]}/ (variant "
                            f"{mine[1]!r}) is ALSO {other!r}'s tag — one of them must change, "
                            f"or their outputs become indistinguishable on disk.")

    # An already-finished run under this id is not an error (a repeat or a deliberate redo is
    # legitimate) but it is the last moment it is cheap to notice.
    try:
        import run_manifest
    except ImportError:
        run_manifest = None
    for track in ("e2e", "two_staged") if run_manifest else ():
        rec = run_manifest.load(results_dir(cfg), run_manifest.run_id_for(cfg, track=track))
        done = sorted(s for s, v in (rec or {}).get("stages", {}).items() if v.get("finished"))
        if done:
            notes.append(f"run id {rec['run_id']!r} already has finished stage(s) "
                         f"{', '.join(done)} — re-running one of those needs REPEAT=<n> for a "
                         f"repeat, or PHASE2_OVERWRITE=1 to redo it in place.")
    if cfg.get("backend") and cfg["backend"] != cfg.get("results_model"):
        notes.append(f'backend {cfg["backend"]!r} != results_model {cfg["results_model"]!r} — '
                     f'this run processes another model\'s outputs (judge/eval setup).')
    if cfg.get("input_tag") != cfg.get("run_tag"):
        notes.append(f'input_tag {cfg["input_tag"]!r} != run_tag {cfg["run_tag"]!r} — the relation '
                     f'stage reads an EXISTING type-stage run rather than this chain\'s own.')
    for s in ("4", "6d", "7", "7b", "6e"):
        sec = cfg.get(s, {})
        if sec.get("inline_vocab_candidates") is False and sec.get("consolidate") is False:
            notes.append(f"{s}: inline_vocab_candidates=false with consolidate=false — the inline "
                         f"stage cannot reuse a mint and nothing collapses the duplicates, so this "
                         f"run measures the UNCONSOLIDATED upper bound on vocabulary size.")
    for s in ("7", "7b", "6e"):
        sec = cfg.get(s, {})
        if sec.get("parallel_llm") and sec.get("inline_vocab_candidates"):
            errs.append(f"{s}: parallel_llm has no effect while inline_vocab_candidates=true "
                        f"(each article's candidates then depend on the previous articles).")
    return errs, notes


# --------------------------------------------------------------------------------------------
# CLI: `python -m experiment_config list | show <name> | check <name>`
# --------------------------------------------------------------------------------------------
def _main(argv):
    cmd = argv[1] if len(argv) > 1 else "list"
    name = argv[2] if len(argv) > 2 else None
    if cmd == "list":
        print(f'  {"experiment":<24} {"tree":<18} {"writes":<22} description')
        for n in available():
            c = resolve(n)
            sfx = (f"_{c.get('variant')}" if c.get("variant") else "") + \
                  (f"_{c['run_tag']}" if c.get("run_tag") else "")
            print(f'  {n:<24} {str(c.get("results_model")):<18} {"*" + sfx + "*":<22} '
                  f'{_read(n).get("description", "")}')
        return 0
    if cmd == "show":
        print(describe(name))
        return 0
    if cmd == "check":
        errs, notes = check(name)
        cfg = resolve(name)
        sfx = (f"_{cfg.get('variant')}" if cfg.get("variant") else "") + \
              (f"_{cfg['run_tag']}" if cfg.get("run_tag") else "")
        print(f"experiment {cfg.get('experiment')!r}: tree results/{cfg['results_model']}/ | "
              f"variant {cfg.get('variant')!r} | run_tag {cfg.get('run_tag')!r} | "
              f"input_tag {cfg.get('input_tag')!r}"
              + (f" | repeat {cfg['repeat']}" if cfg.get("repeat") else ""))
        arts = cfg.get("selected_articles")
        print(f"  writes  *{sfx}*  under results/{cfg['results_model']}/")
        if cfg.get("articles_overridden"):
            print("  ARTICLES OVERRIDDEN by $PHASE2_ARTICLES — repairing part of an existing run; the"
                  "\n  tag is unchanged, so this writes alongside the articles already done.")
        print(f"  articles {len(arts) if arts else 'ALL in the tree'}"
              + (f"   split {cfg['split']}" if cfg.get("split") else "   (no split — set SPLIT=val|test)")
              + (f"   {arts[0]} … {arts[-1]}" if arts else ""))
        for n in notes:
            print(f"  note  {n}")
        for e in errs:
            print(f"  FAIL  {e}")
        if not errs:
            print("  consistent.")
        return 1 if errs else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
