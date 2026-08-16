"""The manual canonicalization sample — is the chosen schema.org term the RIGHT one?

Every canonicalization number the pipeline reports is a count, not a judgement.
`schema_org_reuse_pct` says a schema.org property was chosen; it cannot say the property means what
the relation meant. `cons_merged_into_schema` says 18 minted terms were folded into schema.org; it
cannot say they were the same concept. A run that maps every relation onto `about` scores beautifully
on both. This module draws the sample that puts a number on that, the same way
`grounding_sample.py` does for grounding — and it is the number the reuse claims should be reported
*with*, not instead of.

Four populations, because "canonicalization" is four different decisions:

    type       6d / nb4 : an OIE type  -> a schema.org class or a minted one
    relation   6e / nb7b: a relation phrase -> a schema.org property or a minted one (+ direction)
    merge      consolidation: two vocabulary terms declared the same concept
    reparent   consolidation: one term declared a subclass/subproperty of another

**Stratified by decision source, on purpose.** Uniform sampling would be dominated by whichever way
the run happened to lean, and the two error directions are opposites that both matter:

    a WRONG REUSE   inflates schema_org_reuse_pct — the headline number
    an AVOIDABLE MINT inflates the vocabulary — the number the consolidation story rests on

Each stratum is drawn to a floor so both are measurable, which makes the sample deliberately
non-representative: `score()` reweights every stratum by its true size before reporting a rate.

**The model's own justification is held back** in the key file. The decision has to be shown — it is
what you are judging — but `chosen_type.justification` / the consolidation `reason` is the model
arguing for itself, and reading it first is how a human ends up agreeing with it.

Usage, after a run has finished (no LLM calls, reads only what is on disk):

    import canon_sample as cs
    cs.build(RESULTS_DIR, ARTICLE_ORDER, osfx=_osfx, track="e2e")   # writes sample + csv + key
    ...                                                             # label the CSV by hand
    cs.score(RESULTS_DIR, osfx=_osfx, track="e2e")

or from a shell:

    python -m canon_sample build gemini-2.5-flash _dedup_main_val_r1
    python -m canon_sample score gemini-2.5-flash _dedup_main_val_r1

Files land next to the run's metrics, tagged with the same suffix, so a sample always belongs to
exactly one run:

    _global[/e2e]/canon_manual_sample<osfx>.json    the canonical record (label here or in the CSV)
    _global[/e2e]/canon_manual_sample<osfx>.csv     the same rows, for labelling in a spreadsheet
    _global[/e2e]/canon_manual_key<osfx>.json       held-back justifications + stratum weights
    _global[/e2e]/canon_manual_strata<osfx>.json    stratum sizes + draw counts, for reweighting
"""

from __future__ import annotations

import csv as _csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

__all__ = ["build", "score", "pool", "draw", "grounding_articles", "VERDICTS", "SAMPLE_N", "SEED"]

# The manual grounding sample, so a canonicalization row can say whether the same term was also
# grounded by hand. Sharing the ARTICLES with that sample is the point — reading the article is what
# annotation costs, ticking the box is not — but NOT the draw: grounding judges assertion instances
# and this judges decisions, and there are ~2.6 instances per decision. An instance-weighted sample
# labels `about -> about` twenty-one times and misses half the minted terms, which is precisely
# where wrong canonicalization concentrates. Same articles, own draw, linked afterwards.
GROUNDING_SAMPLE = "grounding_manual_sample.json"

SAMPLE_N = 240              # ~60 per kind; enough for a ±6pp interval on the reuse-precision number
PER_STRATUM_FLOOR = 20      # so a rare decision kind is still measurable, not just represented
SEED = 20260815

SAMPLE_FILE = "canon_manual_sample{osfx}.json"
CSV_FILE = "canon_manual_sample{osfx}.csv"
KEY_FILE = "canon_manual_key{osfx}.json"
STRATA_FILE = "canon_manual_strata{osfx}.json"

# The closed label set. Keep it small — every extra option is one more judgement call per row, and
# the free-text `should_be` carries the detail that matters for the write-up anyway.
VERDICTS = {
    "ok":       "the chosen term means what the source meant",
    "wrong":    "wrong sense — the term means something else",
    "broad":    "defensible but too general to be informative (about / mentions / relatedTo)",
    "narrow":   "too specific — asserts more than the text supports",
    "avoidable": "MINT ONLY: a suitable schema.org term existed and was not used",
    "direction": "RELATION ONLY: right property, subject and object the wrong way round",
    "unsure":   "cannot tell from the evidence shown",
}
# Which verdicts count as a correct canonicalization when the rate is computed.
_GOOD = {"ok"}

_TRACK_DIRS = {"e2e": ("e2e", "_global/e2e"), "two_staged": ("two_staged/1_entities", "_global")}


# --------------------------------------------------------------------------------------------
# pool — one row per decision, carrying everything the judgement needs
# --------------------------------------------------------------------------------------------
def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None


def _cands(row, k=6):
    """The labels the retrieval stage actually showed the model, in the order it showed them —
    without this a rater cannot tell 'the model chose badly' from 'nothing better was offered'."""
    out = []
    for c in (row.get("candidates") or [])[:k]:
        lab = c.get("label")
        src = "s" if c.get("source") == "schema.org" else "v"
        out.append(f"{lab}[{src}]")
    return " · ".join(out)


def _src(chosen):
    return "mint" if chosen.get("is_new_to_vocab") else (chosen.get("source") or "?")


def pool(results_dir, articles, osfx="", track="e2e") -> list[dict]:
    """Every canonicalization decision of one run, flattened to one judgeable row each."""
    results_dir = Path(results_dir)
    sub, gdir = _TRACK_DIRS[track]
    G = results_dir / gdir
    rows = []

    for aid in articles:
        # --- type assignments -------------------------------------------------------------
        tv = _read(results_dir / aid / sub / f"entity_type_matching_review{osfx}.json") or {}
        for r in tv.get("types", []):
            ch = r.get("chosen_type") or {}
            rows.append({
                "kind": "type", "article": aid, "source": _src(ch),
                "item": r.get("oie_type"),
                "context": (r.get("embedding_query_text") or "")[:400],
                "examples": ", ".join((r.get("mentions") or [])[:8]),
                "chosen": ch.get("label"),
                "chosen_meaning": ch.get("definition") or ch.get("path") or "",
                "direction": "",
                "candidates": _cands(r),
                "_why": ch.get("justification") or "",
            })

        # --- relation assignments ---------------------------------------------------------
        rv = _read(results_dir / aid / sub.replace("1_entities", "3_nary_graph")
                   / f"nary_relation_matching_review{osfx}.json") \
            or _read(results_dir / aid / sub / f"nary_relation_matching_review{osfx}.json") or {}
        for r in rv.get("relations", []):
            sigs = "; ".join(r.get("observed_signatures") or [])[:200]
            for a in r.get("assignments", []):
                ch = a.get("chosen_property") or {}
                rows.append({
                    "kind": "relation", "article": aid, "source": _src(ch),
                    "item": r.get("relation_phrase"),
                    "context": (r.get("definition") or r.get("embedding_query_text") or "")[:400],
                    "examples": sigs,
                    "chosen": ch.get("label"),
                    "chosen_meaning": ch.get("definition") or ch.get("signature") or "",
                    "direction": ch.get("direction") or "as_is",
                    "candidates": _cands(r),
                    "_why": ch.get("justification") or "",
                })

    # --- consolidation: merges and re-parents ---------------------------------------------
    # Judged one CHILD at a time rather than one group at a time: a group of five merged terms is
    # five separate claims, and lumping them hides the one that was wrong.
    for stem, kind_of in (("nary_relation_consolidation_report", "relation"),
                          ("type_consolidation_report", "type"),
                          ("relation_consolidation_report", "relation")):
        rep = _read(G / f"{stem}{osfx}.json")
        if not rep:
            continue
        for m in rep.get("merges", []):
            for child in m.get("merged", []):
                rows.append({
                    "kind": "merge", "article": "_global",
                    "source": m.get("survivor_source") or "?",
                    "item": child.get("label"),
                    "context": f'merged into "{m.get("survivor_label")}" '
                               f'({m.get("survivor_source")}) · kind {m.get("kind")}',
                    "examples": ", ".join(child.get("articles") or []),
                    "chosen": m.get("survivor_label"),
                    "chosen_meaning": "",
                    "direction": "",
                    "candidates": f'in_shortlist={child.get("in_shortlist")}',
                    "_why": m.get("reason") or "",
                })
        for rp in rep.get("reparents", []):
            for child in rp.get("children", []):
                rows.append({
                    "kind": "reparent", "article": "_global",
                    "source": rp.get("parent_source") or "?",
                    "item": child.get("label"),
                    "context": f'made a {rp.get("kind")} of "{rp.get("parent_label")}" '
                               f'({rp.get("parent_source")}) · was under '
                               f'{child.get("old_parent") or "nothing"}',
                    "examples": "",
                    "chosen": rp.get("parent_label"),
                    "chosen_meaning": "",
                    "direction": "",
                    "candidates": f'in_shortlist={child.get("in_shortlist")}',
                    "_why": rp.get("reason") or "",
                })
    return rows


def grounding_articles(results_dir, track="e2e") -> list[str]:
    """The articles the manual grounding sample was drawn over, so the canonicalization sample can
    be restricted to them. Empty when that sample does not exist yet."""
    g = _read(Path(results_dir) / _TRACK_DIRS[track][1] / GROUNDING_SAMPLE) or []
    return sorted({r.get("article_id") for r in g if r.get("article_id")},
                  key=lambda a: int("".join(c for c in str(a) if c.isdigit()) or 0))


def _grounded_terms(results_dir, track="e2e") -> set:
    """(article, term) pairs that appear in the manual grounding sample.

    Matched at TERM level, not instance level: the grounding row's id is
    `<subject>-<property>-<object>`, so this says "you also hand-checked an assertion that uses this
    property", which is what makes the joint 'grounded AND correctly canonicalized' rate computable
    on the overlap. It is not a claim that the same instance was seen."""
    out = set()
    for r in _read(Path(results_dir) / _TRACK_DIRS[track][1] / GROUNDING_SAMPLE) or []:
        parts = str(r.get("id") or "").split("-")
        if len(parts) >= 3:
            out.add((r.get("article_id"), "-".join(parts[1:-1])))
        out.add((r.get("article_id"), r.get("id")))
    return out


def _stratum(r):
    return (r["kind"], r["source"])


# --------------------------------------------------------------------------------------------
# draw
# --------------------------------------------------------------------------------------------
def draw(rows, n=SAMPLE_N, floor=PER_STRATUM_FLOOR, seed=SEED):
    """-> (sample, strata). Every stratum gets `floor` rows (or all it has), then what is left is
    shared out in proportion to true size, so the common cases are not crowded out either."""
    rng = random.Random(seed)
    by_st = defaultdict(list)
    for r in rows:
        by_st[_stratum(r)].append(r)
    for v in by_st.values():
        rng.shuffle(v)

    alloc = {k: min(floor, len(v)) for k, v in by_st.items()}
    left = n - sum(alloc.values())
    if left > 0:
        head = sum(len(v) - alloc[k] for k, v in by_st.items())
        for k, v in sorted(by_st.items()):
            if head <= 0:
                break
            extra = int(round(left * (len(v) - alloc[k]) / head))
            alloc[k] = min(len(v), alloc[k] + max(0, extra))
    while sum(alloc.values()) > n:                       # trim the biggest stratum first
        k = max(alloc, key=lambda k: (alloc[k], k))
        if alloc[k] <= min(floor, len(by_st[k])):
            break
        alloc[k] -= 1

    picked, drawn = [], Counter()
    for k, m in alloc.items():
        picked.extend(by_st[k][:m])
        drawn[k] = m

    rng.shuffle(picked)
    for i, r in enumerate(picked):
        r["index"] = i
        r["verdict"] = ""                                # <- ok / wrong / broad / narrow / ...
        r["should_be"] = ""                              # <- what it should have been, free text
        r["note"] = ""
        r["half"] = "calibration" if i < len(picked) // 2 else "locked"
        st = _stratum(r)
        r["_stratum"] = "|".join(st)
        r["_p"] = drawn[st] / len(by_st[st])
    strata = {"|".join(k): {"size": len(v), "drawn": drawn.get(k, 0)}
              for k, v in sorted(by_st.items())}
    return picked, strata


# --------------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------------
_CSV_COLS = ["index", "kind", "article", "in_grounding", "item", "context", "examples",
             "chosen", "direction", "chosen_meaning", "candidates",
             "verdict", "should_be", "note"]


def build(results_dir, articles, osfx="", track="e2e", n=SAMPLE_N, seed=SEED, out_dir=None):
    results_dir = Path(results_dir)
    out = Path(out_dir) if out_dir else results_dir / _TRACK_DIRS[track][1]
    out.mkdir(parents=True, exist_ok=True)
    rows = pool(results_dir, articles, osfx, track)
    grounded = _grounded_terms(results_dir, track)
    for r in rows:
        r["in_grounding"] = "y" if (r["article"], r["chosen"]) in grounded else ""
    if not rows:
        raise SystemExit(f"no canonicalization decisions found under {results_dir} for osfx "
                         f"{osfx!r} — has 6d/6e (or nb4/nb7b) run with this tag?")
    sample, strata = draw(rows, n=n, seed=seed)

    blind = [{k: v for k, v in r.items() if not k.startswith("_")} for r in sample]
    (out / SAMPLE_FILE.format(osfx=osfx)).write_text(
        json.dumps(blind, ensure_ascii=False, indent=2))
    # Written GROUPED BY ARTICLE, though `index` keeps the random calibration/locked assignment:
    # annotation cost is dominated by holding one article in your head, and a shuffled sheet makes
    # you reload it on every row. The two orderings are independent, so grouping costs nothing.
    with (out / CSV_FILE.format(osfx=osfx)).open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(blind, key=lambda r: (
            int("".join(c for c in str(r["article"]) if c.isdigit()) or 10**6),
            r["kind"], str(r["item"]))))
    (out / KEY_FILE.format(osfx=osfx)).write_text(json.dumps(
        {str(r["index"]): {"why": r["_why"], "stratum": r["_stratum"], "p": r["_p"],
                           "source": r["source"]} for r in sample}, ensure_ascii=False, indent=2))
    (out / STRATA_FILE.format(osfx=osfx)).write_text(json.dumps(
        {"pool_size": len(rows), "sample_size": len(sample), "seed": seed,
         "verdicts": VERDICTS, "strata": strata}, ensure_ascii=False, indent=2))

    n_g = sum(1 for r in sample if r.get("in_grounding"))
    print(f"pool {len(rows)} decisions   ->   sample {len(sample)}")
    print(f"  {n_g} of them use a term that is also in the manual grounding sample "
          f"-> the overlap where 'grounded AND correctly canonicalized' is computable")
    print(f'  {"stratum":<22} {"pool":>6} {"drawn":>6}')
    for k, v in sorted(strata.items()):
        print(f'  {k:<22} {v["size"]:>6} {v["drawn"]:>6}')
    print(f'\n  label the `verdict` column: {" / ".join(VERDICTS)}')
    print(f"  {out / CSV_FILE.format(osfx=osfx)}")
    print(f"  (the model's own justification is in {KEY_FILE.format(osfx=osfx)} — "
          f"read it AFTER labelling, not before)")
    return sample


# --------------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------------
def _wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _labelled(out, osfx):
    """Labels from the CSV if it has any, else from the JSON — whichever was actually filled in."""
    js = json.loads((out / SAMPLE_FILE.format(osfx=osfx)).read_text())
    csv_path = out / CSV_FILE.format(osfx=osfx)
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if any((r.get("verdict") or "").strip() for r in rows):
            by_i = {int(r["index"]): r for r in rows if str(r.get("index", "")).strip().isdigit()}
            for x in js:                       # CSV carries the labels, JSON the full record
                r = by_i.get(x["index"])
                if r:
                    x["verdict"] = (r.get("verdict") or "").strip().lower()
                    x["should_be"] = r.get("should_be") or ""
                    x["note"] = r.get("note") or ""
    return js


def score(results_dir, osfx="", track="e2e", half=None, out_dir=None):
    """The reuse-precision number, reweighted to the whole run.

    Reports, per kind and overall: how many judged decisions are correct, what that becomes once
    each stratum is weighted by its true size, and — the two that matter for the write-up —
    the precision of schema.org REUSE and the share of MINTS that were avoidable.
    """
    results_dir = Path(results_dir)
    out = Path(out_dir) if out_dir else results_dir / _TRACK_DIRS[track][1]
    items = _labelled(out, osfx)
    key = json.loads((out / KEY_FILE.format(osfx=osfx)).read_text())
    strata = json.loads((out / STRATA_FILE.format(osfx=osfx)).read_text())

    items = [x for x in items if (x.get("verdict") or "").strip().lower() in VERDICTS]
    if half:
        items = [x for x in items if x.get("half") == half]
    if not items:
        print(f"no labels yet — fill `verdict` in {out / CSV_FILE.format(osfx=osfx)}")
        print(f"  one of: {' / '.join(VERDICTS)}")
        return None
    for x in items:
        x["_v"] = x["verdict"].strip().lower()
        x["_k"] = key[str(x["index"])]

    lab = f"{half} half" if half else "all labelled"
    n_ok = sum(x["_v"] in _GOOD for x in items)
    lo, hi = _wilson(n_ok, len(items))
    print(f'[{lab}] {len(items)}/{strata["sample_size"]} labelled · correct '
          f'{n_ok}/{len(items)} = {n_ok/len(items):.1%}  [{lo:.1%}, {hi:.1%}]  (raw, on a '
          f'floor-stratified sample — the reweighted rate below is the one to report)')

    print("\n  verdict breakdown")
    for v, cnt in sorted(Counter(x["_v"] for x in items).items(), key=lambda kv: -kv[1]):
        print(f'    {v:<10} {cnt:>4}   {VERDICTS[v]}')

    print("\n  by decision kind")
    by_kind = defaultdict(list)
    for x in items:
        by_kind[x["kind"]].append(x)
    for k, xs in sorted(by_kind.items()):
        good = sum(x["_v"] in _GOOD for x in xs)
        a, b = _wilson(good, len(xs))
        print(f"    {k:<10} {good:>3}/{len(xs):<4} = {good/len(xs):>6.1%}  [{a:.0%}, {b:.0%}]")

    # The two headline numbers, each answering a claim the pipeline's own metrics cannot.
    reuse = [x for x in items if x["_k"]["source"] == "schema.org"]
    mints = [x for x in items if x["_k"]["source"] == "mint"]
    print("\n  what the reuse numbers are worth")
    if reuse:
        good = sum(x["_v"] in _GOOD for x in reuse)
        a, b = _wilson(good, len(reuse))
        broad = sum(x["_v"] == "broad" for x in reuse)
        print(f"    schema.org reuse PRECISION   {good}/{len(reuse)} = {good/len(reuse):.1%}  "
              f"[{a:.1%}, {b:.1%}]"
              f"\n      of which too general       {broad}/{len(reuse)} — reuse that is technically "
              f"right and says nothing")
        print(f"      -> multiply the run's schema_org_reuse_pct by this before quoting it as "
              f"'correctly canonicalized'")
    if mints:
        avoid = sum(x["_v"] == "avoidable" for x in mints)
        a, b = _wilson(avoid, len(mints))
        print(f"    mints that were AVOIDABLE     {avoid}/{len(mints)} = {avoid/len(mints):.1%}  "
              f"[{a:.1%}, {b:.1%}]  (a schema.org term existed)")

    # Reweight: each stratum's judged accuracy, weighted by how big that stratum really is.
    by_st = defaultdict(list)
    for x in items:
        by_st[x["_k"]["stratum"]].append(x["_v"] in _GOOD)
    num = den = cov = 0.0
    for st, meta in strata["strata"].items():
        den += meta["size"]
        seen = by_st.get(st)
        if seen:
            num += meta["size"] * (sum(seen) / len(seen))
            cov += meta["size"]
    if den:
        print(f"\n  REWEIGHTED to the whole run: {num/den:.1%} of all "
              f"{int(den)} canonicalization decisions are correct")
        print(f"    ({cov/den:.0%} of the population sits in a stratum you have labelled)")
    return {"labelled": len(items), "raw_correct": n_ok / len(items),
            "reweighted_correct": (num / den) if den else None,
            "reuse_precision": (sum(x["_v"] in _GOOD for x in reuse) / len(reuse)) if reuse else None,
            "avoidable_mint_rate": (sum(x["_v"] == "avoidable" for x in mints) / len(mints))
            if mints else None}


def _main(argv):
    import sys
    if len(argv) < 4:
        print(__doc__)
        return 2
    cmd, model, osfx = argv[1], argv[2], argv[3]
    track = argv[4] if len(argv) > 4 else "e2e"
    root = Path(__file__).resolve().parent.parent / "results" / model
    if cmd == "build":
        arts = sorted((p.parent.parent.name for p in root.glob("art*/*/")), key=lambda a: a)
        arts = sorted({p.name for p in root.glob("art*")},
                      key=lambda a: int("".join(c for c in a if c.isdigit()) or 0))
        build(root, arts, osfx=osfx, track=track)
        return 0
    if cmd == "score":
        score(root, osfx=osfx, track=track,
              half=argv[5] if len(argv) > 5 else None)
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
