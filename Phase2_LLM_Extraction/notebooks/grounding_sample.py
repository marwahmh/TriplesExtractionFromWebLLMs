"""The 200-assertion manual grounding sample, drawn over the 10 monitoring articles.

Replaces nb9b's 50-item `build_sample` for the thesis number. Two things it does differently:

**Equal per article.** 20 items each, so a single article the judge reads badly shows up instead of
being averaged away by whichever graph happens to be biggest.

**The negatives are a census, not a sample.** The judge grounds ~98% of assertions, so 200 drawn
uniformly would contain ~4 negatives — enough to say nothing at all about the direction that
matters (a judge that waves through unsupported assertions). Every judge-negative in the ten
articles is taken; positives fill the rest. That makes the sample deliberately non-representative,
so nothing may be read off it directly: `score()` reweights each stratum by its true size, and
reports the corrected rate with a CI alongside the raw agreement.

Usage from nb9b (after section 6 has grounded all ten):

    import grounding_sample as gs
    graphs = {aid: json.loads((RESULTS_DIR/aid/TRACK_DIR/OUT_NAME).read_text()) for aid in ARTICLE_IDS}
    gs.build(graphs, GLOBAL_DIR, verbalise_node, make_relation_verbaliser)   # writes 3 files
    ...                                                                     # label by hand
    gs.score(GLOBAL_DIR)

Files written to results/<model>/_global/:
    grounding_manual_sample.json   what you label — no judge verdict in it, `human` is ""
    grounding_manual_key.json      judge verdict per index (do not open while labelling)
    grounding_manual_strata.json   stratum sizes + draw counts, needed to reweight
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

SAMPLE_N = 200
PER_ARTICLE = 20
MAX_NEGATIVES = 60          # census of judge-negatives, capped so they cannot eat the budget
SEED = 20260813

SAMPLE_FILE = "grounding_manual_sample.json"
KEY_FILE = "grounding_manual_key.json"
STRATA_FILE = "grounding_manual_strata.json"


# --- pool -----------------------------------------------------------------------------------

def _source_of(x, kind):
    """Where the property (relations) or type (nodes) came from: schema.org, minted vocab, or
    never canonicalised. Not a stratum — too thin at n=200 — but reported, because minted terms
    are where the judge's mistakes have concentrated so far."""
    if kind == "relation":
        return x.get("property_source") or "unknown"
    return x.get("type_source") or "not_canonicalized"      # same label the nb9b report uses


def pool(graphs, verbalise_node, make_relation_verbaliser):
    """Every grounded assertion across the given graphs, flattened to one row each."""
    rows = []
    for aid, g in graphs.items():
        v_rel = make_relation_verbaliser(g)
        for x in g.get("nary_nodes", []):
            if "grounding" not in x:
                continue
            rows.append({"article_id": aid, "kind": "node", "id": x.get("id"),
                         "assertion": verbalise_node(x),
                         "judge_quote": x["grounding"].get("quote"),
                         "judge": bool(x["grounding"].get("supported")),
                         "term_source": _source_of(x, "node")})
        for x in g.get("relations", []):
            if "grounding" not in x:
                continue
            rows.append({"article_id": aid, "kind": "relation",
                         "id": f'{x.get("subject_id")}-{x.get("property")}-{x.get("object_id")}',
                         "assertion": v_rel(x),
                         "judge_quote": x["grounding"].get("quote"),
                         "judge": bool(x["grounding"].get("supported")),
                         "term_source": _source_of(x, "relation")})
    return rows


def _stratum(r):
    return (r["article_id"], r["kind"], "y" if r["judge"] else "n")


# --- draw -----------------------------------------------------------------------------------

def draw(rows, n=SAMPLE_N, per_article=PER_ARTICLE, seed=SEED):
    """Returns (sample, strata). Negatives: all of them (up to MAX_NEGATIVES). Positives: an equal
    share per article, split between nodes and relations in the proportion the article's own graph
    has them, so the sample mirrors the graph's shape on the side that isn't a census."""
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for r in rows:
        by_stratum[_stratum(r)].append(r)
    for v in by_stratum.values():
        rng.shuffle(v)

    picked, drawn = [], Counter()

    negatives = [r for r in rows if not r["judge"]]
    if len(negatives) > MAX_NEGATIVES:            # round-robin over articles, so a single graph
        rng.shuffle(negatives)                    # with many negatives cannot take the whole cap
        queues = defaultdict(list)
        for r in negatives:
            queues[r["article_id"]].append(r)
        negatives, order = [], sorted(queues)
        while len(negatives) < MAX_NEGATIVES and any(queues[a] for a in order):
            for a in order:
                if queues[a] and len(negatives) < MAX_NEGATIVES:
                    negatives.append(queues[a].pop())
    for r in negatives:
        picked.append(r)
        drawn[_stratum(r)] += 1

    articles = sorted({r["article_id"] for r in rows})
    budget = n - len(picked)
    share = {aid: budget // len(articles) for aid in articles}
    for aid in articles[: budget % len(articles)]:
        share[aid] += 1

    for aid in articles:
        pos = {k: v for k, v in by_stratum.items() if k[0] == aid and k[2] == "y"}
        total = sum(len(v) for v in pos.values())
        if not total:
            continue
        want = min(share[aid], total)
        # proportional allocation, but never zero nodes when the article has any
        alloc = {}
        for k, v in pos.items():
            alloc[k] = int(round(want * len(v) / total))
        if pos.get((aid, "node", "y")) and not alloc.get((aid, "node", "y")):
            alloc[(aid, "node", "y")] = 1
        while sum(alloc.values()) > want:         # trim the largest stratum first
            k = max(alloc, key=lambda k: alloc[k])
            alloc[k] -= 1
        while sum(alloc.values()) < want:
            k = min((k for k in pos if alloc[k] < len(pos[k])),
                    key=lambda k: alloc[k] / max(1, len(pos[k])), default=None)
            if k is None:
                break
            alloc[k] += 1
        for k, m in alloc.items():
            m = min(m, len(by_stratum[k]))
            picked.extend(by_stratum[k][:m])
            drawn[k] += m

    rng.shuffle(picked)
    for i, r in enumerate(picked):
        r["index"] = i
        r["human"] = ""                            # <- fill with "y" / "n"
        r["half"] = "calibration" if i < len(picked) // 2 else "locked"
        st = _stratum(r)
        r["_stratum"] = "|".join(st)
        r["_p"] = drawn[st] / len(by_stratum[st])  # inclusion probability, for reweighting

    strata = {"|".join(k): {"size": len(v), "drawn": drawn.get(k, 0)}
              for k, v in sorted(by_stratum.items())}
    return picked, strata


def build(graphs, out_dir, verbalise_node, make_relation_verbaliser, n=SAMPLE_N, seed=SEED):
    out_dir = Path(out_dir)
    rows = pool(graphs, verbalise_node, make_relation_verbaliser)
    sample, strata = draw(rows, n=n, seed=seed)

    blind = [{k: v for k, v in r.items() if k not in ("judge", "term_source")} for r in sample]
    (out_dir / SAMPLE_FILE).write_text(json.dumps(blind, ensure_ascii=False, indent=2))
    (out_dir / KEY_FILE).write_text(json.dumps(
        {str(r["index"]): {"judge": r["judge"], "term_source": r["term_source"],
                           "stratum": r["_stratum"], "p": r["_p"]} for r in sample}, indent=2))
    (out_dir / STRATA_FILE).write_text(json.dumps(
        {"pool_size": len(rows), "sample_size": len(sample),
         "judge_grounded": sum(r["judge"] for r in rows), "strata": strata}, indent=2))

    neg = sum(1 for r in sample if not r["judge"])
    print(f"pool {len(rows)} assertions over {len({r['article_id'] for r in rows})} articles "
          f"({sum(not r['judge'] for r in rows)} judge-negative)")
    print(f"sample {len(sample)}: {neg} negative (census) + {len(sample)-neg} positive, "
          f"{len(sample)//2} calibration / {len(sample)-len(sample)//2} locked")
    for aid in sorted({r["article_id"] for r in sample}):
        got = [r for r in sample if r["article_id"] == aid]
        print(f"   {aid:6} {len(got):3}  ({sum(1 for r in got if r['kind']=='node')} node / "
              f"{sum(1 for r in got if r['kind']=='relation')} relation, "
              f"{sum(1 for r in got if not r['judge'])} negative)")
    print("wrote", (out_dir / SAMPLE_FILE))
    return sample


# --- score ----------------------------------------------------------------------------------

def _wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(out_dir, half=None):
    """Agreement between your labels and the judge, plus the stratum-reweighted grounding rate.

    half=None scores everything, "calibration" the half you may look at while tuning the judge
    prompt, "locked" the half that should be scored once, at the end, and reported.
    """
    out_dir = Path(out_dir)
    labelled = json.loads((out_dir / SAMPLE_FILE).read_text())
    key = json.loads((out_dir / KEY_FILE).read_text())
    strata = json.loads((out_dir / STRATA_FILE).read_text())

    items = [x for x in labelled if (x.get("human") or "").strip().lower() in ("y", "n")]
    if half:
        items = [x for x in items if x.get("half") == half]
    if not items:
        print("no labels yet — fill `human` with y/n in", out_dir / SAMPLE_FILE)
        return
    for x in items:
        x["_k"] = key[str(x["index"])]
        x["_h"] = (x["human"] or "").strip().lower() == "y"

    agree = sum(x["_h"] == x["_k"]["judge"] for x in items)
    lenient = sum(x["_k"]["judge"] and not x["_h"] for x in items)     # judge yes, you no
    strict = sum((not x["_k"]["judge"]) and x["_h"] for x in items)    # judge no, you yes
    lab = f"{half} half" if half else "all labelled"
    print(f"[{lab}] {len(items)}/{len(labelled)} labelled · raw agreement {agree}/{len(items)} "
          f"= {agree/len(items):.1%}   (raw, on a deliberately negative-heavy sample — "
          f"do not report this as the grounding rate)")

    pos = [x for x in items if x["_k"]["judge"]]
    neg = [x for x in items if not x["_k"]["judge"]]
    if pos:
        lo, hi = _wilson(lenient, len(pos))
        print(f"  judge too lenient: {lenient}/{len(pos)} = {lenient/len(pos):.1%} "
              f"of its 'supported' verdicts are wrong  [{lo:.1%}, {hi:.1%}]")
    if neg:
        lo, hi = _wilson(strict, len(neg))
        print(f"  judge too strict:  {strict}/{len(neg)} = {strict/len(neg):.1%} "
              f"of its 'unsupported' verdicts are wrong  [{lo:.1%}, {hi:.1%}]")

    # corrected rate: within each stratum, the labelled fraction that is truly supported, weighted
    # by the stratum's real size in the pool.
    by_st = defaultdict(list)
    for x in items:
        by_st[x["_k"]["stratum"]].append(x["_h"])
    num = den = 0.0
    covered = 0
    for st, meta in strata["strata"].items():
        seen = by_st.get(st)
        den += meta["size"]
        if seen:
            num += meta["size"] * (sum(seen) / len(seen))
            covered += meta["size"]
    judged = strata["judge_grounded"] / strata["pool_size"]
    print(f"  judge's own grounding rate      {judged:.1%}  ({strata['judge_grounded']}/{strata['pool_size']})")
    if den:
        print(f"  human-corrected, reweighted     {num/den:.1%}  "
              f"({covered/den:.0%} of the pool sits in a stratum you have labelled)")

    per_art = defaultdict(lambda: [0, 0])
    for x in items:
        a = per_art[x["article_id"]]
        a[0] += x["_h"] == x["_k"]["judge"]
        a[1] += 1
    print("  per article:", "  ".join(f"{a} {k}/{n}" for a, (k, n) in sorted(per_art.items())))

    per_src = defaultdict(lambda: [0, 0])
    for x in items:
        s = per_src[x["_k"]["term_source"]]
        s[0] += x["_h"] == x["_k"]["judge"]
        s[1] += 1
    print("  by term source:", "  ".join(f"{s} {k}/{n}" for s, (k, n) in sorted(per_src.items())))
