"""A gold-labelled benchmark for the nb9b grounding judge: 250 real assertions + 250 corrupted.

**Why this exists.** nb9b's own validation sample (section 5) draws items *stratified on the judge's
own verdict*, so it measures agreement with a human on the population the judge produced. It cannot
measure the thing that actually matters for a grounding claim — **would this judge catch a fact the
extractor got wrong?** — because the judge grounds ~98% of what it sees, so a uniform sample contains
almost no true negatives and the recall estimate rests on a handful of items.

This module builds the missing half by *manufacturing* negatives. Two disjoint halves:

* **250 clean** assertions drawn from the graphs as they are. Their gold label is unknown until you
  label them by hand — most will be supported, and the ones that are not are the extractor's real
  errors. This half measures the judge's **false-positive rate** (rejecting good facts).
* **250 corrupted** assertions, each a real assertion with one deterministic, meaning-changing edit
  (a swapped entity, an altered number, a flipped quantifier, a reversed arrow, a wrong type…).
  Every edit is *verified absent from the source article* before it is accepted, so the gold label
  is "not supported" by construction. This half measures the judge's **recall on bad facts**, broken
  down by the kind of error — which is the number a thesis needs and the one nobody has.

The two halves are drawn from **disjoint** base assertions and shuffled together, so no item in the
sheet is the recognisable twin of another and hand-labelling stays honest.

**The corrupted half is a hypothesis, not a fact.** An edit can land on something the article
happens to support anyway. That is why every corrupted item also carries a `human` field: label the
ones the judge accepts, and `score()` will separate "the judge missed it" from "the corruption did
not corrupt". The count of failed corruptions is reported, never hidden.

Usage from nb9b (section 8), after the config / client / matcher / prompt cells have run:

    import judge_benchmark as jb
    graphs   = {aid: jb.load_json(RESULTS_DIR / aid / TRACK_DIR / GRAPH_NAME) for aid in ARTICLE_IDS}
    articles = {aid: load_article(aid) for aid in ARTICLE_IDS}
    jb.build(graphs, articles, GLOBAL_DIR, verbalise_node, make_relation_verbaliser, norm_text)

    items = jb.load_items(GLOBAL_DIR)                      # then run the judge over them
    for aid, group in jb.by_article(items):
        set_article(articles[aid]);  run_judge(group, jb.assertion_of, f"bench {aid}")
    jb.save_verdicts(items, GLOBAL_DIR, tag="qwen3-32b-think_seed1234", config={...})

    # label judge_benchmark_sheet.csv by hand, then
    jb.score(GLOBAL_DIR, tag="qwen3-32b-think_seed1234")
    jb.compare(GLOBAL_DIR)                                 # every judge run side by side

Files written to results/<model>/_global/:
    judge_benchmark_items.json          the 500 items, blind (no gold, no corruption info)
    judge_benchmark_sheet.csv           the same, grouped by article — this is what you label
    judge_benchmark_key.json            gold + corruption provenance (do not open while labelling)
    judge_benchmark_meta.json           pool sizes, realised corruption mix, seed, source graph
    judge_benchmark_verdicts_<tag>.json one per judge run, so judges/seeds can be compared
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

N_CLEAN = 250
N_CORRUPT = 250
SEED = 20260817

ITEMS_FILE = "judge_benchmark_items.json"
SHEET_FILE = "judge_benchmark_sheet.csv"
KEY_FILE = "judge_benchmark_key.json"
META_FILE = "judge_benchmark_meta.json"
VERDICTS_FILE = "judge_benchmark_verdicts_{tag}.json"

# How many of the 250 corruptions each operator should contribute. These are *targets*: an operator
# whose pool runs dry (only ~330 assertions in this corpus carry a number at all) hands its shortfall
# to the others, and `meta` records the mix that was actually built. Weighted towards the swaps
# because they are the failure the pipeline actually makes — a canonicalisation step attaching the
# right property to the wrong entity — and away from `hedge_flip`, whose lexicon only fires on a
# few dozen assertions.
TARGET_MIX = {
    "object_swap":    55,   # right subject + property, an object from another article's graph
    "property_swap":  45,   # right endpoints, a property the corpus never uses with this object type
    "number_edit":    40,   # a quantity/date/year moved to a value the article does not state
    "direction_flip": 30,   # subject and object exchanged (entity->entity edges only)
    "subject_swap":   30,
    "type_change":    25,   # node type, or a relation's object type, replaced by an unrelated one
    "hedge_flip":     25,   # "more than" -> "fewer than", "suspected" -> "confirmed", "rose" -> "fell"
}

# Which failure_type nb9b's prompt should return for each operator, if the judge is reading the
# error correctly: "intrinsic" = the article contradicts it, "extrinsic" = the article never says it.
# Scored as a secondary diagnostic — a judge that catches the item but calls a contradiction
# "extrinsic" is not reading the article, it is failing to find a match.
DESIGNED_FAILURE = {
    "object_swap": "extrinsic", "subject_swap": "extrinsic", "property_swap": "extrinsic",
    "type_change": "extrinsic", "number_edit": "intrinsic", "hedge_flip": "intrinsic",
    "direction_flip": "intrinsic",
}


def load_json(path):
    return json.loads(Path(path).read_text())


def assertion_of(item):
    """The verbaliser to hand nb9b's `run_judge`: the assertion text is already baked into the item."""
    return item["assertion"]


# --- normalisation --------------------------------------------------------------------------
# `build` takes nb9b's own `norm_text` so the "is this edit absent from the article?" test uses
# exactly the normalisation the quote matcher uses. This fallback exists only so the module can be
# imported and unit-tested outside the notebook.

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", "…": "..."})


def _fallback_norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s or "").translate(_QUOTES)).strip().casefold()


# --- the pool -------------------------------------------------------------------------------

def pool_items(graphs):
    """Every node and relation across the run, one row each, carrying enough to corrupt it later.

    `pos` is the item's index inside its graph's own list, which is how a corruption re-finds the
    base assertion in a copied graph — object identity does not survive `deepcopy`.
    """
    rows = []
    for aid, g in graphs.items():
        hubs = {n.get("id") for n in g.get("nary_nodes", [])}
        for i, n in enumerate(g.get("nary_nodes", [])):
            rows.append({"article_id": aid, "kind": "node", "pos": i,
                         "id": n.get("id"), "item": n,
                         "term_source": n.get("type_source") or "not_canonicalized"})
        for i, r in enumerate(g.get("relations", [])):
            rows.append({"article_id": aid, "kind": "relation", "pos": i,
                         "id": f'{r.get("subject_id")}-{r.get("property")}-{r.get("object_id")}',
                         "item": r, "hub_subject": r.get("subject_id") in hubs,
                         "term_source": r.get("property_source") or "unknown"})
    return rows


def _corpus_pools(graphs):
    """Corpus-wide vocabularies the corruption operators draw replacements from.

    Everything is keyed by article too, because a replacement must come from a *different* article:
    an object lifted from the same graph is often still supported by the same text.
    """
    objects_by_type = defaultdict(list)      # type -> [(article_id, text)]
    subjects_by_type = defaultdict(list)
    types_by_object = defaultdict(set)       # object_type -> {properties seen with it}
    node_types, parents = set(), {}
    props = set()
    for aid, g in graphs.items():
        for n in g.get("nary_nodes", []):
            if n.get("type"):
                node_types.add(n["type"])
                if n.get("type_parent"):
                    parents[n["type"]] = n["type_parent"]
        for r in g.get("relations", []):
            p, ot, st = r.get("property"), r.get("object_type"), r.get("subject_type")
            if p:
                props.add(p)
            if ot and r.get("object_text"):
                objects_by_type[ot].append((aid, r["object_text"]))
                types_by_object[ot].add(p)
                node_types.add(ot)
            if st and r.get("subject_text"):
                subjects_by_type[st].append((aid, r["subject_text"]))
                node_types.add(st)
    return {"objects_by_type": objects_by_type, "subjects_by_type": subjects_by_type,
            "props_by_object_type": types_by_object, "types": sorted(node_types),
            "parents": parents, "props": sorted(props)}


# --- corruption operators -------------------------------------------------------------------
# Every operator returns (patch, description) or None. `patch` is a dict of field -> new value to
# apply to a COPY of the base assertion; returning None means "this operator does not apply here, or
# its edit could not be verified absent from the article" and the builder moves on to another item.
# The absence test (`absent`) is the whole safety argument for the gold label: an edit whose text the
# article already contains is not a negative, it is a paraphrase.

_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]

# Applied on word boundaries, first match wins. Each pair changes what the assertion CLAIMS —
# a direction, a certainty, or a bound — without changing what it is about, which is the class of
# error a grounding judge is most likely to wave through.
#
# The third field says the word is only a hedge when a NUMBER follows it. "about" in "about 500
# deaths" is a bound and flipping it to "exactly" makes a false claim; "about" in "Mayor angry about
# burglary" is a preposition, and flipping it there produces word salad rather than a wrong fact —
# a negative any judge catches for the wrong reason.
_HEDGE_PAIRS = [
    ("more than", "fewer than", True), ("at least", "at most", True), ("up to", "no fewer than", True),
    ("nearly", "well over", True), ("almost", "well over", True), ("approximately", "exactly", True),
    ("around", "exactly", True), ("about", "exactly", True), ("over", "under", True),
    ("under", "over", True), ("above", "below", True), ("below", "above", True),
    ("may", "will", False), ("could", "will", False), ("might", "will", False),
    ("expected to", "confirmed to have", False), ("planned", "completed", False),
    ("proposed", "enacted", False), ("suspected", "confirmed", False), ("alleged", "proven", False),
    ("possible", "confirmed", False), ("potential", "confirmed", False),
    ("reported", "denied", False), ("accused", "convicted", False),
    ("increase", "decrease", False), ("increased", "decreased", False),
    ("rose", "fell", False), ("rise", "fall", False), ("grew", "shrank", False),
    ("higher", "lower", False), ("record-breaking", "unremarkable", False),
    ("highest", "lowest", False), ("largest", "smallest", False),
    ("began", "ended", False), ("opened", "closed", False), ("launched", "cancelled", False),
    ("approved", "rejected", False), ("supports", "opposes", False), ("won", "lost", False),
]


def _fmt_like(value, raw, grouped=True):
    """Render a number in the same shape as the one it replaces ("2,000" -> "4,800", "3.5" -> "8.4").
    `grouped=False` for years, where a thousands separator would be wrong ("2,018")."""
    if "." in raw:
        return f"{value:,.1f}" if "," in raw else f"{value:.1f}"
    n = int(round(value))
    return f"{n:,}" if grouped and n >= 1000 else str(n)


def _edit_number(text, rng, absent):
    """Move one number to a value the article does not state. Years shift by a few, percentages by
    a dozen points, everything else scales — a 3x edit is unmistakable, a +1 edit would test the
    judge's arithmetic rather than its grounding."""
    for m in reversed(list(_NUM_RE.finditer(text or ""))):   # last number first: usually the salient one
        raw = m.group(0)
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        pct = text[m.end():m.end() + 1] == "%"
        year = len(raw) == 4 and 1900 <= val <= 2100 and val == int(val)
        if pct:
            cands = [val + d for d in (12, 19, 27, -11, -18) if 1 <= val + d <= 99]
        elif year:
            cands = [val + d for d in (-7, -4, 3, 5, 9)]
        else:
            cands = [val * f for f in (0.4, 0.55, 1.8, 2.4, 3.3)]
        rng.shuffle(cands)
        for v in cands:
            new = _fmt_like(v, raw, grouped=not year)
            if new == raw:
                continue
            out = text[:m.start()] + new + text[m.end():]
            if absent(out) and absent(new):
                return out, f"{raw} -> {new}"
    # no usable digit: try a month name instead ("18 June" -> "18 October")
    for i, mo in enumerate(_MONTHS):
        if re.search(rf"\b{mo}\b", text or "", re.I):
            for j in (i + 4, i + 7, i + 2):
                new_mo = _MONTHS[j % 12]
                out = re.sub(rf"\b{mo}\b", new_mo, text, count=1, flags=re.I)
                if absent(out) and absent(new_mo):
                    return out, f"{mo} -> {new_mo}"
    return None


def _flip_hedge(text, absent):
    """Flip one quantifier/certainty/direction word. Skipped on text that is a bare proper name
    ("New York City" has no lowercase token), and on a match that sits inside one — otherwise the
    operator renames an entity instead of changing a claim, which is a different negative and an
    obviously broken sentence."""
    if not text or not any(w[:1].islower() for w in re.findall(r"\w+", text)):
        return None
    for a, b, needs_number in _HEDGE_PAIRS:
        for m in re.finditer(rf"\b{re.escape(a)}\b", text, re.I):
            before, after = text[:m.start()], text[m.end():]
            if needs_number and not re.match(r"[\s£$€]*\d", after):
                continue                       # a bound with nothing to bound: it is a preposition
            neighbours = re.findall(r"\w+", before)[-1:] + re.findall(r"\w+", after)[:1]
            if m.group(0)[:1].isupper() and any(w[:1].isupper() for w in neighbours):
                continue                       # mid-name capital: "New York", "First Republic"
            repl = b.capitalize() if m.group(0)[:1].isupper() else b
            out = before + repl + after
            if absent(out) and absent(repl):
                return out, f"{m.group(0)} -> {repl}"
    return None


def _pick_foreign(candidates, article_id, current, absent, rng, tries=40):
    """A replacement surface form from ANOTHER article, verified absent from this one."""
    if not candidates:
        return None
    cur = (current or "").strip().casefold()
    for _ in range(tries):
        aid, text = rng.choice(candidates)
        if aid == article_id or not text or text.strip().casefold() == cur:
            continue
        if len(text) > 90:                       # a sentence-length "entity" makes an unreadable item
            continue
        if absent(text):
            return text
    return None


def _unrelated_type(current, pools, absent, rng):
    """A type that is neither the original, nor its parent/child, nor a lexical variant of it."""
    parents = pools["parents"]
    kin = {current, parents.get(current)} | {t for t, p in parents.items() if p == current}
    cur_tokens = set(re.findall(r"[a-z]+", _fallback_norm(current)))
    cands = [t for t in pools["types"]
             if t not in kin and not (cur_tokens & set(re.findall(r"[a-z]+", _fallback_norm(t))))]
    rng.shuffle(cands)
    for t in cands[:40]:
        if absent(t):
            return t
    return None


def corrupt(row, op, graphs, pools, absent, rng):
    """Apply operator `op` to the pooled row. -> (patch, description) or None if it does not apply."""
    it, aid, kind = row["item"], row["article_id"], row["kind"]

    if kind == "node":
        if op == "type_change":
            t = _unrelated_type(it.get("type"), pools, absent, rng)
            return ({"type": t}, f'type {it.get("type")} -> {t}') if t else None
        if op == "number_edit":
            r = _edit_number(it.get("label"), rng, absent)
            return ({"label": r[0]}, f"label: {r[1]}") if r else None
        if op == "hedge_flip":
            r = _flip_hedge(it.get("label"), absent)
            return ({"label": r[0]}, f"label: {r[1]}") if r else None
        if op in ("object_swap", "subject_swap"):     # a node has one surface form: its label
            new = _pick_foreign(pools["subjects_by_type"].get(it.get("type"), []),
                                aid, it.get("label"), absent, rng)
            return ({"label": new}, f'label -> "{new}"') if new else None
        return None

    # --- relations ---------------------------------------------------------------------------
    if op == "object_swap":
        new = _pick_foreign(pools["objects_by_type"].get(it.get("object_type"), []),
                            aid, it.get("object_text"), absent, rng)
        return ({"object_text": new}, f'object "{it.get("object_text")}" -> "{new}"') if new else None

    if op == "subject_swap":
        if row.get("hub_subject"):        # a hub's label is also its context line: swapping it there
            return None                   # would contradict the sibling list and give the item away
        new = _pick_foreign(pools["subjects_by_type"].get(it.get("subject_type"), []),
                            aid, it.get("subject_text"), absent, rng)
        return ({"subject_text": new}, f'subject "{it.get("subject_text")}" -> "{new}"') if new else None

    if op == "property_swap":
        ot = it.get("object_type")
        seen = pools["props_by_object_type"].get(ot, set())          # never used with this object type
        sibs = {r.get("property") for r in graphs[aid].get("relations", [])
                if r.get("subject_id") == it.get("subject_id")}
        cur_tokens = set(re.findall(r"[a-z]+", _fallback_norm(it.get("property") or "")))
        cands = [p for p in pools["props"]
                 if p not in seen and p not in sibs
                 and not (cur_tokens & set(re.findall(r"[a-z]+", _fallback_norm(p))))]
        if not cands:
            return None
        p = rng.choice(cands)
        return {"property": p}, f'property {it.get("property")} -> {p}'

    if op == "direction_flip":
        # NOTE — this is the least reliable operator, and the only one with a measured failure rate:
        # on the 2026-08-17 build a human pass judged 6 of its 30 items still supported. The flip
        # only creates a false claim when the property is genuinely asymmetric, and much of this
        # corpus's vocabulary is not: `affiliation`, `hasPart`, `communicatedVia`, `recordedIn` all
        # read the same both ways, and flipping a schema.org edge that the extractor wrote in the
        # schema-correct direction (`CreativeWork.author -> Person`) produces the MORE natural
        # sentence, not a wrong one. Those items are not a judge failure, and `score()` drops them
        # as failed corruptions once they are labelled. Treat this operator's recall as the softest
        # number in the report, and do not raise its share of the mix without an asymmetry check.
        if row.get("hub_subject") or it.get("subject_type") == it.get("object_type"):
            return None                   # a hub as an object reads as nonsense rather than as a
        if not it.get("subject_text") or not it.get("object_text"):   # wrong claim; same-type flips
            return None                                              # are often genuinely symmetric
        return ({"subject_text": it["object_text"], "subject_type": it.get("object_type"),
                 "object_text": it["subject_text"], "object_type": it.get("subject_type")},
                f'reversed: {it.get("subject_text")} <-> {it.get("object_text")}')

    if op == "number_edit":
        r = _edit_number(it.get("object_text"), rng, absent)
        return ({"object_text": r[0]}, f"object: {r[1]}") if r else None

    if op == "hedge_flip":
        for field in ("object_text", "subject_text"):
            if row.get("hub_subject") and field == "subject_text":
                continue
            r = _flip_hedge(it.get(field), absent)
            if r:
                return {field: r[0]}, f"{field}: {r[1]}"
        return None

    if op == "type_change":
        t = _unrelated_type(it.get("object_type"), pools, absent, rng)
        return ({"object_type": t}, f'object type {it.get("object_type")} -> {t}') if t else None

    return None


# --- building the sheet ---------------------------------------------------------------------

def _verbalise(row, graphs, verbalise_node, make_relation_verbaliser, patch=None):
    """Verbalise a (possibly patched) assertion using nb9b's own verbalisers.

    The patch is applied inside a COPY of the whole graph, not just to the assertion, because
    `make_relation_verbaliser` appends the hub's sibling edges as context — and it identifies the
    edge to exclude by object identity. Patching a detached copy would leave the ORIGINAL edge in
    the context line, so a corrupted object would sit next to its own uncorrupted twin.
    """
    aid = row["article_id"]
    g = graphs[aid]
    if patch:
        g = copy.deepcopy(g)
        key = "nary_nodes" if row["kind"] == "node" else "relations"
        g[key][row["pos"]].update(patch)
    item = g["nary_nodes" if row["kind"] == "node" else "relations"][row["pos"]]
    text = verbalise_node(item) if row["kind"] == "node" else make_relation_verbaliser(g)(item)
    return text, item


def _draw_clean(rows, n, rng):
    """Proportional per article, and inside an article proportional to its own node/relation mix, so
    the clean half mirrors the shape of what the pipeline emits rather than a shape we chose."""
    by_article = defaultdict(list)
    for r in rows:
        by_article[r["article_id"]].append(r)
    articles = sorted(by_article)
    share = {a: n * len(by_article[a]) // len(rows) for a in articles}
    for a in sorted(articles, key=lambda a: -len(by_article[a]))[: n - sum(share.values())]:
        share[a] += 1
    picked = []
    for a in articles:
        pool = by_article[a][:]
        rng.shuffle(pool)
        nodes = [r for r in pool if r["kind"] == "node"]
        rels = [r for r in pool if r["kind"] == "relation"]
        want = min(share[a], len(pool))
        n_node = min(len(nodes), round(want * len(nodes) / max(1, len(pool))))
        picked += nodes[:n_node] + rels[: want - n_node]
    return picked


def build(graphs, articles, out_dir, verbalise_node, make_relation_verbaliser, norm_text=None,
          n_clean=N_CLEAN, n_corrupt=N_CORRUPT, seed=SEED, mix=None, source=None, overwrite=False):
    """Draw the clean half, manufacture the corrupted half, write the sheet + key + meta.

    `graphs`/`articles` are {article_id: ...}. `norm_text` should be nb9b's, so the absence test
    that licenses every gold label matches the matcher the judge's quotes are checked against.

    Refuses to overwrite an existing benchmark. Rebuilding renumbers every item, which silently
    invalidates both the hand labels in the sheet and every verdict file scored against it.
    """
    out_dir = Path(out_dir)
    if (out_dir / SHEET_FILE).exists() and not overwrite:
        done = sum(1 for x in load_items(out_dir) if (x.get("human") or "").strip())
        raise FileExistsError(
            f"{out_dir / SHEET_FILE} already exists ({done} item(s) labelled). Rebuilding "
            f"renumbers the items and orphans the labels and every judge_benchmark_verdicts_*.json "
            f"scored against them. Pass overwrite=True only if you mean to start over.")
    norm = norm_text or _fallback_norm
    rng = random.Random(seed)
    rows = pool_items(graphs)
    pools = _corpus_pools(graphs)
    art_norm = {aid: norm(txt) for aid, txt in articles.items()}

    clean = _draw_clean(rows, n_clean, rng)
    used = {(r["article_id"], r["kind"], r["pos"]) for r in clean}
    rest = [r for r in rows if (r["article_id"], r["kind"], r["pos"]) not in used]
    rng.shuffle(rest)

    entries, seen_text = [], set()
    for r in clean:
        text, item = _verbalise(r, graphs, verbalise_node, make_relation_verbaliser)
        entries.append({"row": r, "assertion": text, "origin": "clean", "op": None, "note": None,
                        "original": text, "evidence": item.get("evidence")})
        seen_text.add(norm(text))

    def try_corrupt(r, op):
        """Corrupt, verbalise, and accept only if the result is a NEW assertion.

        Both guards matter. An edit that leaves the verbalisation unchanged is not a negative at
        all. A verbalisation that duplicates another item is worse than useless: node label swaps
        borrow a label from another article's graph, and a donor label can be the very assertion
        that graph contributed to the clean half — the same sentence in the sheet twice, once
        gold-supported and once gold-unsupported, which teaches the annotator to pattern-match.
        """
        absent = lambda s: bool(s) and norm(s) not in art_norm[r["article_id"]]
        got = corrupt(r, op, graphs, pools, absent, rng)
        if not got or any(v is None for v in got[0].values()):
            return None
        patch, note = got
        base, _ = _verbalise(r, graphs, verbalise_node, make_relation_verbaliser)
        text, item = _verbalise(r, graphs, verbalise_node, make_relation_verbaliser, patch=patch)
        if norm(text) == norm(base) or norm(text) in seen_text:
            return None
        seen_text.add(norm(text))
        return {"row": r, "assertion": text, "origin": "corrupt", "op": op, "note": note,
                "original": base, "evidence": item.get("evidence")}

    # Best-effort per item: an operator that does not apply hands the item back, so walk the
    # remaining pool once per operator in target order and take what sticks. A second pass then
    # refills whatever an exhausted operator (only ~330 assertions here carry a number at all)
    # left short, using any operator that applies.
    mix = dict(mix or TARGET_MIX)
    n_corrupted, taken = 0, set()
    for op in sorted(mix, key=lambda o: -mix[o]):
        want = mix[op]
        for r in rest:
            if n_corrupted >= n_corrupt or want <= 0:
                break
            k = (r["article_id"], r["kind"], r["pos"])
            if k in taken:
                continue
            e = try_corrupt(r, op)
            if e:
                taken.add(k); entries.append(e); n_corrupted += 1; want -= 1
    for r in rest:
        if n_corrupted >= n_corrupt:
            break
        k = (r["article_id"], r["kind"], r["pos"])
        if k in taken:
            continue
        for op in rng.sample(list(mix), len(mix)):
            e = try_corrupt(r, op)
            if e:
                taken.add(k); entries.append(e); n_corrupted += 1
                break

    rng.shuffle(entries)
    items, key = [], {}
    for i, e in enumerate(entries):
        r = e["row"]
        items.append({"index": i, "article_id": r["article_id"], "kind": r["kind"],
                      "assertion": e["assertion"], "cited_evidence": e["evidence"],
                      "human": "", "label_source": "", "evidence": "", "human_note": ""})
        key[str(i)] = {"origin": e["origin"], "op": e["op"], "edit": e["note"],
                       "gold": None if e["origin"] == "clean" else "n",
                       "designed_failure": DESIGNED_FAILURE.get(e["op"]),
                       "source_id": r["id"], "article_id": r["article_id"], "kind": r["kind"],
                       "term_source": r.get("term_source"), "original_assertion": e["original"]}

    (out_dir / ITEMS_FILE).write_text(json.dumps(items, ensure_ascii=False, indent=2))
    (out_dir / KEY_FILE).write_text(json.dumps(key, ensure_ascii=False, indent=2))
    realised = Counter(k["op"] for k in key.values() if k["op"])
    meta = {"seed": seed, "source_graph": source, "pool_size": len(rows),
            "articles": sorted(articles), "n_clean": sum(1 for k in key.values() if k["origin"] == "clean"),
            "n_corrupt": sum(1 for k in key.values() if k["origin"] == "corrupt"),
            "target_mix": mix, "realised_mix": dict(realised.most_common()),
            "kinds": dict(Counter(i["kind"] for i in items))}
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    write_sheet(items, out_dir)

    print(f"pool {len(rows)} assertions over {len(graphs)} articles "
          f"({meta['kinds'].get('node', 0)} node / {meta['kinds'].get('relation', 0)} relation drawn)")
    print(f"built {len(items)} items: {meta['n_clean']} clean + {meta['n_corrupt']} corrupted")
    for op, c in realised.most_common():
        print(f"   {op:15} {c:3}   (target {mix.get(op, 0)})")
    short = {op: mix[op] - realised.get(op, 0) for op in mix if realised.get(op, 0) < mix[op]}
    if short:
        print(f"   ! short of target (pool ran dry): {short}")
    print("wrote", (out_dir / SHEET_FILE), "- label the `human` column (y/n), then score()")
    return items


# `label_source` is the provenance of the label in `human`, and it is load-bearing: a benchmark
# whose "human" column was in fact filled by a model measures agreement between two models, not
# against a human, and that distinction has to survive into whatever the thesis reports. Anything
# non-empty means NOT yet reviewed by a person — `score()` counts those separately and says so.
# `evidence` is the sentence the labeller grounded the decision on, verbatim from the article.
SHEET_COLS = ["index", "article_id", "kind", "human", "label_source", "evidence", "human_note",
              "assertion"]
LABEL_FIELDS = ("human", "label_source", "evidence", "human_note")


def write_sheet(items, out_dir):
    """CSV grouped by article: you read one article, then label everything drawn from it. Order
    inside an article stays the shuffled one, so clean and corrupted items stay interleaved."""
    out_dir = Path(out_dir)
    key = lambda x: ([int(t) if t.isdigit() else t
                      for t in re.split(r"(\d+)", x["article_id"])], x["index"])
    with (out_dir / SHEET_FILE).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLS, extrasaction="ignore")
        w.writeheader()
        for x in sorted(items, key=key):
            w.writerow(x)


def load_items(out_dir, with_labels=True):
    """The items, with any `human` labels from the CSV sheet merged in (the sheet wins — it is the
    file you actually edit; the JSON is the canonical record of what was asked)."""
    out_dir = Path(out_dir)
    items = load_json(out_dir / ITEMS_FILE)
    sheet = out_dir / SHEET_FILE
    if with_labels and sheet.exists():
        by_idx = {x["index"]: x for x in items}
        with sheet.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                x = by_idx.get(int(row["index"]))
                if x and (row.get("human") or "").strip():
                    x["human"] = row["human"].strip().lower()
                    for f in LABEL_FIELDS[1:]:
                        x[f] = row.get(f) or ""
    return items


def by_article(items):
    """[(article_id, [items…])] — the judge sees one article per call, so group before judging."""
    groups = defaultdict(list)
    for x in items:
        groups[x["article_id"]].append(x)
    return sorted(groups.items(),
                  key=lambda kv: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", kv[0])])


def save_verdicts(items, out_dir, tag, config=None):
    """Freeze one judge run. `tag` names the judge AND its sampling settings (e.g.
    "qwen3-32b-think_seed1234") — two seeds of the same model are two runs, and the gap between
    them is this benchmark's reproducibility measurement."""
    out_dir = Path(out_dir)
    verdicts = {str(x["index"]): {k: x.get("grounding", {}).get(k)
                                  for k in ("supported", "failure_type", "note", "quote",
                                            "quote_status")}
                for x in items}
    path = out_dir / VERDICTS_FILE.format(tag=tag)
    path.write_text(json.dumps({"tag": tag, "config": config or {}, "verdicts": verdicts},
                               ensure_ascii=False, indent=2))
    n = sum(v["supported"] is not None for v in verdicts.values())
    print(f"wrote {path.name}: {n}/{len(verdicts)} judged"
          f"{'' if n == len(verdicts) else '  ! some batches failed — re-run before scoring'}")
    return path


def verdict_tags(out_dir):
    prefix, suffix = VERDICTS_FILE.split("{tag}")
    return sorted(p.name[len(prefix):-len(suffix)]
                  for p in Path(out_dir).glob(VERDICTS_FILE.format(tag="*")))


# --- scoring --------------------------------------------------------------------------------

def _wilson(k, n, z=1.96):
    if not n:
        return (0.0, 1.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def gold_labels(out_dir, items=None):
    """Resolve every item to a gold label, or to a reason it has none.

    Corrupted items are gold-negative by construction — unless you labelled one "y", which means the
    edit landed on something the article supports anyway. Those are dropped and counted: a failed
    corruption is a bug in this module, not a mistake by the judge, and scoring it as a miss would
    flatter or damn the judge for the wrong reason.
    """
    out_dir = Path(out_dir)
    items = items if items is not None else load_items(out_dir)
    key = load_json(out_dir / KEY_FILE)
    gold, unlabelled, failed = {}, [], []
    for x in items:
        k = key[str(x["index"])]
        h = (x.get("human") or "").strip().lower()
        if k["origin"] == "corrupt":
            if h == "y":
                failed.append(x["index"])
            else:
                gold[x["index"]] = False
        elif h in ("y", "n"):
            gold[x["index"]] = (h == "y")
        else:
            unlabelled.append(x["index"])
    return gold, unlabelled, failed, key, items


def score(out_dir, tag=None, verbose=True, examples=5):
    """Score one judge run against the gold labels. The headline is **recall on the corrupted half**
    — how much of a deliberately broken graph this judge would catch."""
    out_dir = Path(out_dir)
    tag = tag or (verdict_tags(out_dir) or [None])[0]
    if not tag:
        print("no verdict file yet — run the judge and call save_verdicts() first")
        return
    run = load_json(out_dir / VERDICTS_FILE.format(tag=tag))
    verdicts = run["verdicts"]
    gold, unlabelled, failed, key, items = gold_labels(out_dir)
    by_idx = {x["index"]: x for x in items}

    scored = [(i, g, verdicts.get(str(i), {})) for i, g in sorted(gold.items())
              if verdicts.get(str(i), {}).get("supported") is not None]
    unjudged = len(gold) - len(scored)
    if not scored:
        print("nothing to score: no item has both a gold label and a verdict")
        return

    tp = sum(1 for _, g, v in scored if not g and not v["supported"])   # caught a bad assertion
    fn = sum(1 for _, g, v in scored if not g and v["supported"])       # waved a bad one through
    fp = sum(1 for _, g, v in scored if g and not v["supported"])       # rejected a good one
    tn = sum(1 for _, g, v in scored if g and v["supported"])
    n_bad, n_good = tp + fn, fp + tn

    rec, prec = (tp / n_bad if n_bad else None), (tp / (tp + fp) if tp + fp else None)

    # per corruption type — the diagnostic. A judge can look strong overall and be blind to one
    # class (number edits are the usual one), which pooled recall hides completely.
    per_op = defaultdict(lambda: [0, 0])
    for i, g, v in scored:
        op = key[str(i)]["op"]
        if op:
            per_op[op][0] += not v["supported"]
            per_op[op][1] += 1

    drafted = sum(1 for i, _, _ in scored
                  if (by_idx[i].get("label_source") or "").strip()
                  and key[str(i)]["origin"] == "clean")

    if verbose:
        print(f"=== {tag} ===")
        print(f"scored {len(scored)} items  ({n_bad} gold-unsupported, {n_good} gold-supported)")
        if drafted:
            print(f"  ! {drafted} of the clean labels are still model-drafted (label_source set) — "
                  f"this is model-vs-model agreement until a person reviews them")
        if unlabelled:
            print(f"  {len(unlabelled)} clean item(s) still unlabelled — excluded "
                  f"(label the `human` column in {SHEET_FILE})")
        if failed:
            print(f"  {len(failed)} corruption(s) you marked supported — excluded as failed corruptions "
                  f"(indices {failed[:12]}{'…' if len(failed) > 12 else ''})")
        if unjudged:
            print(f"  {unjudged} item(s) have no verdict (judge batch failed) — excluded")

        print(f"\n  RECALL on unsupported   {tp}/{n_bad} = {rec:.1%} {_ci(tp, n_bad)}"
              if n_bad else "\n  no gold-unsupported items")
        if n_good:
            print(f"  false-positive rate     {fp}/{n_good} = {fp/n_good:.1%} {_ci(fp, n_good)}"
                  f"   (good assertions the judge rejected)")
        if prec is not None:
            print(f"  precision               {tp}/{tp+fp} = {prec:.1%} {_ci(tp, tp + fp)}")
        if rec and prec:
            print(f"  F1                      {2*rec*prec/(rec+prec):.3f}")
        print(f"  accuracy                {(tp+tn)}/{len(scored)} = {(tp+tn)/len(scored):.1%}")

        if per_op:
            print("\n  recall by corruption type")
            for op, (k, n) in sorted(per_op.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1])):
                print(f"    {op:15} {k:3}/{n:<3} = {k/n:6.1%} {_ci(k, n)}")

        # did it catch it for the right reason?
        df = [(key[str(i)]["designed_failure"], v.get("failure_type"))
              for i, g, v in scored if not g and not v["supported"] and key[str(i)]["designed_failure"]]
        if df:
            ok = sum(a == b for a, b in df)
            print(f"\n  failure_type matches the edit's design: {ok}/{len(df)} = {ok/len(df):.1%}"
                  f"   {dict(Counter(f'{a}->{b}' for a, b in df).most_common(4))}")

        # a judge that "supports" a corrupted assertion AND locates a real sentence for it is grounding
        # on the neighbourhood rather than on the claim — a different failure from simply being lax
        hall = [(i, v) for i, g, v in scored if not g and v["supported"] and v.get("quote_status") == "found"]
        if fn:
            print(f"  of the {fn} missed, {len(hall)} came with a sentence that IS in the article "
                  f"(grounded on the neighbourhood, not on the claim)")

        per_kind = defaultdict(lambda: [0, 0])
        for i, g, v in scored:
            c = per_kind[key[str(i)]["kind"]]
            c[0] += (not g) == (not v["supported"]); c[1] += 1
        print("  by kind: " + "   ".join(f"{k} {a}/{b} = {a/b:.0%}" for k, (a, b) in sorted(per_kind.items())))

    if verbose and examples:
        miss = [(i, v) for i, g, v in scored if not g and v["supported"]][:examples]
        for i, v in miss:
            k = key[str(i)]
            print(f"\n  MISSED [{i}] {k['article_id']} ({k['op']}: {k['edit']})")
            print(f"    {by_idx[i]['assertion'].splitlines()[0][:110]}")
            print(f"    judge quote: {str(v.get('quote'))[:110]}")
        over = [(i, v) for i, g, v in scored if g and not v["supported"]][:max(1, examples // 2)]
        for i, v in over:
            print(f"\n  FALSE ALARM [{i}] {key[str(i)]['article_id']} "
                  f"({v.get('failure_type')}: {str(v.get('note'))[:70]})")
            print(f"    {by_idx[i]['assertion'].splitlines()[0][:110]}")

    return {"tag": tag, "n": len(scored), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": rec, "fpr": (fp / n_good if n_good else None), "precision": prec,
            "accuracy": (tp + tn) / len(scored),
            "by_op": {op: {"caught": k, "n": n} for op, (k, n) in per_op.items()},
            "unlabelled": len(unlabelled), "failed_corruptions": len(failed), "unjudged": unjudged,
            "model_drafted_labels": drafted}


def _ci(k, n):
    lo, hi = _wilson(k, n)
    return f"[{lo:.0%}, {hi:.0%}]"


def compare(out_dir, tags=None):
    """Every judge run side by side, plus pairwise verdict agreement.

    Two runs of the SAME model under different seeds bound this judge's reproducibility; two
    different models are the actual comparison. Both read off the same 500 items, which is the
    point of freezing them.
    """
    out_dir = Path(out_dir)
    tags = tags or verdict_tags(out_dir)
    if not tags:
        print("no verdict files in", out_dir)
        return
    rows = [score(out_dir, t, verbose=False, examples=0) for t in tags]
    rows = [r for r in rows if r]
    print(f"\n{'judge':38} {'n':>4} {'recall':>8} {'FPR':>7} {'prec':>7} {'acc':>7}")
    for r in rows:
        f = lambda v: "  -  " if v is None else f"{v:.1%}"
        print(f"{r['tag']:38} {r['n']:>4} {f(r['recall']):>8} {f(r['fpr']):>7} "
              f"{f(r['precision']):>7} {f(r['accuracy']):>7}")

    if len(tags) > 1:
        print("\npairwise verdict agreement (all judged items, gold-independent)")
        runs = {t: load_json(out_dir / VERDICTS_FILE.format(tag=t))["verdicts"] for t in tags}
        for a, b in [(a, b) for i, a in enumerate(tags) for b in tags[i + 1:]]:
            both = [i for i in runs[a] if runs[a][i]["supported"] is not None
                    and runs[b].get(i, {}).get("supported") is not None]
            same = sum(runs[a][i]["supported"] == runs[b][i]["supported"] for i in both)
            print(f"  {a}  vs  {b}:  {same}/{len(both)} = {same/max(1,len(both)):.1%}")
    return rows
