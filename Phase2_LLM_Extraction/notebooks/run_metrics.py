"""Every number a run produces, collected from the files it already wrote.

The notebooks print plenty while they run, but a printout dies with the kernel and a thesis table
needs the same quantity across a dozen configurations. This module reads a finished run's output
files and returns one metrics dict per stage, which the notebooks persist as

    results/<model>/_global/<track>/metrics_<stage><tag>.json

so a run's numbers survive, and `compare()` can put several runs side by side long after they were
executed. Nothing here calls an LLM or re-derives a decision: it only counts what is on disk.

Usage in a notebook (the reporting cell at the end of each stage):

    import run_metrics
    m = run_metrics.relations(RESULTS_DIR, articles=ARTICLE_ORDER, isfx=_isfx, osfx=_osfx,
                              track="e2e", experiment=EXPERIMENT)
    run_metrics.show(m)

From a shell, across experiments:

    python -m run_metrics compare gemini-2.5-flash e2e relations _dedup_main _dedup_no_rerank
    python -m run_metrics tracks  gemini-2.5-flash main_val          # staged vs e2e, repeat-aware
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

__all__ = ["extraction", "oie", "types", "relations", "show", "save", "load", "compare",
           "compare_tracks", "table", "TRACKS"]


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------
def _read(p: Path):
    """Parsed JSON, or None when the file is absent — a stage that was not run is not an error."""
    try:
        return json.loads(Path(p).read_text())
    except (FileNotFoundError, NotADirectoryError):
        return None


def _pct(num, den):
    return round(100.0 * num / den, 1) if den else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 2) if xs else None


# Where each track's per-article files and global vocabularies live. Three tracks, because the
# staged pipeline has two independent relation halves — the binary one (nb6 -> nb7) and the reified
# n-ary one (nb6b -> nb7b) — and they must not be summed together or overwrite each other's metrics.
TRACKS = {
    "e2e":        {"global": "_global/e2e", "graph": "e2e",
                   "ext": "nary_graph{osfx}_extended.json",
                   "review": "nary_relation_matching_review{osfx}.json",
                   "typed": "nary_graph{osfx}_typed.json",
                   "type_review": "entity_type_matching_review{osfx}.json",
                   "vocab": "nary_relation_vocabulary", "report": "nary_relation_consolidation_report"},
    "two_staged": {"global": "_global", "graph": "two_staged/3_nary_graph",
                   "ext": "nary_graph{osfx}_extended.json",
                   "review": "nary_relation_matching_review{osfx}.json",
                   "typed": "entities_typed{osfx}.json",
                   "type_review": "entity_type_matching_review{osfx}.json",
                   "vocab": "nary_relation_vocabulary", "report": "nary_relation_consolidation_report"},
    "binary":     {"global": "_global", "graph": "two_staged/2_binary_relations",
                   "ext": "relations{osfx}.json",
                   "review": "relation_matching_review{osfx}.json",
                   "typed": "entities_typed{osfx}.json",
                   "type_review": "entity_type_matching_review{osfx}.json",
                   "vocab": "relation_vocabulary", "report": "relation_consolidation_report"},
}


def _global_dir(results_dir: Path, track: str) -> Path:
    return Path(results_dir) / TRACKS[track]["global"]


def _tokens(paths) -> dict:
    """Sum the token ledgers a stage wrote (nb_parallel.usage_report / the batch cells).
    Returns zeros rather than None so a missing ledger reads as 'not recorded', not as free."""
    tot = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0,
           "total_tokens": 0, "llm_seconds": 0.0, "found": 0}
    for p in paths:
        d = _read(p)
        if not d:
            continue
        tot["found"] += 1
        t = d.get("totals") or {}
        for k in ("prompt_tokens", "output_tokens", "thoughts_tokens", "total_tokens"):
            tot[k] += t.get(k) or 0
        tot["calls"] += t.get("calls") or len(d.get("per_article") or [])
        tot["llm_seconds"] += t.get("llm_seconds") or t.get("seconds") or 0.0
    tot["llm_seconds"] = round(tot["llm_seconds"], 1)
    return tot


def _components(node_ids, edges):
    """Size of the largest connected component of the undirected graph, and the isolated-node count.
    Fragmentation is one of the things the staged/e2e comparison is actually about, so it is
    measured rather than eyeballed in the viewer."""
    adj = defaultdict(set)
    for s, o in edges:
        if s in node_ids and o in node_ids:
            adj[s].add(o); adj[o].add(s)
    seen, sizes = set(), []
    for n in node_ids:
        if n in seen:
            continue
        stack, size = [n], 0
        seen.add(n)
        while stack:
            cur = stack.pop(); size += 1
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb); stack.append(nb)
        sizes.append(size)
    return (max(sizes) if sizes else 0), sum(1 for s in sizes if s == 1)


# --------------------------------------------------------------------------------------------
# stage 1 — extraction  (6c single-call · 6b staged reification)
# --------------------------------------------------------------------------------------------
def extraction(results_dir, articles, sfx="", track="e2e", experiment=None, graph_name=None):
    """Shape and cost of the extracted graphs, per article and summed.

    `graph_name` overrides the filename when a track does not follow `nary_graph<sfx>.json`
    (the staged track writes `two_staged/3_nary_graph/nary_graph<sfx>.json`)."""
    results_dir = Path(results_dir)
    sub = TRACKS[track]["graph"]
    per, tok, reviews = [], Counter(), []
    for aid in articles:
        g = _read(results_dir / aid / sub / (graph_name or f"nary_graph{sfx}.json"))
        if not g:
            continue
        ents, nodes, rels = g.get("entities", []), g.get("nary_nodes", []), g.get("relations", [])
        ids = {e["id"] for e in ents} | {n["id"] for n in nodes}
        big, isolated = _components(ids, [(r.get("subject_id"), r.get("object_id")) for r in rels])
        kinds = Counter(n.get("node_kind") for n in nodes)
        u = (g.get("token_usage") or {}).get("combined") or {}
        for k in ("prompt_tokens", "output_tokens", "thoughts_tokens", "total_tokens"):
            tok[k] += u.get(k) or 0
        rs = g.get("review_stats") or {}
        reviews.append(rs)
        per.append({
            "article": aid,
            "entities": len(ents), "nary_nodes": len(nodes), "relations": len(rels),
            "node_kinds": dict(kinds),
            "main_events": sum(1 for n in nodes if n.get("is_main")),
            "distinct_entity_types": len({e.get("type") for e in ents}),
            "distinct_relation_phrases": len({r.get("relation_phrase") for r in rels}),
            "entities_with_normalized_value": sum(1 for e in ents if e.get("normalized_value")),
            "relations_per_node": round(len(rels) / len(ids), 2) if ids else None,
            "largest_component_share": _pct(big, len(ids)),
            "isolated_nodes": isolated,
            "mean_evidence_spans": _mean([len(r.get("evidence") or []) for r in rels]),
            "mean_confidence": _mean([r.get("confidence") for r in rels]),
            "reviewed": bool(g.get("review_notes")),
            # what the second call added to THIS article, so a single bad review is visible rather
            # than averaged away
            "review_added_relations": rs.get("added_relations"),
            "review_added_entities": rs.get("added_entities"),
            "review_relations_before": rs.get("relations_before"),
            "total_tokens": u.get("total_tokens"),
        })
    tot = {k: sum(r[k] for r in per) for k in ("entities", "nary_nodes", "relations")} if per else {}
    return _stamp("extraction", experiment, track, per, {
        **tot,
        "articles": len(per),
        "main_events": sum(r["main_events"] for r in per),
        "distinct_entity_types": sum(r["distinct_entity_types"] for r in per),
        "distinct_relation_phrases": sum(r["distinct_relation_phrases"] for r in per),
        "mean_largest_component_share": _mean([r["largest_component_share"] for r in per]),
        "isolated_nodes": sum(r["isolated_nodes"] for r in per),
        "mean_evidence_spans": _mean([r["mean_evidence_spans"] for r in per]),
        "reviewed_articles": sum(1 for r in per if r["reviewed"]),
        **_review_effect(reviews),
        **{k: v for k, v in tok.items()},
    })


# --------------------------------------------------------------------------------------------
# stage 0 — staged-track open extraction  (nb2 entities · nb6 binary relations)
# --------------------------------------------------------------------------------------------
def oie(results_dir, articles, osfx="", experiment=None):
    """The staged track's two open-extraction stages, which produce no n-ary graph and so are not
    covered by `extraction()`: nb2's entities+types and nb6's binary triples over them.

    Both are read if present; a stage that has not been run contributes zeros and shows up as 0 in
    `entity_files` / `relation_files`, so a partial pipeline is visible rather than silently averaged
    away."""
    results_dir = Path(results_dir)
    per, n_ent_files, n_rel_files = [], 0, 0
    for aid in articles:
        e = _read(results_dir / aid / "two_staged/1_entities" / f"oie_entities{osfx}.json")
        r = _read(results_dir / aid / "two_staged/2_binary_relations" / f"oie_relations{osfx}.json")
        if not e and not r:
            continue
        n_ent_files += bool(e); n_rel_files += bool(r)
        ents = (e or {}).get("entities", [])
        rels = (r or {}).get("relations", [])
        per.append({
            "article": aid,
            "entities": len(ents),
            "distinct_types": len({x.get("type") for x in ents}),
            "entities_with_normalized_value": sum(1 for x in ents if x.get("normalized_value")),
            "relations": len(rels),
            "distinct_phrases": len({x.get("relation_phrase") for x in rels}),
            "relations_per_entity": round(len(rels) / len(ents), 2) if ents else None,
            "mean_confidence": _mean([x.get("confidence") for x in rels]),
        })
    G = Path(results_dir) / "_global"
    summary = {
        "articles": len(per),
        "entity_files": n_ent_files, "relation_files": n_rel_files,
        "entities": sum(p["entities"] for p in per),
        "distinct_types": sum(p["distinct_types"] for p in per),
        "relations": sum(p["relations"] for p in per),
        "distinct_phrases": sum(p["distinct_phrases"] for p in per),
        "mean_relations_per_entity": _mean([p["relations_per_entity"] for p in per]),
        **_tokens([G / f"token_usage_nb2_oie_entities{osfx}.json",
                   G / f"token_usage_nb6_oie_relations{osfx}.json"]),
    }
    return _stamp("oie", experiment, "binary", per, summary)


# --------------------------------------------------------------------------------------------
# stage 2 — type canonicalization  (6d · nb4)
# --------------------------------------------------------------------------------------------
def types(results_dir, articles, isfx="", osfx="", track="e2e", experiment=None):
    """Reuse-vs-mint on the type side, plus what consolidation did to the grown vocabulary."""
    results_dir = Path(results_dir)
    G = _global_dir(results_dir, track)
    spec = TRACKS[track]
    sub = "e2e" if track == "e2e" else "two_staged/1_entities"
    review_name = spec["type_review"].format(osfx=osfx)
    typed_name = spec["typed"].format(osfx=osfx)

    # As on the relation side: the review file holds the DECISIONS, the typed graph the RESULT.
    # Measure each article from whichever exists, and report how many of each were found.
    per, ranks, chosen_found, decisions = [], [], 0, []
    src_types, src_entities = Counter(), Counter()
    n_reviews = n_typed = 0
    for aid in articles:
        rv = _read(results_dir / aid / sub / review_name)
        tg = _read(results_dir / aid / sub / typed_name)
        if not rv and not tg:
            continue
        n_reviews += bool(rv); n_typed += bool(tg)
        rows = (rv or {}).get("types", [])
        for r in rows:
            ch = r.get("chosen_type") or {}
            src = "mint" if ch.get("is_new_to_vocab") else (ch.get("source") or "?")
            src_types[src] += 1
            labels = [c.get("label") for c in (r.get("candidates") or [])]
            decisions.append((r.get("candidates"), ch.get("label"), ch.get("source")))
            if ch.get("label") in labels:               # did retrieval actually offer what was chosen?
                chosen_found += 1
                ranks.append(labels.index(ch["label"]) + 1)
        ents = (tg or {}).get("entities", [])
        for e in ents:
            src_entities["mint" if e.get("type_is_new_to_vocab") else (e.get("type_source") or "?")] += 1
        per.append({"article": aid, "distinct_oie_types": len(rows), "entities": len(ents),
                    "minted": sum(1 for r in rows if (r.get("chosen_type") or {}).get("is_new_to_vocab")),
                    "schema_org": sum(1 for r in rows
                                      if (r.get("chosen_type") or {}).get("source") == "schema.org")})

    _run = 0
    for r in per:      # running vocabulary size in ARTICLE ORDER: the growth curve (see relations())
        _run += r["minted"]
        r["cum_minted"] = _run

    raw = _read(G / f"type_vocabulary{osfx}.json") or {}
    con = _read(G / f"type_vocabulary{osfx}_consolidated.json") or {}
    rep = _read(G / f"type_consolidation_report{osfx}.json") or {}
    n_types = sum(src_types.values())

    summary = {
        "articles": len(per),
        "review_files": n_reviews, "typed_files": n_typed,
        "distinct_oie_types": n_types,
        "type_decisions": dict(src_types),
        "schema_org_reuse_pct": _pct(src_types.get("schema.org", 0), n_types),
        "mint_pct": _pct(src_types.get("mint", 0), n_types),
        "entities_typed": sum(src_entities.values()),
        "entity_type_sources": dict(src_entities),
        "entity_schema_org_pct": _pct(src_entities.get("schema.org", 0), sum(src_entities.values())),
        # retrieval quality: the LLM can only reuse what the shortlist showed it
        "chosen_in_candidates_pct": _pct(chosen_found, n_types),
        "mean_rank_of_chosen": _mean(ranks),
        "chosen_at_rank1_pct": _pct(sum(1 for r in ranks if r == 1), len(ranks)),
        # what the cross-encoder promoted into that shortlist, and whether it was taken
        **_rerank_effect(decisions),
        # vocabulary economy
        "vocab_raw": _raw_vocab_size(rep, "n_types_removed", len(raw.get("types", []))),
        "vocab_consolidated": len(con.get("types", [])) if con else None,
        "vocab_reduction_pct": _pct(
            _raw_vocab_size(rep, "n_types_removed", len(raw.get("types", []))) - len(con.get("types", [])),
            _raw_vocab_size(rep, "n_types_removed", len(raw.get("types", [])))) if con and raw else None,
        **_consolidation_summary(rep, removed_key="n_types_removed"),
        **_tokens([G / f"token_usage_6d_type_canon{osfx}.json",
                   G / f"token_usage_types{osfx}.json"]),
    }
    return _stamp("types", experiment, track, per, summary)


# --------------------------------------------------------------------------------------------
# stage 3 — relation canonicalization  (6e · nb7 · nb7b)
# --------------------------------------------------------------------------------------------
def relations(results_dir, articles, isfx="", osfx="", track="e2e", experiment=None):
    """Reuse-vs-mint on the property side, phrase splitting, inverse reuse, dedup, consolidation."""
    results_dir = Path(results_dir)
    G = _global_dir(results_dir, track)
    spec = TRACKS[track]
    sub = spec["graph"]
    ext_name = spec["ext"].format(osfx=osfx)
    review_name = spec["review"].format(osfx=osfx)
    vocab_stem, report_stem = spec["vocab"], spec["report"]

    # The review file carries the DECISIONS (candidates offered, assignment split, direction) and
    # the extended graph carries the RESULT (the deduped triples). A consolidation-only re-run
    # writes only the second, so each article is measured from whichever of the two it has, and
    # the summary says how many of each were found rather than silently reporting zeros.
    per, ranks, chosen_found, n_phrases = [], [], 0, 0
    decisions = []
    prop_src, dirs = Counter(), Counter()
    n_reviews = n_exts = 0
    for aid in articles:
        rv = _read(results_dir / aid / sub / review_name)
        ex = _read(results_dir / aid / sub / ext_name)
        # The consolidated graph is the one that gets reported; the raw one is what the per-article
        # build produced. Reading both is what turns consolidation's counts (a vocabulary story) into
        # its effect on the TRIPLES (the story the write-up is actually making).
        co = _read(results_dir / aid / sub / ext_name.replace(".json", "_consolidated.json"))
        if not rv and not ex:
            continue
        rows = (rv or {}).get("relations", [])
        n_reviews += bool(rv)
        n_exts += bool(ex)
        n_phrases += len(rows)
        for r in rows:
            labels = [c.get("label") for c in (r.get("candidates") or [])]
            for a in r.get("assignments", []):
                ch = a.get("chosen_property") or {}
                prop_src["mint" if ch.get("is_new_to_vocab") else (ch.get("source") or "?")] += 1
                dirs[ch.get("direction") or "as_is"] += 1
                decisions.append((r.get("candidates"), ch.get("label"), ch.get("source")))
                if ch.get("label") in labels:
                    chosen_found += 1
                    ranks.append(labels.index(ch["label"]) + 1)
        recs = (ex or {}).get("relations", [])
        crecs = (co or {}).get("relations", [])
        # 6e rewrites properties in place, so the two files are index-aligned; when they are not
        # (a partial re-run), fall back to the aggregate counts and report no per-triple diff.
        aligned = bool(crecs) and len(crecs) == len(recs)
        prop_changed = sum(1 for a, b in zip(recs, crecs) if a.get("property") != b.get("property")) \
            if aligned else None
        to_schema = sum(1 for a, b in zip(recs, crecs)
                        if a.get("property_source") != "schema.org"
                        and b.get("property_source") == "schema.org") if aligned else None
        per.append({
            "article": aid,
            "phrases": len(rows),
            "mints": sum(1 for r in rows for a in r.get("assignments", [])
                         if (a.get("chosen_property") or {}).get("is_new_to_vocab")),
            "split_phrases": sum(1 for r in rows if (r.get("n_assignments") or 1) > 1),
            "instances": sum(len(a.get("instances") or []) for r in rows
                             for a in r.get("assignments", [])),
            "triples": len(recs),
            "merged_triples": sum(1 for r in recs if len(r.get("relation_phrases") or []) > 1
                                  or len(r.get("evidence") or []) > 1),
            "inverse_reused": sum(1 for r in recs
                                  if "inverse" in (r.get("phrase_directions") or {}).values()),
            "schema_org_triples": sum(1 for r in recs if r.get("property_source") == "schema.org"),
            "vocab_triples": sum(1 for r in recs if r.get("property_source") == "vocab"),
            "triples_consolidated": len(crecs),
            "schema_org_triples_consolidated": sum(1 for r in crecs
                                                   if r.get("property_source") == "schema.org"),
            "cons_property_changed": prop_changed,
            "cons_triples_to_schema_org": to_schema,
            # running vocabulary size after this article: canonicalization is incremental, so this
            # is the growth curve — the evidence for (or against) minting saturating on a corpus.
            "cum_minted": None,
        })

    raw = _read(G / f"{vocab_stem}{osfx}.json") or {}
    con = _read(G / f"{vocab_stem}{osfx}_consolidated.json") or {}
    rep = _read(G / f"{report_stem}{osfx}.json") or {}
    n_assign = sum(prop_src.values())
    n_trip = sum(r["triples"] for r in per)
    n_sch = sum(r["schema_org_triples"] for r in per)
    n_trip_c = sum(r["triples_consolidated"] for r in per)
    n_sch_c = sum(r["schema_org_triples_consolidated"] for r in per)
    _run = 0
    for r in per:                       # cumulative mints in ARTICLE ORDER — the order IS the run
        _run += r["mints"] if r.get("mints") is not None else 0
        r["cum_minted"] = _run

    summary = {
        "articles": len(per),
        "review_files": n_reviews, "extended_files": n_exts,
        "distinct_phrases": n_phrases,
        "split_phrases": sum(r["split_phrases"] for r in per),
        "instances": sum(r["instances"] for r in per),
        "triples": n_trip,
        "merged_triples": sum(r["merged_triples"] for r in per),
        "inverse_reused": sum(r["inverse_reused"] for r in per),
        "assignment_sources": dict(prop_src),
        "assignment_mint_pct": _pct(prop_src.get("mint", 0), n_assign),
        # BEFORE global consolidation. The number to report is the `_consolidated` one below —
        # consolidation is what folds mints into schema.org, and it is also what absorbs most of the
        # run-to-run sampling noise. Both are kept so the difference is visible rather than assumed.
        "triple_schema_org_pct": _pct(n_sch, n_trip),
        "triples_consolidated": n_trip_c,
        "triple_schema_org_pct_consolidated": _pct(n_sch_c, n_trip_c),
        "cons_triples_property_changed": sum(r["cons_property_changed"] or 0 for r in per),
        "cons_triples_property_changed_pct": _pct(sum(r["cons_property_changed"] or 0 for r in per),
                                                  n_trip_c),
        "cons_triples_to_schema_org": sum(r["cons_triples_to_schema_org"] or 0 for r in per),
        "directions": dict(dirs),
        # retrieval quality
        "chosen_in_candidates_pct": _pct(chosen_found, n_assign),
        "mean_rank_of_chosen": _mean(ranks),
        "chosen_at_rank1_pct": _pct(sum(1 for r in ranks if r == 1), len(ranks)),
        # what the cross-encoder promoted into that shortlist, and whether it was taken
        **_rerank_effect(decisions),
        # vocabulary economy
        "vocab_raw": _raw_vocab_size(rep, "n_props_removed", len(raw.get("properties", []))),
        "vocab_consolidated": len(con.get("properties", [])) if con else None,
        "vocab_reduction_pct": _pct(
            _raw_vocab_size(rep, "n_props_removed", len(raw.get("properties", []))) - len(con.get("properties", [])),
            _raw_vocab_size(rep, "n_props_removed", len(raw.get("properties", [])))) if con and raw else None,
        **_consolidation_summary(rep, removed_key="n_props_removed"),
        **_tokens([G / f"token_usage{osfx}.json", G / f"token_usage_relations{osfx}.json"]),
    }
    return _stamp("relations", experiment, track, per, summary)


def _rerank_effect(decisions: list) -> dict:
    """What the inline cross-encoder actually changed, from the shortlists the run already saved.

    `rerank_candidates` stamps every surviving candidate with the rank e5 had given it
    (`embed_rank`), so a candidate whose embed_rank is deeper than the shortlist is one the LLM would
    NOT have seen without the cross-encoder. That makes the reranker's effect measurable INSIDE the
    reported run, per decision — which the `no-rerank` ablation cannot do: the ablation shows the
    end-to-end difference between two runs, not which candidate the promotion put in front of the
    model, nor whether the model then took it.

    The number to quote for "what did the reranker buy" is `rr_chosen_promoted_schema`: decisions
    where a schema.org term the bi-encoder had ranked too deep to show was promoted into the
    shortlist AND chosen — reuse that would otherwise have been a mint.

    `decisions` is a list of (candidates, chosen_label, chosen_source) per selection call.
    """
    shown = promoted = promoted_schema = 0
    chosen_promoted = chosen_promoted_schema = 0
    n_with_chosen = 0
    embed_ranks, gains = [], []
    reranked = False
    for cands, label, source in decisions:
        cands = cands or []
        depth = len(cands)
        if not any(c.get("embed_rank") is not None for c in cands):
            continue                       # this decision ran without the cross-encoder
        reranked = True
        shown += depth
        for c in cands:
            er = c.get("embed_rank")
            if er is not None and er > depth:
                promoted += 1
                promoted_schema += c.get("source") == "schema.org"
        hit = next((i for i, c in enumerate(cands) if c.get("label") == label), None)
        if hit is None:
            continue                       # the LLM minted, or chose something not in the shortlist
        n_with_chosen += 1
        er = cands[hit].get("embed_rank")
        if er is not None:
            embed_ranks.append(er)
            gains.append(er - (hit + 1))
            if er > depth:
                chosen_promoted += 1
                chosen_promoted_schema += (source or cands[hit].get("source")) == "schema.org"
    if not reranked:
        return {"rr_reranked": False}
    return {
        "rr_reranked": True,
        "rr_shown_candidates": shown,
        # how much of the shortlist the cross-encoder rewrote, and how much of that was schema.org
        "rr_promoted": promoted,
        "rr_promoted_pct": _pct(promoted, shown),
        "rr_promoted_schema_org": promoted_schema,
        "rr_promoted_schema_org_pct": _pct(promoted_schema, promoted),
        # ... and how often the promotion was the thing the LLM then picked
        "rr_decisions_with_chosen_in_shortlist": n_with_chosen,
        "rr_chosen_promoted": chosen_promoted,
        "rr_chosen_promoted_pct": _pct(chosen_promoted, n_with_chosen),
        "rr_chosen_promoted_schema": chosen_promoted_schema,
        "rr_mean_embed_rank_of_chosen": _mean(embed_ranks),
        "rr_mean_positions_gained": _mean(gains),
    }


def _review_effect(stats: list) -> dict:
    """What the second (review/repair) call added, summed over the articles that ran it.

    Read from `review_stats`, which 6c/nb6b write into each graph. The `no-review` ablation measures
    the DOWNSTREAM consequence of dropping the pass; this measures the pass itself — how many facts
    it recovered from the text the first call had missed, on the very run being reported.
    """
    stats = [s for s in stats if s]
    if not stats:
        return {"review_ran": False}
    ok = [s for s in stats if s.get("patch_parsed")]

    def tot(k):
        return sum(s.get(k) or 0 for s in stats)

    rel_before, rel_after = tot("relations_before"), tot("relations_after")
    return {
        "review_ran": True,
        "review_articles": len(stats),
        "review_failed_articles": len(stats) - len(ok),
        "review_added_entities": tot("added_entities"),
        "review_added_nary_nodes": tot("added_nary_nodes"),
        "review_added_relations": tot("added_relations"),
        "review_removed_nary_nodes": tot("removed_nary_nodes"),
        "review_removed_relations": tot("removed_relations"),
        "review_fixed_evidence": tot("fixed_evidence"),
        "review_guard_dropped": tot("guard_dropped"),
        "review_relations_before": rel_before,
        "review_relations_after": rel_after,
        # the headline: the share of the graph that exists only because of the second call
        "review_relations_delta_pct": _pct(rel_after - rel_before, rel_before),
        "review_nodes_delta_pct": _pct(tot("nary_nodes_after") - tot("nary_nodes_before"),
                                       tot("nary_nodes_before")),
        "review_entities_delta_pct": _pct(tot("entities_after") - tot("entities_before"),
                                          tot("entities_before")) if tot("entities_before") else None,
    }


def _raw_vocab_size(rep: dict, removed_key: str, from_file: int) -> int:
    """How big the vocabulary would be with nothing collapsed.

    With `consolidate_every` on, each pass feeds its merged result back as the working vocabulary, so
    the "raw" file on disk is NOT the uncollapsed vocabulary any more — it is whatever survived the
    last pass. Reading it makes the reduction look like nothing happened (59 -> 58 when 22 terms had
    in fact been removed). For a periodic run the uncollapsed size is survivors + everything removed,
    which equals every id ever issued.
    """
    if rep.get("consolidation_passes"):
        return (rep.get("vocab_size_after") or 0) + (rep.get(removed_key) or 0)
    return from_file


def _consolidation_summary(rep: dict, removed_key: str) -> dict:
    """The consolidation report's headline counts, flattened with a `cons_` prefix. Includes the
    `in_shortlist` audit: what share of the pairs the judge merged were ones the embedder had
    actually proposed — a low share means the judge is reaching past its shortlist."""
    if not rep:
        return {"consolidated": False}
    flags = [m.get("in_shortlist") for r in rep.get("merges", []) for m in r.get("merged", [])]
    flags += [c.get("in_shortlist") for r in rep.get("reparents", []) for c in r.get("children", [])]
    flags = [f for f in flags if f is not None]
    return {
        "consolidated": True,
        "cons_calls": 2 if rep.get("two_call_consolidation") else 1,
        "cons_pool": rep.get("candidate_pool"),
        "cons_reranked": rep.get("consolidation_reranked"),
        "cons_vocab_before": rep.get("vocab_size_before"),
        "cons_vocab_after": rep.get("vocab_size_after"),
        "cons_merge_groups": rep.get("n_merges"),
        "cons_removed": rep.get(removed_key),
        "cons_merged_into_schema": rep.get("n_merged_into_schema"),
        "cons_reparented": rep.get("n_reparented"),
        "cons_reparented_under_schema": rep.get("n_reparented_under_schema"),
        "cons_generalize_groups": rep.get("n_generalize_groups"),
        "cons_generalized": rep.get("n_generalized"),
        "cons_blocked_degrounding": rep.get("n_blocked_degrounding"),
        "cons_parent_conflicts": rep.get("n_parent_conflicts"),
        "cons_keep_overrides": rep.get("n_keep_overrides"),
        "cons_offkind_dropped": rep.get("n_offkind_dropped"),
        "cons_self_parent_fixed": len(rep.get("self_parent_fixed") or []),
        "cons_in_shortlist_pct": _pct(sum(1 for f in flags if f), len(flags)),
    }


def _stamp(stage, experiment, track, per_article, summary):
    import time
    return {"stage": stage, "experiment": experiment, "track": track,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "summary": summary, "per_article": per_article}


# --------------------------------------------------------------------------------------------
# persist / present
# --------------------------------------------------------------------------------------------
def save(metrics: dict, results_dir, osfx="", track="e2e", section=None, articles=None,
         manifest=True) -> Path:
    """Write to `_global[/e2e]/metrics_<stage><osfx>.json`, and CLOSE this stage's run manifest.

    The metrics file answers "what were the numbers"; the manifest answers "of what run" — the
    config, the commit, the corpus order, the files, the token cost. They are written together
    because a metrics file that outlives the memory of how it was produced is the exact failure
    this pipeline kept hitting. `manifest=False` writes the numbers only.
    """
    p = _global_dir(Path(results_dir), track) / f'metrics_{metrics["stage"]}{osfx}.json'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    if manifest:
        try:
            import run_manifest
            # osfx + track scope the manifest's output scan to THIS run's files, so a second run
            # on another GPU sharing this model tree cannot be credited with them.
            try:
                run_manifest.finish(metrics=metrics, section=section, results_dir=results_dir,
                                    articles=articles, osfx=osfx, track=track)
            except TypeError:
                # A kernel that imported run_manifest BEFORE osfx/track existed: this module gets
                # reloaded by the notebooks' metrics cell, run_manifest does not (reloading it would
                # wipe the open stage it is holding). Falling back keeps a finished stage from dying
                # at its last line over a stale import — which cost a completed 6d once.
                print("  (stale run_manifest in this kernel — closing without output scoping)")
                run_manifest.finish(metrics=metrics, section=section, results_dir=results_dir,
                                    articles=articles)
        except ImportError:
            pass
    return p


def load(results_dir, stage, osfx="", track="e2e"):
    return _read(_global_dir(Path(results_dir), track) / f"metrics_{stage}{osfx}.json")


def show(metrics: dict, per_article=True, width=34):
    """Print one stage's metrics: the per-article table, then every summary number."""
    s, per = metrics["summary"], metrics["per_article"]
    head = f'{metrics["stage"].upper()}  ·  {metrics.get("track")}  ·  experiment {metrics.get("experiment")!r}'
    print(head + "\n" + "=" * max(len(head), 74))
    if per_article and per:
        cols = [k for k in per[0] if k != "article" and not isinstance(per[0][k], dict)]
        print(f'{"article":<9}' + "".join(f"{c[:13]:>15}" for c in cols))
        for r in per:
            print(f'{r["article"]:<9}' + "".join(f'{str(r.get(c)):>15}' for c in cols))
        print("-" * max(len(head), 74))
    for k, v in s.items():
        print(f"  {k:<{width}} {v}")


def table(rows: list[dict], keys: list[str], label_key="experiment"):
    """One line per run, `keys` across the columns — the shape a thesis table wants."""
    w = max(len(str(r.get(label_key, ""))) for r in rows) + 2 if rows else 12
    out = [f'{label_key:<{w}}' + "".join(f"{k[:15]:>17}" for k in keys)]
    for r in rows:
        out.append(f'{str(r.get(label_key, "")):<{w}}'
                   + "".join(f'{str(r.get(k)):>17}' for k in keys))
    return "\n".join(out)


DEFAULT_COMPARE_KEYS = {
    "oie": ["entities", "distinct_types", "relations", "distinct_phrases", "total_tokens"],
    "extraction": ["entities", "nary_nodes", "relations", "isolated_nodes",
                   "mean_largest_component_share", "total_tokens"],
    "types": ["distinct_oie_types", "schema_org_reuse_pct", "mint_pct", "vocab_raw",
              "vocab_consolidated", "chosen_at_rank1_pct", "total_tokens"],
    "relations": ["distinct_phrases", "triples", "triple_schema_org_pct", "assignment_mint_pct",
                  "vocab_raw", "vocab_consolidated", "inverse_reused", "total_tokens"],
}


def compare(results_dir, stage, tags, track="e2e", keys=None):
    """Several finished runs side by side, read from their saved metrics files.

    `tags` are output suffixes (`_osfx`), i.e. what each run wrote — e.g. `_dedup_main`,
    `_dedup_no_rerank`. Runs with no metrics file are skipped rather than faked."""
    keys = keys or DEFAULT_COMPARE_KEYS.get(stage, [])
    rows = []
    for t in tags:
        m = load(results_dir, stage, t, track)
        if not m:
            continue
        rows.append({"experiment": m.get("experiment") or t, **m["summary"]})
    return table(rows, keys)


# --------------------------------------------------------------------------------------------
# staged vs e2e — the decision that lets the test runs be e2e only
# --------------------------------------------------------------------------------------------
# Each key with the direction that counts as BETTER, or None where "more" is not self-evidently
# better and the number is reported rather than scored. Raw graph size is the main one: a track that
# emits more relations may be finding more or may be splitting the same facts more finely, and only
# grounding precision / MINE recall can tell those apart — so it is printed, never scored.
_TRACK_KEYS = {
    "extraction": [("relations", None), ("nary_nodes", None), ("entities", None),
                   ("isolated_nodes", "lower"), ("mean_largest_component_share", "higher"),
                   ("mean_evidence_spans", "higher"), ("total_tokens", "lower")],
    "types": [("distinct_oie_types", None), ("schema_org_reuse_pct", "higher"),
              ("vocab_consolidated", "lower"), ("chosen_at_rank1_pct", "higher"),
              ("total_tokens", "lower")],
    "relations": [("triples", None), ("triple_schema_org_pct_consolidated", "higher"),
                  ("vocab_consolidated", "lower"), ("chosen_at_rank1_pct", "higher"),
                  ("inverse_reused", None), ("total_tokens", "lower")],
}


def _repeat_tags(variant, base, repeats):
    """The osfx of each repeat of one arm: `_dedup_main_val_r1`, `_r2`, ... (no repeats -> the bare
    tag). The spread across these is the only thing that makes a track difference interpretable."""
    if not repeats:
        return [f"_{variant}_{base}"]
    return [f"_{variant}_{base}_r{n}" for n in repeats]


def _agg(values):
    """-> (mean, min, max, n) over the repeats that exist."""
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None, None, None, 0
    return round(statistics.fmean(vals), 2), min(vals), max(vals), len(vals)


def compare_tracks(results_dir, variant="dedup", base="main_val", repeats=(1, 2, 3),
                   staged_track="two_staged"):
    """staged vs e2e on the validation split, as the table that decides whether test can be e2e only.

    The comparison is only meaningful against the run-to-run spread: identical settings on this
    corpus have produced 69 vs 91 raw minted properties, so a gap narrower than the repeats' own
    range is not a gap. Each scored key is therefore marked:

        better / worse   the tracks' ranges do NOT overlap, and e2e is on the good side (or not)
        ~within spread   the ranges overlap — the tracks are indistinguishable on this key
        (report)         no self-evident direction; printed for the write-up, not scored

    NOT included, and decisive: grounding precision (nb9/9b), NLI (9c) and MINE recall (nb10). This
    reads the pipeline's own metrics files, which measure economy and shape, never correctness. A
    track that wins every row here can still lose on grounding — check those before concluding.
    """
    out = [f"staged ({staged_track}) vs e2e   ·   variant {variant!r}   ·   arm {base!r}   ·   "
           f"repeats {list(repeats) or '(none)'}", "=" * 100]
    verdicts = []
    for stage, keys in _TRACK_KEYS.items():
        rows = {}
        for track in ("e2e", staged_track):
            # 6c writes its extraction metrics under the VARIANT alone (no run tag exists yet), so
            # the e2e extraction of every repeat of an arm is the same one file.
            tags = ([f"_{variant}"] if (track == "e2e" and stage == "extraction")
                    else _repeat_tags(variant, base, repeats))
            mets = [m for m in (load(results_dir, stage, t, track) for t in tags) if m]
            rows[track] = mets
        if not any(rows.values()):
            continue
        out.append(f"\n{stage.upper()}   e2e: {len(rows['e2e'])} run(s)   "
                   f"{staged_track}: {len(rows[staged_track])} run(s)")
        out.append(f'  {"metric":<38} {"e2e (mean [min-max])":<26} {"staged (mean [min-max])":<26} verdict')
        for key, direction in keys:
            a = _agg([m["summary"].get(key) for m in rows["e2e"]])
            b = _agg([m["summary"].get(key) for m in rows[staged_track]])
            if a[3] == 0 and b[3] == 0:
                continue

            def fmt(t):
                return "—" if t[3] == 0 else (f"{t[0]} [{t[1]}-{t[2]}]" if t[3] > 1 else f"{t[0]}")

            if direction is None or a[3] == 0 or b[3] == 0:
                verdict = "(report)"
            elif a[1] > b[2] or a[2] < b[1]:          # ranges disjoint: a real difference
                e2e_higher = a[0] > b[0]
                good = e2e_higher if direction == "higher" else not e2e_higher
                verdict = "e2e BETTER" if good else "e2e worse"
                verdicts.append((stage, key, good))
            else:
                verdict = "~within spread"
                verdicts.append((stage, key, None))
            out.append(f"  {key:<38} {fmt(a):<26} {fmt(b):<26} {verdict}")

    scored = [v for _, _, v in verdicts if v is not None]
    out.append("\n" + "=" * 100)
    if not scored:
        out.append("  every scored key is within the repeats' own spread — on economy and shape the "
                   "two tracks are\n  indistinguishable on this arm. That IS the 'e2e is close' "
                   "result; decide on grounding and cost.")
    else:
        out.append(f"  scored keys: {sum(1 for v in scored if v)} favour e2e, "
                   f"{sum(1 for v in scored if not v)} favour {staged_track}, "
                   f"{sum(1 for _, _, v in verdicts if v is None)} within spread.")
    out.append("  Grounding (nb9/9b), NLI (9c) and MINE recall (nb10) are NOT in this table and are "
               "the correctness\n  half of the decision — add them before concluding.")
    return "\n".join(out)


def _main(argv):
    if len(argv) >= 3 and argv[1] == "tracks":
        _, _, model, *rest = argv
        base = rest[0] if rest else "main_val"
        variant = rest[1] if len(rest) > 1 else "dedup"
        reps = tuple(int(x) for x in rest[2].split(",")) if len(rest) > 2 else (1, 2, 3)
        root = Path(__file__).resolve().parent.parent / "results" / model
        print(compare_tracks(root, variant=variant, base=base, repeats=reps))
        return 0
    if len(argv) >= 5 and argv[1] == "compare":
        _, _, model, track, stage, *tags = argv
        root = Path(__file__).resolve().parent.parent / "results" / model
        print(compare(root, stage, tags, track))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv))
