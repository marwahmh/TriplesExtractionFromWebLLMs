"""The full-article grounding judge — the one implementation, shared by nb9b and nb9d.

nb9b **runs** the judge over a whole split and prunes what it rejects; nb9d **measures** the judge
against a gold benchmark. Those two only mean anything together if they are the same judge, so the
prompt, the verbalisers, the quote matcher and the batching all live here rather than in either
notebook. A prompt edited in one place is measured in the other; that is the entire point.

Usage from a notebook:

    import grounding_core as gc
    gc.set_llm(gemini_json)                    # whatever backend the notebook activated
    gc.configure(judge_batch=20)               # optional knobs

    gc.set_article(article_text)               # thread-local; do this per article
    gc.run_judge(items, gc.verbalise_node, "nodes")        # verdicts attached in place

Nothing here imports IPython or touches the results/ layout, so it runs the same under Jupyter,
a SLURM job, or pytest.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# --- knobs ------------------------------------------------------------------------------------
# Citation/quote matching. These no longer decide what the judge SEES (it gets the whole article);
# they only score the extractor's quotes and locate the judge's returned sentence.
FUZZY_THRESHOLD = 0.85      # difflib ratio; >= this against the best window -> "partial"
MAX_SPAN_CHARS = 400        # elided quote whose fragments span more than this is not one quote
MAX_SENTENCE_CHARS = 600    # cap when widening a located quote to its sentence (reporting only)

# The article is re-sent with every batch (~1.2k tokens for a news story), so batching costs almost
# nothing on the input side — it exists to keep the OUTPUT list short, which is where a long run of
# verdicts degrades or hits MAX_TOKENS. 60 assertions ~ 2k output tokens; use 20 for a THINKING
# judge, which reasons about every assertion in the batch before emitting any of them.
JUDGE_BATCH = 60
JUDGE_RETRIES = 3
JUDGE_RETRY_BACKOFF = 5     # seconds; linear backoff = BACKOFF * attempt

_LLM = None                      # process-wide default
_TL = threading.local()          # per-thread override, for judging with several backends at once


def set_llm(fn):
    """Point the judge at a backend: any `f(system_prompt, user_message) -> parsed JSON`."""
    global _LLM
    _LLM = fn
    return fn


def set_thread_llm(fn):
    """Bind a backend to THIS thread only, leaving other threads alone.

    Needed to benchmark two models at the same time: without it every worker shares one module-level
    client, so the second judge activated silently redirects the first one's calls. Use together with
    `llm_backend.make_llm`, which returns a client bound to one server rather than mutating globals.
    """
    _TL.llm = fn
    return fn


def _llm():
    fn = getattr(_TL, "llm", None) or _LLM
    if fn is None:
        raise RuntimeError("no backend set — call grounding_core.set_llm(...) first")
    return fn


def configure(**kw):
    """Set knobs by name: configure(judge_batch=20, fuzzy_threshold=0.9)."""
    g = globals()
    for k, v in kw.items():
        key = k.upper()
        if key not in g:
            raise KeyError(f"unknown knob {k!r}")
        g[key] = v
    return {k: g[k.upper()] for k in kw}


# --- text normalisation and quote location ----------------------------------------------------

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                         "–": "-", "—": "-", "…": "..."})


def fix_mojibake(s: str) -> str:
    """Undo UTF-8-read-as-Latin-1 damage ("Â£" -> "£") seen in some scraped article files.
    Cheap and safe: only re-decodes when the tell-tale Â/â sequences are present."""
    if not s or ("Â" not in s and "ã" not in s and "â" not in s):
        return s
    try:
        return s.encode("latin-1", "ignore").decode("utf-8", "ignore")
    except Exception:
        return s


def norm_text(s: str) -> str:
    """Normalise for matching: repair mojibake, unify quotes/dashes, collapse space, casefold."""
    s = fix_mojibake(s or "")
    s = unicodedata.normalize("NFKC", s).translate(_QUOTES)
    return re.sub(r"\s+", " ", s).strip().casefold()


def build_index(raw: str):
    """Normalise the article while keeping a map from every normalised char back to its RAW offset.
    That map is what lets a matched quote be resolved to real character offsets in the article."""
    raw = fix_mojibake(raw)
    chars, idx, prev_space = [], [], False
    for i, ch in enumerate(unicodedata.normalize("NFKC", raw).translate(_QUOTES)):
        if ch.isspace():
            if prev_space or not chars:
                continue
            chars.append(" "); idx.append(i); prev_space = True
        else:
            chars.append(ch.casefold()); idx.append(i); prev_space = False
    return "".join(chars), idx, raw


# The extractor sometimes elides the middle of a long quote ("Belgium recorded ... 39% more than
# usual"). Rather than failing those, split on the ellipsis and locate the fragments in order --
# the span from the first to the last fragment IS the real quote, elided middle included.
_ELLIPSIS_RE = re.compile(r"\s*(?:\.\.\.|…)\s*")


def best_window_ratio(needle: str, haystack: str) -> float:
    """Best difflib ratio of `needle` against any same-length window of `haystack` (coarse stride)."""
    n = len(needle)
    if n == 0 or len(haystack) < n:
        return 0.0
    stride = max(1, n // 4)
    best = 0.0
    for i in range(0, len(haystack) - n + 1, stride):
        r = SequenceMatcher(None, needle, haystack[i:i + n]).ratio()
        if r > best:
            best = r
            if best >= 0.99:
                break
    return best


# Punctuation a quote may legitimately differ by at its edges. A judge quoting a clause out of a
# sentence with attribution ends it with a full stop where the article carries on with a comma
# ("... actually healthy." vs "... actually healthy," she said). That is a faithful contiguous copy,
# not an invention, and treating it as a failed match mislabels ~20% of all quotes as untrustworthy.
_EDGE_PUNCT = ' .,;:!?"\'-'


def locate_quote(qn: str, art_norm: str, idx_map):
    """Find one normalised quote in the article. -> (status, ratio, (start, end) | None) in RAW offsets."""
    if not qn:
        return "missing", 0.0, None
    for cand in (qn, qn.strip(_EDGE_PUNCT)):
        if cand and cand in art_norm:
            s = art_norm.index(cand)
            return "found", 1.0, (idx_map[s], idx_map[s + len(cand) - 1] + 1)

    frags = [f for f in _ELLIPSIS_RE.split(qn) if len(f) > 3]
    if len(frags) > 1:                                   # elided quote: walk the fragments in order
        pos, first, last = 0, None, None
        for f in frags:
            i = art_norm.find(f, pos)
            if i < 0:
                first = None; break
            if first is None:
                first = i
            last = i + len(f); pos = last
        if first is not None:
            span = (idx_map[first], idx_map[last - 1] + 1)
            if span[1] - span[0] <= MAX_SPAN_CHARS:      # guard: fragments matched far apart
                return "partial", 1.0, span
            return "missing", 0.0, None

    r = best_window_ratio(qn, art_norm)
    return ("partial", r, None) if r >= FUZZY_THRESHOLD else ("missing", r, None)


_SENT_END = re.compile(r"[.!?]")


def expand_to_sentences(raw: str, start: int, end: int):
    """Widen a span outward to sentence boundaries, so the reader sees the qualifying clauses the
    extractor may have cut ("...very high" -> "...very high compared to summer norms")."""
    left = raw.rfind(".", 0, start)
    left = 0 if left < 0 else left + 1
    m = _SENT_END.search(raw, end)
    right = len(raw) if not m else m.end()
    if right - left > MAX_SENTENCE_CHARS:                # pathological paragraph: keep the raw span
        return raw[start:end].strip()
    return raw[left:right].strip()


def flatten_evidence(ev):
    """Evidence is a str, list[str], or list[list[str]] (relations). -> flat list[str]."""
    out = []
    if ev is None:
        return out
    if isinstance(ev, str):
        return [ev]
    for item in ev:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, list):
            out.extend(x for x in item if isinstance(x, str))
    return out


def check_evidence(evidence, art_norm, idx_map, raw):
    """-> (status, detail, resolved). `resolved` carries, per quote, its RAW offsets and the
    recovered sentence(s) -- the article's own words, not the extractor's retyped quote."""
    quotes = flatten_evidence(evidence)
    if not quotes:
        return "none", {"n_quotes": 0, "found": 0, "partial": 0, "missing": 0, "worst_ratio": None}, []

    tally = {"found": 0, "partial": 0, "missing": 0}
    worst, resolved = 1.0, []
    for q in quotes:
        st, r, span = locate_quote(norm_text(q), art_norm, idx_map)
        tally[st] += 1
        worst = min(worst, r)
        item = {"quote": q, "status": st, "ratio": round(r, 3), "start": None, "end": None,
                "matched_text": None, "sentence": None}
        if span:
            item.update({"start": span[0], "end": span[1],
                         "matched_text": raw[span[0]:span[1]].strip(),
                         "sentence": expand_to_sentences(raw, *span)})
        resolved.append(item)

    status = ("found"   if tally["missing"] == 0 and tally["partial"] == 0 else
              "partial" if tally["missing"] == 0 else "missing")
    return status, {"n_quotes": len(quotes), **tally, "worst_ratio": round(worst, 3)}, resolved


# --- the article THIS thread is working on ----------------------------------------------------
# The judge prompt is built from the article text itself, so which article is "current" has to be
# per-thread: a batch cell grounds several at once, and a shared global would pair one article's
# text with another's assertions.

_ART = threading.local()


def set_article(raw):
    """Index `raw` and make it THIS thread's current article. -> (norm, idx_map, raw)."""
    ctx = build_index(raw)
    _ART.ctx = ctx
    return ctx


def article_ctx():
    """(norm, idx_map, raw) for the article this thread is on."""
    ctx = getattr(_ART, "ctx", None)
    if ctx is None:
        raise RuntimeError("no article set on this thread — call set_article(text) first")
    return ctx


def evidence_pass(graph, art_norm, idx_map, raw):
    """Score the EXTRACTOR's own citations. Recorded, never gating."""
    for kind in ("nary_nodes", "relations"):
        for item in graph.get(kind, []):
            st, detail, resolved = check_evidence(item.get("evidence"), art_norm, idx_map, raw)
            item.setdefault("grounding", {}).update(
                {"evidence_status": st, "evidence_detail": detail, "evidence_resolved": resolved})
    return graph


# --- the prompt ---------------------------------------------------------------------------------
# The "AN ASSERTION HAS PARTS" block is the 2026-08-19 fix. Measured on the 500-item benchmark, the
# judge caught changed numbers and swapped entities at 92-100% but wrong PROPERTIES at 47-62%,
# wrong TYPES at 52-60% and reversed ARROWS at 33-50% — and those three classes are where this
# pipeline's real errors actually are. The old prompt never told the judge that the property is part
# of the claim, so both endpoints appearing in the article was enough for it to pass.
#
# TYPES ARE OUT OF SCOPE, DELIBERATELY. Grounding asks one question — does the article state this
# triple? — and type quality is a different question answered by a different method (is this the best
# available schema.org class? is a minted type warranted?), evaluated after the canonicalisation
# stages rather than here. Two facts make the separation the right one: only 22% of the types in the
# benchmark's CORRECT assertions appear as a word anywhere in their article (Continent, Person,
# DiseaseOutbreak are inferred, not stated), so a judge told to verify types would reject most good
# assertions on outside knowledge it was also told not to use; and none of the 22 real extractor
# errors in the benchmark were type errors. So types are not shown to the judge at all.
#
# Consequence for the benchmark: its 25 type_change corruptions are now OUT OF SCOPE and excluded
# from scoring by judge_benchmark.OUT_OF_SCOPE_OPS. A judge that passes them is behaving correctly.
#
# EVERY EXAMPLE IN THIS PROMPT IS SCHEMATIC ("X", "SUBJECT", a placeholder property). Keep it that
# way. The benchmark that scores this judge is built from the same corpus the judge reads, so an
# illustration lifted from an article — a real figure, a real entity, a real bad property — is test
# data leaking into the instrument measuring it, and it would inflate exactly the class it
# illustrates. The earlier version of this prompt violated that; any run tagged with its
# prompt_sha1 carries the caveat.
#
# nb9d measures whether an edit here helps; do not change this without re-running it.

GROUNDING_SYSTEM_PROMPT = """You are a strict grounding judge for a knowledge-graph extraction pipeline. You will be given the full text of a news article, followed by a batch of assertions extracted from it.

For each assertion, decide whether the article supports it. Judge only against the article text shown. Do not use outside knowledge, and do not assume anything the article does not state, even if you believe it is true.

A relation is written as: "SUBJECT" --[property]--> "OBJECT". Check two things:

1. The endpoints. Does the article support the subject and the object as described?

2. The property. Read the property name as plain English and ask whether the article states that specific relation between those two things. It is NOT enough that both endpoints appear in the article, and NOT enough that the article connects them in some way. The article may relate the same two things in several ways; the relation this property names must be one of them. If the property reverses the direction the article gives, the assertion is "intrinsic".

A node is written as: a <kind> exists, described as "LABEL". Decide whether the article supports the existence of it as described.

Do not accept an assertion merely because all of its words appear somewhere in the article. Ask what it claims, then ask whether the article states that.

For each assertion, return:
- "supported": true if a careful reader would agree that the article states or directly entails the assertion; false otherwise.
- "failure_type": null when supported. Otherwise exactly one of:
    "intrinsic" - the article contradicts the assertion (wrong number, wrong entity, reversed direction, a different relation, overstated certainty).
    "extrinsic" - the article does not mention it. It may be true in the world, but this text does not state it.
- "note": one short clause giving the specific mismatch, not a restatement of the verdict. Name the figure the article gives instead, the entity it never mentions, or the relation it states in place of the asserted one.
- "quote": when supported, the article text that shows it, copied VERBATIM. Give the shortest continuous span that covers the support. If the support is in different parts of the article, give a list of such spans instead of one. Do not paraphrase, and do not merge separate passages into a single string. Use null when not supported.

Be strict but not pedantic. The assertion may paraphrase the article; it need not reuse the article's wording. What you must not accept is a claim the article never makes.

Some assertions are one part of an n-ary fact. A single fact is often split across several relations that share one hub node, with separate edges for its related information. Such an assertion is shown with a line reading "Context - the same <hub> also asserts:", which lists the sibling parts of that fact. Take this context into consideration when making the judgement.

Return JSON only: {"results": [{"index": <int>, "supported": <bool>, "failure_type": <string|null>, "note": "<string>", "quote": "<string|null>"}]}. Return exactly one result per assertion, echoing its index."""


# --- verbalisers --------------------------------------------------------------------------------

# Types are NOT verbalised — see the note above the prompt. They were dropped rather than shown with
# an instruction to ignore them, because a model attends to what is in front of it and "do not weigh
# this" is a weaker guarantee than absence. The disambiguation they supposedly bought was measured
# and is not there: across the benchmark, only 4 of 668 (article, surface form) pairs carry more than
# one type, and all four are artificial. `node_kind` stays — whether the article ASSERTS something or
# reports someone CLAIMING it is part of what grounding rules on.

def verbalise_node(node):
    kind = (node.get("node_kind") or "node").lower()
    article = "An" if kind[:1] in "aeiou" else "A"
    return f'{article} {kind} exists, described as "{node.get("label")}".'


def verbalise_relation(rel):
    return f'"{rel.get("subject_text")}" --[{rel.get("property")}]--> "{rel.get("object_text")}".' 


def make_relation_verbaliser(graph):
    """Verbalise a relation, appending its n-ary sibling edges as disambiguating context.

    The context disambiguates what an edge MEANS, which is a different question from what the judge
    is allowed to read (here: the whole article). No evidence is pooled.
    """
    nodes_by_id = {n["id"]: n for n in graph.get("nary_nodes", [])}
    rels_by_subject = defaultdict(list)
    for r in graph.get("relations", []):
        rels_by_subject[r.get("subject_id")].append(r)

    def verbalise(rel):
        base = verbalise_relation(rel)
        hub = nodes_by_id.get(rel.get("subject_id"))
        if hub is None:                                  # subject is a plain entity, not a hub
            return base
        sibs = [s for s in rels_by_subject.get(rel.get("subject_id"), []) if s is not rel]
        if not sibs:
            return base
        parts = "; ".join(f'{s.get("property")} "{s.get("object_text")}"' for s in sibs)
        kind = (hub.get("node_kind") or "fact").lower()
        return (base + f'\n(Judge only the arrow above. Context - the same {kind} '
                f'"{hub.get("label")}" also asserts: {parts}.)')
    return verbalise


# --- judging ------------------------------------------------------------------------------------

def as_quotes(q):
    """The judge may return one span or several. -> list[str] either way."""
    if q is None:
        return []
    if isinstance(q, str):
        return [q] if q.strip() else []
    return [x for x in q if isinstance(x, str) and x.strip()]


def _verify_quote(q):
    """Locate each span the judge returned, with the SAME matcher used on the extractor's citations.

    Several spans are allowed because support is not always in one place: an article can name the
    subject in one paragraph and state the relation in another, and a judge forced to pick one span
    has to throw away half of its own evidence. Each span is checked on its own; the reported status
    is the WORST of them, so one invented span still condemns the verdict.
    -> (status, ratio): "found" | "partial" | "missing" | None (no quote)."""
    quotes = as_quotes(q)
    if not quotes:
        return None, None
    art_norm, art_idx, _ = article_ctx()
    rank = {"found": 0, "partial": 1, "missing": 2}
    worst, worst_r = "found", 1.0
    for one in quotes:
        st, r, _span = locate_quote(norm_text(one), art_norm, art_idx)
        if rank[st] > rank[worst]:
            worst = st
        worst_r = min(worst_r, r)
    return worst, round(worst_r, 3)


def _citation_match(judge_quote, cited_quotes):
    """Did the extractor cite the sentence the judge actually grounded the assertion on?
    True when some cited quote is that sentence, or a clause of it (or fuzzily equal). None when
    there is nothing to compare (unsupported, no quote, or no citation)."""
    spans = [norm_text(x) for x in as_quotes(judge_quote)]
    spans = [x for x in spans if x]
    if not spans or not cited_quotes:
        return None
    for c in cited_quotes:
        cn = norm_text(c)
        if not cn:
            continue
        for jq in spans:
            if cn in jq or jq in cn or SequenceMatcher(None, jq, cn).ratio() >= FUZZY_THRESHOLD:
                return True
    return False


def judge_batch(items, verbalise):
    """items: list of dicts -> attach grounding verdicts in place. The ARTICLE is sent whole;
    nothing per-assertion is selected, so the judge — not this module — decides what supports what."""
    payload = [{"index": i, "assertion": verbalise(it)} for i, it in enumerate(items)]
    msg = ("ARTICLE:\n\"\"\"\n" + article_ctx()[2].strip() + "\n\"\"\"\n\n"
           "Assertions to judge:\n" + json.dumps(payload, ensure_ascii=False, indent=1))
    out = _llm()(GROUNDING_SYSTEM_PROMPT, msg)
    by_idx = {r.get("index"): r for r in out.get("results", []) if isinstance(r, dict)}
    for i, it in enumerate(items):
        r = by_idx.get(i)
        g = it.setdefault("grounding", {})
        if r is None:                       # judge skipped it -> keep, but mark unjudged
            g.update({"supported": None, "failure_type": None, "note": "judge returned no verdict",
                      "quote": None, "quote_status": None, "citation_match": None})
            continue
        q = r.get("quote")
        qs, qr = _verify_quote(q)
        g.update({"supported": bool(r.get("supported")), "failure_type": r.get("failure_type"),
                  "note": (r.get("note") or "")[:200], "quote": q,
                  "quote_status": qs, "quote_ratio": qr,
                  "citation_match": _citation_match(q, flatten_evidence(it.get("evidence")))})
    return items


def run_judge(items, verbalise, label, log=print):
    """Every item is judged — the citation check no longer filters anything out."""
    for start in range(0, len(items), JUDGE_BATCH):
        chunk = items[start:start + JUDGE_BATCH]
        for attempt in range(1, JUDGE_RETRIES + 1):
            try:
                judge_batch(chunk, verbalise)
                break                               # success -> stop retrying
            except Exception as e:                  # transient API/JSON error -> back off and retry
                if attempt < JUDGE_RETRIES:
                    wait = JUDGE_RETRY_BACKOFF * attempt
                    log(f"  ! {label} batch {start}-{start+len(chunk)} attempt {attempt}/{JUDGE_RETRIES} "
                        f"failed: {type(e).__name__}: {e} — retrying in {wait:.0f}s")
                    time.sleep(wait)
                else:
                    log(f"  ! {label} batch {start}-{start+len(chunk)} GAVE UP: {type(e).__name__}: {e}")
                    for it in chunk:
                        it.setdefault("grounding", {}).update(
                            {"supported": None, "failure_type": None,
                             "note": f"judge failed: {type(e).__name__}",
                             "quote": None, "quote_status": None, "citation_match": None})
        n = min(start + JUDGE_BATCH, len(items))
        log(f"  {label}: judged {n}/{len(items)}")
    return items


def ground_graph(graph, article_text, log=print):
    """Judge one whole graph: nodes first, then the relations their nodes did not orphan.

    Nodes go first because a node asserts its own existence and type — nothing else in the graph
    tests that — and because a relation whose endpoint node failed cannot stand, so doing nodes
    first saves judging edges that are about to drop.
    """
    art_norm, idx_map, raw = set_article(article_text)
    graph = evidence_pass(graph, art_norm, idx_map, raw)

    nodes = graph.get("nary_nodes", [])
    run_judge(nodes, verbalise_node, "nodes", log=log)
    for n in nodes:
        n["grounding"]["grounded"] = bool(n["grounding"].get("supported"))
    dropped = {n["id"] for n in nodes if not n["grounding"]["grounded"]}

    rels = graph.get("relations", [])
    for r in rels:
        g = r.setdefault("grounding", {})
        g["orphaned"] = (r.get("subject_id") in dropped) or (r.get("object_id") in dropped)
        if g["orphaned"]:
            g.update({"supported": False, "failure_type": "orphan",
                      "note": "endpoint node failed grounding",
                      "quote": None, "quote_status": None, "citation_match": None})
    live = [r for r in rels if not r["grounding"]["orphaned"]]
    run_judge(live, make_relation_verbaliser(graph), "relations", log=log)
    for r in rels:
        r["grounding"]["grounded"] = (bool(r["grounding"].get("supported"))
                                      and not r["grounding"]["orphaned"])
    return graph


# --- reporting ------------------------------------------------------------------------------------

def grounding_report(graph):
    rows = {}
    for kind in ("nary_nodes", "relations"):
        items = graph.get(kind, [])
        if not items:
            continue
        g = [x.get("grounding", {}) for x in items]
        n_ok = sum(bool(x.get("grounded")) for x in g)
        rows[kind] = {"total": len(items), "grounded": n_ok,
                      "rate": round(n_ok / len(items), 3),
                      "failures": dict(Counter(x.get("failure_type") for x in g if not x.get("grounded"))),
                      "citation_status": dict(Counter(x.get("evidence_status") for x in g))}
    # citation precision: of SUPPORTED items carrying a judge quote, how many did the extractor cite?
    for kind in ("nary_nodes", "relations"):
        items = [x.get("grounding", {}) for x in graph.get(kind, [])]
        cm = [x.get("citation_match") for x in items
              if x.get("grounded") and x.get("citation_match") is not None]
        if kind in rows:
            rows[kind]["citation_precision"] = (
                {"n": len(cm), "matched": sum(cm), "rate": round(sum(cm) / len(cm), 3)} if cm else None)
            qs = Counter(x.get("quote_status") for x in items if x.get("quote_status"))
            rows[kind]["judge_quote_status"] = dict(qs)
    # relation grounding split by where the PROPERTY came from — schema.org vs minted extension
    by_src = {}
    for r in graph.get("relations", []):
        src = r.get("property_source") or "unknown"
        d = by_src.setdefault(src, {"total": 0, "grounded": 0})
        d["total"] += 1; d["grounded"] += bool(r.get("grounding", {}).get("grounded"))
    for d in by_src.values():
        d["rate"] = round(d["grounded"] / d["total"], 3) if d["total"] else None
    rows["relations_by_property_source"] = by_src
    # node grounding by type source. Claim/Observation hubs are not type-canonicalized by 6d, so they
    # carry no type_source — bucketed honestly rather than as "unknown" (which reads like missing data).
    by_t = {}
    for n in graph.get("nary_nodes", []):
        src = n.get("type_source") or "not_canonicalized"
        d = by_t.setdefault(src, {"total": 0, "grounded": 0})
        d["total"] += 1; d["grounded"] += bool(n.get("grounding", {}).get("grounded"))
    for d in by_t.values():
        d["rate"] = round(d["grounded"] / d["total"], 3) if d["total"] else None
    rows["nodes_by_type_source"] = by_t
    return rows


def _decision(g):
    s = g.get("supported")
    return "supported" if s else ("unjudged" if s is None else "rejected")


def build_judgment_log(graph, article_id):
    """One flat row per node and per relation — the reviewable record of every decision."""
    v_rel = make_relation_verbaliser(graph)

    def row(kind, id_, assertion, item, subject=None, prop=None, obj=None):
        g = item.get("grounding", {})
        return {"article_id": article_id, "kind": kind, "id": id_, "assertion": assertion,
                "subject": subject, "property": prop, "object": obj,
                "citation_status": g.get("evidence_status"), "orphaned": g.get("orphaned"),
                "decision": _decision(g), "supported": g.get("supported"),
                "grounded": g.get("grounded"), "failure_type": g.get("failure_type"),
                "justification": g.get("note"), "judge_quote": g.get("quote"),
                "judge_quote_status": g.get("quote_status"),
                "citation_match": g.get("citation_match"),
                "evidence_cited": flatten_evidence(item.get("evidence"))}

    rows = [row("node", n.get("id"), verbalise_node(n), n) for n in graph.get("nary_nodes", [])]
    rows += [row("relation", f'{r.get("subject_id")}-{r.get("property")}-{r.get("object_id")}',
                 v_rel(r), r, r.get("subject_text"), r.get("property"), r.get("object_text"))
             for r in graph.get("relations", [])]
    return rows


JUDGMENT_COLS = ["article_id", "kind", "id", "assertion", "subject", "property", "object",
                 "citation_status", "orphaned", "decision", "supported", "grounded",
                 "failure_type", "justification", "judge_quote", "judge_quote_status",
                 "citation_match", "evidence_cited"]


# --- pruning --------------------------------------------------------------------------------------

def prune_graph(graph, drop_unjudged=False, drop_orphan_entities=True):
    """-> (pruned graph, report). Everything the judge rejected is removed.

    This is the artefact downstream evaluation should consume: a graph whose every remaining
    assertion the judge found supported by the source article. Two deliberate choices:

    **Unjudged items are KEPT by default.** `supported is None` means a judge call failed, not that
    the fact is wrong. Dropping them would silently convert API flakiness into a lower recall score
    for the extractor. They are counted in the report so the number is never invisible; pass
    drop_unjudged=True if you want the strict reading.

    **Entities left dangling are dropped**, since an entity no surviving relation mentions is not
    a fact the graph still asserts. Pass drop_orphan_entities=False to keep the full entity list.
    """
    keep = lambda g: bool(g.get("grounded")) or (g.get("supported") is None and not drop_unjudged)

    nodes = graph.get("nary_nodes", [])
    rels = graph.get("relations", [])
    kept_nodes = [n for n in nodes if keep(n.get("grounding", {}))]
    kept_ids = {n.get("id") for n in kept_nodes}
    # a relation also goes if the judge kept it but its hub did not survive (drop_unjudged can
    # remove a hub whose edges were judged fine)
    kept_rels = [r for r in rels if keep(r.get("grounding", {}))
                 and not (r.get("subject_id") in {n.get("id") for n in nodes} - kept_ids)
                 and not (r.get("object_id") in {n.get("id") for n in nodes} - kept_ids)]

    entities = graph.get("entities", [])
    referenced = {r.get("subject_id") for r in kept_rels} | {r.get("object_id") for r in kept_rels}
    kept_entities = ([e for e in entities if e.get("id") in referenced]
                     if drop_orphan_entities else list(entities))

    pruned = {k: v for k, v in graph.items() if k not in ("nary_nodes", "relations", "entities")}
    pruned.update({"entities": kept_entities, "nary_nodes": kept_nodes, "relations": kept_rels})

    report = {
        "nary_nodes": {"before": len(nodes), "after": len(kept_nodes), "removed": len(nodes) - len(kept_nodes)},
        "relations": {"before": len(rels), "after": len(kept_rels), "removed": len(rels) - len(kept_rels)},
        "entities": {"before": len(entities), "after": len(kept_entities),
                     "removed": len(entities) - len(kept_entities)},
        "unjudged_kept": sum(1 for x in kept_nodes + kept_rels
                             if x.get("grounding", {}).get("supported") is None),
        "drop_unjudged": drop_unjudged, "drop_orphan_entities": drop_orphan_entities,
        "removed_by_failure_type": dict(Counter(
            x.get("grounding", {}).get("failure_type")
            for x in nodes + rels if not keep(x.get("grounding", {})))),
    }
    for k in ("nary_nodes", "relations", "entities"):
        b = report[k]["before"]
        report[k]["kept_rate"] = round(report[k]["after"] / b, 3) if b else None
    pruned["prune_report"] = report
    return pruned, report
