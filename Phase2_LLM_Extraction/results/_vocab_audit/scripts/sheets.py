"""Turn each model's 250-relation sample into two judgment sheets:
schema.org-sourced elements and non-schema.org (minted / never-canonicalised) elements."""
import json, os, csv, re, math, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import CLASSES, PROPS, OUT, MODELS

STOP = set("a an the of to in on for by with and or is are as at from that this it its".split())

def toks(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s or "")
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]

def build_index(items):
    docs, df = {}, Counter()
    for lab, rec in items.items():
        t = toks(lab) * 3 + toks(rec.get("comment", ""))
        docs[lab] = Counter(t)
        for w in set(t): df[w] += 1
    N = len(docs)
    idf = {w: math.log(1 + N / (1 + c)) for w, c in df.items()}
    return docs, idf

CLS_IDX = build_index(CLASSES)
PRP_IDX = build_index(PROPS)

def candidates(term, definition, kind, k=6):
    docs, idf = CLS_IDX if kind == "type" else PRP_IDX
    q = Counter(toks(term) * 3 + toks(definition))
    scores = []
    for lab, d in docs.items():
        s = sum(qv * min(d.get(w, 0), 3) * idf.get(w, 0) for w, qv in q.items())
        if s: scores.append((s, lab))
    scores.sort(reverse=True)
    return [lab for _, lab in scores[:k]]

def usage_str(r, kind, side=None):
    if kind == "property":
        return f'{r["subject_text"]} [{r["subject_type"]}] --{r["property"]}--> {r["object_text"]} [{r["object_type"]}]'
    return f'"{r[side+"_text"]}" ({r[side+"_node_kind"]}) as {r[side+"_type"]}'

def build(model):
    rows = list(csv.DictReader(open(os.path.join(OUT, model, "sample_relations.csv"))))
    # element_key -> record
    acc = {}
    def add(key, base, usage, arm):
        e = acc.setdefault(key, dict(base, n=0, n_e2e=0, n_staged=0, usages=[], defs=set(), justs=set(), parents=set()))
        e["n"] += 1
        e["n_e2e" if arm == "e2e" else "n_staged"] += 1
        if usage not in e["usages"]: e["usages"].append(usage)
        if base.get("definition"): e["defs"].add(base["definition"])
        if base.get("justification"): e["justs"].add(base["justification"])
        if base.get("parent"): e["parents"].add(base["parent"])

    for r in rows:
        src = r["property_source"]
        add(("property", r["property"], src),
            {"element_kind": "property", "element": r["property"], "source": src,
             "definition": r["property_definition"], "parent": r["property_parent"],
             "justification": r["property_justification"]},
            usage_str(r, "property"), r["arm"])
        for side in ("subject", "object"):
            t, s = r[f"{side}_type"], r[f"{side}_type_source"] or ""
            if not t: continue
            add(("type", t, s),
                {"element_kind": "type", "element": t, "source": s,
                 "definition": r[f"{side}_type_definition"], "parent": r[f"{side}_type_parent"],
                 "justification": r[f"{side}_type_justification"]},
                usage_str(r, "type", side), r["arm"])

    schema_rows, other_rows = [], []
    for (kind, el, src), e in sorted(acc.items(), key=lambda kv: (-kv[1]["n"], kv[0][0], kv[0][1])):
        ref = (PROPS if kind == "property" else CLASSES).get(el)
        common = {
            "model": model, "element_kind": kind, "element": el,
            "n_occurrences": e["n"], "n_e2e": e["n_e2e"], "n_staged": e["n_staged"],
            "example_usages": " || ".join(e["usages"][:3]),
        }
        if src == "schema.org":
            schema_rows.append({**common,
                "in_schema_org": "yes" if ref else "NO",
                "schema_definition": (ref or {}).get("comment", "")[:500],
                "schema_domain": "/".join((ref or {}).get("domain", []))[:200],
                "schema_range": "/".join((ref or {}).get("range", []))[:200],
                "model_justification": " | ".join(sorted(e["justs"]))[:600],
                "justification_is_fallback": "yes" if any("fallback" in j.lower() for j in e["justs"]) else "no",
                "fits_meaning": "", "is_best_fit": "", "better_fit": "",
                "issue_code": "", "judge_notes": "", "hand_verified": "", "hand_notes": ""})
        else:
            defn = " | ".join(sorted(e["defs"]))[:400]
            other_rows.append({**common,
                "source": "minted" if src == "vocab" else "never_canonicalised",
                "vocab_definition": defn,
                "vocab_parent": "/".join(sorted(e["parents"]))[:200],
                "model_justification": " | ".join(sorted(e["justs"]))[:600],
                "exists_verbatim_in_schema_org": "yes" if ref else "no",
                "schema_org_candidates": ", ".join(candidates(el, defn, kind)),
                "fits_meaning": "", "schemaorg_alternative": "", "suggested_schemaorg": "",
                "issue_code": "", "judge_notes": "", "hand_verified": "", "hand_notes": ""})
    return schema_rows, other_rows

SCHEMA_COLS = ["model", "element_kind", "element", "in_schema_org", "n_occurrences", "n_e2e",
               "n_staged", "schema_definition", "schema_domain", "schema_range",
               "example_usages", "model_justification", "justification_is_fallback",
               "fits_meaning", "is_best_fit", "better_fit", "issue_code", "judge_notes",
               "hand_verified", "hand_notes"]
OTHER_COLS = ["model", "element_kind", "element", "source", "n_occurrences", "n_e2e", "n_staged",
              "vocab_definition", "vocab_parent", "exists_verbatim_in_schema_org",
              "schema_org_candidates", "example_usages", "model_justification",
              "fits_meaning", "schemaorg_alternative", "suggested_schemaorg", "issue_code",
              "judge_notes", "hand_verified", "hand_notes"]

if __name__ == "__main__":
    for m in MODELS:
        s, o = build(m)
        d = os.path.join(OUT, m)
        with open(os.path.join(d, "judgments_schemaorg.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SCHEMA_COLS, extrasaction="ignore"); w.writeheader(); w.writerows(s)
        with open(os.path.join(d, "judgments_minted.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=OTHER_COLS, extrasaction="ignore"); w.writeheader(); w.writerows(o)
        mint = sum(1 for r in o if r["source"] == "minted")
        print(f"{m:20s} schemaorg_rows={len(s):4d}  other_rows={len(o):4d} (minted={mint}, never_canonicalised={len(o)-mint})")
