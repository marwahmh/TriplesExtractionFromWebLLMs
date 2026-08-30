import csv, os, sys, json
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import OUT, MODELS
SHORT = {"gemini-2.5-flash":"Gemini 2.5 Flash","deepseek-v4-flash":"DeepSeek V4 Flash",
         "llama33-70b":"Llama 3.3 70B","qwen3-32b-think":"Qwen3 32B (think)"}

summary = {}
for m in MODELS:
    d = {}
    sc = list(csv.DictReader(open(os.path.join(OUT, m, "judgments_schemaorg.csv"))))
    mi = list(csv.DictReader(open(os.path.join(OUT, m, "judgments_minted.csv"))))
    rel = list(csv.DictReader(open(os.path.join(OUT, m, "sample_relations.csv"))))
    d["n_relations"] = len(rel)
    d["arm"] = dict(Counter(r["arm"] for r in rel))
    d["property_source"] = dict(Counter("minted" if r["property_source"]=="vocab" else "schema.org" for r in rel))
    # weight per element AND per occurrence
    def block(rows, key, weight):
        c = Counter()
        for r in rows:
            c[r[key] or "(blank)"] += (int(r["n_occurrences"]) if weight else 1)
        return dict(c)
    d["schemaorg"] = {
        "n_elements": len(sc), "n_occurrences": sum(int(r["n_occurrences"]) for r in sc),
        "by_kind": dict(Counter(r["element_kind"] for r in sc)),
        "fits_meaning": block(sc, "fits_meaning", False),
        "fits_meaning_weighted": block(sc, "fits_meaning", True),
        "is_best_fit": block(sc, "is_best_fit", False),
        "is_best_fit_weighted": block(sc, "is_best_fit", True),
        "issue_code": block(sc, "issue_code", False),
    }
    minted = [r for r in mi if r["source"] == "minted"]
    unmat  = [r for r in mi if r["source"] != "minted"]
    for name, rows in (("minted", minted), ("never_canonicalised", unmat)):
        d[name] = {
            "n_elements": len(rows), "n_occurrences": sum(int(r["n_occurrences"]) for r in rows),
            "by_kind": dict(Counter(r["element_kind"] for r in rows)),
            "fits_meaning": block(rows, "fits_meaning", False),
            "schemaorg_alternative": block(rows, "schemaorg_alternative", False),
            "issue_code": block(rows, "issue_code", False),
            "label_has_whitespace": dict(Counter(r["label_has_whitespace"] for r in rows)),
        }
    # domain/range
    dm = {"ok":0,"bad":0}; rg = {"ok":0,"bad":0}
    for r in sc:
        if r["element_kind"] != "property": continue
        dm["ok"] += int(r["domain_ok"] or 0); dm["bad"] += int(r["domain_bad"] or 0)
        rg["ok"] += int(r["range_ok"] or 0);  rg["bad"] += int(r["range_bad"] or 0)
    d["domain_range"] = {"domain": dm, "range": rg}
    summary[m] = d

with open(os.path.join(OUT, "audit_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

def pct(n, t): return f"{100*n/t:.0f}%" if t else "-"

print("=" * 100)
print("SCHEMA.ORG-SOURCED ELEMENTS (per distinct element)")
print(f'{"model":20s} {"n":>4s} {"fits yes":>9s} {"partial":>8s} {"no":>5s} {"best-fit yes":>13s} {"dom ok":>8s} {"rng ok":>8s}')
for m in MODELS:
    s = summary[m]["schemaorg"]; n = s["n_elements"]
    f = s["fits_meaning"]; b = s["is_best_fit"]; d = summary[m]["domain_range"]
    dt = d["domain"]["ok"]+d["domain"]["bad"]; rt = d["range"]["ok"]+d["range"]["bad"]
    print(f'{SHORT[m]:20s} {n:4d} {pct(f.get("yes",0),n):>9s} {pct(f.get("partial",0),n):>8s} {pct(f.get("no",0),n):>5s} '
          f'{pct(b.get("yes",0),n):>13s} {pct(d["domain"]["ok"],dt):>8s} {pct(d["range"]["ok"],rt):>8s}')

for track in ("minted", "never_canonicalised"):
    print()
    print("=" * 100)
    print(f"{track.upper()} ELEMENTS")
    print(f'{"model":20s} {"n":>4s} {"fits yes":>9s} {"partial":>8s} {"no":>5s} {"sc exact":>9s} {"sc broader":>11s} {"sc none":>8s} {"no issue":>9s} {"whitespace":>11s}')
    for m in MODELS:
        s = summary[m][track]; n = s["n_elements"] or 1
        f = s["fits_meaning"]; a = s["schemaorg_alternative"]; i = s["issue_code"]
        print(f'{SHORT[m]:20s} {s["n_elements"]:4d} {pct(f.get("yes",0),n):>9s} {pct(f.get("partial",0),n):>8s} {pct(f.get("no",0),n):>5s} '
              f'{pct(a.get("exact",0),n):>9s} {pct(a.get("broader",0),n):>11s} {pct(a.get("none",0),n):>8s} '
              f'{pct(i.get("NONE",0),n):>9s} {pct(s["label_has_whitespace"].get("yes",0),n):>11s}')

print()
print("=" * 100)
print("ISSUE CODES (all models pooled)")
for track, sheet in (("schema.org","schemaorg"),("minted","minted"),("never_canonicalised","never_canonicalised")):
    c = Counter()
    for m in MODELS:
        key = "schemaorg" if track == "schema.org" else track
        for k, v in summary[m][key]["issue_code"].items(): c[k] += v
    tot = sum(c.values())
    print(f"\n-- {track} (n={tot})")
    for k, v in c.most_common():
        print(f"   {k:28s} {v:4d}  {pct(v,tot)}")
