import csv, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample import OUT, MODELS

for name in ("sample_relations", "judgments_schemaorg", "judgments_minted"):
    allrows, cols = [], None
    for m in MODELS:
        rows = list(csv.DictReader(open(os.path.join(OUT, m, f"{name}.csv"))))
        if cols is None: cols = list(rows[0].keys())
        allrows.extend(rows)
    with open(os.path.join(OUT, f"ALL_{name}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(allrows)
    print(f"ALL_{name}.csv  {len(allrows)} rows")

# occurrence-weighted view of the schema.org sheet
print("\nOccurrence-weighted (a term counted once per relation it appears in):")
for m in MODELS:
    sc = list(csv.DictReader(open(os.path.join(OUT, m, "judgments_schemaorg.csv"))))
    tot = sum(int(r["n_occurrences"]) for r in sc)
    ok  = sum(int(r["n_occurrences"]) for r in sc if r["fits_meaning"] == "yes")
    best= sum(int(r["n_occurrences"]) for r in sc if r["is_best_fit"] == "yes")
    none= sum(int(r["n_occurrences"]) for r in sc if r["issue_code"] == "NONE")
    print(f"  {m:20s} occurrences={tot:4d}  fits=yes {100*ok/tot:.0f}%  best-fit=yes {100*best/tot:.0f}%  no issue {100*none/tot:.0f}%")

print("\nMinted, occurrence-weighted:")
for m in MODELS:
    mi = [r for r in csv.DictReader(open(os.path.join(OUT, m, "judgments_minted.csv"))) if r["source"]=="minted"]
    tot = sum(int(r["n_occurrences"]) for r in mi)
    ok  = sum(int(r["n_occurrences"]) for r in mi if r["fits_meaning"] == "yes")
    reuse = sum(int(r["n_occurrences"]) for r in mi if r["schemaorg_alternative"] in ("exact","broader"))
    print(f"  {m:20s} occurrences={tot:4d}  fits=yes {100*ok/tot:.0f}%  schema.org alternative existed {100*reuse/tot:.0f}%")

# singleton rate: elements used exactly once in the 250-relation sample
print("\nSingleton rate (element appears in exactly one sampled relation):")
for m in MODELS:
    for sheet, lbl in (("judgments_schemaorg","schema.org"),("judgments_minted","non-schema.org")):
        rows = list(csv.DictReader(open(os.path.join(OUT, m, f"{sheet}.csv"))))
        s = sum(1 for r in rows if int(r["n_occurrences"]) == 1)
        print(f"  {m:20s} {lbl:15s} {s:4d}/{len(rows):4d} = {100*s/len(rows):.0f}%")
