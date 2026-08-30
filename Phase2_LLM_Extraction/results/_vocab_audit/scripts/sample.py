"""Draw a stratified sample of 250 consolidated relations per model (validation split,
both arms) and write the audit CSV plus the two judgment sheets (schema.org / minted)."""
import json, glob, os, csv, random, re
from collections import Counter, defaultdict

ROOT = "/Users/marwah/Documents/Thesis/TriplesExtractionFromWebLLMs/Phase2_LLM_Extraction"
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results", "_vocab_audit")
os.makedirs(OUT, exist_ok=True)

MODELS = ["gemini-2.5-flash", "deepseek-v4-flash", "llama33-70b", "qwen3-32b-think"]
# Single-call track only. The staged track is a different pipeline with a different
# hub-typing regime, and mixing the two made every rate an average over two populations.
ARMS = {
    "e2e": "{m}/art*/e2e/nary_graph_dedup_main_val_r1_extended_consolidated.json",
}
N_PER_MODEL = 250
SEED = 1234

# ---- schema.org ground truth -------------------------------------------------
sf = os.path.join(ROOT, "schema_org_files")
CLASSES = {c["label"]: c for c in json.load(open(os.path.join(sf, "schema_classes.json")))}
PROPS = {p["label"]: p for p in json.load(open(os.path.join(sf, "schema_properties.json")))}

def load_graph(path):
    d = json.load(open(path))
    nodes = {}
    for e in d["entities"]:
        nodes[e["id"]] = dict(e, node_kind="Entity")
    for h in d["nary_nodes"]:
        nodes[h["id"]] = dict(h, text=h.get("label"), node_kind=h.get("node_kind", "Hub"))
    return d, nodes

def collect(model):
    """rows: one per relation, with node context attached."""
    rows = []
    for arm, pat in ARMS.items():
        for f in sorted(glob.glob(os.path.join(RES, pat.format(m=model)))):
            art = os.path.basename(f.split("/art")[0] + "") if False else None
            m = re.search(r"/(art\d+)/", f)
            art = m.group(1)
            d, nodes = load_graph(f)
            for i, r in enumerate(d["relations"]):
                s = nodes.get(r["subject_id"], {})
                o = nodes.get(r["object_id"], {})
                rows.append({
                    "model": model, "arm": arm, "article": art, "rel_idx": i,
                    "subject_id": r["subject_id"],
                    "subject_text": r.get("subject_text") or s.get("text", ""),
                    "subject_node_kind": s.get("node_kind", ""),
                    "subject_type": r.get("subject_type") or s.get("type", ""),
                    "subject_type_source": s.get("type_source"),
                    "subject_type_definition": s.get("type_definition") or "",
                    "subject_type_parent": s.get("type_parent") or "",
                    "subject_type_justification": s.get("type_justification") or "",
                    "subject_aliases": "; ".join(s.get("aliases") or []),
                    "property": r.get("property", ""),
                    "property_source": r.get("property_source"),
                    "property_definition": r.get("property_definition") or "",
                    "property_parent": r.get("property_parent") or "",
                    "property_justification": r.get("property_justification") or "",
                    "relation_phrases": "; ".join(r.get("relation_phrases") or []),
                    "object_id": r["object_id"],
                    "object_text": r.get("object_text") or o.get("text", ""),
                    "object_node_kind": o.get("node_kind", ""),
                    "object_type": r.get("object_type") or o.get("type", ""),
                    "object_type_source": o.get("type_source"),
                    "object_type_definition": o.get("type_definition") or "",
                    "object_type_parent": o.get("type_parent") or "",
                    "object_type_justification": o.get("type_justification") or "",
                    "object_aliases": "; ".join(o.get("aliases") or []),
                    "evidence": " || ".join(
                        s2 for ev in (r.get("evidence") or [])
                        for s2 in (ev if isinstance(ev, list) else [ev])
                    )[:1200],
                })
    return rows

def stratified_sample(rows, n_total, seed):
    """Draw PROPORTIONALLY: the schema.org / minted split of the sample reproduces the
    split of the population it is drawn from, so a rate measured on the sample is a
    rate over the model's own graphs and needs no re-weighting. Within a stratum the
    draw is round-robin over articles so no article dominates."""
    rnd = random.Random(seed)
    picked = []
    per_arm = {a: n_total // len(ARMS) for a in ARMS}
    for arm in ARMS:
        pool = [r for r in rows if r["arm"] == arm]
        by_src = defaultdict(list)
        for r in pool:
            by_src["minted" if r["property_source"] == "vocab" else "schema.org"].append(r)
        want = min(per_arm[arm], len(pool))
        quota = {}
        # largest-remainder apportionment against the population shares
        exact = {k: want * len(v) / len(pool) for k, v in by_src.items()}
        for k, e in exact.items():
            quota[k] = min(int(e), len(by_src[k]))
        left = want - sum(quota.values())
        for k in sorted(exact, key=lambda k: -(exact[k] - int(exact[k]))):
            if left <= 0: break
            take = min(left, len(by_src[k]) - quota[k])
            quota[k] += take
            left -= take
        for k, q in quota.items():
            cand = by_src.get(k, [])
            # round-robin over articles so no single article dominates
            byart = defaultdict(list)
            for r in cand:
                byart[r["article"]].append(r)
            for a in byart:
                rnd.shuffle(byart[a])
            order = sorted(byart)
            rnd.shuffle(order)
            sel, i = [], 0
            while len(sel) < q:
                progressed = False
                for a in order:
                    if i < len(byart[a]):
                        sel.append(byart[a][i]); progressed = True
                        if len(sel) == q: break
                if not progressed: break
                i += 1
            picked.extend(sel)
    return picked

if __name__ == "__main__":
    summary = []
    for model in MODELS:
        rows = collect(model)
        samp = stratified_sample(rows, N_PER_MODEL, SEED)
        for i, r in enumerate(samp, 1):
            r["sample_id"] = f"{model.split('-')[0][:4].upper()}-{i:03d}"
        os.makedirs(os.path.join(OUT, model), exist_ok=True)
        p = os.path.join(OUT, model, "sample_relations.csv")
        cols = ["sample_id", "model", "arm", "article", "subject_id", "subject_text",
                "subject_node_kind", "subject_type", "subject_type_source",
                "property", "property_source", "object_id", "object_text",
                "object_node_kind", "object_type", "object_type_source",
                "relation_phrases", "property_definition", "property_parent",
                "property_justification", "subject_type_definition", "subject_type_parent",
                "subject_type_justification", "object_type_definition", "object_type_parent",
                "object_type_justification", "subject_aliases", "object_aliases", "evidence"]
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(samp)
        # distinct element counts
        props = Counter((r["property"], r["property_source"]) for r in samp)
        types = Counter()
        for r in samp:
            for side in ("subject", "object"):
                types[(r[f"{side}_type"], r[f"{side}_type_source"])] += 1
        sc_p = sum(1 for k in props if k[1] == "schema.org")
        mi_p = sum(1 for k in props if k[1] == "vocab")
        sc_t = sum(1 for k in types if k[1] == "schema.org")
        mi_t = sum(1 for k in types if k[1] == "vocab")
        un_t = sum(1 for k in types if k[1] is None)
        arm_c = Counter(r["arm"] for r in samp)
        src_c = Counter(r["property_source"] for r in samp)
        print(f"{model:20s} n={len(samp)} arms={dict(arm_c)} psrc={dict(src_c)} | "
              f"distinct props sc={sc_p} minted={mi_p} | distinct types sc={sc_t} minted={mi_t} unmatched={un_t}")
        summary.append((model, len(samp), sc_p, mi_p, sc_t, mi_t, un_t))
