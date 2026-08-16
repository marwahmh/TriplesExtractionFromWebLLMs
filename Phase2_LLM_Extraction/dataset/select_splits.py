"""Split the corpus into a VALIDATION set (20) and a TEST set (the rest), once and reproducibly.

Every component claim in the write-up — what the review pass adds, what the reranker promotes, what
consolidation merges — is measured on the validation set, and the reported end-to-end numbers come
from the test set. That only means anything if the two are DISJOINT and fixed before the runs, which
is what this script produces: `dataset/splits.json`, keyed by URL.

Selection, in order:

1. **Two sets of articles are pinned into validation**, because for them the choice is already made:
   - the **10 monitoring articles** (`dataset/monitoring_sample.json`), which carry the 200
     manually-checked grounding facts — the human-verified anchor has to sit on the side where the
     components are tuned and judged;
   - the **4 development articles** (art1, art2, art4, art5 as of 2026-08-15), the ones the pipeline
     was built against. Every prompt, threshold and retrieval setting was chosen while looking at
     their outputs, so a test score computed on them would be measuring a corpus the method has
     already been fitted to. They are validation by definition, not by sampling.
2. **Two per domain**, so no domain's behaviour is measured on a single article — capped at
   `n_domain - 1` so **every domain keeps at least one article in test**. Culture has exactly 2
   articles and therefore contributes 1, not 2.
3. **Top up to 20** with whatever is closest to the corpus median length.

Within a domain the article closest to the **corpus median length** wins, not the longest. Length
drives graph size, fragmentation and how much there is to canonicalize, so a validation set skimmed
off the top of each domain would measure the components on a corpus systematically longer than the
test set they are reported against. Only stubs (below `MIN_CHARS`) are excluded, and no outlet may
take more than `MAX_PER_SOURCE` unpinned slots — otherwise BBC, at a third of the corpus, would
dominate the set the component numbers are computed on.

Keyed by URL, NOT by artN: `art<N>` is the row position in `articles.csv` and silently renumbers
when the CSV is edited, which would move articles between splits without anything saying so.
`resolve()` re-derives the current ids from each article's `URL:` header and shouts when they moved.

    python dataset/select_splits.py             # select + write dataset/splits.json
    python dataset/select_splits.py --check     # verify the existing file against the corpus
    python dataset/select_splits.py --freeze    # pin the CURRENT validation set; new articles then
                                                # only ever go to test (idempotent from then on)
    python dataset/select_splits.py --unfreeze  # allow a fresh draw (invalidates existing runs)

Read from a config via `"split": "val" | "test"` (see notebooks/experiment_config.py), which is what
turns a split into part of a run's identity: the tag becomes `main_val` / `main_test`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "dataset" / "articles.csv"
ARTICLES = ROOT / "dataset" / "articles"
MONITORING = ROOT / "dataset" / "monitoring_sample.json"
SPLITS = ROOT / "dataset" / "splits.json"

# --- selection knobs ---------------------------------------------------------------------
VAL_SIZE = 20             # 2 per domain over 10 domains, minus what the domain caps forbid
PER_DOMAIN = 2            # target per domain, before the "leave one in test" cap
# A STUB cut, not a length floor. An earlier version required 2400 chars, which — combined with
# "longest first within a domain" — made every sampled article long: validation's median came out at
# 5290 against the corpus's 3842, so the components were being measured on a corpus systematically
# longer than the one they are reported on. Only genuine stubs are excluded now.
MIN_CHARS = 900
# Unpinned validation slots one outlet may take. Set so the validation set MIRRORS the corpus rather
# than being outlet-balanced against it: BBC is 26/79 (33%) of the corpus, and three domains
# (Health, Environment, Culture) have no non-BBC article above MIN_CHARS left once the monitoring
# article is pinned — a tighter cap silently drops those domains below their quota instead.
MAX_PER_SOURCE = 5

# The articles the pipeline was DEVELOPED on — pinned to validation so no test number is computed on
# an article the method was fitted to. Held as URLs, like every other identity here: `art<N>` is the
# row position in articles.csv and renumbers when the CSV is edited, which would silently pin the
# wrong four. (These were art1 / art2 / art4 / art5 when the split was created.)
DEV_URLS = [
    "https://www.bbc.com/news/articles/c3ry307rxqro",              # art1  Climate       heatwave deaths
    "https://www.bbc.com/news/articles/crlw1eeez19o",              # art2  Entertainment Swift wedding
    "https://www.bbc.com/sport/tennis/articles/cgjyjjxzvydo",      # art4  Sports        US Open prize money
    "https://www.bbc.com/news/articles/c75ykve4zzxo",              # art5  Health        Ebola trial
]
# ------------------------------------------------------------------------------------------


def load_rows() -> list[dict]:
    """Every article in the corpus, with its current art id and length. No exclusions: the test set
    is 'everything not in validation', so nothing may be dropped here."""
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    for i, r in enumerate(rows, 1):
        p = ARTICLES / f"art{i}.txt"
        r["art_id"] = f"art{i}"
        r["chars"] = len(p.read_text(encoding="utf-8")) if p.exists() else 0
    return rows


def _url_of(art_id: str) -> str | None:
    p = ARTICLES / f"{art_id}.txt"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines()[:5]:
        if line.startswith("URL:"):
            return line[4:].strip()
    return None


def sha(art_id: str) -> str:
    return hashlib.sha256((ARTICLES / f"{art_id}.txt").read_bytes()).hexdigest()[:16]


def monitoring_urls() -> list[str]:
    """The monitoring sample's URLs — the manually-grounded articles."""
    if not MONITORING.exists():
        return []
    return [a["url"] for a in json.loads(MONITORING.read_text(encoding="utf-8"))["articles"]]


def pinned_urls() -> list[str]:
    """Everything that must be in validation: the manually-grounded sample + the development set.

    They consume the VAL_SIZE budget rather than adding to it — the pins ARE the validation set's
    core, and the domain quota only fills what is left over."""
    seen, out = set(), []
    for u in monitoring_urls() + DEV_URLS:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def frozen_urls() -> list[str] | None:
    """The validation URLs an existing splits.json has FROZEN, or None if it has not been frozen.

    Freezing exists because the sampled slots are chosen relative to the CORPUS MEDIAN length: grow
    the corpus and the median moves, so a plain re-run can hand back a different validation set —
    silently invalidating every run already made against the old one. Once frozen, re-running only
    ever grows the test set, which is exactly what adding articles should do.
    """
    if not SPLITS.exists():
        return None
    man = json.loads(SPLITS.read_text(encoding="utf-8"))
    return [a["url"] for a in man["validation"]] if man.get("frozen") else None


def select(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """-> (validation, test). Deterministic: same corpus in, same split out.

    With a FROZEN splits.json the validation set is taken verbatim and every article outside it —
    including ones added since — goes to test."""
    frozen = frozen_urls()
    if frozen is not None:
        by_url = {r["url"]: r for r in rows}
        val = [by_url[u] for u in frozen if u in by_url]
        missing = [u for u in frozen if u not in by_url]
        if missing:
            raise SystemExit("frozen validation article(s) are no longer in articles.csv:\n  "
                             + "\n  ".join(missing))
        test = [r for r in rows if r["url"] not in set(frozen)]
        print(f"  validation FROZEN: {len(val)} article(s) kept verbatim; "
              f"{len(test)} in test ({len(rows)} in the corpus)")
        return (sorted(val, key=lambda r: (r["domain"], r["url"])),
                sorted(test, key=lambda r: (r["domain"], r["url"])))
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_domain[r["domain"]].append(r)
    # Ordered by closeness to the CORPUS median length, so the sampled slots reproduce the length
    # range the test set contains instead of skimming the longest articles off each domain. Article
    # length drives graph size, fragmentation and how much there is to canonicalize at all, so a
    # validation set that is uniformly long measures the components on a corpus that does not exist.
    target = statistics.median([r["chars"] for r in rows])
    for v in by_domain.values():
        v.sort(key=lambda r: (abs(r["chars"] - target), r["url"]))

    pins = set(pinned_urls())
    chosen: dict[str, dict] = {r["url"]: r for r in rows if r["url"] in pins}
    missing = pins - set(chosen)
    if missing:
        # A pinned URL that is not in the corpus means the corpus moved under the pin — silently
        # dropping it would put a development or manually-graded article back into test.
        raise SystemExit("pinned article(s) not found in articles.csv — check DEV_URLS and "
                         f"monitoring_sample.json against the current corpus:\n  "
                         + "\n  ".join(sorted(missing)))
    src_used = Counter()                                     # counts UNPINNED picks only

    def quota(domain: str) -> int:
        """Target for a domain, never taking its last article: a domain with no test article cannot
        be reported on at all."""
        return min(PER_DOMAIN, max(len(by_domain[domain]) - 1, 0))

    def free_in(domain: str) -> list:
        return [r for r in by_domain[domain]
                if r["url"] not in chosen and r["chars"] >= MIN_CHARS
                and src_used[r["source"]] < MAX_PER_SOURCE]

    def take(domain: str) -> bool:
        if sum(1 for c in chosen.values() if c["domain"] == domain) >= quota(domain):
            return False
        free = free_in(domain)
        if len(free) <= 1:            # never take a domain's last remaining test article
            return False
        r = free[0]                   # closest to the corpus median
        chosen[r["url"]] = r
        src_used[r["source"]] += 1
        return True

    for domain in sorted(by_domain, key=lambda d: (-len(by_domain[d]), d)):
        while sum(1 for c in chosen.values() if c["domain"] == domain) < quota(domain):
            if not take(domain):
                break

    # Top up to VAL_SIZE with whatever is closest to the corpus median, wherever it is: the quota cap
    # is lifted here, the "leave one in test" rule is not.
    while len(chosen) < VAL_SIZE:
        pool = [r for d in by_domain for r in free_in(d) if len(free_in(d)) > 1]
        if not pool:
            print(f"  WARN  only {len(chosen)} validation articles satisfy the constraints "
                  f"(wanted {VAL_SIZE}) — relax MIN_CHARS or MAX_PER_SOURCE.")
            break
        r = min(pool, key=lambda r: (abs(r["chars"] - target), r["url"]))
        chosen[r["url"]] = r
        src_used[r["source"]] += 1

    val = sorted(chosen.values(), key=lambda r: (r["domain"], r["url"]))
    test = sorted((r for r in rows if r["url"] not in chosen),
                  key=lambda r: (r["domain"], r["url"]))
    return val, test


def _entry(r: dict, pins: set) -> dict:
    return {"domain": r["domain"], "source": r["source"], "art_id": r["art_id"],
            "title": r["title"], "url": r["url"], "date": r.get("Date", ""),
            "chars": r["chars"], "sha256_16": sha(r["art_id"]),
            "monitoring": r["url"] in pins,
            "development": r["url"] in set(DEV_URLS)}


def write(val: list[dict], test: list[dict]) -> dict:
    pins = set(monitoring_urls())
    lens = sorted(r["chars"] for r in val)
    payload = {
        "purpose": "Fixed VALIDATION / TEST split. Component effects (review pass, reranker, "
                   "consolidation) are measured on validation; reported end-to-end numbers come "
                   "from test. Disjoint by construction.",
        "selection": {"val_size": VAL_SIZE, "per_domain": PER_DOMAIN, "min_chars": MIN_CHARS,
                      "max_per_source": MAX_PER_SOURCE,
                      "pinned_monitoring": "dataset/monitoring_sample.json (manually grounded)",
                      "pinned_development": DEV_URLS,
                      "rule": "pins first (monitoring + development articles), then 2 per domain "
                              "capped at n_domain-1 so every domain keeps a test article, then "
                              "topped up to val_size from the largest domains"},
        "frozen": bool(frozen_urls() is not None or "--freeze" in sys.argv),
        "counts": {"validation": len(val), "test": len(test), "total": len(val) + len(test)},
        "validation_lengths": {"min": lens[0], "median": lens[len(lens) // 2], "max": lens[-1]},
        "validation": [_entry(r, pins) for r in val],
        "test": [_entry(r, pins) for r in test],
    }
    SPLITS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def resolve(split: str, path: Path = SPLITS, strict: bool = True) -> list[str]:
    """Current art ids for one split, re-derived from each article's URL.

    Use this instead of hard-coding ids: it survives an articles.csv edit (which renumbers artN) and
    fails loudly when an article has vanished. Returned in corpus order — the order IS the run, since
    every canonicalization stage is incremental."""
    man = json.loads(Path(path).read_text(encoding="utf-8"))
    if split not in ("validation", "test"):
        raise KeyError(f"split must be 'validation' or 'test', not {split!r}")
    by_url = {}
    for p in ARTICLES.glob("art*.txt"):
        u = _url_of(p.stem)
        if u:
            by_url[u] = p.stem
    ids, moved, lost = [], [], []
    for a in man[split]:
        cur = by_url.get(a["url"])
        if cur is None:
            lost.append(a["title"])
            continue
        if cur != a["art_id"]:
            moved.append(f'{a["art_id"]} -> {cur}  {a["title"][:50]}')
        ids.append(cur)
    if moved:
        print(f"RENUMBERED since splits.json was written ({split}):")
        for m in moved:
            print("  ", m)
        print("   results/<model>/artN/ still points at the OLD text — re-check before reusing it.")
    if lost:
        msg = f"{split}: articles missing from dataset/articles/: " + "; ".join(lost)
        if strict:
            raise SystemExit(msg)
        print("WARNING:", msg)
    return sorted(ids, key=lambda a: int("".join(c for c in a if c.isdigit()) or 0))


def _report(val, test):
    dv, dt = Counter(r["domain"] for r in val), Counter(r["domain"] for r in test)
    print(f"  validation {len(val)}   test {len(test)}   total {len(val) + len(test)}")
    print(f'  {"domain":<24} {"val":>4} {"test":>5}')
    for d in sorted(set(dv) | set(dt)):
        flag = "   <- no test article!" if dt[d] == 0 else ""
        print(f"  {d:<24} {dv[d]:>4} {dt[d]:>5}{flag}")
    print(f'  {"sources (val)":<24} ' + ", ".join(f"{s} {n}" for s, n in
                                                  Counter(r["source"] for r in val).most_common()))
    mon, dev = set(monitoring_urls()), set(DEV_URLS)
    print(f"  monitoring articles in val : {sum(1 for r in val if r['url'] in mon)}/{len(mon)}")
    print(f"  development articles in val: {sum(1 for r in val if r['url'] in dev)}/{len(dev)}"
          f"   ({', '.join(r['art_id'] for r in val if r['url'] in dev)})")
    leaked = [r["art_id"] for r in test if r["url"] in (mon | dev)]
    print(f"  LEAKED INTO TEST           : {leaked or 'none'}")


def main():
    rows = load_rows()
    if "--unfreeze" in sys.argv:
        man = json.loads(SPLITS.read_text(encoding="utf-8"))
        man["frozen"] = False
        SPLITS.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
        print("unfrozen — the NEXT run will re-draw validation from scratch. Every result already "
              "computed against the old validation set becomes unattributable; re-freeze at once "
              "unless that is what you want.")
        return 0
    if "--check" in sys.argv:
        if not SPLITS.exists():
            raise SystemExit(f"{SPLITS} does not exist — run without --check first.")
        man = json.loads(SPLITS.read_text(encoding="utf-8"))
        v, t = set(a["url"] for a in man["validation"]), set(a["url"] for a in man["test"])
        assert not (v & t), f"splits OVERLAP on {len(v & t)} article(s)"
        corpus = {r["url"] for r in rows}
        print(f"  disjoint: yes | validation {len(v)} | test {len(t)}")
        if v | t != corpus:
            print(f"  corpus DRIFTED: {len(corpus - (v | t))} article(s) in no split, "
                  f"{len((v | t) - corpus)} in a split but not in articles.csv — re-run to fix.")
        pinned = set(pinned_urls())
        missing = pinned - v
        print(f"  pins in validation: {len(pinned - missing)}/{len(pinned)}"
              + (f"   MISSING: {sorted(missing)}" if missing else ""))
        if missing:
            raise SystemExit("a pinned article is not in validation — re-run to fix.")
        for s in ("validation", "test"):
            ids = resolve(s, strict=False)
            print(f"  {s:<11} -> {len(ids)} ids: {ids[:6]}{' ...' if len(ids) > 6 else ''}")
        return 0
    val, test = select(rows)
    write(val, test)
    if "--freeze" in sys.argv:
        print("\n  FROZEN — re-running now only ever grows the test set. New articles land in test;"
              "\n  validation stays exactly these 20. Use --unfreeze to re-draw (and invalidate).")
    print(f"wrote {SPLITS.relative_to(ROOT)}")
    _report(val, test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
