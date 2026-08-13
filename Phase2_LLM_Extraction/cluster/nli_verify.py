#!/usr/bin/env python
"""Score every extracted assertion with the entailment verifier, on a GPU.

This is the expensive half of notebook 9c, lifted out so it can run on a Lab-IA A6000 instead of a
laptop. It reads the judgment logs nb9/nb9b wrote, scores each assertion against its article, and
writes results/<model>/<id>/<track>/nli_verdicts.json — exactly the file 9c reads back. The cheap
half (agreement, kappa, disagreement sheet, threshold calibration) stays in the notebook and needs
no GPU and no model.

Both halves import notebooks/nli_core.py, so the scoring here is the same code the notebook would
have run locally — not a re-implementation that could drift.

    python nli_verify.py --root /path/to/Phase2_LLM_Extraction --results-model gemini-2.5-flash
    python nli_verify.py --articles art1 art5 --batch-size 64 --force
"""

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None,
                   help="Phase2_LLM_Extraction root (default: inferred from this file's location)")
    p.add_argument("--results-model", default="gemini-2.5-flash",
                   help="results/<RESULTS_MODEL>/ — whose graphs are being verified")
    p.add_argument("--track", default="e2e", help="results/<id>/<TRACK>/")
    p.add_argument("--articles", nargs="*", default=None,
                   help="article ids (default: every one that has a judgment log)")
    p.add_argument("--sources", nargs="*", default=["evidence", "fulltext"],
                   choices=["evidence", "fulltext"],
                   help="which judgment log(s) supply the assertions to score")
    p.add_argument("--model", default=None, help="verifier checkpoint (default: nli_core's)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--window", type=int, default=3, help="sentences per localisation window")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64,
                   help="an A6000 handles 64-128 at 512 tokens; halves itself on OOM")
    p.add_argument("--no-sentence-pass", action="store_true",
                   help="skip the localisation pass (~10x fewer forward passes, no best-window)")
    p.add_argument("--device", default=None, help="cuda / mps / cpu (default: best available)")
    p.add_argument("--force", action="store_true", help="re-score articles that already have verdicts")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    root = args.root or Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "notebooks"))
    import nli_core                                     # noqa: E402  (needs the path above)

    results_dir = root / "results" / args.results_model
    article_dir = root / "dataset" / "articles"
    if not results_dir.is_dir():
        sys.exit(f"no such results dir: {results_dir}")

    def natkey(s):
        import re
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]

    def usable(a):
        return ((article_dir / f"{a}.txt").exists()
                and nli_core.has_claims(results_dir, a, args.track, args.sources))

    if args.articles:
        missing = [a for a in args.articles if not usable(a)]
        if missing:
            print(f"!! skipped — no {args.sources} judgment log or no article text: {missing}")
        articles = [a for a in args.articles if usable(a)]
    else:
        articles = sorted((p.name for p in results_dir.iterdir() if p.is_dir() and usable(p.name)),
                          key=natkey)
    if not articles:
        sys.exit(f"nothing to score under {results_dir} for sources={args.sources}")

    model_name = args.model or nli_core.DEFAULT_MODEL
    print(f"loading {model_name} ...", flush=True)
    t0 = time.perf_counter()
    verifier = nli_core.Verifier(model_name, device=args.device,
                                 max_len=args.max_len, batch_size=args.batch_size)
    print(f"{model_name} on {verifier.device}  ({verifier.n_params/1e6:.0f}M params, "
          f"{time.perf_counter()-t0:.1f}s)", flush=True)

    # Never score with an unverified model: a checkpoint can load without a single warning and
    # still carry an untrained head, which would produce a whole corpus of meaningless verdicts.
    print("\n--- orientation probe ---", flush=True)
    verifier.probe(threshold=args.threshold)

    config = {"model": model_name, "threshold": args.threshold, "window": args.window,
              "max_len": args.max_len, "sentence_pass": not args.no_sentence_pass,
              "probe_auc": verifier.probe_auc, "device": verifier.device,
              "scored_on": "cluster"}

    print(f"\n--- scoring {len(articles)} article(s): {articles} ---", flush=True)
    total = 0
    for aid in articles:
        out_path = results_dir / aid / args.track / nli_core.VERDICTS_NAME
        claims = nli_core.load_claims(results_dir, aid, args.track, args.sources)
        if out_path.exists() and not args.force:
            # A verdict file can be truncated or empty if an earlier run died mid-write, and an
            # unreadable cache must not wedge the job — just re-score it.
            try:
                cached = json.loads(out_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
                print(f"{aid:8} unreadable cache ({type(exc).__name__}) — re-scoring", flush=True)
                cached = {}
            if cached.get("model") == model_name and len(cached.get("scores", {})) == len(claims):
                print(f"{aid:8} {len(claims):5} claims — cached, skipping", flush=True)
                continue
        t0 = time.perf_counter()
        text = (article_dir / f"{aid}.txt").read_text(encoding="utf-8")
        scored = verifier.score_claims(text, [c["assertion"] for c in claims],
                                       window=args.window,
                                       sentence_pass=not args.no_sentence_pass,
                                       threshold=args.threshold)
        by_id = {c["id"]: s for c, s in zip(claims, scored)}
        nli_core.write_verdicts(results_dir, aid, args.track, by_id, config)
        verifier.free_gpu()
        n_sup = sum(s["nli_supported"] for s in scored)
        total += len(claims)
        print(f"{aid:8} {len(claims):5} claims — {n_sup:5} supported "
              f"({n_sup/max(1,len(claims)):6.1%})  {time.perf_counter()-t0:6.1f}s", flush=True)

    print(f"\ndone — {total} assertion(s) scored across {len(articles)} article(s)")
    print(f"verdicts written to {results_dir}/<id>/{args.track}/{nli_core.VERDICTS_NAME}")
    print("now run sections 5-7 of 9c_nli_verification.ipynb locally (no GPU needed)")


if __name__ == "__main__":
    main()
