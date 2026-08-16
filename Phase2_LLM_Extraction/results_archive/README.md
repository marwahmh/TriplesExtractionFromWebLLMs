# results_archive/ — superseded outputs, kept for provenance

Frozen. Nothing here is read by a notebook and nothing here should be reported. It is kept because
the reasoning in the write-up refers to these runs and because re-deriving them costs API budget.

## 2026-08-15_exploration/

The whole `results/` tree as it stood on 2026-08-15, moved aside when the repo was prepared for the
reporting runs. What it contains:

- **Four development articles only** — art1, art2, art4, art5 — not the corpus in
  `dataset/articles.csv`. Every count in these files is over those four.
- Five model trees: `gemini-2.5-flash` (the bulk), `qwen3-14b`, `qwen3-32b`, `qwen3-32b-think`,
  `llama33-70b`.
- Run tags invented one at a time as the pipeline was being designed: `dedup`, `dedup_rr`,
  `dedup_schemainline_rr`, `dedup_rr_concise`, `dedup_rr_concise2`, `dedup_rr_orig2`,
  `dedup_rr_orig2_mixedCons`, `..._SepMergeReparent`, `..._SepMergeReparent_v3`. They name *what was
  changed*, not *which experiment produced them*, and several were produced by notebooks whose
  settings have since been edited — which is exactly the reachability problem the manifest in
  `results/` now fixes.
- Grounding judgments, NLI verdicts and the manual grounding sample for those four articles.

The measured comparison between these arms, and the reasoning drawn from it, is written up in
[`../notebooks/e2e/CURRENT_CONFIG.txt`](../notebooks/e2e/CURRENT_CONFIG.txt) §5–§7 — including the
run-to-run variance measurement (69 vs 91 raw mints under identical settings) that is the reason
reporting runs are repeated three times.

To read one of these files from a notebook, point `results_model` at a copy under `results/`; the
notebooks resolve their root by looking for a sibling `results/` + `dataset/`, so they will not
reach into this directory on their own.
