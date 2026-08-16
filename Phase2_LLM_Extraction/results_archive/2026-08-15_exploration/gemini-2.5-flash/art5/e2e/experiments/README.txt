E2E retrieval-experiment result files (archived). Full write-up with aims + findings:
  notebooks/e2e/experiments/EXPERIMENTS.txt

Baseline = 6e (files WITHOUT a suffix, in the parent e2e/ folder).
Suffix -> experiment:
  _fullschema  6f  full schema.org catalog instead of top-10 retrieval
  _hybrid      6g  embedding + HARD type-signature lane (RRF)
  _dualemb     6i  embedding + SOFT signature-embedding lane (RRF)
  _quota       6j  separate top-10 schema.org + top-3 vocab      [WINNER]
  _recon       6k  schema-only inline + per-article mint reconsolidation

File kinds:
  nary_graph_dedup_extended_<suffix>.json             canonicalized graph
  nary_relation_matching_review_dedup_<suffix>.json   per-phrase candidate/choice audit
  nary_relation_vocabulary_dedup_<suffix>.json        grown property vocabulary   (_global/e2e/experiments)
  token_usage_dedup_<suffix>.json                     API token usage             (_global/e2e/experiments)
  *_vs_*.json                                         cross-experiment comparison reports (_global/e2e/experiments)
