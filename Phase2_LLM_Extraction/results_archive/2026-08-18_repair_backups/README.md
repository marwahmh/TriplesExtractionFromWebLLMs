# 2026-08-18 repair backups

Pre-repair copies of files changed while closing out the validation runs, at the same relative paths
they hold under `results/`. Two repairs, both written up in [`../../REPORTING_PLAN.md`](../../REPORTING_PLAN.md) §6b:

- `*.json.bak-idcollision` — gemini staged. A one-article repair run for `art2` re-issued property
  ids `P0558`–`P0563`, already held by six consolidation-generalize terms, so six of art2's relations
  resolved to the wrong property. art2's copies were renumbered to `P0602`–`P0607`.
- `*.json.bak-typerefresh` — all three models, staged track. `nb4`'s post-consolidation refresh read
  the e2e field names on staged entity records, skipped every entity, and reported `0 nodes changed`;
  the type consolidation was replayed over the per-article data. Type fields only — the relation
  consolidation in `*_extended_consolidated.json` was asserted unchanged.

To restore one: copy it back over the live file with the `.bak-*` suffix stripped, then re-run the
affected stage's metrics cell (or `notebooks/run_metrics.py`) so the numbers match the files again.
