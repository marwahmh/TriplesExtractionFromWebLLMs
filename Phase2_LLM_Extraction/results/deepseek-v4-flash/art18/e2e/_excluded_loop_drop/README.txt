art18 6e-stage outputs, moved aside 2026-08-26.

art18 is one of five test articles dropped from the deepseek-v4-flash main_test run
(configs/deepseek-loop-drop.json): its 6c extraction is a repetition loop, 801 Event
hubs built from 2 distinct labels.

The two files dated 17:20 are from the CRASHED run of 6e (the one art28 killed). The
_extended_consolidated.json dated 18:42 is not a real result either: the clean 75-article
re-run never processed art18, but refresh_ext_after_consolidation() globs the tree and
rewrote the stale extended graph against the new vocabulary.

Nothing was corrupted -- all 45 of art18's relations resolved to schema.org properties,
so no vocabulary ids dangled and the run's closing invariant check passed. They are moved
here only so downstream stages that glob art*/e2e/ do not silently pick them up.

metrics_relations_dedup_main_test.json was already computed over the correct 75 and needs
no change. The run_e2e.sh console banner said "6526 over 76 article(s)" because it globs;
the true figure is 6481 over 75.

The 6c/6d outputs for art18 are left in place, so a re-extraction can still use them.
