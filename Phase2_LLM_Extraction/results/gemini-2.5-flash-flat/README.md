# gemini-2.5-flash-flat — the naive baseline tree

Written by `notebooks/e2e/6f_flat_schema_baseline.ipynb`. Same model as `results/gemini-2.5-flash/`
(gemini-2.5-flash, dynamic thinking, temperature 0, no output cap), same articles, **no pipeline**:
the whole schema.org vocabulary (933 classes + 1,521 properties, the pool nb1/nb5 built for
retrieval) is placed in the system prompt after a one-sentence task ("extract a typed knowledge
graph of the entities and the relations between them, using only the labels listed") and a JSON
skeleton of `{entities: [{id, text, type}], relations: [{subject_id, property, object_id}]}`.
ONE call. No retrieval, no shortlist, no minting, no review, no consolidation, and no instructions
about coreference, evidence, events or dates, so the graphs carry entities and relations only.

It is a separate tree so nothing here can be mistaken for a pipeline run, and so nb9b / nb10 can
be pointed at it by name.

## Files

    <art>/e2e/nary_graph_dedup_flat_<split>_r<n>_extended_consolidated.json
        The graph. The name copies the pipeline's FINAL graph name on purpose: nb10 derives
        `nary_graph_dedup_<MINE_TAG>_extended_consolidated.json` from its tag and nb9b takes
        GROUND_GRAPH, so both score this arm without an edit. Nothing in it was extended or
        consolidated (`"consolidated": false`, `"method": "flat_schema_single_call"`).
        Every entity / node carries `type_source` and every relation `property_source`:
        `schema.org` when the emitted label is exactly a listed term, `invalid` otherwise
        (`type_check` / `property_check` separate a case-only miss from an invented term).
    _global/e2e/metrics_flat_dedup_flat_<split>_r<n>.json
        Term validity (strict / lenient), triples on a valid property, distinct valid vocabulary,
        graph sizes, token cost, per-article rows.
    _global/e2e/token_usage_dedup_flat_<split>_r<n>.json
        Per-article prompt / output / thinking tokens.

## Scoring it with the pipeline's instruments (cluster, free)

    GROUND_MODEL=gemini-2.5-flash-flat GROUND_TAG=flat_val_r1 \
    GROUND_GRAPH=nary_graph_dedup_flat_val_r1_extended_consolidated.json   -> nb9b
    MINE_MODELS=gemini-2.5-flash-flat MINE_TAG=flat_val_r1                   -> nb10

There is no `_runs/` manifest here: the notebook is not a stage of the config-driven chain.
