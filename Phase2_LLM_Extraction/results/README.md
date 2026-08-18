# results/ — outputs of reporting runs

Empty on purpose. Everything produced before 2026-08-15 was exploratory (four development
articles, hand-typed run tags like `rr_orig2_mixedCons_SepMergeReparent_v3`) and now lives in
[`../results_archive/2026-08-15_exploration/`](../results_archive/2026-08-15_exploration/). Nothing
in this tree should be a number you cannot trace back to a config file and a git commit.

## Layout

```
results/<model>/                          one tree per results_model (config key)
  _runs/<run_id>.json                     THE RUN MANIFEST — see below
  _global/e2e/                            cross-article vocabularies, hierarchies,
  _global/                                consolidation reports, metrics_<stage><tag>.json
  art<N>/e2e/                             per-article graphs for the e2e track
  art<N>/two_staged/                      per-article outputs for the staged track
RUNS.md                                   generated index of every manifest in this tree
```

A file's name carries which run wrote it: `nary_graph_dedup_main_r1_extended_consolidated.json` is
`<base>_<variant>_<run_tag>_<stage suffixes>`. `variant` names the extraction arm (6c/6b), `run_tag`
names the canonicalization run.

## The run manifest

Every stage that starts writes `results/<model>/_runs/<run_id>.json` through
[`notebooks/run_manifest.py`](../notebooks/run_manifest.py). It is written automatically — the
notebooks' config cell opens it, the metrics cell at the bottom of each notebook closes it. One
manifest per `(track, variant, run_tag)`, holding for each stage:

- the **fully resolved config** the run actually read, not a description of it
- **git commit + branch + dirty flag** at the moment the stage started
- start / finish timestamps, host, python version, backend and model ids
- the **article order** (the vocabulary stages are incremental — order is part of the run)
- every **output file** the stage wrote, with size and mtime
- **token usage**, summed from the `token_usage*` files the stage produced. `"complete": false` means
  a ledger it read covers only part of the run, so the total is a floor — the ledger's `coverage`
  block names the articles. `RUNS.md` marks these with `*`.
- the **stage metrics** dict, the same one `metrics_<stage><tag>.json` holds
- **`attempts`**, when a later run superseded a finished stage under the same `run_id` (a repair run
  reusing a `run_tag`, typically): the previous finished record is kept here rather than dropped,
  oldest first, so the run it describes is still accountable.

So a number in the thesis resolves to a manifest, and a manifest resolves to a commit and a config.

Two known defects in the **validation** runs — partial token ledgers on the periodic stages, and
gemini staged `art2` coming from a repair run — are written up in
[`../REPORTING_PLAN.md`](../REPORTING_PLAN.md) §6b. Structural numbers are unaffected; cost numbers
should come from the test-split runs.

```bash
cd notebooks
python -m run_manifest list                      # every run on disk, one line each
python -m run_manifest list gemini-2.5-flash     # one tree
python -m run_manifest show e2e__dedup__main_r1  # the whole record
python -m run_manifest index                     # regenerate results/RUNS.md
python -m run_manifest diff main_r1 main_r2      # what differs between two runs' configs
```

## Naming a reporting run

`run_tag` defaults to the experiment name, so `main` writes `*_dedup_main*` without anyone typing a
tag. Repeats get an index — sampling is not deterministic even at temperature 0, so an arm is worth
running three times and reported as mean + range:

```bash
REPEAT=1 EXPERIMENT=main ./run_e2e.sh     # -> run_tag main_r1
REPEAT=2 EXPERIMENT=main ./run_e2e.sh     # -> run_tag main_r2
REPEAT=3 EXPERIMENT=main ./run_e2e.sh     # -> run_tag main_r3
```

A stage refuses to start when a manifest says that stage already finished under the same run id —
that is the guard against silently overwriting the run whose numbers are already written up. Bypass
it deliberately with `PHASE2_OVERWRITE=1` (or `"overwrite": true` in the config) when you mean to
redo a run in place.
