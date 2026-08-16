# configs/ — one file per experiment

Every notebook in both tracks starts with the same line:

```python
EXPERIMENT = "main"
```

and reads everything else from `configs/<EXPERIMENT>.json`. Changing what a run does means editing
(or adding) a config file and re-running — not hunting for the right knob in nine notebooks.

## Why

A run of this pipeline is defined by ~30 settings spread over three to five notebooks that have to
agree with each other: all of them must point at the same `results/<model>/` tree, the type stage's
`run_tag` must be the relation stage's `input_tag`, both must canonicalize the same extraction
`variant`. When those live in notebook cells, the notebook always describes the *next* run, never
the one whose numbers you are writing up — which is what `notebooks/e2e/CURRENT_CONFIG.txt` was
hand-maintained to work around, and why it drifted. A config file *is* that snapshot, kept honest by
being the thing the run actually reads.

## Layout

| file | what it is |
|---|---|
| `_base.json` | every default, in one place. Not runnable on its own — experiments extend it. |
| `main.json` | the reported configuration, and what loads when no experiment is named. Every ablation below is measured against it. |
| `pilot.json` | `main` on the 4 development articles, for smoke-testing a change cheaply. |
| `consolidate-at-end.json` | ONE global re-consolidation after the last article — the pipeline's behaviour up to 2026-08-16, now an ablation against the every-5 default. |
| `no-review.json`, `no-rerank.json`, `no-rerank-inline.json`, `no-rerank-cons.json`, `inline-vocab.json`, `one-call-consolidation.json`, `no-generalize.json`, `no-guard-schema-parent.json`, `no-consolidation.json`, `concise-prompt.json` | ablations: each isolates ONE mechanism. `no-rerank` turns off both cross-encoder stages; the `-inline` / `-cons` pair splits them, because they act on different pools. |
| `qwen3-14b.json`, `qwen3-32b.json`, `qwen3-32b-think.json`, `llama33-70b.json` | open-weight model arms (need the vLLM tunnel — see `cluster/README.md`). |
| `legacy.json` | HISTORICAL: the UNTAGGED filenames used before run tags existed. The files it reached now live in `results_archive/`, so it reads nothing in a clean tree; a run made with it cannot be attributed to a configuration and must not be reported. |

An experiment file states only what it *changes*:

```json
{
  "extends": "main",
  "experiment": "no-rerank",
  "description": "ABLATION: no cross-encoder anywhere ...",
  "6d": {"rerank": false, "rerank_consolidate": false},
  "6e": {"rerank": false, "rerank_consolidate": false}
}
```

No `run_tag`: it defaults to `no_rerank`, after the experiment. The outputs of this run are
`*_dedup_no_rerank*`, and its record is `results/gemini-2.5-flash/_runs/e2e__dedup__no_rerank.json`.

`extends` chains, so an ablation can be a delta on another ablation. A section may also override a
top-level key for its stage alone — `legacy.json` uses that to give the staged n-ary notebooks
`"variant": ""`, because their graphs on disk predate the variant convention.

## Keys

Top level, shared by every stage:

| key | meaning |
|---|---|
| `results_model` | which `results/<name>/` tree to read and write. Also selects the backend by default. |
| `backend` | which model does the work. `null` = same as `results_model`. Set it differently only to process one model's outputs with another (a judge setup). |
| `selected_articles` | list of ids, or `null` for the whole corpus. **The order is the run**: canonicalization is incremental, so what article *k* mints depends on what *k−1* added. |
| `variant` | names the reification prompt ARM (6c / nb6b). Every later stage reads `nary_graph<_variant>.json`, so a new prompt needs a new variant or it silently overwrites the arm it was meant to be compared with. |
| `run_tag` | tags everything a run writes, so a new configuration never overwrites an older one's outputs. **Leave it unset** and it becomes the experiment name (`no-rerank` -> `no_rerank`) — unique by construction, because config filenames are. Set it only when the tag must differ from the name. |
| `repeat` | *n*-th sample of the SAME arm: appends `_r<n>` to `run_tag`, so `main` run three times is `main_r1/2/3`. Usually given as `REPEAT=<n>` in the environment rather than written into the file. |
| `input_tag` | which upstream run the canonicalization stages read. `null` = same as `run_tag` (repeat included), which is right when the whole chain runs as one experiment. Set it to read an EXISTING type-stage run while varying only the relation stage. |
| `split` | `"validation"` / `"test"`: takes the article list from `dataset/splits.json` and appends `_val` / `_test` to the tag, so the two can never overwrite each other. Usually given as `SPLIT=val` in the environment. |
| `overwrite` | allow a stage to re-run over its own finished outputs. Normally false — see *Re-running* below. |

Then one section per notebook — `"2"`, `"4"`, `"6"`, `"6b"`, `"7"`, `"7b"` (staged) and `"6c"`,
`"6d"`, `"6e"` (e2e). The staged notebooks take the **same knobs as their e2e twins**, which is what
makes "run both tracks under one experiment and compare" a real operation rather than a hand-matching
exercise.

The knobs that decide results, in the order they bite:

| key | stage | meaning |
|---|---|---|
| `prompt` | 6c/6b, 7/7b/6e | which prompt arm to send. Arms live in the notebook as data; the config picks one. |
| `review` | 6c/6b | run the second review/repair call over the draft graph. |
| `inline_vocab_candidates` | type + relation stages | `false` = the per-article selection call sees schema.org only and cannot reuse an earlier mint, so duplicates are produced by design and collapsed globally afterwards. `true` = the grown vocabulary is offered inline. **Judge `false` on post-consolidation vocabulary size, not on the inline mint rate.** |
| `top_k_match` | " | how many candidates the selection call sees. |
| `rerank` / `rerank_from` | " | cross-encode the shortlist e5 fetched, then cut to `top_k_match`. The depth is the ceiling — the cross-encoder can only promote what e5 already retrieved. |
| `anchor_mode` / `anchor_penalty` | relation stages | how each property is represented in the document pool (one blended vector vs. several surface anchors). |
| `parallel_llm` / `workers` | relation stages | overlap the per-article calls. Only valid while `inline_vocab_candidates` is `false`, since that is what makes the calls independent; results are unchanged either way, because they are still *applied* in article order. |
| `consolidate` | " | `false` skips the global re-consolidation LLM pass but still writes the `*_consolidated` files (identical to the raw ones), so the chain runs and the metrics record a real zero. |
| `consolidate_every` | " | **`5` by default**: re-consolidate every 5 articles and feed the merged vocabulary back, so article 6 builds on the collapsed term set. Bounds both the vocabulary and the consolidation prompt — the largest in the pipeline, and the one that grows with the vocabulary. The passes are accumulated into one cumulative report and the per-article graphs are refreshed once at the end, so a term merged in pass 1 and again in pass 2 still resolves to its final survivor. `0` restores the single end-of-run pass (`consolidate-at-end.json`). |
| `consolidate_include_schema` | " | whether schema.org joins the consolidation candidate pool, letting a mint be folded into a standard term. |
| `subtype_mode` / `subprop_mode` | " | `subclass`/`subproperty` keeps a narrower term and re-parents it; `merge` collapses it; `off` merges only exact synonyms. |
| `allow_generalize` | relation stages | consolidation may mint ONE new general property to collapse several specifics into. |
| `two_call_consolidation` | " | merge first, then build hierarchy over the survivors — so "subproperty" cannot be used to dodge a merge. |
| `guard_schema_parent` | " | refuse any re-parent that moves a property off a schema.org parent onto a minted one. |
| `fresh` | canonicalization | `true` rebuilds the vocabulary from empty over every selected article (what a reported run does). |

## Commands

```bash
cd notebooks
python -m experiment_config list           # every experiment, its tree and the tag it writes
python -m experiment_config show main      # the fully-resolved config (paste into an appendix)
python -m experiment_config check main     # cross-stage consistency + tag collisions, before spending GPU time
python -m run_manifest list                # what has actually been RUN, and from which commit
```

## Re-running

Every stage records itself in `results/<model>/_runs/<run_id>.json` and **refuses to start** when
that record says the same stage already finished under the same tag. The refusal names the three
ways past it, which are the three things you might actually mean:

| you mean | do |
|---|---|
| another sample of the same arm (variance) | `REPEAT=2 ./run_e2e.sh` -> tag `main_r2`, its own manifest |
| a different configuration | a new experiment file; its tag comes from its name |
| replace this run in place | `PHASE2_OVERWRITE=1`, or `"overwrite": true` in the config |

Sampling is not deterministic at temperature 0 — two identical runs of this corpus gave 69 vs 91 raw
minted properties (65 vs 67 after consolidation). A reported arm is three repeats, given as mean +
range, on POST-consolidation numbers.

## The four environment overrides

They exist so that a matrix of runs needs no config file per cell, and each one becomes part of the
run's tag — the run stays identifiable from its filenames alone.

| variable | does | tag |
|---|---|---|
| `EXPERIMENT=no-rerank` | which config file | `no_rerank` |
| `SPLIT=val` \| `test` | which articles, from `dataset/splits.json` | `…_val` / `…_test` |
| `MODEL=qwen3-32b-think` | which results tree + backend, without a config per (experiment, model) | *(the tree changes, not the tag)* |
| `REPEAT=2` | another sample of the same arm | `…_r2` |

```bash
MODEL=llama33-70b SPLIT=val REPEAT=2 EXPERIMENT=no-rerank ./run_e2e.sh 6d 6e
#  -> results/llama33-70b/…_dedup_no_rerank_val_r2…
```

`REPEAT` re-runs the CANONICALIZATION under a new tag; it deliberately does not re-extract, because
6c's output is named by `variant` alone and a second extraction sample is a new `variant`, not a
repeat. The guard in `run_manifest.py` enforces that and explains it if you hit it.

## Adding one

Copy the closest existing file, change `experiment` and `description`, and state only the keys that
differ. Leave `run_tag` out: it defaults to the experiment name, which is what stops it overwriting
the run you are comparing it to, and `check` reports it as an error if some other experiment already
writes that tag in the same tree. Then `python -m experiment_config check <name>` before you start.
