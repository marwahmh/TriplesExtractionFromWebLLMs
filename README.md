# Triples Extraction from the Web with LLMs

Master's thesis project: building knowledge graphs from web news articles with LLMs, and
canonicalizing the extracted vocabulary against **schema.org**.

The work has two phases:

1. **Phase 1 — What structured data already exists on the web.** Large-scale analysis of the
   [Web Data Commons](http://webdatacommons.org/) (WDC) JSON-LD corpus and Common Crawl to measure
   which schema.org classes news and fact-checking sites actually annotate, motivating why LLM
   extraction is needed at all.
2. **Phase 2 — LLM extraction pipeline.** Extract entities and relations from 100 news articles
   with an LLM (Gemini and open-weight models), reify facts into n-ary event graphs, and
   canonicalize every free-form type and relation to schema.org via embedding retrieval plus an LLM
   judge — minting a new term only when nothing in schema.org fits. Evaluated for precision
   (grounding judges) and recall (MINE-style benchmark).

## Repository layout

| Directory | What it is |
|---|---|
| [Phase2_LLM_Extraction/](Phase2_LLM_Extraction/) | **The main pipeline** — notebooks, configs, run scripts, results. Has its own detailed [README](Phase2_LLM_Extraction/README.md). |
| [WDC_download_analysis/](WDC_download_analysis/) | Phase 1: streaming/parallel analysis of the WDC JSON-LD dump — per-class and per-domain statistics, figures, and a LaTeX report ([wdc_analysis_report.tex](WDC_download_analysis/wdc_analysis_report.tex)). |
| [CommonCrawlExtraction/](CommonCrawlExtraction/) | Phase 1: extracting embedded JSON-LD directly from Common Crawl HTML records ([extract_jsonld_commoncrawl.ipynb](CommonCrawlExtraction/extract_jsonld_commoncrawl.ipynb)). |
| [SourceDomainsExtraction/](SourceDomainsExtraction/) | Curated news + fact-check domain lists used to filter WDC/Common Crawl and to pick article sources. |
| [wdc_class_data/](wdc_class_data/) | Raw per-class WDC quad dumps (NewsArticle, ClaimReview, Event, …) used by the Phase 1 notebooks. |
| [Master-Thesis-Document/](Master-Thesis-Document/) | The thesis manuscript (LaTeX, `Thesis.tex` + `ChapterN.tex`). |
| [jdse2026/](jdse2026/) | LNCS extended abstract submitted to JDSE 2026. |
| [poster_iswc2026/](poster_iswc2026/) | ISWC 2026 poster sources. |

## Setup

Python 3.10+ with Jupyter. The pipeline needs:

```bash
pip install google-genai sentence-transformers numpy pandas beautifulsoup4 requests
export GEMINI_API_KEY=...   # https://aistudio.google.com/apikey  (or GOOGLE_API_KEY)
```

The embedding retrieval uses a local sentence-transformer (`intfloat/e5-base-v2`, downloaded on
first use) — no API key needed for that step. The Phase 1 notebooks only need the standard
scientific stack (pandas, matplotlib) plus enough disk for the WDC dumps.

## Running the pipeline (Phase 2)

Everything below runs from `Phase2_LLM_Extraction/`. Full details — the notebook-by-notebook table,
config keys, provenance/manifest system, ablations — are in
[Phase2_LLM_Extraction/README.md](Phase2_LLM_Extraction/README.md).

**1. One-time schema.org enrichment.** Run `notebooks/0_shared/1_schema_enrichment.ipynb` and
`5_property_enrichment.ipynb` once to parse and Gemini-enrich the schema.org class and property
pools into `schema_org_files/` (already present in this repo).

**2. Dataset.** `dataset/articles/art1..100.txt` is the scraped 100-article news corpus
(`dataset/build_dataset.ipynb` rebuilds it from `dataset/articles.csv`). The corpus is split once
into validation (20 articles) and test (80) by `python dataset/select_splits.py`.

**3. Run a track on Gemini (the default backend).** Every run is configured by one file in
`configs/` — `main.json` is the reported configuration, and its base (`configs/_base.json`) already
selects Gemini: `"backend": null` means "call the Gemini API directly via `google-genai`",
`"gemini_model": "gemini-2.5-flash"` is the model, and `"results_model": "gemini-2.5-flash"` is the
results tree the run writes to. So running on Gemini requires no backend setup at all — only the
API key:

```bash
export GEMINI_API_KEY=...             # or GOOGLE_API_KEY; from https://aistudio.google.com/apikey
cd Phase2_LLM_Extraction
python -m notebooks.experiment_config check main   # sanity-check the config before spending tokens

# e2e track: one extraction call per article (6c), then type (6d) and relation (6e) canonicalization
SPLIT=val  SKIP_SERVER_CHECK=1 ./run_e2e.sh        # 20 validation articles first (cheap dry run)
SPLIT=test SKIP_SERVER_CHECK=1 ./run_e2e.sh        # the 80 reported test articles

# staged track: entities (2) -> types (4) -> relations (6) -> properties (7)
SPLIT=val SKIP_SERVER_CHECK=1 ./run_staged.sh      # or a subset: ./run_staged.sh 2 4 6b 7b
```

`SKIP_SERVER_CHECK=1` matters on Gemini: the run scripts otherwise preflight a local vLLM server,
which a Gemini run does not need. Useful overrides, each of which becomes part of the run tag so
runs never overwrite each other: `EXPERIMENT=<name>` picks another config (an ablation),
`REPEAT=2` re-runs the same arm under a new tag (temperature 0 is greedy, not deterministic), and
`PHASE2_OVERWRITE=1` deliberately redoes a finished run in place — without it a stage refuses to
start over an existing manifest.

Which stages actually call Gemini: the extraction notebooks (2, 6, 6b, 6c) on every article; the
enrichment notebooks (1, 5) once; and the judge + consolidation steps inside the canonicalization
notebooks (4, 7, 7b, 6d, 6e). The embedding retrieval inside those same notebooks is local and
free. Rate limits are handled by the `workers` config key (default 4): the Gemini paid tier is
comfortable at 4–8 parallel articles, the free tier (10 requests/min) wants `"workers": 1–2` —
override it in your experiment file rather than editing notebooks. Thinking is on
(`thinking_budget: -1`, dynamic), and every notebook ends with a token report — prompt, output, and
thinking tokens per article, thinking broken out because Gemini bills it as output but excludes it
from the API's output count — persisted to `results/gemini-2.5-flash/_global/token_usage_<nb>.json`.

To run interactively instead, open the notebooks under `notebooks/e2e/` or `notebooks/two_staged/`
in order; the first cell of each is just `EXPERIMENT = "main"`. Note the canonicalization notebooks
(4, 7, 7b, 6d, 6e) must run their articles sequentially — they grow one shared vocabulary, so order
changes the output — while the extraction notebooks parallelize freely.

Outputs land in `results/<model>/art<N>/` (per-article graphs) and `results/<model>/_global/`
(grown vocabularies, consolidation reports). Every run writes a manifest to
`results/<model>/_runs/` recording config, git commit, outputs, and token spend; a stage refuses to
silently overwrite a finished run.

**4. Evaluation.**

```bash
./run_grounding.sh gpt-oss-120b   # precision: judge every extracted assertion against the article text (nb9b)
./run_mine.sh llama33-70b         # recall: MINE-style benchmark, 15 expected facts per article (nb10)
```

Both sweep every finished model/track/split combination with a single judge (one judge across all
arms is required for comparability; `9d_judge_benchmark.ipynb` is the 500-item human-labelled
benchmark that picked it). `notebooks/0_shared/8_kg_visualization.ipynb` renders any graph as PNG +
interactive HTML.

**Open-weight models.** Qwen3, Llama 3.3, etc. are served with vLLM on the Lab-IA GPU cluster; the
notebooks swap backends via `notebooks/llm_backend.py`. Setup and per-session runbook:
[Phase2_LLM_Extraction/cluster/README.md](Phase2_LLM_Extraction/cluster/README.md).

## Building the thesis PDF

```bash
cd Master-Thesis-Document
pdflatex -synctex=1 Thesis.tex && bibtex Thesis && pdflatex -synctex=1 Thesis.tex && pdflatex -synctex=1 Thesis.tex
```
