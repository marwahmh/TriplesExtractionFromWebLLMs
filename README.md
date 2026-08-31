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

**3. Pick an experiment, run a track.** Every run is configured by one file in `configs/`
(`main.json` is the reported configuration). Headless:

```bash
cd Phase2_LLM_Extraction
EXPERIMENT=main ./run_e2e.sh          # e2e track:    6c -> 6d -> 6e  (one extraction call, then canonicalize)
EXPERIMENT=main ./run_staged.sh       # staged track: 2 -> 4 -> 6 -> 7  (entities, types, relations, properties)
SPLIT=val ./run_e2e.sh                # validation articles (SPLIT=test for the reported set)
MODEL=qwen3-32b-think ./run_e2e.sh    # open-weight model via a vLLM server instead of Gemini
```

Or interactively: open the notebooks under `notebooks/two_staged/` and `notebooks/e2e/` in order and
set the `EXPERIMENT` name in the first cell. Sanity-check a config before spending tokens with
`python -m experiment_config check main` (from `notebooks/`).

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
