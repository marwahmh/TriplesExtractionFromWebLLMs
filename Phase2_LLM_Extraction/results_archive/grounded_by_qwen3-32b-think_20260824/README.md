Per-article grounding outputs judged by qwen3-32b-think, copied 2026-08-24 before re-grounding
everything with gpt-oss-120b.

nary_graph_grounded_fulltext.json / _pruned.json carry NO judge tag in their filename, so a
re-ground with a different judge replaces them in place. The _global/grounding_summary_<judge>_*
files ARE tagged and were not at risk. Kept so the qwen-judged graphs remain available as a
cross-judge comparison and as nb10's original input.
