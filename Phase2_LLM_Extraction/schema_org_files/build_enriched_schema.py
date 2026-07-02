"""Build schema_classes_enriched.json from schema_classes.json + hand-authored definitions.

Enriched definitions live in enriched_defs/*.json (each a flat {label: definition} map, merged
in filename order so later files can override earlier ones). They replace the often-terse
schema.org comment with a richer Claude-authored gloss: expanded definition + synonyms/aka +
example instances + occasional "distinct from <sibling>" cue. These discriminative tokens are
what a bi-encoder needs to separate look-alike sibling types. Kept short on purpose — long
generic prose dilutes a mean-pooled embedding.

This is a one-time, reusable artifact: enrich the WHOLE schema.org vocabulary once, then reuse
schema_classes_enriched.json across every article. Types still missing a definition fall back
to their original schema.org comment (enriched=False), so the output is always a complete
933-row file with the same shape as schema_classes.json.
"""
import glob
import json
import re
from pathlib import Path

# All paths are relative to this script's own folder (schema_org_files/), so it runs correctly
# regardless of the current working directory.
HERE = Path(__file__).parent

# Contrastive clauses like "Distinct from State, which is ..." help the LLM judge (which reads
# negation correctly) but HURT the bi-encoder embedding: a mean-pooled model has no notion of
# negation, so naming the confusable type just injects its tokens and pulls the two vectors
# CLOSER. So we keep these clauses in enriched_comment (for the judge) but strip them from the
# text we actually embed.
_CONTRAST_RE = re.compile(r"\s*(?:Distinct from|Not to be confused with)[^.]*\.", re.IGNORECASE)

def embedding_comment(text):
    """Drop contrastive/negation clauses so only additive, embed-friendly text remains."""
    return _CONTRAST_RE.sub("", text).strip()

# Merge every part file in enriched_defs/ (sorted) into one label -> definition map.
ENRICHED = {}
for fp in sorted(glob.glob(str(HERE / "enriched_defs" / "*.json"))):
    part = json.loads(Path(fp).read_text())
    ENRICHED.update({k: v for k, v in part.items() if v and v.strip()})

src = json.loads((HERE / "schema_classes.json").read_text())
out = []
n_enriched = 0
for c in src:
    label = c["label"]
    enriched_comment = ENRICHED.get(label)
    is_enriched = enriched_comment is not None
    # enriched_comment keeps the full text (incl. "Distinct from ..."); the embedded text drops
    # the contrastive clause because it backfires for a bi-encoder.
    comment_for_embed = embedding_comment(enriched_comment) if is_enriched else c["comment"]
    n_enriched += is_enriched
    # Same embedding_text format as the baseline ("Label (path): comment"), so the ONLY thing
    # that changes for enriched types is the comment text -> isolates the effect of enrichment.
    head = f"{label} ({c['path']})" if c["path"] else label
    out.append({
        "id": c["id"],
        "label": label,
        "comment": c["comment"],                     # keep original schema.org comment for reference
        "enriched_comment": enriched_comment or "",  # "" when not enriched
        "enriched": is_enriched,
        "path": c["path"],
        "embedding_text": f"{head}: {comment_for_embed}",
    })

(HERE / "schema_classes_enriched.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(f"wrote {len(out)} classes, {n_enriched} enriched "
      f"({n_enriched/len(out):.0%}) -> schema_classes_enriched.json")
missing = [c["label"] for c in out if not c["enriched"]]
if missing:
    print(f"{len(missing)} still using original comment, e.g.: {missing[:15]}")
