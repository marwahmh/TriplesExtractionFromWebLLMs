"""Self-contained interactive knowledge-graph viewer (vis-network, no CDN).

Used by notebooks/0_shared/8_kg_visualization.ipynb. `write_html()` takes the same
(nodes, edges) structures the notebook's `load_kg()` produces and emits ONE offline
HTML file with a control panel:

  * filters      — entity type, vocabulary source (schema.org / minted / open) for both
                   relations and entity types, node kind, confidence, leaf/isolated nodes
  * colour modes — by entity type, by vocabulary source, by node kind
  * layouts      — force, concentric (n-ary inner ring / entities outer), hierarchical
  * overlap      — post-stabilization label-aware de-overlap pass ("Tidy")
  * inspection   — click a node/edge for its full record (definition, justification,
                   evidence, confidence), searchable triples table synced to the filters

The vis-network bundle shipped with pyvis is inlined, so the file works offline and
survives being emailed / dropped in a thesis appendix.
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------------------
# vis-network bundle (inlined so the HTML is self-contained)
# ---------------------------------------------------------------------------------------

_VIS_CACHE: dict[str, str] = {}


def _vis_js() -> str:
    """Return the vis-network UMD bundle source, preferring the copy pyvis ships."""
    if "js" in _VIS_CACHE:
        return _VIS_CACHE["js"]
    cands: list[Path] = []
    try:
        import pyvis  # noqa: F401

        base = Path(pyvis.__file__).parent
        cands += sorted(base.glob("**/vis-network.min.js"), reverse=True)
    except Exception:
        pass
    for p in cands:
        try:
            _VIS_CACHE["js"] = p.read_text(encoding="utf-8", errors="ignore")
            return _VIS_CACHE["js"]
        except Exception:
            continue
    raise RuntimeError(
        "vis-network.min.js not found — `pip install pyvis` (its bundled copy is inlined)."
    )


# ---------------------------------------------------------------------------------------
# payload construction
# ---------------------------------------------------------------------------------------

# raw-record keys that are noise in the details panel (ids/duplicates of shown fields)
_SKIP_FIELDS = {
    "id", "mentions", "type_oie", "phrase_directions", "subject_id", "object_id",
}
_FIELD_ORDER = [
    "text", "label", "node_kind", "type", "type_source", "type_parent", "type_definition",
    "type_justification", "type_vocab_id", "type_is_new_to_vocab", "aliases",
    "normalized_value", "normalized_end", "is_main",
    "subject_text", "property", "property_source", "property_parent", "property_definition",
    "property_justification", "vocab_id", "property_is_new_to_vocab", "object_text",
    "relation_phrases", "confidence", "evidence",
]
_PRETTY = {
    "text": "text", "label": "label", "node_kind": "node kind", "type": "type",
    "type_source": "type source", "type_parent": "parent type", "type_definition": "definition",
    "type_justification": "justification", "type_vocab_id": "vocab id",
    "type_is_new_to_vocab": "new to vocab", "aliases": "aliases",
    "normalized_value": "normalized", "normalized_end": "normalized end", "is_main": "main event",
    "subject_text": "subject", "property": "property", "property_source": "property source",
    "property_parent": "parent property", "property_definition": "definition",
    "property_justification": "justification", "vocab_id": "vocab id",
    "property_is_new_to_vocab": "new to vocab", "object_text": "object",
    "relation_phrases": "relation phrase(s)", "confidence": "confidence", "evidence": "evidence",
}


def _flatten_strings(v) -> list[str]:
    """Evidence arrives as str | [str] | [[str]] depending on the stage that wrote it."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            out += _flatten_strings(x)
        return out
    return [str(v)]


def _fields(raw: dict | None) -> list[list[str]]:
    """Ordered [label, value] pairs from a raw entity / n-ary / relation record."""
    if not raw:
        return []
    keys = [k for k in _FIELD_ORDER if k in raw] + [
        k for k in raw if k not in _FIELD_ORDER and k not in _SKIP_FIELDS
    ]
    out = []
    for k in keys:
        if k in _SKIP_FIELDS or k == "evidence":
            continue
        v = raw.get(k)
        if v in (None, "", [], {}, False):
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in _flatten_strings(v)) or ", ".join(map(str, v))
        out.append([_PRETTY.get(k, k.replace("_", " ")), str(v)])
    return out


def _bucket(src) -> str:
    """schema.org | vocab (minted) | open (not yet canonicalized) — same language for
    property sources and entity-type sources."""
    if src == "schema.org":
        return "schema.org"
    if src in (None, "", "open"):
        return "open"
    return "vocab"


def build_payload(nodes: dict, edges: list, type_colors: dict, other_color: str,
                  theme: dict, aid: str = "", source_file: str = "",
                  type_path=None, prop_path=None) -> dict:
    """(nodes, edges) as produced by the notebook loader -> the JSON the viewer consumes.

    `type_path(label, parent)` / `prop_path(label, parent)` are optional resolvers returning the
    term's ancestor chain as [(term, source), ...] root-first; the viewer shows it on hover."""
    def _path(fn, label, parent):
        if not fn or not label:
            return []
        try:
            return [[t, s] for t, s in fn(label, parent)]
        except Exception:
            return []
    deg: dict[str, int] = {i: 0 for i in nodes}
    for e in edges:
        if e["s"] in nodes and e["o"] in nodes:
            deg[e["s"]] += 1
            deg[e["o"]] += 1

    jnodes = []
    for i, n in nodes.items():
        kind = "nary" if n.get("is_nary") else ("literal" if n.get("is_lit") else "entity")
        raw = n.get("raw") or {}
        tsrc = _bucket(n.get("nary_type_source") if kind == "nary" else n.get("type_source"))
        label = n["label"]
        if n.get("norm"):
            label = f'{label} ({n["norm"]})'
        jnodes.append({
            "id": i,
            "label": label,
            "plain": n["label"],
            "type": n.get("type") or "Unknown",
            "kind": kind,
            "naryKind": n.get("nary_kind"),
            "naryType": n.get("nary_type"),
            "typeSource": tsrc,
            "isMain": bool(n.get("is_main")),
            "typePath": _path(type_path,
                              n.get("nary_type") or n.get("type") if kind == "nary" else n.get("type"),
                              n.get("nary_type_parent") if kind == "nary" else n.get("type_parent")),
            "deg": deg.get(i, 0),
            "conf": raw.get("confidence"),
            "fields": _fields(raw),
            "evidence": _flatten_strings(raw.get("evidence") or n.get("evidence")),
        })

    jedges = []
    for k, e in enumerate(edges):
        if e["s"] not in nodes or e["o"] not in nodes:
            continue
        raw = e.get("raw") or {}
        jedges.append({
            "id": f"e{k}",
            "from": e["s"],
            "to": e["o"],
            "label": e.get("property") or "",
            "source": _bucket(e.get("source")),
            "propPath": _path(prop_path, e.get("property"), raw.get("property_parent")),
            "conf": e.get("confidence"),
            "phrases": [p for p in (e.get("phrases") or []) if p],
            "fields": _fields(raw),
            "evidence": _flatten_strings(raw.get("evidence") or e.get("evidence")),
        })

    return {
        "articleId": aid,
        "sourceFile": source_file,
        "nodes": jnodes,
        "edges": jedges,
        "typeColors": type_colors,
        "otherColor": other_color,
        "theme": theme,
    }


# ---------------------------------------------------------------------------------------
# HTML template  (tokens: __VIS_JS__, __DATA__, __TITLE__)
# ---------------------------------------------------------------------------------------

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script>__VIS_JS__</script>
<style>
:root{
  --bg:#f5f5f3; --surface:#ffffff; --panel:#fbfbfa; --ink:#14140f; --muted:#6b6b62;
  --line:#e3e3dd; --accent:#2a78d6; --accent-soft:#e8f0fb; --shadow:0 1px 2px rgba(0,0,0,.05),0 8px 24px rgba(0,0,0,.06);
  --schema:#5b5a54; --vocab:#b06a2c; --open:#7f8c99;
}
html[data-theme="dark"]{
  --bg:#14161a; --surface:#1b1e24; --panel:#181b20; --ink:#e9e9e4; --muted:#9a9a92;
  --line:#2b2f36; --accent:#6aa6f0; --accent-soft:#1f2a3a; --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.35);
  --schema:#a8a79e; --vocab:#e09a54; --open:#8fa0b0;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0}
body{
  font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
  background:var(--bg); color:var(--ink); overflow:hidden;
}
#app{display:flex; height:100%; width:100%}

/* ---------- sidebar ---------- */
#panel{
  width:296px; flex:0 0 296px; background:var(--panel); border-right:1px solid var(--line);
  display:flex; flex-direction:column; overflow:hidden; transition:margin-left .18s ease;
}
#panel.collapsed{margin-left:-296px}
#panel header{padding:14px 16px 10px; border-bottom:1px solid var(--line)}
#panel header h1{margin:0; font-size:14px; font-weight:650; letter-spacing:-.01em}
#panel header .sub{color:var(--muted); font-size:11px; margin-top:3px; word-break:break-all}
#controls{overflow-y:auto; padding:6px 0 24px; flex:1}
.sect{border-bottom:1px solid var(--line)}
.sect>summary{
  cursor:pointer; padding:9px 16px; font-size:10.5px; font-weight:700; letter-spacing:.09em;
  text-transform:uppercase; color:var(--muted); list-style:none; display:flex; justify-content:space-between;
  align-items:center; user-select:none;
}
.sect>summary::-webkit-details-marker{display:none}
.sect>summary::after{content:"›"; transform:rotate(90deg); transition:transform .15s; font-size:14px; color:var(--muted)}
.sect:not([open])>summary::after{transform:rotate(0deg)}
.sect .body{padding:2px 16px 14px}
label.row{display:flex; align-items:center; gap:8px; padding:3px 0; cursor:pointer; font-size:12.5px}
label.row:hover{color:var(--accent)}
label.row input{accent-color:var(--accent); margin:0; flex:0 0 auto}
label.row .cnt{margin-left:auto; color:var(--muted); font-variant-numeric:tabular-nums; font-size:11px}
.sw{width:11px;height:11px;border-radius:3px;flex:0 0 auto;border:1px solid rgba(0,0,0,.12)}
.sw.line{height:0;border:none;border-top:3px solid;border-radius:0;width:16px}
.sw.dash{border-top-style:dashed}
.mini{display:flex;gap:6px;margin:2px 0 8px}
.mini button{
  font:inherit; font-size:11px; padding:3px 8px; border-radius:6px; cursor:pointer;
  border:1px solid var(--line); background:var(--surface); color:var(--muted);
}
.mini button:hover{border-color:var(--accent); color:var(--accent)}
.seg{display:flex; border:1px solid var(--line); border-radius:7px; overflow:hidden; background:var(--surface)}
.seg button{
  flex:1; font:inherit; font-size:11.5px; padding:6px 4px; border:0; border-right:1px solid var(--line);
  background:transparent; color:var(--muted); cursor:pointer;
}
.seg button:last-child{border-right:0}
.seg button.on{background:var(--accent-soft); color:var(--accent); font-weight:600}
select,input[type="search"],input[type="text"]{
  width:100%; font:inherit; font-size:12.5px; padding:6px 8px; border:1px solid var(--line);
  border-radius:7px; background:var(--surface); color:var(--ink);
}
input[type="range"]{width:100%; accent-color:var(--accent)}
.fieldlab{display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin:8px 0 3px}
.typelist{max-height:212px; overflow-y:auto; margin-right:-6px; padding-right:6px}
.hint{font-size:11px; color:var(--muted); margin-top:6px; line-height:1.4}

/* ---------- main ---------- */
#main{flex:1; position:relative; min-width:0; display:flex; flex-direction:column}
#topbar{
  display:flex; align-items:center; gap:8px; padding:8px 12px; background:var(--surface);
  border-bottom:1px solid var(--line); z-index:5;
}
#topbar .title{font-weight:650; font-size:13px; margin-right:2px}
#topbar .stats{color:var(--muted); font-size:11.5px; font-variant-numeric:tabular-nums}
#topbar .grow{flex:1}
#search{width:230px}
button.tb{
  font:inherit; font-size:12px; padding:5px 10px; border-radius:7px; cursor:pointer;
  border:1px solid var(--line); background:var(--surface); color:var(--ink);
}
button.tb:hover{border-color:var(--accent); color:var(--accent)}
button.tb.on{background:var(--accent-soft); border-color:var(--accent); color:var(--accent)}
#canvasWrap{flex:1; position:relative; min-height:0}
#net{position:absolute; inset:0; background:var(--surface)}
#legend{
  position:absolute; left:12px; bottom:12px; background:var(--surface); border:1px solid var(--line);
  border-radius:10px; padding:9px 11px; box-shadow:var(--shadow); font-size:11.5px; max-width:270px;
  max-height:44%; overflow:auto;
}
#legend .lrow{display:flex;align-items:center;gap:7px;padding:1.5px 0}
#legend .lhead{font-size:10px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:6px 0 3px}
#legend .lhead:first-child{margin-top:0}
#legendHead{
  display:flex; align-items:center; gap:6px; width:100%; border:0; background:none; cursor:pointer;
  font:inherit; font-size:10px; font-weight:700; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); padding:0 0 6px;
}
#legendHead .car{margin-left:auto; font-size:12px; transition:transform .15s}
#legend.min #legendHead{padding-bottom:0}
#legend.min #legendHead .car{transform:rotate(-90deg)}
#legend.min #legendBody{display:none}
#details{
  position:absolute; right:12px; top:12px; width:314px; max-height:calc(100% - 24px); overflow:auto;
  background:var(--surface); border:1px solid var(--line); border-radius:12px; box-shadow:var(--shadow);
  padding:13px 15px; display:none;
}
#details.show{display:block}
#details .dh{display:flex; align-items:flex-start; gap:8px; margin-bottom:2px}
#details .dt{font-weight:650; font-size:13.5px; line-height:1.3; flex:1; word-break:break-word}
#details .x{cursor:pointer; color:var(--muted); border:0; background:none; font-size:17px; line-height:1; padding:0 2px}
#details .chips{display:flex; flex-wrap:wrap; gap:5px; margin:7px 0 9px}
.crumb{display:flex; flex-wrap:wrap; align-items:center; gap:3px 4px; margin:6px 0 2px; font-size:11px}
.crumb .seg{padding:1px 5px; border-radius:5px; background:var(--panel); border:1px solid var(--line)}
.crumb .seg.schema{color:var(--schema)}
.crumb .seg.vocab{color:var(--vocab); border-color:var(--vocab)}
.crumb .seg.open{color:var(--open); border-style:dashed}
.crumb .seg.last{font-weight:650}
.crumb .sep{color:var(--muted)}
.chip{font-size:10.5px; padding:2px 7px; border-radius:20px; border:1px solid var(--line); color:var(--muted)}
.chip.schema{border-color:var(--schema); color:var(--schema)}
.chip.vocab{border-color:var(--vocab); color:var(--vocab)}
.chip.open{border-color:var(--open); color:var(--open)}
.chip.main{border-color:var(--accent); color:var(--accent); font-weight:600}
#details dl{margin:0; display:grid; grid-template-columns:88px 1fr; gap:4px 10px; font-size:12px}
#details dt{color:var(--muted)}
#details dd{margin:0; word-break:break-word}
#details .ev{margin-top:10px}
#details .ev .q{
  border-left:2px solid var(--line); padding:3px 0 3px 9px; margin:5px 0; color:var(--muted);
  font-size:11.5px; font-style:italic;
}
#details .acts{display:flex; gap:6px; margin-top:11px}
/* ---------- triples drawer ---------- */
#drawer{
  border-top:1px solid var(--line); background:var(--surface); height:216px; display:none;
  flex-direction:column; flex:0 0 auto;
}
#drawer.show{display:flex}
#drawer .dhead{display:flex;align-items:center;gap:8px;padding:6px 12px;border-bottom:1px solid var(--line)}
#drawer .dhead .t{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
#tblWrap{overflow:auto; flex:1}
table{border-collapse:collapse; width:100%; font-size:12px}
th{
  position:sticky; top:0; background:var(--surface); text-align:left; font-weight:600; color:var(--muted);
  padding:6px 10px; border-bottom:1px solid var(--line); font-size:11px; z-index:1;
}
td{padding:5px 10px; border-bottom:1px solid var(--line); vertical-align:top}
tbody tr{cursor:pointer}
tbody tr:hover{background:var(--accent-soft)}
td.prop{font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;vertical-align:1px}
#toggle{
  position:absolute; left:8px; top:50%; z-index:6; transform:translateY(-50%);
}
.vis-tooltip{
  position:absolute; visibility:hidden; padding:7px 10px; white-space:pre-wrap; font-family:inherit;
  font-size:12px; color:var(--ink); background:var(--surface); border:1px solid var(--line);
  border-radius:8px; box-shadow:var(--shadow); max-width:340px; z-index:20; pointer-events:none;
}
#loading{
  position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
  background:var(--surface); color:var(--muted); font-size:13px; z-index:9;
}
</style>
</head>
<body>
<div id="app">
  <aside id="panel">
    <header>
      <h1 id="ptitle">Knowledge graph</h1>
      <div class="sub" id="psub"></div>
    </header>
    <div id="controls">

      <details class="sect" open><summary>Colour by</summary><div class="body">
        <div class="seg" id="colormode">
          <button data-v="type" class="on">Entity type</button>
          <button data-v="source">Vocabulary</button>
          <button data-v="kind">Node kind</button>
        </div>
        <div class="hint" id="colorhint">Hue = entity / event type. Ring = where the type came from.</div>
      </div></details>

      <details class="sect" open><summary>Filtered out</summary><div class="body">
        <div class="seg" id="filtermode">
          <button data-v="fade" class="on">Fade</button>
          <button data-v="hide">Hide</button>
        </div>
        <div class="fieldlab" style="margin-top:9px"><span>Ghost opacity</span><span id="ghostVal">35%</span></div>
        <input type="range" id="ghost" min="5" max="100" step="5" value="35">
        <div class="hint">What happens to whatever the filters below exclude. <b>Fade</b> keeps it on
          the canvas as a ghost — the slider sets how faint — so the layout stays put and you keep the
          context; <b>Hide</b> removes it. The Structure section's hide options always remove.</div>
      </div></details>

      <details class="sect" open><summary>Vocabulary source</summary><div class="body">
        <div class="fieldlab"><span>Relations (edges)</span></div>
        <div id="edgeSrc"></div>
        <div class="fieldlab" style="margin-top:9px"><span>Node types</span></div>
        <div id="nodeSrc"></div>
      </div></details>

      <details class="sect" open><summary>Node types</summary><div class="body">
        <div class="mini"><button id="tAll">All</button><button id="tNone">None</button>
          <button id="tTop">Top 8</button></div>
        <div class="typelist" id="typeList"></div>
      </div></details>

      <details class="sect"><summary>Structure</summary><div class="body">
        <div id="kindList"></div>
        <label class="row"><input type="checkbox" id="hideIso" checked><span>Hide isolated nodes</span></label>
        <label class="row"><input type="checkbox" id="hideLeaf"><span>Hide leaf entities (degree 1)</span></label>
        <label class="row"><input type="checkbox" id="mainOnly"><span>Main event's neighbourhood only</span></label>
        <div class="fieldlab"><span>Min. confidence</span><span id="confVal">0.00</span></div>
        <input type="range" id="conf" min="0" max="1" step="0.05" value="0">
      </div></details>

      <details class="sect" open><summary>Layout &amp; labels</summary><div class="body">
        <div class="fieldlab"><span>Layout</span></div>
        <select id="layout">
          <option value="force">Force-directed (default)</option>
          <option value="concentric">Radial — events inside, participants fanned out</option>
          <option value="hier">Hierarchical (top-down)</option>
        </select>
        <div class="fieldlab" style="margin-top:9px"><span>Spacing</span><span id="spreadVal">150</span></div>
        <input type="range" id="spread" min="60" max="480" step="10" value="150">
        <div class="fieldlab" style="margin-top:9px"><span>Node labels</span></div>
        <div class="seg" id="nodelab">
          <button data-v="auto" class="on">Auto</button>
          <button data-v="all">All</button>
          <button data-v="hubs">Hubs</button>
          <button data-v="off">Off</button>
        </div>
        <div class="hint">Auto shows every label when you zoom in, and only hubs / n-ary nodes in the
          zoomed-out overview — labels keep a constant size on screen.</div>
        <div class="fieldlab"><span>Edge labels</span></div>
        <div class="seg" id="edgelab">
          <button data-v="never">Off</button>
          <button data-v="select" class="on">On select</button>
          <button data-v="always">Always</button>
        </div>
        <div class="fieldlab" style="margin-top:9px"><span>Label size (on screen)</span><span id="fontVal">13</span></div>
        <input type="range" id="font" min="8" max="30" step="1" value="13">
        <label class="row" style="margin-top:6px"><input type="checkbox" id="physics"><span>Physics running</span></label>
        <label class="row"><input type="checkbox" id="dim" checked><span>Dim non-neighbours on select</span></label>
      </div></details>

    </div>
  </aside>

  <div id="main">
    <div id="topbar">
      <button class="tb" id="collapse" title="Hide panel">☰</button>
      <span class="title" id="mtitle">Knowledge graph</span>
      <span class="stats" id="stats"></span>
      <span class="grow"></span>
      <input type="search" id="search" placeholder="Search nodes…  ( / )">
      <button class="tb" id="tidy" title="Nudge overlapping labels apart">Tidy</button>
      <button class="tb" id="fit" title="Fit to view">Fit</button>
      <button class="tb" id="tbl" title="Show triples table">Triples</button>
      <button class="tb" id="png" title="Download the current view as PNG">PNG</button>
      <button class="tb" id="theme" title="Light / dark">◐</button>
      <button class="tb" id="reset" title="Reset all filters">Reset</button>
    </div>
    <div id="canvasWrap">
      <div id="net"></div>
      <div id="loading">laying out the graph…</div>
      <div id="legend">
        <button id="legendHead">Legend<span class="car">▾</span></button>
        <div id="legendBody"></div>
      </div>
      <div id="details">
        <div class="dh"><div class="dt" id="dTitle"></div><button class="x" id="dClose">×</button></div>
        <div id="dPath"></div>
        <div class="chips" id="dChips"></div>
        <dl id="dFields"></dl>
        <div class="ev" id="dEvidence"></div>
        <div class="acts">
          <button class="tb" id="dFocus">Focus</button>
          <button class="tb" id="dEgo">Isolate neighbourhood</button>
          <button class="tb" id="dCopy">Copy JSON</button>
        </div>
      </div>
    </div>
    <div id="drawer">
      <div class="dhead"><span class="t">Triples</span><span class="stats" id="tblCount"></span>
        <span class="grow" style="flex:1"></span>
        <input type="search" id="tblSearch" placeholder="filter…" style="width:180px">
        <button class="tb" id="tblCopy">Copy TSV</button>
        <button class="tb" id="tblClose">×</button></div>
      <div id="tblWrap"><table id="tripleTable"><thead><tr>
        <th style="width:26%">subject</th><th style="width:22%">property</th>
        <th style="width:26%">object</th><th style="width:12%">source</th><th style="width:14%">conf.</th>
      </tr></thead><tbody></tbody></table></div>
    </div>
  </div>
</div>

<script>
"use strict";
const DATA = __DATA__;

/* ------------------------------------------------------------------ palette */
const T = DATA.theme;
const SRC_COLOR = {"schema.org": T.schema, "vocab": T.vocab, "open": T.open};
const SRC_LABEL = {"schema.org": "schema.org", "vocab": "minted (ex:)", "open": "open / uncanonicalized"};
const KIND_COLOR = {entity: "#2a78d6", nary: "#e0a21a", literal: "#9a998f"};
const KIND_LABEL = {entity: "entity", nary: "n-ary node (event / observation / claim)", literal: "literal"};
const NARY_COLOR = {Event: "#e34948", Observation: "#1baf7a", Claim: "#4a3aa7"};

const nodeById = {}, edgeById = {};
DATA.nodes.forEach(n => nodeById[n.id] = n);
DATA.edges.forEach(e => edgeById[e.id] = e);

const TYPES = {};                                   // type -> count, ranked by frequency
DATA.nodes.forEach(n => TYPES[n.type] = (TYPES[n.type] || 0) + 1);
const TYPE_RANK = Object.keys(TYPES).sort((a, b) => TYPES[b] - TYPES[a] || a.localeCompare(b));
const HAS_NARY = DATA.nodes.some(n => n.kind === "nary");
const HAS_LIT  = DATA.nodes.some(n => n.kind === "literal");

function typeColor(t) { return DATA.typeColors[t] || DATA.otherColor; }
function nodeBase(n) {
  if (STATE.colorMode === "source") return SRC_COLOR[n.typeSource];
  if (STATE.colorMode === "kind")
    return n.kind === "nary" ? (NARY_COLOR[n.naryKind] || KIND_COLOR.nary) : KIND_COLOR[n.kind];
  return typeColor(n.type);
}
function borderOf(n) {
  if (STATE.colorMode === "source") return shade(SRC_COLOR[n.typeSource], -0.25);
  return SRC_COLOR[n.typeSource];
}
function shade(hex, amt) {                          // amt<0 darken, >0 lighten
  const c = hex.replace("#", "");
  const v = c.length === 3 ? c.split("").map(x => x + x).join("") : c;
  const rgb = [0, 2, 4].map(i => parseInt(v.slice(i, i + 2), 16));
  const out = rgb.map(x => Math.round(amt < 0 ? x * (1 + amt) : x + (255 - x) * amt));
  return "#" + out.map(x => x.toString(16).padStart(2, "0")).join("");
}
function mix(hex, other, w) {                       // w = weight of `other`
  const p = h => { const c = h.replace("#", ""); const v = c.length === 3 ? c.split("").map(x => x + x).join("") : c;
                   return [0, 2, 4].map(i => parseInt(v.slice(i, i + 2), 16)); };
  const a = p(hex), b = p(other);
  return "#" + a.map((x, i) => Math.round(x * (1 - w) + b[i] * w).toString(16).padStart(2, "0")).join("");
}

/* ------------------------------------------------------------------ state */
const STATE = {
  colorMode: "type", filterMode: "fade", ghost: 0.35,   // ghost = opacity of filtered-out items
  types: new Set(TYPE_RANK),
  edgeSrc: new Set(["schema.org", "vocab", "open"]),
  nodeSrc: new Set(["schema.org", "vocab", "open"]),
  kinds: new Set(["entity", "nary", "literal"]),
  conf: 0, hideIso: true, hideLeaf: false, mainOnly: false,
  edgeLabels: "select", nodeLabels: "auto", font: 13, spread: 150, layout: "force",
  dim: true, physics: false, selection: null, ego: null, search: "",
};
const DEFAULTS = JSON.parse(JSON.stringify({
  types: TYPE_RANK, edgeSrc: [...STATE.edgeSrc], nodeSrc: [...STATE.nodeSrc], kinds: [...STATE.kinds],
}));

/* ------------------------------------------------------------------ network */
const nodesDS = new vis.DataSet(DATA.nodes.map(n => ({
  id: n.id,
  label: n.label,
  shape: n.kind === "nary" ? "star" : (n.kind === "literal" ? "box" : "dot"),
  size: n.kind === "nary" ? 25 + 2.4 * Math.min(n.deg, 14) : 10 + 2.6 * Math.min(n.deg, 12),
  borderWidth: n.isMain ? 4 : 2,
  title: tipOfNode(n),
  widthConstraint: {maximum: 150},
})));
const edgesDS = new vis.DataSet(DATA.edges.map(e => ({
  id: e.id, from: e.from, to: e.to, title: tipOfEdge(e),
  dashes: e.source === "vocab", arrows: {to: {enabled: true, scaleFactor: 0.55}},
  smooth: {enabled: true, type: "dynamic"},
})));

/** "Thing › Event › SocietalEvent* › HolidayWeekend*" — * marks a minted (ex:) segment. */
function pathText(chain) {
  if (!chain || chain.length < 2) return "";
  return chain.map(p => p[0] + (p[1] === "vocab" ? "*" : "")).join(" › ");
}
function tipOfNode(n) {
  let s = n.plain + "\n" + (n.kind === "nary" ? (n.naryKind || "n-ary") + " · " : "") +
          (n.kind === "nary" ? (n.naryType || n.type) : n.type) + "  [" + n.typeSource + "]";
  if (n.isMain) s += "  · MAIN";
  const p = pathText(n.typePath);
  if (p) s += "\n" + p;
  if (n.evidence.length) s += "\n“" + trunc(n.evidence[0], 200) + "”";
  return s;
}
function tipOfEdge(e) {
  let s = (nodeById[e.from] ? nodeById[e.from].plain : e.from) + "  —" + e.label + "→  " +
          (nodeById[e.to] ? nodeById[e.to].plain : e.to) + "\n[" + e.source + "]";
  const p = pathText(e.propPath);
  if (p) s += "\n" + p;
  if (e.phrases.length) s += "\nphrase: " + e.phrases.join(", ");
  if (e.evidence.length) s += "\n“" + trunc(e.evidence[0], 200) + "”";
  return s;
}
function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n) + "…" : s; }

const container = document.getElementById("net");
const network = new vis.Network(container, {nodes: nodesDS, edges: edgesDS}, baseOptions());

function baseOptions() {
  return {
    nodes: {
      font: {size: STATE.font, face: "-apple-system, Segoe UI, Inter, Helvetica, Arial",
             color: cssVar("--ink"), strokeWidth: 4, strokeColor: cssVar("--surface"), multi: false},
      scaling: {min: 8, max: 40},
      shadow: false,
    },
    edges: {
      color: {inherit: false, opacity: 1},
      width: 1.3, selectionWidth: w => w * 2.4,
      font: {size: Math.max(9, STATE.font - 4), face: "-apple-system, Segoe UI, Inter, Helvetica, Arial",
             color: cssVar("--muted"), strokeWidth: 5, strokeColor: cssVar("--surface"), align: "middle"},
      arrowStrikethrough: false,
    },
    physics: {
      enabled: true, solver: "barnesHut", maxVelocity: 42, minVelocity: 0.7,
      barnesHut: {gravitationalConstant: -12000 - 60 * DATA.nodes.length, centralGravity: 0.25,
                  springLength: STATE.spread, springConstant: 0.035, damping: 0.55, avoidOverlap: 0.55},
      stabilization: {enabled: true, iterations: 600, updateInterval: 40, fit: true},
    },
    layout: {improvedLayout: DATA.nodes.length <= 200, randomSeed: 42},
    interaction: {hover: true, tooltipDelay: 140, navigationButtons: false, keyboard: false,
                  multiselect: true, hideEdgesOnDrag: DATA.edges.length > 400},
  };
}
function cssVar(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

/* ------------------------------------------------------------------ filtering */
function visibleSets() {
  const nodeOK = n =>
    STATE.kinds.has(n.kind) && STATE.types.has(n.type) &&
    STATE.nodeSrc.has(n.typeSource) &&
    (n.conf == null || n.conf >= STATE.conf);
  const edgeOK = e => STATE.edgeSrc.has(e.source) && (e.conf == null || e.conf >= STATE.conf);

  // What PASSES the filters (drives the stats, the labels and the triples table)…
  const passN = new Set(DATA.nodes.filter(nodeOK).map(n => n.id));
  const passE = new Set(DATA.edges.filter(e => edgeOK(e) && passN.has(e.from) && passN.has(e.to))
                                  .map(e => e.id));
  // …and what is RENDERED. In "fade" mode the filtered-out ones stay on the canvas as ghosts, so the
  // layout doesn't jump every time a box is ticked; the Structure options below always remove.
  let nOK = STATE.filterMode === "fade" ? new Set(DATA.nodes.map(n => n.id)) : new Set(passN);
  let eVis = DATA.edges.filter(e => (STATE.filterMode === "fade" || passE.has(e.id)) &&
                                    nOK.has(e.from) && nOK.has(e.to));

  if (STATE.mainOnly) {                                     // keep main events + what they touch
    const seeds = new Set(DATA.nodes.filter(n => n.isMain).map(n => n.id));
    if (seeds.size) {
      const keep = new Set(seeds);
      eVis.forEach(e => { if (seeds.has(e.from)) keep.add(e.to); if (seeds.has(e.to)) keep.add(e.from); });
      nOK = new Set([...nOK].filter(i => keep.has(i)));
      eVis = eVis.filter(e => nOK.has(e.from) && nOK.has(e.to));
    }
  }
  if (STATE.ego) {                                          // isolate a node's neighbourhood
    const keep = new Set([STATE.ego]);
    eVis.forEach(e => { if (e.from === STATE.ego) keep.add(e.to); if (e.to === STATE.ego) keep.add(e.from); });
    nOK = new Set([...nOK].filter(i => keep.has(i)));
    eVis = eVis.filter(e => nOK.has(e.from) && nOK.has(e.to));
  }
  const deg = {};
  eVis.forEach(e => { deg[e.from] = (deg[e.from] || 0) + 1; deg[e.to] = (deg[e.to] || 0) + 1; });
  if (STATE.hideLeaf) {
    const drop = new Set([...nOK].filter(i => nodeById[i].kind === "entity" && (deg[i] || 0) <= 1));
    if (drop.size < nOK.size) {
      nOK = new Set([...nOK].filter(i => !drop.has(i)));
      eVis = eVis.filter(e => nOK.has(e.from) && nOK.has(e.to));
      Object.keys(deg).forEach(k => delete deg[k]);
      eVis.forEach(e => { deg[e.from] = (deg[e.from] || 0) + 1; deg[e.to] = (deg[e.to] || 0) + 1; });
    }
  }
  if (STATE.hideIso) nOK = new Set([...nOK].filter(i => (deg[i] || 0) > 0));
  eVis = eVis.filter(e => nOK.has(e.from) && nOK.has(e.to));

  return {
    shown: nOK,                                             // rendered (ghosts included)
    shownEdges: eVis,
    nodes: new Set([...nOK].filter(i => passN.has(i))),      // rendered AND passing the filters
    edges: eVis.filter(e => passE.has(e.id)),
  };
}

let VIS = {nodes: new Set(), edges: [], shown: new Set(), shownEdges: []};
let NEIGH = null;                                            // ids highlighted around the selection

/** vis font sizes are in CANVAS units, so a fitted 150-node graph renders them unreadably small.
 *  Divide by the current zoom instead: labels keep a constant ON-SCREEN size (STATE.font px). */
function fontScale() {
  // clamped: labels stay readable when zoomed out without growing so large (in canvas units)
  // that they dwarf the graph and make the layout impossible to untangle.
  const s = Math.min(3, Math.max(0.45, network ? network.getScale() : 1));
  return 1 / s;
}

/** Level of detail: in the zoomed-out overview only hubs and n-ary nodes keep their label, so the
 *  picture stays readable; zooming in brings the rest back. Selection and search always win. */
function labelVisible(n, scale, hit) {
  if (hit || (NEIGH && NEIGH.has(n.id))) return true;
  if (STATE.nodeLabels === "off") return false;
  if (STATE.nodeLabels === "all") return true;
  if (STATE.nodeLabels === "hubs") return n.kind === "nary" || n.deg >= 4;
  if (scale >= 0.75) return true;                            // auto
  if (scale >= 0.40) return n.kind === "nary" || n.deg >= 2;
  return n.kind === "nary" || n.deg >= 5;
}

function applyStyles() {
  const surf = cssVar("--surface"), ink = cssVar("--ink"), muted = cssVar("--muted");
  const q = STATE.search.trim().toLowerCase();
  const scale = network ? network.getScale() : 1;
  const F = STATE.font * fontScale();
  const nUpd = DATA.nodes.map(n => {
    const shown = VIS.shown.has(n.id);
    const ghost = shown && !VIS.nodes.has(n.id);             // filtered out, kept as a ghost
    const dimmed = !ghost && STATE.dim && NEIGH && !NEIGH.has(n.id);
    const fade = ghost ? (1 - STATE.ghost) : (dimmed ? 0.78 : 0);
    const base = nodeBase(n);
    const bg = fade ? mix(base, surf, fade) : base;
    const bd = fade ? mix(borderOf(n), surf, fade) : borderOf(n);
    const hit = q && n.plain.toLowerCase().includes(q);
    return {
      id: n.id, hidden: !shown,
      label: ((!ghost || STATE.ghost >= 0.7) && labelVisible(n, scale, hit)) ? n.label : "",
      widthConstraint: {maximum: F * 12},                    // wrap at a constant on-screen width
      color: {background: bg, border: hit ? cssVar("--accent") : bd,
              highlight: {background: shade(base, 0.18), border: cssVar("--accent")},
              hover: {background: shade(base, 0.12), border: cssVar("--accent")}},
      borderWidth: hit ? 5 : (ghost ? 1 : (n.isMain ? 4 : 2)),
      font: {size: F * (n.kind === "nary" ? 1.2 : 1),
             color: fade ? mix(ink, surf, Math.min(fade, 0.72)) : ink,
             strokeWidth: F * 0.28, strokeColor: surf},
      size: (n.kind === "nary" ? 25 + 2.4 * Math.min(n.deg, 14) : 10 + 2.6 * Math.min(n.deg, 12)) *
            (ghost ? 0.72 + 0.28 * STATE.ghost : 1),
    };
  });
  nodesDS.update(nUpd);

  const shownE = new Set(VIS.shownEdges.map(e => e.id));
  const passE  = new Set(VIS.edges.map(e => e.id));
  const eUpd = DATA.edges.map(e => {
    const shown = shownE.has(e.id);
    const ghost = shown && !passE.has(e.id);
    const dimmed = !ghost && STATE.dim && NEIGH && !(NEIGH.has(e.from) && NEIGH.has(e.to));
    const col = SRC_COLOR[e.source];
    const showLab = !ghost && (STATE.edgeLabels === "always" ||
                    (STATE.edgeLabels === "select" && NEIGH && NEIGH.has(e.from) && NEIGH.has(e.to)));
    return {
      id: e.id, hidden: !shown,
      color: {color: mix(col, surf, ghost ? (1 - STATE.ghost) * 0.85 : (dimmed ? 0.82 : 0)),
              opacity: ghost ? Math.max(0.12, STATE.ghost) : (dimmed ? 0.5 : 0.85),
              highlight: cssVar("--accent"), hover: cssVar("--accent")},
      label: showLab ? e.label : "",
      font: {size: F * 0.85, color: muted, strokeWidth: F * 0.3, strokeColor: surf},
      width: ghost ? 0.6 + 0.5 * STATE.ghost : (dimmed ? 0.8 : 1.3),
    };
  });
  edgesDS.update(eUpd);
}

function refresh(relayout) {
  VIS = visibleSets();
  applyStyles();
  const ghosts = VIS.shown.size - VIS.nodes.size;
  document.getElementById("stats").textContent =
    VIS.nodes.size + " / " + DATA.nodes.length + " nodes · " +
    VIS.edges.length + " / " + DATA.edges.length + " relations" +
    (ghosts > 0 ? "  ·  " + ghosts + " faded" : "");
  buildTable();
  if (relayout) restabilize();
}

/* ------------------------------------------------------------------ layouts */
function restabilize() {
  if (STATE.layout === "hier") {
    network.setOptions({
      layout: {hierarchical: {enabled: true, direction: "UD", sortMethod: "directed",
               levelSeparation: Math.max(110, STATE.spread), nodeSpacing: Math.max(120, STATE.spread * 0.9),
               treeSpacing: 200, shakeTowards: "roots"}},
      physics: {enabled: false},
    });
    setTimeout(() => { network.fit({animation: true}); setTimeout(applyStyles, 1050); }, 60);
    return;
  }
  network.setOptions({layout: {hierarchical: {enabled: false}}});
  if (STATE.layout === "concentric") { concentric(); return; }
  network.setOptions({
    physics: {enabled: true,
      barnesHut: {gravitationalConstant: -12000 - 60 * DATA.nodes.length, centralGravity: 0.25,
                  springLength: STATE.spread, springConstant: 0.035, damping: 0.55, avoidOverlap: 0.55},
      stabilization: {enabled: true, iterations: 600, updateInterval: 40, fit: true}},
  });
  network.stabilize(600);
}

/** Event-centric radial layout: every n-ary node sits on an inner ring, and its participants fan
 *  out in rows inside that node's own angular sector. Rows are sized from the nodes' MEASURED label
 *  boxes, so it is overlap-free by construction, deterministic, and it reads like the data model. */
function concentric() {
  network.setOptions({physics: {enabled: false}});
  const ids = [...VIS.shown];
  if (!ids.length) return;
  network.redraw();                                          // boxes must reflect the current labels
  const M = {}, GAP = 16, ROWGAP = 26;
  ids.forEach(i => {
    const b = network.getBoundingBox(i);
    M[i] = {w: Math.max(b.right - b.left, 30), h: Math.max(b.bottom - b.top, 24)};
  });
  const adj = {};
  VIS.shownEdges.forEach(e => {
    (adj[e.from] = adj[e.from] || []).push(e.to);
    (adj[e.to] = adj[e.to] || []).push(e.from);
  });
  // the inner ring is the n-ary nodes; a binary graph falls back to its highest-degree nodes
  let ring = ids.filter(i => nodeById[i].kind === "nary");
  if (!ring.length) {
    ring = ids.slice().sort((a, b) => nodeById[b].deg - nodeById[a].deg)
              .slice(0, Math.max(1, Math.round(ids.length / 8)));
  }
  const ringSet = new Set(ring);
  const kids = {}, orphans = [];
  ring.forEach(i => kids[i] = []);
  ids.forEach(i => {
    if (ringSet.has(i)) return;
    const anchor = (adj[i] || []).find(j => ringSet.has(j));
    if (anchor) kids[anchor].push(i); else orphans.push(i);
  });

  const k = ring.length;
  const R1 = Math.max(240, ring.reduce((s, i) => s + M[i].w + GAP, 0) / (2 * Math.PI));
  const angleOf = {};
  ring.forEach((id, idx) => {
    const a = 2 * Math.PI * idx / k - Math.PI / 2;
    angleOf[id] = a;
    network.moveNode(id, R1 * Math.cos(a), R1 * Math.sin(a));
  });

  const half = (Math.PI / k) * 0.92;                          // each event's angular sector
  let maxR = R1;
  ring.forEach(id => {
    const a0 = angleOf[id];
    const ch = kids[id];
    let r = R1 + M[id].h / 2 + 40, idx = 0;
    while (idx < ch.length) {
      const arc = 2 * half * r;                               // arc length available at this radius
      let used = 0, row = [], rowH = 0;
      while (idx < ch.length && used + M[ch[idx]].w + GAP <= arc) { used += M[ch[idx]].w + GAP; row.push(ch[idx]); idx++; }
      if (!row.length) { row.push(ch[idx]); used = M[ch[idx]].w + GAP; idx++; }   // never stall
      let acc = -used / 2;
      row.forEach(c => {
        const w = M[c].w + GAP;
        const ang = a0 + (acc + w / 2) / r;
        acc += w; rowH = Math.max(rowH, M[c].h);
        network.moveNode(c, r * Math.cos(ang), r * Math.sin(ang));
      });
      r += rowH + ROWGAP;
    }
    maxR = Math.max(maxR, r);
  });

  let r = maxR + 40, idx = 0;                                 // whatever hangs off no event: outer rings
  while (idx < orphans.length) {
    const arc = 2 * Math.PI * r;
    let used = 0, row = [], rowH = 0;
    while (idx < orphans.length && used + M[orphans[idx]].w + GAP <= arc) { used += M[orphans[idx]].w + GAP; row.push(orphans[idx]); idx++; }
    if (!row.length) { row.push(orphans[idx]); used = M[orphans[idx]].w + GAP; idx++; }
    let acc = 0;
    row.forEach(c => {
      const w = M[c].w + GAP;
      const ang = 2 * Math.PI * (acc + w / 2) / used;
      acc += w; rowH = Math.max(rowH, M[c].h);
      network.moveNode(c, r * Math.cos(ang), r * Math.sin(ang));
    });
    r += rowH + ROWGAP;
  }
  setTimeout(() => { network.fit({animation: true}); setTimeout(applyStyles, 1050); }, 40);
}
function hash(s) { let h = 0; for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0; return Math.abs(h); }

/** Label-aware de-overlap: push nodes apart until their label boxes stop intersecting. */
function deoverlap(iters) {
  const ids = [...VIS.shown];
  if (ids.length < 2) return;
  network.redraw();                                          // make sure label boxes are current
  const PAD = 6;
  const it = ids.map(id => {
    const b = network.getBoundingBox(id), p = network.getPosition(id);
    return {id, x: p.x, y: p.y, w: (b.right - b.left) / 2 + PAD, h: (b.bottom - b.top) / 2 + PAD};
  });
  for (let s = 0; s < iters; s++) {
    let moved = 0;
    for (let a = 0; a < it.length; a++) {
      for (let b = a + 1; b < it.length; b++) {
        const A = it[a], B = it[b];
        const dx = B.x - A.x, dy = B.y - A.y;
        const ox = (A.w + B.w) - Math.abs(dx), oy = (A.h + B.h) - Math.abs(dy);
        if (ox > 0 && oy > 0) {
          moved++;
          if (ox < oy) {                                  // resolve along the cheaper axis
            const push = (ox / 2 + 0.5) * (dx >= 0 ? 1 : -1);
            A.x -= push; B.x += push;
          } else {
            const push = (oy / 2 + 0.5) * (dy >= 0 ? 1 : -1);
            A.y -= push; B.y += push;
          }
        }
      }
    }
    if (!moved) break;
  }
  it.forEach(o => network.moveNode(o.id, o.x, o.y));
}

/* ------------------------------------------------------------------ selection */
function neighboursOf(id) {
  const s = new Set([id]);
  VIS.shownEdges.forEach(e => { if (e.from === id) s.add(e.to); if (e.to === id) s.add(e.from); });
  return s;
}
network.on("click", p => {
  if (p.nodes.length) { select("node", p.nodes[0]); }
  else if (p.edges.length) { select("edge", p.edges[0]); }
  else { clearSelection(); }
});
network.on("doubleClick", p => { if (p.nodes.length) { STATE.ego = p.nodes[0]; refresh(true); } });
let _zoomT = null;                                           // keep label size constant on screen
network.on("zoom", () => { clearTimeout(_zoomT); _zoomT = setTimeout(applyStyles, 130); });

function clearSelection() {
  STATE.selection = null; NEIGH = null;
  document.getElementById("details").classList.remove("show");
  network.unselectAll(); applyStyles();
}
function select(kind, id) {
  STATE.selection = {kind, id};
  if (kind === "node") { NEIGH = neighboursOf(id); network.selectNodes([id]); renderNode(nodeById[id]); }
  else {
    const e = edgeById[id];
    NEIGH = new Set([e.from, e.to]); network.selectEdges([id]); renderEdge(e);
  }
  applyStyles();
}
function chip(txt, cls) { return '<span class="chip ' + (cls || "") + '">' + esc(txt) + "</span>"; }
function esc(s) { return String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c])); }

/** Breadcrumb of the term's ancestors, each segment coloured by where it came from. */
function pathHTML(chain) {
  if (!chain || !chain.length) return "";
  return '<div class="crumb">' + chain.map((p, i) =>
      '<span class="seg ' + (p[1] === "vocab" ? "vocab" : p[1] === "open" ? "open" : "schema") +
      (i === chain.length - 1 ? " last" : "") + '">' + esc(p[0]) + "</span>").join('<span class="sep">›</span>') +
    "</div>";
}

function renderDetails(title, chips, fields, evidence, chain) {
  document.getElementById("dTitle").textContent = title;
  document.getElementById("dPath").innerHTML = pathHTML(chain);
  document.getElementById("dChips").innerHTML = chips.join("");
  if (chain && chain.length > 1)                             // the breadcrumb already shows the parent
    fields = fields.filter(f => f[0] !== "parent type" && f[0] !== "parent property");
  document.getElementById("dFields").innerHTML =
    fields.map(f => "<dt>" + esc(f[0]) + "</dt><dd>" + esc(f[1]) + "</dd>").join("");
  document.getElementById("dEvidence").innerHTML = evidence.length
    ? '<div class="fieldlab"><span>evidence</span></div>' +
      evidence.map(t => '<div class="q">' + esc(t) + "</div>").join("")
    : "";
  document.getElementById("details").classList.add("show");
}
function renderNode(n) {
  const kindLbl = n.kind === "nary" ? (n.naryKind || "n-ary") : n.kind;
  const typeLbl = n.kind === "nary" ? (n.naryType || n.type) : n.type;   // n-ary: kind ≠ schema type
  const chips = [chip(kindLbl)];
  if (typeLbl && typeLbl !== kindLbl) chips.push(chip(typeLbl));
  chips.push(chip(SRC_LABEL[n.typeSource], n.typeSource === "schema.org" ? "schema"
                  : n.typeSource === "vocab" ? "vocab" : "open"));
  if (n.isMain) chips.push(chip("main event", "main"));
  chips.push(chip(n.deg + " links"));
  renderDetails(n.plain, chips, n.fields, n.evidence, n.typePath);
}
function renderEdge(e) {
  const s = nodeById[e.from], o = nodeById[e.to];
  const chips = [chip(SRC_LABEL[e.source], e.source === "schema.org" ? "schema"
                 : e.source === "vocab" ? "vocab" : "open")];
  if (e.conf != null) chips.push(chip("confidence " + e.conf));
  renderDetails((s ? s.plain : e.from) + "  →  " + (o ? o.plain : e.to),
                chips, [["property", e.label]].concat(e.fields), e.evidence, e.propPath);
}
document.getElementById("dClose").onclick = clearSelection;
document.getElementById("dFocus").onclick = () => {
  if (!STATE.selection) return;
  if (STATE.selection.kind === "node") {
    network.focus(STATE.selection.id, {scale: 1.1, animation: true}); setTimeout(applyStyles, 1050);
  }
};
document.getElementById("dEgo").onclick = () => {
  if (!STATE.selection) return;
  STATE.ego = STATE.ego ? null : (STATE.selection.kind === "node" ? STATE.selection.id : null);
  document.getElementById("dEgo").textContent = STATE.ego ? "Show whole graph" : "Isolate neighbourhood";
  refresh(true);
};
document.getElementById("dCopy").onclick = () => {
  if (!STATE.selection) return;
  const obj = STATE.selection.kind === "node" ? nodeById[STATE.selection.id] : edgeById[STATE.selection.id];
  navigator.clipboard && navigator.clipboard.writeText(JSON.stringify(obj, null, 2));
};

/* ------------------------------------------------------------------ sidebar */
function checkRow(id, label, count, swatch, checked, onchange) {
  const l = document.createElement("label"); l.className = "row";
  const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = checked; cb.dataset.id = id;
  l.appendChild(cb);
  if (swatch) l.insertAdjacentHTML("beforeend", swatch);
  const sp = document.createElement("span"); sp.textContent = label; l.appendChild(sp);
  if (count != null) { const c = document.createElement("span"); c.className = "cnt"; c.textContent = count; l.appendChild(c); }
  cb.onchange = () => onchange(id, cb.checked);
  return l;
}
function sw(color) { return '<span class="sw" style="background:' + color + '"></span>'; }
function swLine(color, dashed) {
  return '<span class="sw line' + (dashed ? " dash" : "") + '" style="border-color:' + color + '"></span>';
}

function buildTypeList() {
  const host = document.getElementById("typeList"); host.innerHTML = "";
  TYPE_RANK.forEach(t => host.appendChild(checkRow(t, t, TYPES[t], sw(typeColor(t)), STATE.types.has(t),
    (id, on) => { on ? STATE.types.add(id) : STATE.types.delete(id); refresh(false); })));
}
function buildSrcLists() {
  const eh = document.getElementById("edgeSrc"); eh.innerHTML = "";
  const nh = document.getElementById("nodeSrc"); nh.innerHTML = "";
  ["schema.org", "vocab", "open"].forEach(s => {
    const ec = DATA.edges.filter(e => e.source === s).length;
    const nc = DATA.nodes.filter(n => n.typeSource === s).length;
    eh.appendChild(checkRow(s, SRC_LABEL[s], ec, swLine(SRC_COLOR[s], s === "vocab"), STATE.edgeSrc.has(s),
      (id, on) => { on ? STATE.edgeSrc.add(id) : STATE.edgeSrc.delete(id); refresh(false); }));
    nh.appendChild(checkRow(s, SRC_LABEL[s], nc, sw(SRC_COLOR[s]), STATE.nodeSrc.has(s),
      (id, on) => { on ? STATE.nodeSrc.add(id) : STATE.nodeSrc.delete(id); refresh(false); }));
  });
}
function buildKindList() {
  const host = document.getElementById("kindList"); host.innerHTML = "";
  [["nary", HAS_NARY], ["entity", true], ["literal", HAS_LIT]].forEach(([k, present]) => {
    if (!present) return;
    const c = DATA.nodes.filter(n => n.kind === k).length;
    host.appendChild(checkRow(k, KIND_LABEL[k].split(" (")[0], c, sw(KIND_COLOR[k]), STATE.kinds.has(k),
      (id, on) => { on ? STATE.kinds.add(id) : STATE.kinds.delete(id); refresh(false); }));
  });
}
function buildLegend() {
  const L = document.getElementById("legendBody");
  let h = "";
  if (STATE.colorMode === "type") {
    h += '<div class="lhead">node colour — type</div>';
    TYPE_RANK.filter(t => DATA.typeColors[t]).forEach(t => {
      h += '<div class="lrow">' + sw(typeColor(t)) + esc(t) + " <span style='color:var(--muted)'>(" + TYPES[t] + ")</span></div>";
    });
    const other = TYPE_RANK.filter(t => !DATA.typeColors[t]);
    if (other.length) h += '<div class="lrow">' + sw(DATA.otherColor) + "other (" + other.length + " types)</div>";
  } else if (STATE.colorMode === "source") {
    h += '<div class="lhead">node colour — where the type came from</div>';
    ["schema.org", "vocab", "open"].forEach(s => { h += '<div class="lrow">' + sw(SRC_COLOR[s]) + esc(SRC_LABEL[s]) + "</div>"; });
  } else {
    h += '<div class="lhead">node colour — kind</div>';
    if (HAS_NARY) Object.keys(NARY_COLOR).forEach(k => { h += '<div class="lrow">' + sw(NARY_COLOR[k]) + esc(k) + "</div>"; });
    h += '<div class="lrow">' + sw(KIND_COLOR.entity) + "entity</div>";
    if (HAS_LIT) h += '<div class="lrow">' + sw(KIND_COLOR.literal) + "literal</div>";
  }
  h += '<div class="lhead">relations</div>';
  ["schema.org", "vocab", "open"].forEach(s => {
    if (!DATA.edges.some(e => e.source === s)) return;
    h += '<div class="lrow">' + swLine(SRC_COLOR[s], s === "vocab") + esc(SRC_LABEL[s]) + "</div>";
  });
  if (STATE.colorMode !== "source") {
    h += '<div class="lhead">node ring — type source</div>';
    ["schema.org", "vocab", "open"].forEach(s => {
      if (!DATA.nodes.some(n => n.typeSource === s)) return;
      h += '<div class="lrow"><span class="sw" style="background:transparent;border:2.5px solid ' +
           SRC_COLOR[s] + ';border-radius:50%"></span>' + esc(SRC_LABEL[s]) + "</div>";
    });
  }
  if (HAS_NARY) h += '<div class="lhead">shape</div><div class="lrow">★ n-ary node · thick ring = main event</div>';
  L.innerHTML = h;
}

document.getElementById("legendHead").onclick = () =>
  document.getElementById("legend").classList.toggle("min");

/* segmented controls */
function seg(id, key, after) {
  const host = document.getElementById(id);
  host.querySelectorAll("button").forEach(b => b.onclick = () => {
    host.querySelectorAll("button").forEach(x => x.classList.remove("on"));
    b.classList.add("on"); STATE[key] = b.dataset.v; after && after();
  });
}
seg("colormode", "colorMode", () => { buildTypeList(); buildLegend(); applyStyles();
  document.getElementById("colorhint").textContent = STATE.colorMode === "source"
    ? "Hue = schema.org vs. minted vs. open, for node types and relations alike."
    : STATE.colorMode === "kind"
      ? "Hue = node kind (Event / Observation / Claim / entity)."
      : "Hue = entity / event type. Ring = where the type came from."; });
seg("edgelab", "edgeLabels", () => applyStyles());
seg("nodelab", "nodeLabels", () => { applyStyles(); deoverlap(40); });
seg("filtermode", "filterMode", () => refresh(false));

document.getElementById("tAll").onclick = () => { STATE.types = new Set(TYPE_RANK); buildTypeList(); refresh(false); };
document.getElementById("tNone").onclick = () => { STATE.types = new Set(); buildTypeList(); refresh(false); };
document.getElementById("tTop").onclick = () => { STATE.types = new Set(TYPE_RANK.slice(0, 8)); buildTypeList(); refresh(false); };

document.getElementById("hideIso").onchange = e => { STATE.hideIso = e.target.checked; refresh(false); };
document.getElementById("hideLeaf").onchange = e => { STATE.hideLeaf = e.target.checked; refresh(false); };
document.getElementById("mainOnly").onchange = e => { STATE.mainOnly = e.target.checked; refresh(true); };
document.getElementById("conf").oninput = e => {
  STATE.conf = parseFloat(e.target.value);
  document.getElementById("confVal").textContent = STATE.conf.toFixed(2); refresh(false);
};
document.getElementById("layout").onchange = e => { STATE.layout = e.target.value; refresh(true); };
document.getElementById("spread").oninput = e => {
  STATE.spread = parseInt(e.target.value, 10);
  document.getElementById("spreadVal").textContent = STATE.spread;
};
document.getElementById("spread").onchange = () => refresh(true);
document.getElementById("font").oninput = e => {
  STATE.font = parseInt(e.target.value, 10);
  document.getElementById("fontVal").textContent = STATE.font; applyStyles();
};
document.getElementById("physics").onchange = e => {
  STATE.physics = e.target.checked; network.setOptions({physics: {enabled: STATE.physics}});
};
document.getElementById("dim").onchange = e => { STATE.dim = e.target.checked; applyStyles(); };
document.getElementById("ghost").oninput = e => {
  STATE.ghost = parseInt(e.target.value, 10) / 100;
  document.getElementById("ghostVal").textContent = e.target.value + "%";
  applyStyles();
};

/* topbar */
function fitView(nodeIds) {                                  // fit, then resync the on-screen label size
  network.fit(nodeIds ? {nodes: nodeIds, animation: true} : {animation: true});
  setTimeout(applyStyles, 1050);
}
document.getElementById("collapse").onclick = () => {
  document.getElementById("panel").classList.toggle("collapsed");
  setTimeout(() => network.redraw(), 200);
};
document.getElementById("fit").onclick = () => fitView();
document.getElementById("tidy").onclick = () => { network.setOptions({physics: {enabled: false}});
  document.getElementById("physics").checked = false; STATE.physics = false; deoverlap(70); };
document.getElementById("png").onclick = () => {
  const cv = container.querySelector("canvas"); if (!cv) return;
  const a = document.createElement("a");
  a.download = (DATA.articleId || "kg") + ".png"; a.href = cv.toDataURL("image/png"); a.click();
};
document.getElementById("theme").onclick = () => {
  const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", cur);
  network.setOptions(baseOptions()); network.setOptions({physics: {enabled: STATE.physics}});
  applyStyles(); buildLegend();
};
document.getElementById("reset").onclick = () => {
  STATE.types = new Set(DEFAULTS.types); STATE.edgeSrc = new Set(DEFAULTS.edgeSrc);
  STATE.nodeSrc = new Set(DEFAULTS.nodeSrc); STATE.kinds = new Set(DEFAULTS.kinds);
  STATE.conf = 0; STATE.hideIso = true; STATE.hideLeaf = false; STATE.mainOnly = false; STATE.ego = null;
  STATE.filterMode = "fade"; STATE.ghost = 0.35;
  document.querySelectorAll("#filtermode button").forEach(b => b.classList.toggle("on", b.dataset.v === "fade"));
  document.getElementById("ghost").value = 35; document.getElementById("ghostVal").textContent = "35%";
  STATE.search = ""; document.getElementById("search").value = "";
  document.getElementById("conf").value = 0; document.getElementById("confVal").textContent = "0.00";
  document.getElementById("hideIso").checked = true; document.getElementById("hideLeaf").checked = false;
  document.getElementById("mainOnly").checked = false;
  buildTypeList(); buildSrcLists(); buildKindList(); clearSelection(); refresh(true);
};
const searchBox = document.getElementById("search");
searchBox.oninput = () => { STATE.search = searchBox.value; applyStyles(); };
searchBox.onkeydown = ev => {
  if (ev.key !== "Enter") return;
  const q = searchBox.value.trim().toLowerCase(); if (!q) return;
  const hit = DATA.nodes.find(n => VIS.shown.has(n.id) && n.plain.toLowerCase().includes(q));
  if (hit) { select("node", hit.id); network.focus(hit.id, {scale: 1.2, animation: true}); setTimeout(applyStyles, 1050); }
};
document.addEventListener("keydown", ev => {
  if (ev.key === "/" && document.activeElement !== searchBox) { ev.preventDefault(); searchBox.focus(); }
  if (ev.key === "Escape") { clearSelection(); }
});

/* ------------------------------------------------------------------ triples table */
const drawer = document.getElementById("drawer");
document.getElementById("tbl").onclick = () => {
  drawer.classList.toggle("show");
  document.getElementById("tbl").classList.toggle("on");
  buildTable();                                              // fill it the first time it opens
  setTimeout(() => network.redraw(), 60);
};
document.getElementById("tblClose").onclick = () => {
  drawer.classList.remove("show"); document.getElementById("tbl").classList.remove("on");
};
document.getElementById("tblSearch").oninput = buildTable;
document.getElementById("tblCopy").onclick = () => {
  const tsv = tableRows().map(e => [nodeById[e.from].plain, e.label, nodeById[e.to].plain,
                                    e.source, e.conf == null ? "" : e.conf].join("\t")).join("\n");
  navigator.clipboard && navigator.clipboard.writeText(tsv);
};
function tableRows() {
  const q = (document.getElementById("tblSearch").value || "").trim().toLowerCase();
  return VIS.edges.filter(e => {
    if (!q) return true;
    return (nodeById[e.from].plain + " " + e.label + " " + nodeById[e.to].plain).toLowerCase().includes(q);
  }).sort((a, b) => a.label.localeCompare(b.label) ||
                    nodeById[a.from].plain.localeCompare(nodeById[b.from].plain));
}
function buildTable() {
  if (!drawer.classList.contains("show")) return;
  const rows = tableRows();
  document.getElementById("tblCount").textContent = rows.length + " shown";
  const body = document.querySelector("#tripleTable tbody");
  body.innerHTML = rows.map(e => {
    const s = nodeById[e.from], o = nodeById[e.to];
    return '<tr data-id="' + e.id + '"><td><span class="dot" style="background:' + nodeBase(s) + '"></span>' +
      esc(s.plain) + "</td><td class='prop'>" + esc(e.label) + '</td><td><span class="dot" style="background:' +
      nodeBase(o) + '"></span>' + esc(o.plain) + "</td><td style='color:" + SRC_COLOR[e.source] + "'>" +
      esc(SRC_LABEL[e.source]) + "</td><td>" + (e.conf == null ? "—" : e.conf) + "</td></tr>";
  }).join("");
  body.querySelectorAll("tr").forEach(tr => tr.onclick = () => {
    select("edge", tr.dataset.id);
    const e = edgeById[tr.dataset.id];
    fitView([e.from, e.to]);
  });
}

/* ------------------------------------------------------------------ boot */
document.getElementById("ptitle").textContent = DATA.articleId ? "KG · " + DATA.articleId : "Knowledge graph";
document.getElementById("mtitle").textContent = DATA.articleId ? "Knowledge graph — " + DATA.articleId : "Knowledge graph";
document.getElementById("psub").textContent = DATA.sourceFile || "";
buildTypeList(); buildSrcLists(); buildKindList(); buildLegend();
refresh(false);
/* every stabilization run (first paint, "Spacing", layout switch) ends the same way:
   freeze the physics, nudge overlapping labels apart, fit. */
network.on("stabilizationIterationsDone", () => {
  network.setOptions({physics: {enabled: false}});
  STATE.physics = false; document.getElementById("physics").checked = false;
  network.fit({animation: false}); applyStyles();            // labels get their final size first…
  deoverlap(60); network.fit({animation: false}); applyStyles();   // …then separate, refit, resize
  document.getElementById("loading").style.display = "none";
});
setTimeout(() => { document.getElementById("loading").style.display = "none"; }, 12000);
</script>
</body>
</html>
"""


def write_html(out_path, aid, nodes, edges, type_colors, other_color,
               theme: dict | None = None, source_file: str = "", title: str | None = None,
               type_path=None, prop_path=None) -> Path:
    """Render the interactive viewer for one article to `out_path` (a self-contained .html)."""
    theme = theme or {"schema": "#5b5a54", "vocab": "#b06a2c", "open": "#7f8c99"}
    payload = build_payload(nodes, edges, type_colors, other_color, theme, aid, source_file,
                            type_path=type_path, prop_path=prop_path)
    # `<\/` keeps a stray "</script>" inside evidence text from closing the script block early
    data_js = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html_doc = (_TEMPLATE
                .replace("__DATA__", data_js)
                .replace("__TITLE__", title or f"Knowledge graph — {aid}")
                .replace("__VIS_JS__", _vis_js()))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    return out
