import sys, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))
import nli_core, torch

rows = [r for r in nli_core.load_claims(ROOT/"results/gemini-2.5-flash", "art1", "e2e")
        if r["kind"] == "relation" and r.get("subject") and r.get("object")]
random.Random(1).shuffle(rows)
sample = rows[:10]
v = nli_core.Verbaliser(batch_size=8)
print("verbaliser:", v.model_name, "on", v.device, flush=True)
# raw generation, no cleaning
texts = [v.tok.apply_chat_template(
            [{"role":"system","content":nli_core.VERBALISE_SYSTEM},
             {"role":"user","content":v._prompt(r)}],
            tokenize=False, add_generation_prompt=True) for r in sample]
enc = v.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
enc = {k: t.to(v.device) for k, t in enc.items()}
with torch.inference_mode():
    gen = v.model.generate(**enc, max_new_tokens=48, do_sample=False,
                           pad_token_id=v.tok.pad_token_id)
for r, g in zip(sample, gen):
    raw = v.tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nPROMPT   : {v._prompt(r)!r}")
    print(f"RAW      : {raw!r}")
    print(f"CLEANED  : {v._clean(raw, r)}")
    print(f"TEMPLATE : {nli_core.template_hypothesis(r)}")
