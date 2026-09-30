import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.novelty.checker import NoveltyChecker

MODEL = "aubmindlab/aragpt2-base"
PROMPTS = {
    "الصبر": "من أجمل ما قيل عن الصبر:",
    "الأمل": "من أجمل ما قيل عن الأمل:",
    "الصداقة": "من أجمل ما قيل عن الصداقة:",
    "الحياة": "من أجمل ما قيل عن الحياة:",
    "الحكمة": "من أجمل ما قيل عن الحكمة:",
}
SAMPLES_PER_PROMPT = 4

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL).to(device).eval()
tok.pad_token = tok.pad_token or tok.eos_token

splits = [pd.read_parquet(f"data/processed/{s}.parquet") for s in ("train", "val", "test")]
checker = NoveltyChecker().fit(pd.concat(splits)["text_normalized"].tolist())

results = []
for theme, prompt in PROMPTS.items():
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            do_sample=True, top_p=0.95, temperature=0.9,
            max_new_tokens=40,
            num_return_sequences=SAMPLES_PER_PROMPT,
            pad_token_id=tok.pad_token_id,
        )
    for seq in out:
        text = tok.decode(seq[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        r = checker.check(text)
        results.append({"theme": theme, "text": text, "cosine": r.cosine,
                        "edit_similarity": r.edit_similarity, "memorized": r.memorized})

with open("reports/zero_shot_samples.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"memorized: {sum(r['memorized'] for r in results)}/{len(results)}")