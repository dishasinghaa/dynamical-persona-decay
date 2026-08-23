# Re-extract Phase 1's direction vector at layer 29 (winner of the sweep).
# Reuses the 30 pairs already generated - no new text generation needed,
# just new forward passes at the better layer.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TARGET_LAYER = 29

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
)
model.eval()

with open("phase1_pairs.json") as f:
    pairs = json.load(f)
print(f"Loaded {len(pairs)} pairs, extracting at layer {TARGET_LAYER}")

def get_activation(text, layer=TARGET_LAYER):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    captured = {}

    def hook(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured["value"] = hs[0, -1, :].detach().cpu()

    handle = model.model.layers[layer].register_forward_hook(hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return captured["value"]

safe_acts, unsafe_acts = [], []

for i, p in enumerate(pairs):
    safe_text = f"{p['constraint']} {p['question']} {p['safe_completion']}"
    unsafe_text = f"{p['constraint']} {p['question']} {p['unsafe_completion']}"
    safe_acts.append(get_activation(safe_text))
    unsafe_acts.append(get_activation(unsafe_text))
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(pairs)} done")

safe_acts = torch.stack(safe_acts)
unsafe_acts = torch.stack(unsafe_acts)

d = unsafe_acts.mean(dim=0) - safe_acts.mean(dim=0)
d_normalized = d / d.norm()

safe_proj = safe_acts @ d_normalized
unsafe_proj = unsafe_acts @ d_normalized
pooled_std = (safe_proj.std() + unsafe_proj.std()) / 2
separation_ratio = (unsafe_proj.mean() - safe_proj.mean()) / pooled_std

print(f"\nLayer {TARGET_LAYER} (all 30 pairs):")
print(f"Safe:   mean={safe_proj.mean():.3f}, std={safe_proj.std():.3f}")
print(f"Unsafe: mean={unsafe_proj.mean():.3f}, std={unsafe_proj.std():.3f}")
print(f"Separation/std: {separation_ratio:.3f}")

torch.save({
    "d": d,
    "d_normalized": d_normalized,
    "safe_acts": safe_acts,
    "unsafe_acts": unsafe_acts,
    "target_layer": TARGET_LAYER,
}, "phase1_direction_layer29.pt")

print("\nSaved to phase1_direction_layer29.pt")
print("Next: rerun Phase 2 cosine baseline, pointing it at this new file,")
print("to see if the trajectory looks cleaner near the leak point.")
