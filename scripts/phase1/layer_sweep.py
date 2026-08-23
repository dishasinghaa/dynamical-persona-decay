# Layer sweep: reuses the 30 pairs already generated in Phase 1 (no new
# generation needed - just new forward passes), and checks separation
# quality at several candidate layers. Pick whichever layer gives the
# cleanest safe-vs-unsafe separation before committing Phase 3's compute
# to a possibly bad layer choice.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="sdpa",
    output_hidden_states=True,
)
model.eval()

NUM_LAYERS = model.config.num_hidden_layers
# Candidate layers roughly at 25%, 35%, 50% (your original), 65%, 80% depth
CANDIDATE_LAYERS = sorted(set([
    round(NUM_LAYERS * 0.25),
    round(NUM_LAYERS * 0.35),
    round(NUM_LAYERS * 0.50),
    round(NUM_LAYERS * 0.65),
    round(NUM_LAYERS * 0.80),
]))
print(f"Model has {NUM_LAYERS} layers. Sweeping: {CANDIDATE_LAYERS}")

with open("phase1_pairs.json") as f:
    all_pairs = json.load(f)

# Use a subset for speed - 15 pairs is plenty for a diagnostic sweep
pairs = all_pairs[:15]
print(f"Using {len(pairs)} pairs for the sweep")

def get_activation_all_layers(text):
    """One forward pass, return last-token activation at EVERY layer."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
    # out.hidden_states: tuple of (num_layers+1) tensors, each (1, seq, hidden)
    # index 0 = embeddings, index i = output of layer i
    return [out.hidden_states[l][0, -1, :].detach().cpu() for l in CANDIDATE_LAYERS]

safe_acts_per_layer = {l: [] for l in CANDIDATE_LAYERS}
unsafe_acts_per_layer = {l: [] for l in CANDIDATE_LAYERS}

for i, p in enumerate(pairs):
    safe_text = f"{p['constraint']} {p['question']} {p['safe_completion']}"
    unsafe_text = f"{p['constraint']} {p['question']} {p['unsafe_completion']}"

    safe_all = get_activation_all_layers(safe_text)
    unsafe_all = get_activation_all_layers(unsafe_text)

    for j, l in enumerate(CANDIDATE_LAYERS):
        safe_acts_per_layer[l].append(safe_all[j])
        unsafe_acts_per_layer[l].append(unsafe_all[j])

    if (i + 1) % 5 == 0:
        print(f"  {i+1}/{len(pairs)} done")

# ── Compute separation quality at each layer ─────────────────────────

print(f"\n{'Layer':<8}{'Safe mean':<12}{'Safe std':<12}{'Unsafe mean':<14}{'Unsafe std':<12}{'Separation/std':<15}")

layer_scores = {}
for l in CANDIDATE_LAYERS:
    safe = torch.stack(safe_acts_per_layer[l])
    unsafe = torch.stack(unsafe_acts_per_layer[l])

    d = unsafe.mean(dim=0) - safe.mean(dim=0)
    d_norm = d / d.norm()

    safe_proj = safe @ d_norm
    unsafe_proj = unsafe @ d_norm

    # "separation quality": gap between means, relative to pooled std
    pooled_std = (safe_proj.std() + unsafe_proj.std()) / 2
    separation_ratio = (unsafe_proj.mean() - safe_proj.mean()) / pooled_std

    layer_scores[l] = separation_ratio.item()
    print(f"{l:<8}{safe_proj.mean():<12.3f}{safe_proj.std():<12.3f}"
          f"{unsafe_proj.mean():<14.3f}{unsafe_proj.std():<12.3f}{separation_ratio:<15.3f}")

best_layer = max(layer_scores, key=layer_scores.get)
print(f"\nBest layer by separation/std ratio: {best_layer} (score: {layer_scores[best_layer]:.3f})")

plt.figure(figsize=(8, 4))
plt.bar([str(l) for l in CANDIDATE_LAYERS], [layer_scores[l] for l in CANDIDATE_LAYERS])
plt.xlabel("Layer")
plt.ylabel("Separation quality (gap / pooled std)")
plt.title("Which layer best separates safe vs unsafe activations?")
plt.axhline(0, color='gray', linewidth=0.5)
plt.tight_layout()
plt.savefig("layer_sweep_scores.png", dpi=100)
plt.show()

print(f"\nUse layer {best_layer} for Phase 1 re-extraction + Phase 2/3, "
      f"if it's meaningfully better than layer 18's original score.")
