# Dynamical Persona Decay: Predicting Safety-Instruction Eviction Before It Happens

[`writeup/WRITEUP.md`](writeup/WRITEUP.md) — start there.

## What this is

Tests whether internal activations in a language model predict "eviction"
of an early safety instruction (e.g. forgetting not to reveal a password)
*before* the leak is visible in generated text — using a distributional
signal (Wasserstein distance across resampled rollouts) rather than a
single trajectory.

**Headline result:** leaked vs. held rollout populations diverge
internally ~130 tokens before the secret is textually stated, an effect
specific to the extracted direction (confirmed via orthogonal control)
and replicated across three independent domains.

## Repo structure

```
scripts/
  phase0/          # establishing the phenomenon (eviction is real & probabilistic)
  phase1/          # extracting the contrastive safety direction d, layer selection
  phase2/          # single-trajectory baseline (negative result)
  phase3/          # population-level W1 test (positive result) + qualitative example
  controls/        # orthogonal control, natural-vs-instructed check, trained-probe
                    # cross-check, manual label spot-check
  generalization/  # testing across 7 prompts / 5 domains
  steering/         # causal intervention: constant and conditional steering
writeup/
  WRITEUP.md       # full executive summary + main write-up
```

Scripts are numbered/named in the order they were run, matching the
narrative in the write-up. Each script is self-contained (loads its own
model, generates its own data) rather than depending on a shared
pipeline — reflects the actual iterative process of a 20-hour sprint,
including a few scripts that exist specifically because an earlier one
had a bug (see `steering/constant_steering.py` vs. the original prefill
bug described in the write-up).

## Requirements

```
transformers
accelerate
torch
scipy
scikit-learn
matplotlib
numpy
```

All experiments were run on Qwen2.5-3B-Instruct via a free-tier Colab T4 GPU.

## Notes

- `phase1/layer_sweep.py` and `phase1/reextract_layer29_diagnostic.py`
  document a real methodological correction: a small-sample layer sweep
  initially suggested layer 29 was best; re-running on the full dataset
  showed layer 18 was actually better. Kept in the repo as an honest
  record rather than removed.
- `controls/trained_probe_crosscheck.py` produced an inconclusive result
  (likely overfitting at n=14) — included for completeness, discussed
  honestly as a limitation in the write-up.
