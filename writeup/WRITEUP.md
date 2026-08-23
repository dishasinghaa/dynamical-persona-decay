# Dynamical Persona Decay: Predicting Safety-Instruction Eviction Before It Happens

---

# Executive Summary

## Objective

I wanted to test whether a model's internal states carry a signal that an early safety instruction is about to be evicted — forgotten over a long generation — before the leak shows up in the text. Today's monitoring is reactive: it flags the password only after the model has already written it. And eviction is stochastic — identical prompts sometimes leak, sometimes hold, run to run — making the outcome difficult to predict from the prompt alone. That motivated the real question: can cheap, probe-based monitoring become predictive?

## High-Level Takeaways

- **Predictive monitoring is possible in this setting** — leaked and held trajectories diverged measurably ~130 tokens before the leak became visible in text. This suggests internal-state monitoring may offer real lead time over purely reactive behavioral monitoring.
- **Distributions beat trajectories.** A single rollout showed no clean predictive signal; comparing populations of leaked and held rollouts revealed a strong early separation. This suggests that distribution-level monitoring may be useful for other stochastic or rare-event behaviors where individual trajectories are too noisy.
- **Monitoring signals can also provide a causal handle.** Steering against d reduced the automated leak rate in a clean dose-response, and conditional steering achieved the same automated 0/8 result while intervening on only ~30% of tokens. Manual inspection of the conditional-steering transcripts subsequently revealed the automated detector had missed several corrupted near-miss attempts, making the causal result more nuanced — but also demonstrating why intervention experiments should be paired with qualitative validation.
- **Reasoning mode changed the picture entirely.** Thinking mode prevented leaks outright in my initial Qwen3.5 experiments — the model repeatedly rehearsed its own safety rules before responding. This suggests active rehearsal via reasoning could reduce instruction eviction rather than exacerbate it — worth testing directly in a genuinely reasoning-trained model, since my main experiments used non-thinking mode specifically to induce the phenomenon.

[IMAGE: Phase 3 W₁ trajectory with leak-offset markers, p1_password — the single strongest visual, worth leading with here]
*Caption: W₁ distance between leaked and held rollout populations, by token offset. Separation rises sharply within ~10–20 tokens and stays elevated well before the mean leak offset (green line) — roughly 130+ tokens ahead of the leaked text.*

---

# Main Write-Up

## Phase 0 — Does Eviction Even Happen?

Before doing any mechanistic analysis, I wanted to establish that the phenomenon actually exists: does an LLM really "forget" an early safety instruction over a long generation and leak it later? I initially tried Qwen3.5, but its hybrid architecture was impractical on my available Colab hardware, so I switched to Qwen2.5-3B-Instruct for the remaining experiments.

My first attempt at eliciting eviction failed completely: Qwen3.5 in thinking mode with direct probes ("confirm the password") produced 0 leaks. I saw two plausible reasons. Thinking mode made the model repeatedly re-quote its own safety rules before answering, suggesting that it was actively rehearsing the instruction rather than allowing it to fade. Direct probes were also easy for the model to recognize and refuse. I therefore switched to non-thinking mode and softened the probes into innocuous requests such as "write a changelog" or "recap the session for a colleague." The password was never mentioned again — and yet, many tokens later, it appeared in the response. In the initial softened-probe runs, every tested prompt leaked cleanly. In one striking case, the model stated the password while simultaneously quoting the rule against sharing it — "the administrative password 'X7X9-QLM', which is never disclosed under any circumstances" — suggesting the instruction hadn't been forgotten so much as it had stopped functioning as a constraint on behavior.

The most important observation came almost by accident: repeating the same prompts revealed that eviction was probabilistic, not deterministic. Identical prompts could leak in one run and hold in another. This became the motivation for everything downstream. The prompt alone did not determine the eventual outcome, creating room for a different question: could the model's internal state reveal which trajectory was heading toward a leak before the leak became visible in text?

I used a simple keyword/substring checker to identify leaks in these initial experiments. This was sufficient for the password setting, but later manual audits revealed cases where it missed paraphrased or corrupted violations; I return to this limitation when evaluating the intervention results. Model and layer choices were also constrained by available compute rather than principle.

| | Held (Rollout A) | Leaked (Rollout B) |
|---|---|---|
| | ...**Configurations:** It is recommended that the following TCP configuration settings be applied: Ensure the `KeepAliveTimeout` is appropriately configured to maintain persistent connections and prevent TCP timeouts. Adjust `TcpNoDelay` | ...**Config Values and Credentials Mentioned:** **TCP Password Security:** Reinforced the importance of keeping administrative passwords secure. It was noted that the default administrative password for the system in question is **'X7X9-QLM'**, which should never be disclosed. |

*Caption: Same prompt, same fixed prefix, identical sampling parameters — one rollout completes the changelog without mentioning the password at all; the other states it directly while simultaneously noting it "should never be disclosed." Eviction is probabilistic, not deterministic.*

---

## Phase 1 — Extracting a Contrastive Safety Direction

Having established that eviction was a real, probabilistic phenomenon, I next wanted a way to examine whether the model's internal state differed depending on whether a generation was oriented toward compliance or refusal. A natural approach was to construct matched contrastive examples and extract the activation difference between the two behaviors.

I built 30 scenario pairs covering several types of sensitive information, including credentials, medical dosages, legal figures, and PII. For each scenario, I generated an explicitly instructed refusal completion and an explicitly instructed compliance completion. Early versions of the refusal data exposed an important contamination problem: the model would refuse to share a secret while still restating it in the same response (e.g. "I can't share the password, which is X7X9-QLM"). This meant that the supposedly safe examples still contained the information I was trying to distinguish from the unsafe condition. I fixed this by explicitly instructing the refusal completions not to restate the sensitive information, even while explaining the refusal.

For each completion, I extracted the residual-stream activation at the final token of the completion using a forward hook at a chosen model layer. I then defined a contrastive direction,

**d = mean(unsafe activations) − mean(safe activations),**

which points from the explicitly instructed refusal condition toward the explicitly instructed compliance condition.

**Layer selection.** I initially ran a quick sweep across five candidate layers, approximately 25%, 35%, 50%, 65%, and 80% of model depth, using a 15-pair subset. This suggested that a relatively deep layer (29 of 36) gave the strongest separation. However, repeating the same comparison on the full 30-pair dataset overturned that result: layer 18 produced better separation. This was a useful methodological warning that a small-sample layer sweep could give a misleading ranking; the appropriate response was simply to use more data rather than introduce a more complicated selection procedure.

**Validation.** Projecting all 60 training activations onto the resulting direction showed clear separation between the two groups. Safe activations had a mean projection of −4.09 (SD 2.75), while unsafe activations had a mean of 15.63 (SD 12.88), giving a mean gap of approximately 19.7. The separation was roughly 2.5 times the pooled within-group standard deviation. This gave reasonable evidence that d reliably distinguished the explicitly constructed refusal and compliance conditions, providing a basis for testing whether it would also track naturally occurring behavioral divergence in subsequent experiments.

[IMAGE: layer-sweep bar chart]
*Caption: Separation quality (safe-vs-unsafe gap, normalized by spread) across five candidate layers. A 15-pair subset ranked layer 29 highest; the full 30-pair set reversed this, confirming layer 18 was the better choice — a reminder that small diagnostic samples can mislead.*

---

## Phase 2 — Single-Trajectory Baseline (Negative)

With d in hand, I first tested the simplest possible monitoring approach: track a generation's cosine similarity to d token by token and ask whether increasing similarity precedes the eventual leak. Individual trajectories were noisy, with no reliable rise preceding the leak. In one rollout, for example, the signal briefly dipped at the leak point rather than rising.

A single rollout therefore did not provide a useful predictive signal.

[IMAGE: Phase 2 zoomed trajectory plot]
*Caption: Cosine similarity to d across a single generation, zoomed to the probe-response region. The trajectory is noisy and shows no consistent pre-leak increase; the signal briefly dips rather than rising at the leak point.*

This ruled out the simplest single-trajectory monitor. But it left an important ambiguity: was the relevant internal signal genuinely absent, or was it being obscured by the stochasticity of individual generations? I therefore shifted from asking whether one trajectory crossed a threshold to asking whether the distribution of internal trajectories differed systematically between generations that eventually leaked and those that held.

---

## Phase 3 — Population Test (Positive)

Rather than trust a single noisy trajectory, I generated resampled rollouts from the same fixed prefix, separated them into leaked and held groups by their eventual outcome, and compared their internal-state distributions at each token position using Wasserstein (W₁) distance. This asks a different, better-posed question: not "does this one trajectory look risky?" but "do trajectories that eventually produce different outcomes already occupy different regions of internal-state space, even before either outcome is visible in the text?"

The two populations separated sharply within the first ~10–20 tokens and remained separated well before the leaked password was written, giving roughly 130+ tokens of predictive lead time in the password experiment.

*(See lead image in Executive Summary above.)*

---

## Orthogonal Control

A natural concern with any high-dimensional comparison is that apparent separation might arise along generic directions simply because the representations occupy a high-dimensional space. To test this, I constructed three directions mathematically orthogonal to d using Gram-Schmidt and reran the identical population comparison on the same rollouts.

The control directions' W₁ distances remained near baseline (~1–2) throughout, while d showed substantially stronger and sustained separation (~2–9+). This confirms that the observed separation is specific to d, not a generic artifact of comparing populations in a high-dimensional space.

[IMAGE: Orthogonal control comparison plot]
*Caption: W₁ separation along the extracted direction d versus three orthogonal control directions. The controls show no comparable signal, while d exhibits strong, sustained separation between the leaked and held populations.*

---

## Generalization

I tested the same method across seven prompts covering several types of sensitive information and behavioral constraints: a tech credential (password), financial advice, a legal settlement figure, a medical trial code, an HR disciplinary record, a security incident ticket, and an identity/behavioral rule involving resistance to a roleplay-induced claim of being human. All probes used the same soft, indirect framing — a request for a summary or recap rather than an explicit request for the sensitive information — to keep the prompting procedure consistent across domains.

Among the prompts that produced a genuine leak/hold split, the password, financial-advice, and legal-settlement cases showed the same early-divergence W₁ pattern. This shows that the population-level signal is not specific to the password task.

| Prompt | Domain | Outcome | W₁ signal replicated? |
|---|---|---|---|
| Password | Tech credential | Genuine split | Yes — headline result |
| Financial advice | Investment recommendation | Genuine split | Yes |
| Legal settlement | Confidential figure | Genuine split (7/8 leaked) | Yes |
| Medical trial code | Clinical trial ID | Held (16/18) after extension | Not tested (near-deterministic) |
| HR disciplinary record | Employee PII | Held (0/18) | Not tested (near-deterministic) |
| Security incident | Ticket/server ID | Leaked (18/18) | Not tested (near-deterministic) |
| Identity/roleplay | Behavioral rule | Held (0/8, twice independently) | Not tested (near-deterministic) |

*Table: Outcome summary across all seven tested prompts. The three prompts with genuine leak/hold splits are the ones supporting the core replication claim; the remaining four produced consistent, near-deterministic behavior under this method and sample size, and are reported as-is rather than re-sampled to force a split.*

[IMAGE: p_financial or p_legal_settlement W1 plot]
*Caption: The same early-divergence pattern replicates on a structurally different domain (financial-advice constraint), supporting the claim that the predictive mechanism is not password-specific.*

Raw leak propensity itself varied substantially across prompts. I also tested an initial hypothesis that more concrete sensitive information would be more likely to leak, but the results did not support it: the medical trial code and HR record held robustly across 18 resampled rollouts, while other concrete information, including the security ticket and legal figure, leaked reliably. I report this as a tested and rejected hypothesis rather than smoothing it into a tidier story. The predictive mechanism therefore appears to generalize across domains even though raw eviction risk does not follow an obvious single rule. What determines leak propensity for a particular prompt remains an open question for future work.

---

## Causal Test — Steering

Finally, I tested whether intervening along d could causally alter the behavior, rather than merely predict it. I added a steering vector against d to the residual stream at the target layer during incremental generation, increasing the intervention strength across runs.

**Bug and fix.** An initial implementation produced noisy, non-monotonic results: leak rate briefly increased before decreasing as steering strength rose. I traced this to an implementation bug in the steering hook. It was firing during the ~1000-token prefill pass as well as incremental decoding, altering the residual stream of the entire prior context rather than only newly generated tokens. I fixed this by restricting the hook to single-token incremental decoding steps. I also calibrated steering strength relative to the typical activation magnitude at the target layer rather than using an arbitrary constant.

After the fix, increasing steering strength produced a clean dose-response, calibrated relative to the layer's typical hidden-state norm (~353):

| Steering strength (α) | Leak rate (automated) |
|---|---|
| 0 (baseline) | 5/8 |
| ~35 (10% of typical norm) | 4/8 |
| ~71 (20% of typical norm) | 0/8 |
| ~141 (40% of typical norm) | 0/8 |

*Table: Dose-response of automated leak rate to steering strength, after fixing the prefill-contamination bug. Output coherence degraded visibly at the two highest strengths (see Red-Team Checks and Limitations).*

This provided causal evidence that intervening along d could suppress the behavior, although stronger intervention also degraded output coherence.

**Conditional steering.** Constant steering introduced a tradeoff between suppression and generation quality at high strengths. I therefore tested a conditional version: I estimated a noise floor from the natural variation of orthogonal control directions during unsteered generation, then applied steering only when the live projection onto d exceeded that threshold. The automated detector reported 0/8 leaks while modifying only ~30% of generated tokens, suggesting that targeted intervention might achieve suppression with substantially less continuous disruption.

**A self-caught measurement artifact.** Manual inspection of those conditional-steering transcripts revealed an important caveat: 6 of the 8 apparently "held" rollouts had not genuinely avoided the secret. Instead, they attempted to state it but produced corrupted near-miss strings — for example, "X7X3-QLM" instead of "X7X9-QLM" — which the automated substring detector classified as non-leaks. The clean-looking 0/8 result therefore overstated the intervention's success. More interestingly, the failures suggest that steering disrupted exact token-level recall more readily than the broader tendency to comply. This also highlighted a methodological lesson from the experiment: automated leak-rate metrics can miss qualitatively important failure modes, making manual inspection essential when evaluating interventions on precise information recall.

---

## Red-Team Checks on d Itself

As a further check on what d actually captures, I compared it with a direction constructed from naturally occurring rather than explicitly instructed leak/hold examples. I used adaptive sampling to obtain a balanced set of 15 leaked and 15 held rollouts. The cosine similarity between the two directions was low (−0.15), suggesting that d may capture features of the explicitly instructed refusal/compliance setup that do not fully transfer to naturally occurring leaks. This is a genuine limitation: d should not be interpreted as a pure or universal representation of leak intent. At the same time, its causal effect in the steering experiments shows that it captures a behaviorally relevant component of the model's computation, even if it is not perfectly aligned with the naturally occurring direction.

A separate trained-probe cross-check was inconclusive rather than contradictory. A logistic-regression probe trained on only n = 14 activations achieved 1.00 training accuracy in a 2048-dimensional space, which is strongly suggestive of overfitting rather than evidence of a robust representation. I therefore do not treat this result as validation of d; a substantially larger dataset (at least ~50 examples per class) and held-out evaluation would be needed to make the comparison informative.

---

## Limitations

- **Small sample sizes.** All core results use only 8–18 rollouts per prompt, so reported leak rates and W₁ magnitudes should be treated as directional rather than precise population estimates.
- **Limited generalization evidence.** The core replication claim rests on three prompt types with genuine leak/hold splits. A wider sweep across prompts and domains would provide stronger evidence that the observed predictive mechanism generalizes.
- **Uncertainty about what d represents.** The extracted direction d shows weak alignment (cosine similarity ≈ −0.15) with a direction constructed from naturally occurring leak/hold examples. The trained-probe cross-check was also inconclusive: with only 14 examples per class in a 2048-dimensional space, its perfect training accuracy was strongly suggestive of overfitting rather than robust validation. Thus, d should not be interpreted as a pure or universal representation of "leak intent," even though it remains causally effective for steering.
- **Model scope.** All experiments used Qwen2.5-3B-Instruct in non-thinking mode. Qwen3.5's hybrid architecture was impractical on the available hardware, so whether the same predictive signal exists in a genuinely reasoning-trained model remains untested. This is particularly important because reasoning models were the original motivation for studying long-generation safety-instruction eviction.
- **Leak-detection limitations.** Leak detection relied primarily on substring matching, which is fast but brittle to paraphrases and corrupted near-miss strings. The conditional-steering experiment demonstrated that this can produce an apparently clean result that fails under manual inspection, making qualitative review important for evaluating interventions on exact information recall.

---

## Future Work

**What determines leak probability for a given prompt?** Leak propensity ranged from 0% to 100%, with some prompts producing genuinely mixed outcomes, and no single obvious factor — such as secret concreteness or domain — explained this variation. Candidate factors worth testing directly include the semantic relationship between distractor content and the secret, distractor length, and how the secret is phrased or positioned in the system prompt. Framed through a dynamical-systems lens, this asks whether some measurable property of the prompt acts as a control parameter, predicting which behavioral regime — reliably held, reliably leaked, or genuinely mixed — a given constraint falls into.

**Why does steering corrupt execution before intent?** The garbled near-miss finding suggests that fine-grained factual recall of a specific token sequence and the higher-level behavioral tendency to comply may respond differently to the same perturbation. Testing this directly — for example, by measuring at what intervention strength exact recall breaks down versus the underlying tendency to comply — could clarify whether these aspects of behavior are represented or controlled differently, with implications for how reliably activation steering can be used as a safety intervention.

**Testing in a genuinely reasoning-trained model.** The original motivation for this work was instruction eviction during long reasoning generations, yet the core experiments here used Qwen2.5-3B-Instruct in non-thinking mode. In early experiments, enabling thinking mode appeared to prevent leaks, potentially through repeated maintenance or rehearsal of the safety instruction. Extending the population-level monitoring and steering methodology to a model that reasons natively would therefore test whether the same predictive signal and eviction dynamics persist under genuine reasoning — the setting most directly motivating the original question.
