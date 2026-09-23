# FractlToCalculus — Investigation Summary

This document consolidates everything validated during this session's deep-dive
into the renormalization approach (delay/Hankel embedding → SVD mode discovery →
nonlinear lift → closed-form linear operator), across both its original domain
(chaotic dynamical systems) and an extended test of language modeling.

## 1. The core idea

Every script in this repo implements a version of the same five-step recipe:

1. **Delay-embed** a sequence into a Hankel matrix (a window of recent history per row).
2. **SVD-compress** to discover a low-rank "macro mode" subspace (the renormalization step: separating coherent structure from noise).
3. **Nonlinearly lift** the macro modes (quadratic/pairwise cross-terms, or a neighbor-interaction basis for spatial systems).
4. **Fit a linear operator** via closed-form ridge regression (no gradient descent).
5. **Roll out autonomously** to forecast/generate forward in time.

[renorm_core.py](renorm_core.py) and [metrics.py](metrics.py) factor this recipe into a shared library used by every benchmark below.

## 2. Validated strength: chaotic/physical system forecasting

Using [benchmark_chaos.py](benchmark_chaos.py) on the Lorenz-96 two-scale dataset ([lorenz96_sim.py](lorenz96_sim.py)), evaluated across 20 independent held-out rollout windows (not just one lucky starting point):

| Variant | Prediction horizon (mean±std, sim time) |
|---|---|
| Naive linear operator | 0.162 ± 0.052 |
| **Physics-informed quadratic/neighbor lift** | **0.977 ± 0.173** (best) |
| Auto-discovered SVD modes (linear only) | 0.202 ± 0.033 |
| SVD modes + nonlinear re-lift | 0.654 ± 0.314 |

**Key findings:**
- The lifted/Koopman operator gives a real, ~5x longer forecast horizon than naive linear regression on the same chaotic system — a genuine, non-trivial win.
- A bug was found and fixed in the original `grand_unification.py` (it reconstructed predictions from the *post-step* state instead of the *pre-step* state, an off-by-one that made it look artificially worse). Once fixed, the "auto-discover modes + re-lift" idea is legitimately the second-best approach, not the worst.
- **This is the part of the project with the most solid, reproducible evidence of real value.**

## 3. Language modeling investigation

### 3.1 Memorization check ([language_benchmark.py](language_benchmark.py))
On the original ~550-character corpus (80/20 split): train accuracy 0.793 vs. held-out 0.128 (chance 0.032). Not pure memorization, but a large overfitting gap — too little data for the model's capacity.

### 3.2 Scaling law on real data (tiny-Shakespeare, 1.1MB, the standard char-RNN/nanoGPT benchmark)
Fixed architecture (window=16, rank=32 SVD modes, quadratic lift), increasing training data 450x:

| Train chars | Val accuracy | Val bits/char |
|---|---|---|
| 2,000 | 0.133 | 5.912 |
| 150,000 | 0.217 | 5.891 |
| 900,000 | 0.216 | 5.895 |

**Held-out quality plateaus completely after ~150k characters and never improves again**, even with 6x more data — a hard capacity ceiling, not a data problem.

### 3.3 Capacity sweep ([capacity_sweep.py](capacity_sweep.py))
Doubling rank 32→64 only nudged accuracy 0.217→0.268, while fit time went 27s→86s and feature count 561→2145 (quadratic blow-up). Rank=128 needed **25GB RAM** just to build the feature matrix — infeasible. Unlike attention, this pairwise-polynomial lift's cost explodes combinatorially long before becoming competitive.

### 3.4 Reality check vs. a real transformer ([nanogpt_baseline.py](nanogpt_baseline.py))
A minimal 825k-parameter, 4-layer char-level Transformer, same data/split:

| Iterations | Wall-clock | Val accuracy | Val bits/char |
|---|---|---|---|
| 250 | 3.4 min | 0.274 | 3.61 |
| 1500 | 16.4 min | **0.414** | **2.86** |

The transformer surpasses the renorm approach's *best-ever* capacity-swept result within 3.4 minutes, and keeps improving with no plateau. **Verdict: training is genuinely cheap (seconds) but hits a hard quality ceiling that more data or raw capacity cannot fix — not a viable "cheap GPT" substitute as designed.**

### 3.5 Layering experiments (does depth/hierarchy close the gap?)

| Approach | Result |
|---|---|
| Unsupervised depth-stacking (SVD→tanh→feed-forward), [deep_renorm_language.py](deep_renorm_language.py) | **Hurt** (0.217→0.16-0.17): unsupervised layers discard target-relevant variance |
| Target-aware boosting (residual stacking), [deep_renorm_language.py](deep_renorm_language.py) | Tiny gain, fast diminishing returns (0.217→0.220 over 4x the compute) |
| **Word-level renormalization** ([word_level_benchmark.py](word_level_benchmark.py)) | No plateau yet even at max available data (0.056→0.129 accuracy, 450k word growth); ~1,600x chance vs. char-level's ~14x chance |
| **Hierarchical word+char generator** ([hierarchical_generator.py](hierarchical_generator.py)) | **Real gain**: ~27% bits/char reduction (5.94→4.33) — word-level decides identity in one shot, char-level only spells out-of-vocabulary (4.2%) words |
| Sentence/paragraph bag-of-words context → word prediction ([hierarchical_context_test.py](hierarchical_context_test.py)) | **Negligible** (0.127→0.127, robust null across regularization strength and context-window size) |

**Key insight:** repeating the *identical* trick (classify exact identity from a vocabulary) breaks down at the sentence/paragraph level because natural sentences are essentially never exact repeats (near-100% OOV) — unlike the closed, reused ~12.5k-word vocabulary. The correct generalization is continuous embeddings (bag-of-words + SVD), which was tested rigorously and found to add negligible value to word-level prediction, regardless of local context window (tested 1, 2, 6 words) or regularization (tested 1e-4 to 1e-1).

**Overall pattern:** the one genuine win (word+char hybrid) came from combining *qualitatively different* kinds of decisions (discrete word identity + continuous character fallback) — not from stacking more of the same linear/quadratic regression at a bigger scale. Depth/hierarchy alone, within this closed-form paradigm, does not reliably compound value the way it does in a trained neural network.

## 4. Bottom-line verdict

- **Chaotic/physical time-series forecasting**: genuinely validated. The renormalization recipe reliably beats naive linear baselines on a real chaotic system, is nearly free to train (closed-form, seconds), and the underlying "renormalization spectrum" (SVD singular-value knee) is a real, physically meaningful scale-separation signal — not something asserted, something measured.
- **Language modeling as a cheap GPT substitute**: does not hold up. Every variant tested (single-layer, unsupervised depth, boosting, sentence-level context) hits a capacity ceiling far below even a few minutes of real transformer training, because closed-form linear/quadratic regression is fundamentally less expressive than composed, gradient-trained attention. The one clear win (word+char hybrid) is real but modest, and still ~1.5 bits/char behind a tiny transformer.

## 5. Where the evidence points next

Given the user's read that this is "genuinely a good way of doing one-dimensional chaos predictions" — that is exactly where this session's evidence is strongest and most reproducible. Natural next steps for that direction (not yet executed, for future consideration):

- Test genuinely new 1D time-series domains beyond Lorenz-96 (financial returns, sensor/IoT streams, audio envelopes, epidemiological curves) using the same [renorm_core.py](renorm_core.py) + [metrics.py](metrics.py) harness, to see how far the "physics-informed or auto-discovered lift + closed-form operator" recipe generalizes as a fast, interpretable forecasting tool.
- Quantify robustness across multiple random seeds/forcing parameters of the Lorenz-96 system itself (not yet done) to confirm the benchmark_chaos.py results aren't overfit to one specific trajectory.
- Investigate why `grand_unification`-style re-lifting has high variance across windows (large std band in benchmark_results.png) — likely the more tractable, promising direction for improving the chaos-forecasting side, versus continuing to push the language side.

## 6. File inventory (this session)

| File | Purpose |
|---|---|
| [renorm_core.py](renorm_core.py) | Shared recipe: Hankel embedding, SVD modes, lifts, ridge/pinv fitting, layer stacking, boosting, rollout |
| [metrics.py](metrics.py) | Relative error, prediction horizon, multi-window evaluation |
| [benchmark_chaos.py](benchmark_chaos.py) / [benchmark_results.png](benchmark_results.png) | Quantitative multi-window comparison of the 4 chaos-forecasting variants |
| [language_benchmark.py](language_benchmark.py) | Memorization check + char-level scaling law on tiny-Shakespeare |
| [capacity_sweep.py](capacity_sweep.py) | Rank/window capacity sweep for the char-level model |
| [nanogpt_baseline.py](nanogpt_baseline.py) | Minimal transformer baseline, same data/split |
| [deep_renorm_language.py](deep_renorm_language.py) | Unsupervised depth-stacking and boosting variants |
| [word_level_benchmark.py](word_level_benchmark.py) | Word-level (sparse) scaling law with honest OOV handling |
| [hierarchical_generator.py](hierarchical_generator.py) | Word+char hybrid generator and evaluation |
| [hierarchical_context_test.py](hierarchical_context_test.py) | Sentence/paragraph embedding context test |
| [tinyshakespeare.txt](tinyshakespeare.txt) | Standard 1.1MB char-RNN/nanoGPT benchmark corpus (downloaded) |
