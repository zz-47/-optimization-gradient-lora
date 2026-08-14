# optimization-gradient-lora

LoRA gradient geometry on real SLM weights — norm decay to delta structure to rank–LR coupling to merge criteria. CPU-only, no assumptions.

---

## Units

| # | Unit | Claim to test | Status |
|---|------|-------------|--------|
| 1 | Gradient norm geometry | The LoRA descent is structured: `A` frozen at step 0, norm decay, matrix-set gradient scale | ✅ Complete (4 experiments measured) |
| 2 | The learned delta | The landing is concentrated at the injected rank and aligned with its own gradient | ✅ Complete (4 experiments measured) |
| 3 | Why low-rank GD works | A full fine-tuning produces a similar delta plus trailing noise — the constrained path wins by structure | ⬜ Planned |
| 4 | Rank–learning-rate interaction | The `rank · LR` product sets the achievable floor | ⬜ Planned |
| 5 | Scale/alpha dynamics | `α/r` controls the effective step size; optimal α is linear in rank | ⬜ Planned |
| 6 | Deployment: merge criteria | Gradient-norm convergence gives the early-stop and freeze point | ⬜ Planned |

---

## Repository Structure

```
-optimization-gradient-lora/
├── README.md                          ← this file
├── NOTATION.md                        ← symbol registry across all units
├── requirements.txt                   ← pinned env
├── .venv/                             ← local environment (hidden, gitignored)
├── ogl_cache/                         ← cached matrices, gradients, deltas
├── 1_gradient_norm_geometry.ipynb     ← the descent path: norms, frozen-A, matrices, layers, e2e
├── 2_delta_structure.ipynb            ← the landing: spectrum, alignment, per-matrix contrast
├── 3_why_lowrank_gd.ipynb             ← full-FT vs LoRA delta comparison
├── 4_rank_lr_interaction.ipynb        ← rank·LR product vs floor
├── 5_scale_alpha.ipynb                ← α/r dynamics and optimal α
└── 6_merge_criteria.ipynb             ← early stop, freeze, merge
```

---

## Notation (pre-defined, used across all units)

| Symbol | Meaning |
|--------|---------|
| `W` | Base weight matrix (W_Q, W_V, W_gate, W_up) |
| `X` | Input = `W_E[:576]`, real token embeddings |
| `E` | Injected rank-4 target delta, `‖E‖/‖W‖ = 0.1` |
| `A, B` | LoRA blocks: `A∈R^{r×k}`, `B∈R^{d×r}` |
| `α, r, scale` | alpha=16, rank=8, scale=`α/r`=2 |
| `ΔW` | Learned delta = `scale·B·A` |
| `W_eff` | `W + ΔW`, effective weight in the forward |
| `L` | Relative MSE loss (or CE in the e2e run) |
| `G` | Effective gradient `∂L/∂W_eff` |
| `∇A, ∇B` | Block gradients = `scale·BᵀG`, `scale·G·Aᵀ` |
| `‖·‖_F` | Frobenius norm |
| `σ_i` | Singular values, descending |
| `k90` | Smallest k with `Σ_{i≤k}σ_i² ≥ 0.9·Σσ_i²` |
| `top1` | `σ₁²/Σσ_i²`, energy in the leading direction |
| `strong count` | Number of `σ_i > 1e-3·σ₁` |
| `G₁ / ΣG_t` | Gradient at step 1 / accumulated over the run |
| `cos θ_i` | Principal-angle cosines = `σ_i(U_Δ[:,:k]ᵀ U_G[:,:k])` |

Full definitions and named operations in [`NOTATION.md`](NOTATION.md).

---

## Unit 1 — Gradient norm geometry (complete)

**Design.** Train real LoRA adapters (r=8, α=16, Adam lr 1e-2) to reconstruct outputs of real SmolLM2-135M matrices from real token embeddings. Record per-step Frobenius norms of the two adapter blocks, `‖∇A‖_F` and `‖∇B‖_F`, plus relative MSE. Four objects: raw trajectories on `W_gate`; the frozen-A initial condition; attention-vs-FFN at matched task strength; per-layer trend (0/10/20); plus an end-to-end fidelity run inside the live model under cross-entropy.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Norms decay with loss | `‖∇B‖` falls ≥2 orders | 4×–1.9e3×; layer-10 gate only 2× | ⚠️ Partial — universal decay, depth varies |
| C2 | Frozen-A: `‖∇A‖₀=0`, `‖∇B‖₀>0` | float noise | `0.0` exactly, all matrices/layers/e2e | ✅ Holds — stronger than predicted |
| C3 | Block-norm identities to float noise | ≤1e-6 rel | 1.4e-7–3.5e-7 | ✅ Holds |
| C4 | Attention vs FFN gradient scale | open | spread within attention: W_V 50× W_Q; FFN between | ❌ Reversed axis |
| C5 | Per-layer trend | open | A-wake@1 all layers; floor differs (layer 10 worst) | ⚠️ Phase invariant, floor layer-dependent |
| C6 | Norm decay couples to loss | plateau alignment | lower `‖dB‖end` ↔ lower floor; e2e CE 8.53→6.27 | ✅ Supports |

**Verdict in one line.** The descent is structured, not a scalar grind: `A` is exactly frozen at step 0 (`∇A=(α/r)BᵀG` with `B₀=0`), wakes within one Adam step, and the gradient scale is set by matrix geometry — `W_V`'s step-0 norm runs 50× `W_Q`'s, with gradients confined to 1–2 dominant directions.

---

## Unit 2 — The learned delta (complete)

**Design.** Re-run the same rig but record the landing: the final `ΔW` and the gradient records `G₁` (step 1) and `ΣG_t` (accumulated) from the same loop, for all four matrices. SVD each delta (rank, strong count, `k90`, `top1`) and compare its top subspace against the gradient's via principal-angle cosines.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Delta is concentrated | strong ≈ 4, `k90` small | strong 4/4/5/4; `k90(dW)=4` all four | ✅ Holds |
| C2 | Delta tighter than base | `k90(ΔW) ≪ k90(W)` | 4 vs 65/130/394/396 (16–99×) | ✅ Holds |
| C3 | Top directions align with gradient | cos ≥ 0.9 both records | cos₁ ≥ 0.997 all four; W_V 4th 0.18 vs `G₁` | ⚠️ Partial |
| C4 | Trailing directions are noise | tail cos ≈ 0 | dirs 5–6 keep 0.30–0.76 | ❌ Reversed |
| C5 | Alignment stable across matrices | all four strong | consistent at cos₁; W_V 4th, W_gate strong=5 | ⚠️ Partial |

**Verdict in one line.** The landing lands on the target rank exactly (`k90(ΔW)=4` vs base `k90` 65–396), its dominant direction aligns with the run's gradient (cos₁ ≥ 0.997 on both records) — but the tail is structured, not noise (directions 5–6 keep cosines 0.30–0.76), so the accurate claim is "rank 4 for 90% of the energy, plus a structured tail."

---

## What this series builds toward

The 13-repo study series maps LoRA from decomposition through deployment on CPU-only real SLM weights. This repo is the mechanism core: unit 1 measures the path, unit 2 the landing, unit 3 proves the constrained path wins by structure, units 4–5 map the rank·LR·α operating space, unit 6 turns the norm plateau into a merge criterion. Each notebook is self-contained and every number is measured on real matrices.