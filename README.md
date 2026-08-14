# optimization-gradient-lora

LoRA gradient geometry on real SLM weights — norm decay to delta structure to rank–LR coupling to merge criteria. CPU-only, no assumptions.

---

## Units

| # | Unit | Claim to test | Status |
|---|------|-------------|--------|
| 1 | Gradient norm geometry | The LoRA descent is structured: `A` frozen at step 0, norm decay, matrix-set gradient scale | ✅ Complete (4 experiments measured) |
| 2 | The learned delta | The landing is concentrated at the injected rank and aligned with its own gradient | ✅ Complete (4 experiments measured) |
| 3 | Why low-rank GD works | Full fine-tuning is energy-concentrated but tail-sprawling; the constraint wins by filtering the harmful tail | ✅ Complete (4 experiments measured) |
| 4 | Rank–learning-rate interaction | Rank is a hard capacity limit; LR is a quality dial; the effective step is `∝ α·lr/r` | ✅ Complete (4 experiments measured) |
| 5 | Scale/alpha dynamics | `α` is a weak, Adam-normalized dial; the robust property is the ratio `α/r`; α rescues a smaller rank | ✅ Complete (4 experiments measured) |
| 6 | Deployment: merge criteria | The merge is exact by linearity; the gradient-norm plateau is *not* the loss-floor point — gate stop/freeze on the loss floor | ✅ Complete (4 experiments measured) |

---

## Repository Structure

```
-optimization-gradient-lora/
├── README.md                          ← this file
├── NOTATION.md                        ← symbol registry across all units
├── INTUITION.md                       ← the why: purpose of each gradient + the equation behind each figure
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

Full definitions and named operations in [`NOTATION.md`](NOTATION.md). Intuition — what each gradient does and the equation behind every figure — in [`INTUITION.md`](INTUITION.md).

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

## Unit 3 — Why low-rank GD works (complete)

**Design.** Same task, same inputs, same optimizer, trained twice per matrix: once as LoRA (r=8, cached), once as full fine-tuning where the entire `W` is a free parameter (`ΔW_F = P_final − W`). Compare spectra (`k90`, strong count, `top1`), aligned fraction (energy of `ΔW_F` inside the LoRA core), principal-angle cosines vs `ΔW_L` and vs the true target `E`, and loss floors.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Full-FT spreads beyond injected rank | `k90(F)` ≫ 4 | `k90(F)` 4/6/4/4 but strong 139–150, rank 178–192 | ⚠️ Partial — energy concentrates, count sprawls |
| C2 | LoRA core holds most free energy | aligned ≥ 0.9 | W_Q 0.992; W_V 0.011; W_gate 0.002; W_up 0.003 | ❌ Reversed — W_Q only |
| C3 | Core directions match | cos ≥ 0.9 | W_Q 1.000; others 0.075–0.168 | ❌ Reversed — W_Q only |
| C4 | Full-FT finds target as well | cos(DF,E) ≥ cos(DL,E) | mixed; both low (E is not the natural basis; G=E·XᵀX is) | ⚠️ Mixed |
| C5 | Floors comparable | ratio ~1 | 0.58, 35.6, 0.91, 6.0 | ❌ Reversed — constraint wins |

**Verdict in one line.** The constrained path wins by *filtering a harmful tail*, not by matching full-FT: unconstrained training stays energy-concentrated (k90 ≈ 4) but sprawls across ~145 strong directions, and where the deltas diverge LoRA reaches a 6–35× lower floor. Only W_Q (square in the observable, d = n = 576) pins the solution down and aligns with full-FT (cos 1.000).

---

## Unit 4 — Rank × learning rate (complete)

**Design.** Sweep `r ∈ {1,2,4,8,16}` × `LR ∈ {1e-3, 3e-3, 1e-2, 3e-2}` on the controlled rank-4 task (W_Q), recording loss floor, steps-to-threshold, and final `‖∇B‖`. Multi-matrix check on W_V and W_gate.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Capacity threshold at the true rank | cliff at r=4 | r=1 5.25e-3, r=2 3.10e-3 **identical across all LRs**; r=4 drops to 2e-6 | ✅ Holds — sharp cliff, LR-independent below |
| C2 | Floor follows `rank × LR` | equal product → equal floor | fixed-LR floors worsen with rank (2.04e-6→7.78e-6 at lr=1e-2) | ❌ Not supported — effective step `∝ α/r` |
| C3 | Product too small fails | budget edge | O/x grid is capacity-only; budget edge only in floors (lr=1e-3 → ~2e-5) | ⚠️ Test too weak |
| C4 | LR too large degrades floor | stability edge | lr=3e-2 worse than lr=1e-2 at every r≥4 | ✅ Supported |
| C5 | Surface transfers | W_V/W_gate same shape | same cliff + sweet spot + degradation | ✅ Holds |

**Verdict in one line.** Rank is a hard capacity limit — sub-capacity floors (5.25e-3, 3.10e-3) are LR-independent to three significant figures — and LR is a quality dial with a flat working band (2–8e-6) between a budget edge (lr=1e-3) and a stability edge (lr=3e-2); the effective step is `∝ α·lr/r`, not the `rank × LR` product.

---

## Unit 5 — Scale and alpha (complete)

**Design.** Sweep `α ∈ {2,4,8,16,32,64}` at r=8, lr=1e-2 on the controlled rank-4 task (W_Q). Test the ratio law with equal-`α/r` pairs across r=4/8/16, and the rescue test (can a rank-4 adapter at high α match rank-8 at mid α?).

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | Floor improves with α then degrades | U-shape | floors 2.65e-6–7.06e-6 across α=2..64 (2.7×); no systematic rise | ❌ Reversed — flat band, weak dial |
| C2 | Floor depends on `α/r`, not α alone | equal ratio → equal floor | ratio-4 group tight (2.68/3.37/4.76e-6); ratio-2 degrades at r16 | ✅ Holds, caveat at high rank/low ratio |
| C3 | Optimal α is linear in rank | constant optimal `α/r` | constant ratio keeps floors in-band across r | ✅ Holds |
| C4 | α rescues a smaller rank | r=4 high α ≥ r=8 best | r=4/α32 3.51e-6 beats r=8/α16 7.06e-6 | ✅ Holds |
| C5 | α stability ceiling exists | very large α degrades | none at r=8/α64; hint at r=4 | ⚠️ Partial |

**Verdict in one line.** α is a weak, Adam-normalized step dial (a ~√k effect on the normalized step) with one robust property — the **ratio**: floor follows `α/r`, the constant-ratio rule transfers across ranks, and a rank-4 adapter at high α (3.51e-6) beats a rank-8 config at mid α (7.06e-6).

---

## Unit 6 — Deployment: merge criteria (complete)

**Design.** On the controlled rank-4 task (W_gate layer 0, r=8, α=16, Adam lr 1e-2, 300 steps) record per-step `‖∇B‖` and loss, then read three deployment cuts. C1 early stop: the floor at the plateau step `s*` (first step where `‖∇B‖ < 0.01·‖∇B‖₀`) vs the floor at step 300. C2 freeze: `A` frozen at `s*`, continued to 300, floor vs the unfrozen run. C3 merge: output-level linearity check `‖(W+ΔW)·Xᵀ − (W·Xᵀ + ΔW·Xᵀ)‖/‖·‖` at float precision. C4 per-layer: the plateau step and costs for W_gate at layers 0/10/20.

**Measured findings (SmolLM2-135M, not assumed):**

| # | Claim | Predicted | Measured | Verdict |
|---|---|---|---|---|
| C1 | The plateau is the early-stop point | floor at `s*` within ~1.5× | floor(s*)=1.63e-5 vs floor(300)=2.39e-6, cost 6.8×; layer 20 0.96×, layer 10 0.28× | ❌ Reversed — loss keeps falling ~6× after the gradient crosses the 1% line |
| C2 | Freezing `A` at the plateau is near-lossless | within ~1.5× | floor_frozen=1.74e-5 vs floor_full=2.39e-6, cost 7.3× | ❌ Reversed — `A` still works at `s*` |
| C3 | The merge is exact by linearity | rel err ≤ ~1e-7 | rel err 2.81e-7; `(α/r)B·A` recon error 0.0 | ✅ Holds exactly |
| C4 | The criteria are layer-local | steps and costs differ | s* clusters (110–118); cost spans 0.28–6.4× | ✅ Holds — a universal cut is unsafe |

**Verdict in one line.** The merge is free — exact to 2.8e-7 by linearity — but the gradient-norm plateau is *not* the loss-floor point: the loss improves ~6× after `‖∇B‖` falls 2 orders, and cutting there costs 6.8× at the core matrix (and varies 0.28–6.4× per layer). The corrected recipe: **merge by linearity, gate stop/freeze on the loss floor, not the gradient norm.**

---

## What this series builds toward

The 13-repo study series maps LoRA from decomposition through deployment on CPU-only real SLM weights. This repo is the mechanism core: unit 1 measures the path, unit 2 the landing, unit 3 shows the constraint wins by filtering the harmful tail, units 4–5 map the rank·LR·α operating space, unit 6 turns the norm plateau into a merge criterion (early stop, freeze, exact merge). Each notebook is self-contained and every number is measured on real matrices.