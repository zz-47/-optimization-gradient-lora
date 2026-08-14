# NOTATION — optimization-gradient-lora

Every symbol, metric, and named operation in this repository, defined once. Cross-references: the previous studies defined the weight-matrix symbols (W_Q, W_V, W_gate, …), the LoRA parametrization, and the spectral metrics (k90, top1, strong count).

## Base objects

| Symbol | Shape | Meaning |
|---|---|---|
| `W` | (d, k) | base weight matrix (W_Q, W_V, W_gate, W_up), layer 0 by default |
| `W_E` | (V, k) | token embedding matrix, V = 49152 |
| `X` | (n, k) | input = `W_E[:576]`, real token rows, n = 576 |
| `E` | (d, k) | injected rank-4 target delta, `‖E‖/‖W‖ = 0.1` |
| `Y` | (n, d) | target output = `X·(W + E)ᵀ` |

## LoRA parametrization

| Symbol | Meaning |
|---|---|
| `A` | (r, k) A-block, init ~N(0, 0.02²) |
| `B` | (d, r) B-block, init 0 |
| `α`, `r`, `scale` | alpha = 16, rank = 8, scale = α/r = 2 |
| `ΔW` | (d, k) = scale·B·A, the learned delta |
| `W_eff` | W + ΔW, the matrix used in the forward |

## Loss & gradient

| Symbol | Meaning |
|---|---|
| `R` | residual = `X·W_effᵀ − Y` |
| `L` | relative MSE = `mean(R²)/mean(Y²)` |
| `G` | effective gradient = `∂L/∂W_eff` = `2RᵀX/(n·d·denom)` |
| `denom` | `mean(Y²)`, the loss normalizer |
| `∇A` | `scale·Bᵀ·G`, A-block gradient |
| `∇B` | `scale·G·Aᵀ`, B-block gradient |

## Metrics

| Symbol | Meaning |
|---|---|
| `‖·‖_F` | Frobenius norm = √(Σσᵢ²) |
| `σᵢ` | singular values, descending |
| `k90` | smallest k with `Σ_{i≤k}σᵢ² ≥ 0.9·Σσᵢ²` (effective rank) |
| `top1` | `σ₁²/Σσᵢ²`, leading-direction energy fraction |
| `strong count` | number of `σᵢ > 1e-3·σ₁` |
| `matrix_rank` | number of nonzero singular values at float precision |
| `G₁` | effective gradient at step 1 (the early direction) |
| `ΣG_t` | effective gradient accumulated over all steps (the total direction) |
| `cos θᵢ` | principal-angle cosines = `σᵢ(U_Δ[:,:k]ᵀ U_G[:,:k])` |

## Named operations

- **Frozen-A phase** — with `B₀ = 0`, `∇A = scale·Bᵀ·G = 0` exactly at step 0 while `∇B ≠ 0`; the adapter begins as a B-only learner.
- **Crossover step** — the step where `‖∇A‖` stops being negligible and the adapter becomes two-block.
- **Norm plateau** — the region where `‖∇B‖` flattens as the loss converges; the basis for early stopping.
- **Gradient concentration** — how few directions carry the gradient's energy (read via `k90(G)`).
- **Alignment** — the overlap of the learned delta's column space with the gradient's, read via principal-angle cosines.
- **Matched task strength** — `‖E‖/‖W‖ = 0.1` with the same `X` for every matrix, so differences are matrix-side, not difficulty-side.