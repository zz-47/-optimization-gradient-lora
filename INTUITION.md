# INTUITION — optimization-gradient-lora

The *why* behind every gradient we measured, and the one equation that makes each figure speak. Read this alongside NOTATION.md (the symbols) — this file is the intuition.

## The cast of gradients

| Quantity | Role in training (easy words) | The key equation |
|---|---|---|
| `∇B` | The "forward engine" — how the adapter's output changes. Since `B` multiplies `A` on the left, it reshapes the *amplitude* of each rank direction. | `∇B = (α/r)·G·Aᵀ` |
| `∇A` | The "direction tuner" — how the adapter picks *which* input features matter. At step 0 it is exactly 0 because `B₀=0` | `∇A = (α/r)·Bᵀ·G` |
| `G` | The effective gradient — the "would-be full-FT step". LoRA never takes it directly; it projects it through `B`/`A`. | `G = 2RᵀX/(n·d·mean(Y²))`, with `R = X·W_effᵀ − Y` |
| `G₁` | The direction at step 1 — the first real push. | `G₁ = G` evaluated at `s=1` |
| `ΣG_t` | The total push over the run — where the trajectory *actually went*, integrated. | `ΣG_t = Σ_s G_s` |

**One mental model:** full-FT takes the raw `G`. LoRA *filters* it — `∇B` and `∇A` are `G` projected through the low-rank factors. That projection IS the constraint that wins in study 3.

---

## Study 1 — The descent path (gradient norms over steps)

**Figure: `‖∇A‖` and `‖∇B‖` vs step.**
- What you're seeing: the *push strength* decaying as the adapter approaches the target.
- The equation that makes it work: `∇A₀ = (α/r)·B₀ᵀ·G₀ = 0` exactly, because `B₀ = 0`. **A is frozen at step 0 by construction, not by luck.**
- Why it decays: as `W_eff → W+E`, the residual `R → 0`, so `G → 0`, so both block gradients → 0.
- Specialist read: a *structured* descent (frozen-A start → two-block learning) — a scalar grind would not show `‖∇A‖₀ = 0` exactly.

---

## Study 2 — The landing (spectrum of `ΔW`)

**Figure: singular values of `ΔW` (log scale).**
- What you're seeing: the learned delta's energy concentrated on ~4 directions.
- The equation: `ΔW = (α/r)·B·A`, so `rank(ΔW) ≤ r = 8` — the budget is fixed at the start.
- `k90` reads it: `Σ_{i≤k}σᵢ² ≥ 0.9·Σσᵢ²` → the smallest `k` carrying 90% of energy. Measured `k90(ΔW)=4` while `k90(W)` is 65–396: the adapter spent its budget on the target rank, not the base weight's.

**Figure: principal-angle cosines vs gradient.**
- The equation: `cos θᵢ = σᵢ(U_Δ[:,:k]ᵀ U_G[:,:k])` — how the delta's top space overlaps the gradient's.
- Measured `cos₁ ≥ 0.997`: the landing lies *along* the direction the run pushed. The tail (dirs 5–6, cos 0.3–0.76) is structured, not noise.

---

## Study 3 — Full-FT vs LoRA (why the constraint wins)

**Figure: spectra of `ΔW_F` (full-FT) vs `ΔW_L` (LoRA).**
- What you're seeing: both concentrate energy (`k90 ≈ 4`), but full-FT *sprawls* — ~145 "strong" directions, `rank ≈ 180`.
- The decisive equation: **loss-floor ratio** = `floor(F)/floor(L)` = 0.58, 35.6, 0.91, 6.0 → LoRA wins by **filtering a harmful tail** the unconstrained path chases.
- Why W_Q is the exception: it's square (`d = n = 576`), so `X·ΔWᵀ` observes the full matrix — the unconstrained path pins the same solution and `cos(ΔW_F, ΔW_L) = 1.000`.

---

## Study 4 — Rank × LR (capacity vs step)

**Figure: floor vs rank, one curve per LR (log scale).**
- What you're seeing: hard capacity walls at r<4, then a LR-driven quality band at r≥4.
- The equation: sub-capacity floors are **LR-independent** — r=1 gives 5.25e-3, r=2 gives 3.10e-3 *identically across all four LRs* — because the task is rank 4 and the adapter *cannot represent it* regardless of step size.
- Specialist read: **rank = capacity, LR = quality.** Two knobs, different jobs. Effective step is `∝ α·lr/r`, not `rank × LR`.

---

## Study 5 — α and the ratio law

**Figure: floor vs α (semilogy) — a flat band, not a U.**
- What you're seeing: α sweeps 32×, the floor moves 2.7×.
- The equation: `α` appears *only* as `α/r` — in `ΔW`, `∇A`, `∇B`. The ratio is the invariant.
- Why α is weak: Adam normalizes each gradient by its own magnitude (`m̂/√v̂`), so a constant scale multiplier partially divides itself out — a ~√α effect, not α.
- The rescue: `floor(r=4, α=32) = 3.51e-6 < floor(r=8, α=16) = 7.06e-6` — above capacity, α substitutes for rank.

---

## Study 6 — The deployment recipe (stop / freeze / merge)

**Figure: loss and `‖∇B‖` vs step — the gap.**
- What you're seeing: the gradient crosses 1% of its initial at step 110, **but the loss keeps falling 6.8×** to step 300. The norm plateau is *not* the loss plateau.
- The equation that corrects the naive recipe: `floor(s*)/floor(300) = 6.8` on layer 0 (0.28–6.4× per layer) — cutting at the gradient plateau costs real quality.
- The equation that always holds: `X·(W+ΔW)ᵀ = X·Wᵀ + X·ΔWᵀ` — **exact by linearity** (measured rel err 2.81e-7). The merge is free.

---

## Four takeaways

1. **Structured path** — LoRA descends with `A` frozen at step 0 (by `B₀=0`), norm decay, matrix-set gradient scale.
2. **Filter, not match** — LoRA beats full-FT by filtering a harmful tail; the constraint is the win.
3. **Two knobs** — rank = capacity (hard wall), `α·lr/r` = step (quality). Never three independent knobs.
4. **Merge free, stop on loss** — merge by linearity (exact); gate stop/freeze on the loss floor, not the gradient norm.
