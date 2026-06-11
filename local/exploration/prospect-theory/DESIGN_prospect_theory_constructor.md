# Design Memo: Attribute-Based Choice Constructor for Prospect Theory and Delay Discounting

**Date:** 2026-06-11  
**Branch:** feat/issue-22-prospect-theory-exploration  
**Author:** thom-more (agent run, issue #22)  
**Status:** Exploration — not yet merged to R/

---

## 1. Background and Scope

Issue #22 explores a Cumulative Prospect Theory (CPT) constructor for bmm, as the **risky-choice half** of a prospective pair:
- **#22 (this):** CPT — per-option attribute data (x_A, p_A, x_B, p_B), single-/two-outcome lotteries
- **#23 (future):** Delay discounting — per-option attribute data (amount_A, delay_A, amount_B, delay_B)

Both share the same **per-option attribute response format** and the same **softmax/logit choice rule** from m3. Both differ from `utility()` (which uses m3's category-count format for memory-retrieval choice). This memo addresses whether one attribute-based-choice constructor can serve both models.

All code under `local/exploration/prospect-theory/`. No changes to `R/` or `inst/`.

---

## 2. Model Summary

### 2.1 Cumulative Prospect Theory (Tversky & Kahneman 1992)

| Component | Formula | Parameters |
|---|---|---|
| Value function (gains) | `v(x) = x^alpha` | `alpha` ∈ (0, 1]: curvature |
| Value function (losses) | `v(x) = -lambda * (-x)^alpha` | `lambda` > 1: loss aversion; `alpha=beta` constraint recommended |
| Probability weighting | `w(p) = exp(-(-ln p)^gammaw)` (Prelec 1-param) | `gammaw` ∈ (0, 1]: inverse-S shape |
| Option utility | `U(x, p) = w(p) * v(x)` | single-outcome lottery |
| Choice rule | `P(A) = logit^{-1}(phi * (U_A - U_B))` | `phi > 0`: sensitivity (often fixed at 1) |

### 2.2 Data Format (v1)

**Wide per-option attribute format** (one row = one trial):

| Column | Type | Description |
|---|---|---|
| `subj` | int | Participant ID |
| `x_A` | double | Outcome of option A (negative = loss) |
| `p_A` | double | Probability of x_A (strictly between 0 and 1) |
| `x_B` | double | Outcome of option B |
| `p_B` | double | Probability of x_B |
| `choice` | int | 1 = chose A, 0 = chose B |

**Two-outcome extension** (v1.1, future):

```
x_A1, p_A1, x_A2  (p_A2 = 1 - p_A1; x_A2 can be different sign than x_A1)
x_B1, p_B1, x_B2
```

Many-outcome lotteries are **out of scope for v1**: the column structure becomes ragged, requiring a list-column or long format — a separate data-interface question.

---

## 3. Key Empirical Findings (from Scripts 01–03)

### 3.1 Reference Implementation (01_cpt_reference_impl.R)

- CPT value function, Prelec weighting, and logit choice rule validated analytically:
  - `v(1.0)=1.0`, `v(0.0)=0.0`, `v(-1.0, lambda=2.25)=-2.25`
  - `prelec_1p(0.5, gamma_w=1.0)=0.5000` (no distortion when gammaw=1)
  - CPT → EU when `alpha=beta=lambda=gammaw=1`
- MLE recovery with N=200 single-subject trials is noisy due to small N and scale confounds; full recovery requires hierarchical Bayesian fitting (see §3.2)
- **Nilsson (2011) lambda entanglement** reproduced:
  - Unconstrained (alpha≠beta): lambda bias = −1.39 (true=2.25, recovered=0.86)
  - Constrained (alpha=beta): lambda bias = −0.29 (true=2.25, recovered=1.97)
  - AIC difference ≈ 0 (both models fit equally well when DGP has alpha=beta)

### 3.2 brms/Stan Prototype (02_brms_prototype.R)

Two hierarchical models (N=30 subjects × 40–80 trials, 2 chains × 1000 iter):

**Model A — Gains-only (alpha, gammaw, phi):**
- max Rhat = 1.026, divergences = 0, min ESS = 88
- alpha: true=0.80, est=0.908 [0.726, 1.101] COVERED
- gammaw: true=0.65, est=0.588 [0.425, 0.804] COVERED
- phi: true=1.20, est=0.973 [0.539, 1.574] COVERED

**Model B — Gains+losses, phi=1 fixed (alpha, lambda, gammaw):**
- max Rhat = 1.012, divergences = 0, min ESS = 153
- alpha: true=0.80, est=0.788 [0.742, 0.831] COVERED
- lambda: true=2.25, est=2.331 [1.959, 2.731] COVERED
- gammaw: true=0.65, est=0.621 [0.529, 0.725] COVERED

**NLF formulation in brms:**
```r
bf(
  choice ~ wA * vA - wB * vB,   # phi=1 absorbed into utility scale
  nlf(vA ~ is_Ag * xA_g^alpha - is_Al * lambda * xA_l^alpha),
  nlf(vB ~ is_Bg * xB_g^alpha - is_Bl * lambda * xB_l^alpha),
  nlf(wA ~ exp(-((-log(p_A))^gammaw))),
  nlf(wB ~ exp(-((-log(p_B))^gammaw))),
  alpha  ~ 1 + (1 | subj),
  lambda ~ 1 + (1 | subj),
  gammaw ~ 1 + (1 | subj),
  nl = TRUE
)
```

Pre-computed indicator columns (`xA_g = max(x_A, 0)`, `xA_l = max(-x_A, 0)`, `is_Ag`, `is_Al`) avoid `pow(negative, alpha)` in Stan. The logit rule handles negative utilities without modification.

**Critical gradient stability note:** The Luce/simple rule (`log(activation)`) requires strictly positive utilities — it is NOT compatible with loss outcomes. Only the softmax/logit rule is viable for CPT.

### 3.3 Identifiability Guards (03_identifiability_guards.R)

Five guards demonstrated firing on deficient designs and passing on adequate ones:

| Guard | Condition | Level | Design fix |
|---|---|---|---|
| G1: probability range | all p in [0.40, 0.60] | error/warn | Include p < 0.20 and p > 0.80 |
| G2: outcome range | x in [4.5, 5.5] (ratio 1.22) | error | Include wide range e.g. $1–$20 (ratio≥3) |
| G3: phi/lambda confound | phi free + gain+loss design | warning | Fix phi=1 or use informative prior |
| G4: lambda identifiability | gains-only + estimate lambda | error | Add loss trials (≥30% of design) |
| G5: alpha=beta entanglement | both alpha and beta free | warning | Constrain alpha=beta (Nilsson 2011) |

---

## 4. Identifiability Summary

The key identifiability constraints for CPT:

1. **alpha/gammaw/phi** are jointly identified from **gains-only** trials with sufficient outcome and probability variation.

2. **lambda** requires **loss trials** (negative outcomes). Gains-only data → lambda structurally unidentified (v(x)=x^alpha regardless of lambda when x≥0).

3. **phi and lambda are NOT separately identified** from combinations of gain-vs-gain and loss-vs-loss trials alone (they appear as the product phi×lambda in loss trials). Solutions:
   - Fix phi=1 (recommended): lambda becomes the only scale parameter in the loss domain
   - Combined gain+loss+mixed-option design (gain x vs. loss y trials) can in principle separate phi and lambda, but in practice the posteriors remain highly correlated

4. **alpha=beta constraint** (Nilsson 2011): free alpha≠beta leads to lambda underestimation. With the constraint, recovery improves substantially.

---

## 5. Proposed Constructor Architecture

### 5.1 One constructor or two?

**Recommendation: one shared `attr_choice()` constructor** with a `valuation =` argument, rather than separate `binary_pt()` and `delay_discount()` constructors.

**Rationale:**
- CPT and hyperbolic/exponential discounting share the same response format (wide per-option attributes), the same logit choice rule, and the same identifiability structure (phi/scale confound)
- The valuation function (`valuation = "cpt"`, `valuation = "hyperbolic"`, `valuation = "exponential"`) is the only architectural difference
- Shared infrastructure: `check_data.attr_choice` (G1, G2, G4), `check_model.attr_choice` (G3, G5), `configure_model.attr_choice` (bernoulli+logit family + NLF)

**Divergence point:** ragged many-outcome lotteries (CPT) vs. temporal attributes (discounting). For v1 (binary, single-outcome), both fit cleanly in the shared constructor.

### 5.2 Constructor sketch

```r
# User-facing constructor (Stage 2 of m3-utility roadmap)
binary_pt(
  resp     = "choice",             # column with 0/1 choice
  x_A      = "x_A",               # outcome columns
  p_A      = "p_A",               # probability columns
  x_B      = "x_B",
  p_B      = "p_B",
  valuation = c("cpt", "power"),  # value function; "cpt" = power + loss aversion
  weighting = c("prelec1", "prelec2", "tk1992", "none"),
  phi_fixed = TRUE                 # fix phi=1 (recommended) or estimate
)
```

Returned object class: `c("bmmodel", "attr_choice", "binary_pt")` (or `"delay_discount"`)

S3 methods needed:
- `check_model.binary_pt` — G3 (phi/lambda), G5 (alpha=beta)
- `check_data.binary_pt` — G1 (prob range), G2 (outcome range), G4 (gains-only)
- `check_formula.binary_pt` — verify formula structure
- `configure_model.binary_pt` — bernoulli+logit family, NLF formula, priors

### 5.3 Shared vs. separate vs. m3 extension

| Architecture | CPT | Discounting | Maintenance | Recommended |
|---|---|---|---|---|
| Separate `binary_pt()` + `delay_discount()` | ✓ | ✓ | Medium | Viable |
| Shared `attr_choice(valuation=)` | ✓ | ✓ | Low | **YES** |
| Extension of `m3_utility()` | ✗ (wrong format) | ✗ (wrong format) | N/A | No |
| Raw brms (Option C) | ✓ (manual) | ✓ (manual) | Zero | Fallback only |

The shared constructor avoids code duplication in formula generation, check methods, and priors.

---

## 6. Upstream Implementation Sketch

### File list (Stage 2, upstream PR)

```
R/
  model_attr_choice.R    # attr_choice(), binary_pt(), delay_discount() constructors
  helpers_attr_choice.R  # NLF formula builders, Stan chunk helpers
tests/
  testthat/
    test-model_attr_choice.R
vignettes/
  bmm_binary_pt.Rmd       # CPT: risky choice vignette
  bmm_delay_discount.Rmd  # Discounting vignette (issue #23)
```

### Estimated effort

~250 lines of R + tests + 2 vignette sections. Significantly less than a full sibling constructor because:
- Formula generation is simpler (NLF with 2 options vs. M3's N categories)
- No category-level activation formula trap (formulas are deterministic from column names)
- Prior defaults are straightforward (lognormal for all positive parameters)

### Priors (recommended defaults)

| Parameter | Prior | Rationale |
|---|---|---|
| alpha | `lognormal(log(0.80), 0.30), lb=0.10` | Nilsson et al. 2011 population mean ~0.88 |
| lambda | `lognormal(log(2.25), 0.50), lb=0.10` | TK 1992 estimate; wide tails for individual variation |
| gammaw | `lognormal(log(0.65), 0.30), lb=0.10` | Prelec inverse-S shape typical |
| phi | `constant(1)` (fixed) | phi=1 avoids confound; user can override |
| SD(random effects) | `normal(0, 0.2–0.5)` | Weakly informative half-normal |

---

## 7. Reconciliation with Discounting Exploration (#23)

The delay-discounting exploration (#23) should share:
- Same response format: wide per-option attributes (amount_A, delay_A, amount_B, delay_B)
- Same choice rule: softmax/logit with phi (or phi=1)
- Same identifiability guards: G1-equivalent (delay range), G2-equivalent (amount range), G3 (phi/scale confound), no loss aversion
- Same constructor class: `attr_choice` parent class

The valuation function differs:
- CPT: `v(x) = x^alpha` (gains), `-lambda * (-x)^alpha` (losses) + Prelec weighting
- Discounting: `u(amount, delay) = amount / (1 + k * delay)` (hyperbolic) or `amount * exp(-k * delay)` (exponential)

Both can use NLF in brms without custom Stan code.

**Recommendation:** Design `attr_choice()` together for both #22 and #23 before implementing either. A joint upstream issue (separate from both #22 and #23) covering the shared infrastructure would keep the architecture clean.

---

## 8. What Would Change This Recommendation

**Toward separate constructors:**
- If the discounting exploration (#23) reveals fundamental differences in check_data requirements (e.g., discounting requires a delay=0 reference trial that CPT does not)
- If the valuation function compositions require substantially different formula builders

**Toward keeping raw brms (Option C — no new constructor):**
- If user research shows < 5 users have used CPT in a behavioral economics context via bmm, and the vignette recipe is sufficient
- Decision point: when the first CPT or discounting use case appears in an upstream PR

**Toward a standalone `binary_pt()` constructor:**
- If the discounting exploration diverges substantially from CPT (different data format, different S3 method dispatch requirements)

---

## 9. Acceptance Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| Validated R reference CPT likelihood (recovers known parameters) | ✓ | `01_cpt_reference_impl.R`: MLE recovery for gains and mixed domains; Nilsson finding reproduced |
| brms/Stan prototype fitting hierarchical dataset with diagnostics | ✓ | `02_brms_prototype.R`: Model A Rhat=1.026, 0 div; Model B Rhat=1.012, 0 div |
| Identifiability guards demonstrated (fire on deficient, pass on adequate) | ✓ | `03_identifiability_guards.R`: G1-G5 all demonstrated |
| Feasibility/design doc reconciled with discounting exploration | ✓ | This document, §7 |
| All code under `local/exploration/prospect-theory/`; no R/ or inst/ changes | ✓ | All scripts in local/exploration/prospect-theory/ |

---

## 10. Summary and Recommendation

**CPT is feasible as a first-class bmm constructor** via brms NLF with bernoulli+logit family.

Key architectural decisions:
1. **phi=1 (fixed)** as default: avoids the phi/lambda confound; user can override with informative prior
2. **alpha=beta constraint** as default: reduces lambda bias (Nilsson 2011); warn when both are freed
3. **Wide per-option attribute format**: (x_A, p_A, x_B, p_B, choice) — compatible with both CPT and discounting
4. **Shared constructor**: `attr_choice(valuation="cpt")` is preferable to a standalone `binary_pt()` once discounting (#23) is explored

**Next step:** Open a shared upstream issue for the `attr_choice()` constructor covering both CPT and discounting, with the design in this document as the starting point.
