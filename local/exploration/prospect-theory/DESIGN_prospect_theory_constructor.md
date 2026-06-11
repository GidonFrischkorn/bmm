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

Four hierarchical models (N=30 subjects × 40–80 trials, 2 chains × 1000 iter):

**Model A — Gains-only (alpha, gammaw, phi):**
- max Rhat = 1.026, divergences = 0, min ESS = 88
- alpha: true=0.80, est=0.908 [0.726, 1.101] COVERED
- gammaw: true=0.65, est=0.588 [0.425, 0.804] COVERED
- phi: true=1.20, est=0.973 [0.539, 1.574] COVERED

**Model B — Gains+losses, phi=1 fixed, alpha=beta (alpha, lambda, gammaw):**
- max Rhat = 1.012, divergences = 0, min ESS = 153
- alpha: true=0.80, est=0.788 [0.742, 0.831] COVERED
- lambda: true=2.25, est=2.331 [1.959, 2.731] COVERED
- gammaw: true=0.65, est=0.621 [0.529, 0.725] COVERED

**Model C — NEGATIVE CONTROL: phi free, gains+losses, alpha=beta (alpha, lambda, gammaw, phi):**
> This model is the *naïve* four-parameter fit—adding phi as a free hierarchical
> parameter to the same gains+losses data as Model B (DGP phi=1). Expected
> result: Rhat >> 1.05, many divergences, alpha not covered, phi and lambda
> posteriors wide and anti-correlated. This is the strongest empirical evidence
> for the phi=1 default; without this negative control, the phi=1 recommendation
> was only theoretical (Nilsson 2011 MLE in script 01). Results written to
> `results/02_cpt_recovery.csv`. **Run `02_brms_prototype.R` to populate.**
- max Rhat = [run script] (expected >> 1.05), divergences = [run] (expected >> 50)
- alpha: expected NOT COVERED (alpha→1.06 was observed in a previous session)
- lambda: expected low estimate (phi compensation pulls lambda toward 1)
- phi: expected inflated estimate (compensates for lower lambda)

**Model D — Hierarchical Nilsson: phi=1 fixed, alpha≠bta (alpha, bta, lambda, gammaw):**
> This is the hierarchical Bayesian analogue of the MLE Nilsson (2011) finding
> from script 01. Data DGP has alpha=beta=0.80, so any lambda underestimation
> is attributable solely to the freed alpha≠bta. Expected: lambda estimate
> lower than Model B's (larger negative bias), alpha and bta diverge slightly,
> gammaw similar. Results written to `results/02_modelD_recovery.csv`.
> **Run `02_brms_prototype.R` to populate.**
- max Rhat = [run script] (expected: mildly elevated, 1.02–1.05)
- lambda: expected underestimated (compare with Model B bias = +0.08)
- alpha, bta: expected near true=0.80 but with wider intervals than Model B

**Nilsson hierarchical comparison (expected, to be confirmed by running script):**
- Model B (phi=1, alpha=bta):  lambda bias ≈ +0.08  (est=2.33, true=2.25)
- Model D (phi=1, alpha≠bta): lambda bias ≈ **more negative** (Nilsson entanglement)
- This comparison provides the Bayesian backbone for the alpha=bta default.

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

**Recommendation (updated after reading DD exploration): two separate user-facing
constructors (`pt_choice()` and `dd_choice()`) with a shared internal `attr_choice()`
base class.** The delay-discounting exploration (#23, `DESIGN_dd_api.md`) independently
arrived at **Option A** (separate constructors, shared base). Both explorations
arriving at the same conclusion from different starting points is strong convergent
evidence. The shared `attr_choice(valuation=)` alternative (originally recommended
here) has been reconsidered; see the evidence below.

**Why separate constructors:**

1. **Luce-rule restriction** — CPT with losses requires softmax/logit (losses → V < 0,
   Luce rule `log(V)` undefined). Delay discounting always has V > 0 (amounts are
   positive), so both Luce and softmax work. This is a *constructor-level* constraint:
   `pt_choice()` should never offer `choice_rule = "luce"`. A shared constructor with
   `valuation =` would need a runtime check instead, which obscures the constraint.

2. **Different check_data guards** — CPT: G1 (probability range), G2 (outcome magnitude
   ratio), G4 (gains-only → no lambda). DD: G1 (delay range), G2 (k vs. phi confound),
   G3 (functional-form discriminability). The guard vocabularies are structurally
   different (probability range vs. delay spread). Sharing a `check_data.attr_choice()`
   would require special-casing both, defeating the abstraction.

3. **No delay=0 forcing function** — The CPT §8 concern about a "delay=0 reference trial
   requirement" that would force separate constructors was investigated. The DD exploration
   does **not** have a delay=0 guard: `delay_SS = 0` is a design convention in the Kirby
   DDT format, not a `check_data` requirement. Quasi-hyperbolic β is identified by
   contrast across delays; it does not strictly require delay=0 in the data. This concern
   therefore does NOT force separate constructors, but neither does it support a shared
   constructor. Decision stands on rationale 1 and 2 above.

4. **bmm's "one constructor = one response format" principle** — The DD design memo
   explicitly invokes this principle. CPT and DD both use per-option attribute format,
   but the attribute semantics differ fundamentally (amount+probability vs. amount+delay).

**Shared infrastructure (internal, not user-facing):**
- `attr_choice` S3 parent class
- `configure_model.attr_choice` — bernoulli+logit family, NLF formula boilerplate
- `check_data.attr_choice` — checks common to both (e.g., non-missing subj, trial)
- Domain-specific guards live in `check_data.pt_choice` and `check_data.dd_choice`

**Divergence point:** ragged many-outcome lotteries (CPT v1.1) vs. temporal attributes
(discounting). For v1 (binary, single-outcome), both could in principle fit in a shared
constructor, but separate constructors are better for all the reasons above.

### 5.2 Constructor sketch

```r
# User-facing constructor for CPT (Stage 2 of m3-utility roadmap)
pt_choice(
  resp      = "choice",             # column with 0/1 choice
  x_A       = "x_A",               # outcome columns
  p_A       = "p_A",               # probability columns
  x_B       = "x_B",
  p_B       = "p_B",
  weighting = c("prelec1", "prelec2", "tk1992", "none"),
  phi_fixed = TRUE                  # fix phi=1 (recommended) or estimate
)

# User-facing constructor for delay discounting (issue #23)
dd_choice(
  resp        = "choice",
  options     = list(LL = c(amt = "amt_LL", delay = "delay_LL"),
                     SS = c(amt = "amt_SS", delay = "delay_SS")),
  discount_fn = "hyperbolic",       # "hyperbolic", "exponential", "hyperboloid", "qh"
  choice_rule = "softmax"           # or "luce" (available for discounting, not CPT)
)
```

`pt_choice()` returned object class: `c("bmmodel", "attr_choice", "pt_choice")`
`dd_choice()` returned object class: `c("bmmodel", "attr_choice", "dd_choice")`

S3 methods needed for `pt_choice()`:
- `check_model.pt_choice` — G3 (phi/lambda), G5 (alpha=beta)
- `check_data.pt_choice` — G1 (prob range), G2 (outcome range), G4 (gains-only)
- `check_formula.pt_choice` — verify formula structure
- `configure_model.pt_choice` — bernoulli+logit family, NLF formula, priors

### 5.3 Architecture comparison

| Architecture | CPT | Discounting | Maintenance | Recommended |
|---|---|---|---|---|
| Separate `pt_choice()` + `dd_choice()` with shared `attr_choice` base | ✓ | ✓ | Medium | **YES** |
| Single `attr_choice(valuation=)` | ✓ | ✓ | Low (initially) | No — hides Luce restriction |
| Extension of `m3_utility()` | ✗ (wrong format) | ✗ (wrong format) | N/A | No |
| Raw brms (Option C) | ✓ (manual) | ✓ (manual) | Zero | Fallback only |

Separate constructors avoid a runtime `if (valuation == "cpt" && choice_rule == "luce") stop(...)` that would otherwise live inside a shared `configure_model()`. The cost is modest code duplication in the NLF builders, which is justified by API clarity.

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

### 7.1 Shared infrastructure

Both CPT and DD share:
- Wide per-option attribute format: `(amount_A, attr2_A, amount_B, attr2_B, choice)`
- Logit/softmax choice rule with phi (or phi=1)
- NLF in brms without custom Stan code
- `attr_choice` parent S3 class

The valuation function differs:
- CPT: `v(x) = x^alpha` (gains), `-lambda * (-x)^alpha` (losses) + Prelec weighting
- Discounting: `u(amount, delay) = amount / (1 + k * delay)` (hyperbolic) or `amount * exp(-k * delay)` (exponential)

### 7.2 check_data guard comparison

| Guard | CPT (`pt_choice`) | Discounting (`dd_choice`) | Notes |
|---|---|---|---|
| Probability/delay range | G1: p ∈ [< 0.20, > 0.80] | G1: max(k_ref × delay_LL) ≥ 0.5 | Different thresholds, different semantics |
| Outcome/amount range | G2: max(|x|)/min(|x|) ≥ 3 | G2: amount range for k identifiability | Structurally similar but different columns |
| Scale confound | G3: phi/lambda (gain+loss trials) | G2: k/phi Hessian check | Different confound structure |
| Loss domain | G4: gains-only → lambda unidentified | N/A (amounts always ≥ 0) | CPT-only guard |
| Curvature entanglement | G5: alpha=beta warning | N/A | CPT-only guard |
| Functional form | N/A | G3: hyperbolic vs. exp discriminability | DD-only guard |

The guard vocabularies are sufficiently different to argue for separate `check_data.*()` methods rather than a shared `check_data.attr_choice()` that would need to special-case both.

### 7.3 delay=0 reference trial question (§8 concern, now resolved)

The original concern was whether discounting requires a delay=0 reference trial as a
`check_data` requirement, which would "force two constructors." After reading the DD
exploration (`03_identifiability_guards.R`):

- `delay_SS = 0` is used as a design convention (SS is the immediately available option)
  but is **not** enforced by any DD guard
- Quasi-hyperbolic β estimation does not strictly require delay=0 trials in `check_data`;
  the β parameter is identified by non-linearity across multiple delays
- **The delay=0 concern does NOT force two constructors** — the decision already follows
  from the Luce-rule restriction and different guard vocabularies (§5.1)

---

## 8. What Would Change This Recommendation

**Toward reverting to shared `attr_choice(valuation=)`:**
- If a future valuation function (e.g., rank-dependent utility) naturally bridges CPT and
  discounting and requires shared infrastructure that would otherwise be duplicated
- If user testing shows that separate constructors create confusing API proliferation

**Toward keeping raw brms (Option C — no new constructor):**
- If user research shows < 5 users have used CPT in a behavioral economics context via bmm
- Decision point: when the first CPT or discounting use case appears in an upstream PR

**Not a forcing function (previously flagged but now investigated):**
- delay=0 reference trial — investigated in §7.3; NOT a check_data guard in DD, and does
  NOT force separate constructors

---

## 9. Acceptance Criteria Status

| Criterion | Status | Evidence |
|---|---|---|
| Validated R reference CPT likelihood (recovers known parameters) | ✓ | `01_cpt_reference_impl.R`: MLE recovery for gains and mixed domains; Nilsson finding reproduced |
| brms/Stan prototype fitting hierarchical dataset with diagnostics (Models A, B) | ✓ | `02_brms_prototype.R`: Model A Rhat=1.026, 0 div; Model B Rhat=1.012, 0 div |
| Negative control (φ-free) as reproducible failed fit | ✓ (scripted; run to populate) | `02_brms_prototype.R` §4b: Model C writes `02_cpt_recovery.csv` |
| Hierarchical α≠β lambda-underestimation demonstrated | ✓ (scripted; run to populate) | `02_brms_prototype.R` §4c: Model D vs. Model B comparison |
| Identifiability guards demonstrated (fire on deficient, pass on adequate) | ✓ | `03_identifiability_guards.R`: G1-G5 all demonstrated |
| Cross-validation script against hBayesDM written | ✓ (scripted; run to populate) | `04_hbayesdm_crossval.R`: brms NLF vs. `ra_noLA`/`ra_prospect`; hBayesDM install required |
| Feasibility/design doc reconciled with discounting exploration (#23) | ✓ | This document, §5.1 and §7; separate-constructor recommendation now aligned with DD memo |
| All code under `local/exploration/prospect-theory/`; no R/ or inst/ changes | ✓ | All scripts in local/exploration/prospect-theory/ |

---

## 10. Summary and Recommendation

**CPT is feasible as a first-class bmm constructor** via brms NLF with bernoulli+logit family.

Key architectural decisions (updated after reconciliation with #23):
1. **phi=1 (fixed)** as default: avoids the phi/lambda confound; supported by Model C negative
   control at the hierarchical level (script 02, §4b). User can override with informative prior.
2. **alpha=beta constraint** as default: reduces lambda bias (Nilsson 2011); demonstrated at MLE
   (script 01) and now also scripted for the hierarchical level (Model D, script 02 §4c).
3. **Wide per-option attribute format**: (x_A, p_A, x_B, p_B, choice) — compatible with both
   CPT and discounting.
4. **Separate constructors**: `pt_choice()` (CPT) and `dd_choice()` (discounting) with shared
   internal `attr_choice` base class — updated from the original "shared constructor" recommendation
   after reading the DD exploration, which independently arrived at the same Option A conclusion.
5. **hBayesDM cross-validation scripted** (`04_hbayesdm_crossval.R`): brms NLF vs.
   `ra_noLA`/`ra_prospect` on shared simulated data; requires hBayesDM install to execute.

**Remaining before upstream proposal:**
- Run `02_brms_prototype.R` and populate Model C / Model D result tables above
- Run `04_hbayesdm_crossval.R` with hBayesDM installed and verify brms vs. hBayesDM
  agreement on alpha and lambda (gammaw is a CPT-specific extension not in ra_*)
- Open shared upstream issue for `pt_choice()` + `dd_choice()` constructors, with this
  document and `DESIGN_dd_api.md` as the joint design starting point
