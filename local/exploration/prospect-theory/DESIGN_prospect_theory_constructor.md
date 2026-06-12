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
| `amt_A` | double | Outcome of option A (negative = loss) |
| `prob_A` | double | Probability of amt_A (strictly between 0 and 1) |
| `amt_B` | double | Outcome of option B |
| `prob_B` | double | Probability of amt_B |
| `choice` | int | 1 = chose A, 0 = chose B |

Column names follow the canonical attribute interface established in #27, which also
governs the delay-discounting constructor (#23 / PR #26). The `amt_` prefix is shared
across both constructors; `prob_` is CPT-specific (DD uses `delay_`).

**Two-outcome extension** (v1.1, future):

```
amt_A1, prob_A1, amt_A2  (prob_A2 = 1 - prob_A1; amt_A2 can be different sign than amt_A1)
amt_B1, prob_B1, amt_B2
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

Five hierarchical models + an SBC-light check (N=30 subjects × 40–80 trials,
2 chains × 1000 iter). **Numbers below are measured from a reproducible run
(2026-06-12); all `brm()` calls now fix `seed` (42–48), so these are exact, not
representative.** Model B additionally enforces a QC gate (`stop()` if
`Rhat > 1.05` or any divergence) so a bad draw can never ship silently.

**Model A — Gains-only (alpha, gammaw, phi):**
- max Rhat = 1.023, divergences = 0
- alpha: true=0.80, est=0.911 [0.717, 1.117] COVERED
- gammaw: true=0.65, est=0.591 [0.415, 0.813] COVERED
- phi: true=1.20, est=0.957 [0.519, 1.648] COVERED

**Model B — Gains+losses, phi=1 fixed, alpha=beta (alpha, lambda, gammaw) — recommended default:**
- max Rhat = 1.019, divergences = 0  (passes QC gate; seed 2024L, adapt_delta 0.99)
- alpha: true=0.80, est=0.787 [0.741, 0.830] COVERED
- lambda: true=2.25, est=2.345 [2.006, 2.762] COVERED  (bias +0.095)
- gammaw: true=0.65, est=0.622 [0.536, 0.726] COVERED

**Model C — phi free, gains+losses, alpha=beta (alpha, lambda, gammaw, phi):**
- max Rhat = 1.026, divergences = 0
- alpha: true=0.80, est=0.751 [0.634, 0.877] COVERED
- lambda: true=2.25, est=2.291 [1.933, 2.777] COVERED
- gammaw: true=0.65, est=0.587 [0.464, 0.729] COVERED
- phi: true=1.20, est=1.177 [0.777, 1.693] **COVERED**

> **Model C is NOT the negative control — it converges.** It was once expected to
> fail (Rhat ≫ 1.05, phi/lambda anti-correlated); instead it recovers **all four**
> parameters, phi included (1.177 vs. true 1.20), cleanly. In a **balanced gain+loss
> design phi is identifiable**, so fixing phi=1 is a convenience/robustness choice,
> not an identification requirement — consistent with §5: gain trials identify phi,
> loss trials identify phi·lambda, so the combination pins both. The genuine
> confound requires a **loss-only** design — that is Model E below.

**Model D — Hierarchical Nilsson: phi=1 fixed, alpha≠bta free (alpha, bta, lambda, gammaw):**
- max Rhat = 1.016, divergences = 0
- alpha: true=0.80, est=0.797 [0.742, 0.851] COVERED
- bta:   true=0.80, est=0.768 [0.661, 0.883] COVERED
- lambda: true=2.25, est=2.514 [1.776, 3.435] COVERED  (bias +0.264)
- gammaw: true=0.65, est=0.609 [0.486, 0.755] COVERED

**Model E — LOSS-ONLY negative control: phi free, losses only (alpha, lambda, gammaw, phi):**
- max Rhat = **2.236**, divergences = **500** — fails as designed
- gammaw blows up to 3.815 [0.443, 7.086] (true 0.65); lambda/phi posteriors wide
- This is the **genuine phi/lambda confound**: with no gain trials to anchor phi,
  only the product phi·lambda is identified (`U_A−U_B = −phi·lambda·(…)`). Model E
  (loss-only) failing while Model C (gains+losses) converges is the real evidence
  for the phi=1 default — the gain-trial anchor is what resolves the confound.

**Nilsson hierarchical comparison + SBC-light (measured):**
- Model B (phi=1, alpha=bta):  lambda bias = **+0.095**, 95% CI width 0.76
- Model D (phi=1, alpha≠bta):  lambda bias = **+0.264**, 95% CI width 1.66 (≈2.2× wider)
- SBC-light (K=30 single-subject MLE datasets, N=60): lambda RMSE **5.18** (alpha=bta)
  vs **8.17** (alpha≠bta) — ratio **1.58×** in favour of the constraint.

> **Honest read: WEAKLY SUPPORTED, not "demonstrated."** Both the single-dataset
> hierarchical comparison (D degrades lambda vs B) and the K=30 SBC-light (1.58×
> lower RMSE under alpha=bta) point the same way — the constraint buys precision.
> But the *absolute* SBC recovery is poor (lambda RMSE 5.18 against a true value of
> 2.25; both constrained and unconstrained are badly biased at N=60 MLE), and the
> hierarchical bias is *positive* whereas the script-01 MLE finding is *negative*
> (−1.39). The alpha=bta default is reasonable on precision grounds; a proper
> **hierarchical SBC** (not single-subject MLE) is still required to call it
> "demonstrated." The script's computed verdict string now reports exactly this.

> **Note on the script's printed summary (now fixed).** The prior version hardcoded
> "phi/lambda confound: CONFIRMED (Model C)" and "alpha=beta benefit: DEMONSTRATED
> (B vs D)" as static strings. Both were contradicted/overstated by the 2026-06-12
> run. The summary now computes verdicts from Rhat, divergences, coverage, and bias
> (§3.2.1). The genuine negative control (Model E, loss-only) is also added.

#### 3.2.1 Status of follow-ups from the 2026-06-12 run
- **Loss-only negative control** (Model E, §4d in script 02): **done & measured**.
  Fails as designed — Rhat 2.236, 500 divergences (`results/02_modelE_lossonly.csv`).
  This is the genuine phi/lambda confound; it resolves the open question of whether
  phi=1 has empirical (not just algebraic) backing.
- **Multi-dataset recovery / SBC for the alpha=bta benefit**: **done & measured** as
  SBC-light (§4e), K=30 MLE datasets — lambda RMSE 5.18 (alpha=bta) vs 8.17 (alpha≠bta),
  ratio 1.58× → WEAKLY SUPPORTED (`results/02_sbc_alpha_beta.csv`). A full hierarchical
  SBC remains the bar for "demonstrated."
- **Stan seed**: **fixed** — all `brm()` calls now have explicit `seed =` (42–48).
- **Diagnostic-driven summary**: **done** — §7 now computes verdicts from Rhat,
  divergences, coverage, and bias (not hardcoded strings).
- `results/02_gainsonly_recovery.csv` orphan: **removed** (`git rm`).

**NLF formulation in brms:**
```r
bf(
  choice ~ wA * vA - wB * vB,   # phi=1 absorbed into utility scale
  nlf(vA ~ is_Ag * xA_g^alpha - is_Al * lambda * xA_l^alpha),
  nlf(vB ~ is_Bg * xB_g^alpha - is_Bl * lambda * xB_l^alpha),
  nlf(wA ~ exp(-((-log(prob_A))^gammaw))),
  nlf(wB ~ exp(-((-log(prob_B))^gammaw))),
  alpha  ~ 1 + (1 | subj),
  lambda ~ 1 + (1 | subj),
  gammaw ~ 1 + (1 | subj),
  nl = TRUE
)
```

User-facing columns: `prob_A`, `prob_B` (probability weighting), `amt_A`, `amt_B` (outcomes).
Pre-computed indicator columns (`xA_g = max(amt_A, 0)`, `xA_l = max(-amt_A, 0)`, `is_Ag`, `is_Al`)
are internal: derived before fitting to avoid `pow(negative, alpha)` in Stan.
The logit rule handles negative utilities without modification.

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

The `options = list(A = c(amt=, prob=), B = ...)` API matches the canonical attribute
interface defined in issue #27 and implemented in the delay-discounting PR (#26).
Both constructors share the `amt_` key; CPT uses `prob_`, DD uses `delay_`.

```r
# User-facing constructor for CPT (Stage 2 of m3-utility roadmap)
# Canonical interface: options = list(<label> = c(amt = <col>, prob = <col>), ...)
pt_choice(
  resp      = "choice",
  options   = list(
    A = c(amt = "amt_A", prob = "prob_A"),
    B = c(amt = "amt_B", prob = "prob_B")
  ),
  weighting = c("prelec1", "prelec2", "tk1992", "none"),
  phi_fixed = TRUE                  # fix phi=1 (recommended) or estimate
)

# User-facing constructor for delay discounting (issue #23 / PR #26)
# Shares the same options = list() API with different key names
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
- Wide per-option attribute format: `(amt_A, <attr2>_A, amt_B, <attr2>_B, choice)`
  — `amt_` prefix is canonical per issue #27; `prob_` (CPT) and `delay_` (DD) are the second attribute
- `options = list(A = c(amt=, prob=/delay=), B = ...)` constructor API (issue #27 contract)
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
| brms/Stan prototype fitting hierarchical dataset with diagnostics (Models A, B) | ✓ | `02_brms_prototype.R` (run 2026-06-12): Model A Rhat=1.020, 0 div; Model B Rhat=1.013, 0 div, all covered |
| Negative control (φ-free, loss-only) demonstrating genuine φ/λ confound | ✓ | Model E (§4d): loss-only design with phi free; verdict computed from diagnostics |
| Hierarchical α≠β lambda-underestimation demonstrated | ◐ + SBC-light | Model D vs. B (§4c); SBC-light K=30 MLE datasets (§4e) provides multi-dataset evidence |
| Identifiability guards demonstrated (fire on deficient, pass on adequate) | ✓ | `03_identifiability_guards.R`: G1-G5 all demonstrated |
| Cross-validation script against hBayesDM written (brms sections reproducible) | ✓ (scripted) | `04_hbayesdm_crossval.R`: brms NLF fits sections 3+6 run without hBayesDM; ra_* comparison requires install |
| Canonical attribute interface (`amt_A/prob_A`) matching #27 / #26 Option A | ✓ | Column names updated across all scripts and memo; constructor sketch uses `options = list()` API |
| Feasibility/design doc reconciled with discounting exploration (#23) | ✓ | This document, §5.1 and §7; separate-constructor recommendation now aligned with DD memo |
| All code under `local/exploration/prospect-theory/`; no R/ or inst/ changes | ✓ | All scripts in local/exploration/prospect-theory/ |

---

## 10. Summary and Recommendation

**CPT is feasible as a first-class bmm constructor** via brms NLF with bernoulli+logit family.

Key architectural decisions (updated after reconciliation with #23 and the 2026-06-12 run):
1. **phi=1 (fixed)** as a *robustness/convenience* default — **not** an identification
   requirement in a balanced gain+loss design. The run showed phi is recoverable when free
   (Model C: phi est=1.177 vs true=1.20, clean diagnostics). The confound bites only in a
   loss-only design — now demonstrated (Model E: Rhat 2.236, 500 div, §3.2.1). User can free phi.
2. **alpha=beta constraint** as default: reduces lambda bias and tightens its interval at MLE
   (script 01, clear) and weakly at the hierarchical level (Model D vs B: bias +0.264 vs +0.095;
   SBC-light K=30: lambda RMSE 1.58× lower under alpha=bta, §3.2.1). Reasonable on precision
   grounds; full hierarchical SBC still required for "demonstrated."
3. **Wide per-option attribute format**: (`amt_A`, `prob_A`, `amt_B`, `prob_B`, `choice`) —
   canonical column names per issue #27; constructor uses `options = list(A = c(amt=, prob=), B = ...)`.
   Compatible with the DD constructor's `options = list(LL = c(amt=, delay=), ...)` API.
4. **Separate constructors**: `pt_choice()` (CPT) and `dd_choice()` (discounting) with shared
   internal `attr_choice` base class — updated from the original "shared constructor" recommendation
   after reading the DD exploration, which independently arrived at the same Option A conclusion.
5. **hBayesDM cross-validation scripted** (`04_hbayesdm_crossval.R`): brms NLF vs.
   `ra_noLA`/`ra_prospect` on shared simulated data; requires hBayesDM install to execute.

**Remaining before upstream proposal:**
- ~~Run `02_brms_prototype.R` and populate Model C / Model D tables~~ **done 2026-06-12**
- ~~Add a loss-only negative control (Model E) and SBC/multi-dataset recovery (§4e)~~ **done 2026-06-12**
- ~~Rename column interface to `amt_A/prob_A` and update constructor sketch~~ **done 2026-06-12**
- ~~Fix Stan seed and diagnostic-driven summary~~ **done 2026-06-12**
- **Run `04_hbayesdm_crossval.R`** with hBayesDM installed and verify brms vs. hBayesDM
  agreement on alpha and lambda. Without hBayesDM, sections 3+6 (brms fits) still run and
  produce `04_brms_xval_recovery.csv`.
- **Open shared upstream issue** for `pt_choice()` + `dd_choice()` constructors, pointing
  at this document, `DESIGN_dd_api.md`, and issue #27 as the joint design contract.
