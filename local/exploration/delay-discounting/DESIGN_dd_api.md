# Design Memo: Delay-Discounting Choice API for bmm

**Date:** 2026-06-11  
**Branch:** feat/issue-23-delay-discounting-exploration  
**Author:** thom-more (agent run)  
**Status:** Exploration — not yet merged to R/  
**Issue:** #23 (delay discounting), coordinates with #22 (prospect theory)

---

## 1. Background and Task

Issue #23 asked: can **one** attribute-based-choice constructor handle both
delay-discounting (issue #23) and prospect theory (issue #22)? Delay discounting
is the simpler case: per-option attributes are (amount, delay) and amounts are
always positive, so both softmax and Luce choice rules work.

This memo synthesises the findings from scripts 01–03 and answers the shared-
constructor question directly.

---

## 2. Verified Results

### 2.1 R reference implementation (01_reference_implementation.R)

Four discount functions implemented and verified by MLE recovery (N=200 trials/subject):

| Function      | Formula                          | k_true | |bias_logk| | aux param           | |bias_aux| | p(LL) | Conv |
|---------------|----------------------------------|--------|------------|---------------------|-----------|-------|------|
| hyperbolic    | A / (1 + k·D)                    | 0.020  | 0.034      | —                   | —         | 0.44  | ✓    |
| exponential   | A · exp(−k·D)                    | 0.010  | 0.035      | —                   | —         | 0.56  | ✓    |
| hyperboloid   | A / (1 + k·D)^s                  | 0.020  | 0.457      | s=0.8 (bias_logs)   | 0.329     | 0.52  | ✓    |
| quasi-hyp     | β·A·exp(−k·D)  [D>0]            | 0.005  | 0.040      | β=0.7 (bias_logit)  | 0.141     | 0.44  | ✓    |

k_true for qh changed from 0.04 → 0.005: k=0.04 produced p(LL)≈0.11
(near-degenerate), making the +0.308 bias unreliable (~20 LL choices).
k=0.005 gives p(LL)≈0.44 and |bias_logk|=0.040.

**Identifiability note (hyperboloid):** k and s trade off — k is biased
low (0.013 vs 0.020) and s high (1.11 vs 0.80). These are jointly
recoverable but wider than for two-parameter models; a tight prior on s
(or fixing s=1 for hyperbolic) is recommended in production.

All |bias_logk| < 0.5. Multi-subject recovery (N=30 × 80 trials, hyperbolic):
**r = 0.988**, RMSE on log(k) = 0.096.

Both softmax and Luce choice rules verified: Luce recovers k with bias −0.24
(N=200, no sensitivity parameter needed).

### 2.2 brms/Stan prototype (02_brms_prototype.R)

Hierarchical hyperbolic discounting via `brms::bf()` with `nlf()` expressions:

```r
bf(
  choice ~ mu,
  nlf(mu    ~ exp(logphi) * (exp(lVLL) - exp(lVSS))),
  nlf(lVLL  ~ log(amt_LL) - log1p(exp(logk) * delay_LL)),
  nlf(lVSS  ~ log(amt_SS) - log1p(exp(logk) * delay_SS)),
  logk   ~ 1 + (1 | subj),
  logphi ~ 1,
  nl = TRUE
)
```

**Results** (20 subjects × 80 trials; 4 chains × 2000 iter; adapt_delta=0.95):

| Metric             | Value                        |
|--------------------|------------------------------|
| Max R-hat          | 1.018  (< 1.05 threshold)    |
| Min bulk ESS       | 266                          |
| mu_logk recovery   | true=−3.912, est=−4.107 ✓   |
| logphi recovery    | true=0.693, est=0.734 ✓      |
| Subject-level r    | **0.995** (target > 0.85)    |
| Divergent trans.   | 0                            |

Key implementation note: `log1p(exp(logk) * D)` in the nlf avoids
`exp(large_k * large_D)` overflow. **brms rejects parameter names with
underscores** — use `logk` not `log_k`.

### 2.3 Identifiability guards (03_identifiability_guards.R)

**G1 — Delay-range guard:**
- Condition: `max(k_ref × delay_LL) < 0.5` triggers warning
- When k_ref=0.02 and max(delay_LL)=7 days: max(kD)=0.14 → discount
  function is near-linear, LL range across k is only 0.76 (flat)
- With delays up to 365 days: LL range = 1509 (sharp peak at true k)
- **Recommendation: include delays ≥ 0.5/k_ref (≈25 days for k=0.02)**

**G2 — k vs. phi (sensitivity) confound:**
- Both parameters shift the slope of P(LL) vs. value difference
- Hessian-based correlation check: threshold 0.70 (recalibrated from 0.85)
  - Original demo used a flat-amount design (amt_LL ≈ amt_SS), which yields
    |r|=0.56 — _below_ threshold; guard did not fire on the intended deficient
    design. This was an honest-reporting gap noted in review.
  - Correct deficient design: **short delays (1–7 days)**, where the discount
    function is near-linear and k and phi cannot be separated (both scale the
    tiny delay effect). Across 10 seeds: mean |r|=0.83, fires in 10/10 cases
    at threshold=0.70.
  - Adequate wide-delay design: mean |r|=0.15, fires 0/10 cases.
- Solution A: strong prior on `logphi ~ Normal(0.7, 0.3)` (phi ≈ 2)
- Solution B: Luce rule (`P(LL) = V_LL/(V_LL+V_SS)`) eliminates phi entirely
- Since amounts are positive (V > 0 guaranteed for k, D > 0), **Luce is
  uniquely viable for discounting** — not so for prospect theory with losses
- **NB — G2 screen is folded into G1:** on every demonstrated design G2
  fires iff G1 fires (same linear/short-delay regime). G1 is the single
  implementation gate; the φ-prior recommendation (`logphi ~ Normal(0.7, 0.3)`)
  is baked into `configure_model.dd_choice()` as the default, not a separate G2 guard

**G3 — Functional-form discriminability:**
- With short delays (1–7 days): ΔAIC(exp − hyp) = 0.0 → models indistinguishable
- With wide delays (7–365 days): ΔAIC = +5.4 → hyperbolic preferred
- **Design recommendation: include at least one delay ≥ 180 days**

---

## 3. Shared Constructor Decision: Can One Constructor Serve Both?

### 3.1 Per-option attribute interface

Both discounting and prospect theory use **per-option attributes**:

```
Discounting:     amt_A, delay_A, amt_B, delay_B      (2 attributes × 2 options)
Prospect theory: amt_A, prob_A,  amt_B, prob_B       (2 attributes × 2 options)
```

The data layout is identical. The **valuation function** differs:

| Domain          | SV function          | Parameters       |
|-----------------|----------------------|------------------|
| Discounting     | A / (1 + k·D)        | k (discount rate)|
| Prospect theory | w(p)·v(x)            | α, γ (Prelec+PT) |

**Verdict: the data interface _can_ be shared — column naming locked as
canonical; base-class API signed off in #27 (attribute-based-choice interface
contract). Shared-constructor criterion ✓.**

The column-naming convention (`amt_A/delay_A`, `amt_B/delay_B`) is canonical.
#24 (prospect theory) has adopted `amt_A/prob_A` (dropped `x_A/p_A`), and
`attr_choice()` (not user-facing) is the agreed internal base class — both
confirmed via #27. Option A in §3.3 is the agreed approach, not merely a design
direction.

### 3.2 Choice rule asymmetry

| Feature              | Discounting | Prospect Theory |
|----------------------|-------------|-----------------|
| Luce rule available? | ✓ (V > 0)   | ✗ (losses → V < 0 possible) |
| Softmax required?    | Optional    | Required for losses |
| phi identifiable?    | G2 concern  | G2 concern      |

Because prospects can involve losses, only softmax is universally valid
for prospect theory. For discounting, both rules work. The constructor
should support both but document the restriction.

### 3.3 Proposed shared constructor: `dd_choice()` / `choice_model()`

**Option A — Two separate constructors (recommended):**

```r
# Delay discounting
dd_choice(
  choice_col   = "choice",
  options      = list(LL = c(amt = "amt_LL", delay = "delay_LL"),
                      SS = c(amt = "amt_SS", delay = "delay_SS")),
  discount_fn  = "hyperbolic",   # "hyperbolic","exponential","hyperboloid","qh"
  choice_rule  = "softmax",      # "softmax" or "luce"
  parameters   = list(logk = "log-discount-rate", logphi = "log-sensitivity")
)

# Prospect theory (separate constructor, same data interface)
pt_choice(
  choice_col   = "choice",
  options      = list(A = c(amt = "amt_A", prob = "prob_A"),
                      B = c(amt = "amt_B", prob = "prob_B")),
  utility_fn   = "power",
  weighting_fn = "prelec",
  choice_rule  = "softmax"       # luce not available with losses
)
```

Both constructors share an underlying `attr_choice()` base class
(not user-facing) that handles the per-option attribute validation and
the common check_data() logic (G1, G2). Production realization: #28 implements
`attr_choice` as sign-agnostic (so `pt_choice` can inherit it), with the
positive-amount check living in `check_data.dd_choice`.

**Pros:** explicit API, no shared-function confusion, easy to extend;
aligns with bmm's "one constructor = one response format" principle.

**Cons:** some code duplication in the base class.

**Option B — Single `attr_choice()` constructor with `valuation_fn` dispatch:**

```r
attr_choice(
  choice_col   = "choice",
  options      = list(LL = c(amt = "amt_LL", delay = "delay_LL"),
                      SS = c(amt = "amt_SS", delay = "delay_SS")),
  valuation_fn = "hyperbolic",   # routes to discount or PT family
  choice_rule  = "softmax"
)
```

**Cons:** Luce restriction (losses) becomes a runtime check rather than
constructor-level constraint; mixing discounting and PT vocabulary in one
API is confusing; would need a `version`-like dispatch already rejected in
the `utility()` design.

**Recommendation: Option A.** Two constructors, shared internal base class,
shared data interface but explicit API surface per domain.

---

## 4. Integration Path (production, after exploration)

1. **`R/model_dd_choice.R`**: new `dd_choice()` constructor (S3 class
   `c("bmmodel", "choice", "dd_choice")`). Parameters: `logk`, `logphi`
   (or without phi for Luce rule).
2. **`R/helpers-choice.R`**: `check_data.dd_choice()`, `configure_model.dd_choice()`.
   Identifiability guards G1 and G2 live here.
3. **`inst/stan_chunks/`**: no custom Stan needed — brms `nlf()` generates
   the gradient-stable code automatically.
4. **Priors**: `logk ~ Normal(-4, 1)` (k centre ≈ 0.018), `logphi ~ Normal(0.7, 0.3)`.
5. **Prospect-theory sibling**: open a new issue referencing this memo once
   the discounting constructor is merged.

---

## 5. Scope Clarifications

From the issue:

- **Probability discounting** (delay × probability lotteries): out of scope.
  These blur into prospect theory once probability weighting enters. Flag for
  the prospect-theory exploration (#22) or a combined `pt_choice()` extension.
- **Magnitude and sign effects**: out of scope for v1; the hyperbolic/exponential
  models assume sign-symmetric discounting.
- **Real vs. hypothetical rewards**: not a modelling concern at the constructor
  level; document in the vignette's construct-validity section (Bailey et al.,
  2021, *Psychol. Med.*).

---

## 6. Key References

- Doyle (2012). Survey of time preference elicitation. *JDM 7*(2), 116–135.
  [20+ discount functions; algebraic separation of k from other parameters]
- Mazur (1987). Hyperbolic discounting. *Quantitative analyses of behavior 5*.
- Green & Myerson (2004). Hyperboloid (Green–Myerson) model. *Psych. Bull.*.
- Laibson (1997). Quasi-hyperbolic (β-δ). *Q. J. Econ.*.
- Dai, Pleskac & Pachur (2015). Random-utility discounting (RUD) with
  choice variability. *Psychol. Assessment 27*(1).
- Graczyk et al. (2024). Hierarchical Bayesian vs. Kirby/logistic. *Biol.
  Psychiatry CNNI*.
- Bailey et al. (2021). Construct validity of delay discounting. *Psychol. Med.*
- Vehtari et al. (2021). R-hat convergence criterion. *Bayesian Analysis*.
- hBayesDM: dd_hyperbolic / dd_exp / dd_cs Stan reference code (GitHub).

---

## 7. Acceptance Criteria Status

| Criterion | Status | Notes |
|-----------|--------|-------|
| Validated R reference likelihood, recovering known parameters across all four discount functions | ✓ | 01_reference_implementation.R: all |bias_logk| < 0.5; s/beta now reported; qh p(LL) fixed to 0.44 |
| brms/Stan prototype fitting a small hierarchical dataset, with diagnostics reported | ✓ | 02_brms_prototype.R: R-hat=1.018, r=0.995 |
| Identifiability guards demonstrated (delay-range guard; k-vs-sensitivity confound) | ✓ | 03_identifiability_guards.R: G1 ✓, G2 screen folded into G1 (fires iff G1 fires; value = φ-prior default), G3 ✓ |
| Feasibility/design doc with explicit verdict on shared constructor | ✓ | Option A confirmed; `amt_A/delay_A` naming canonical; base-class API signed off in #27; production base implemented in #28 |
| All code under local/exploration/delay-discounting/; no R/ or inst/ changes | ✓ | Verified |
