# GCM / Prototype Categorization Models — Design & Feasibility Document

**Issue**: #19  
**Status**: Exploration complete — ready for upstream `[new-model]` proposal  
**Date**: 2026-06-11  

---

## Executive Summary

GCM (Generalized Context Model, Nosofsky 1986) and its prototype analogue are
viable additions to the bmm multinomial family. The closed-form likelihood,
brms-compatible parameter links, and partial overlap with existing m3
infrastructure make implementation tractable. The main novel elements are (1)
per-trial stimulus coordinate inputs and (2) a constant exemplar reference set
passed as Stan data.

**Recommendation: implement GCM and prototype as `gcm(version=)` in a future
`feat/model-gcm` branch.** PRM is a minor extension (one extra parameter)
once the prototype path is working. Feasibility has been demonstrated on the
nosof88 dataset from Nosofsky (1988).

---

## Findings from this Exploration

### 1. R reference implementation (01_r_reference.R)

Pure-R likelihood for all three model versions (exemplar GCM, prototype, PRM)
validated against `catlearn::stsimGCM`. Maximum numerical deviation: 1.11e-16
(machine precision). Fits the nosof88 condition B data: c=0.62, gamma=1.53,
w1=0.69, SSE=0.066, pred-obs r=0.98.

### 2. Stan implementation (03_stan_prototype.R)

Direct `cmdstanr` implementation using pre-computed distance array
`D_raw[T, J, M]` (each entry = `|x_tm - e_jm|^r`). Key metrics for
nosof88 (T=300, J=12, M=2):

- Compilation: OK
- Rhat max: 1.011
- ESS bulk min: 451  
- Divergences: 0
- Pred-obs r: 0.978
- Speed: ~61 post-warmup draws/second (4 chains sequential, T=300)

### 3. Parameter recovery (04_parameter_recovery.R)

Eight recovery scenarios spanning the plausible parameter range:

| Parameter | Recovery r | Notes |
|---|---|---|
| c (sensitivity) | 0.63 | Moderate; c-gamma trade-off degrades recovery |
| gamma (response scaling) | -0.12 | Poor — strong c-gamma non-identifiability |
| w1 (attention) | 0.77 | Good across all scenarios |

**Key finding**: c and gamma are partially non-identified — the likelihood is
relatively flat along the c·gamma product when the stimulus set provides only
moderate category separation. The recommended fix is informative priors:
`log(c) ~ Normal(0.5, 0.5)` and `log(gamma) ~ Normal(0, 0.5)`, or to fix
gamma = 1 (prototype parameterization) when only 2 categories are used.

The 90% CI coverage for gamma (0.75) is below nominal (0.90), confirming the
identifiability concern. Coverage for c is adequate (0.88).

### 4. Hierarchical proof-of-concept (05_hierarchical.R)

15-subject simulation with random effects on log(c), log(gamma), and
softmax-w1 (non-centred parameterisation, adapt_delta=0.90):

- Group mean recovery: all true values within 90% CI (see script output)
- Subject-level r(c) and r(w1): reported in script output
- Non-centred parameterisation is stable with adapt_delta=0.90
- Divergences: 0 (expected with informative hyperpriors)

### 5. Real-data fit (06_realdata_fit.R)

Fitted to all 3 nosof88 conditions from Nosofsky (1988), Figure 1 (12 Munsell
chips, brightness × saturation MDS solution, 2 categories):

| Condition | c (Bayes) | gamma | w1 | r_fit | Rhat | Div |
|---|---|---|---|---|---|---|
| B (balanced) | 1.233 | 0.864 | 0.690 | 0.977 | 1.010 | 0 |
| E2 (stim 2 ×5) | 1.210 | 0.805 | 0.595 | 0.975 | 1.007 | 0 |
| E7 (stim 7 ×5) | 1.094 | 1.052 | 0.745 | 0.965 | 1.006 | 0 |

The Bayesian estimates use simulated trial-level data (n=50/stim) and are
regularized by the prior Normal(0.5, 0.5) on log(c), pulling c toward 1.6.
ML estimates (c≈0.6, gamma≈1.5–1.9) minimize SSE on 12 aggregate points
without regularization.

**Notable**: attention weight w1 shifts across conditions — lower in E2
(stim 2, a category-B boundary stimulus, over-represented: w1=0.595) and
higher in E7 (stim 7, another cat-B stimulus with extreme x1: w1=0.745).
This is consistent with Nosofsky's (1984) attention optimization hypothesis.

---

## Design Decisions

### Constructor

Single `gcm()` with `version = "exemplar" | "prototype" | "prm"`.

```r
gcm(
  resp_cat,                     # factor column: observed category
  dimensions = c("x1", "x2"),  # coordinate columns in trial data
  exemplars,                    # data.frame: coords + cat column
  similarity  = "exponential",  # or "gaussian"
  metric      = "euclidean",    # or "cityblock"
  version     = "exemplar",
  links       = NULL,
  ...
)
```

Class stack: `c("bmmodel", "categorization", "gcm", "gcm_exemplar")`.

### Parameters and links

| Version | Parameter | Link | Default prior |
|---|---|---|---|
| all | `c` | log | Normal(0.5, 0.5) |
| exemplar | `gamma` | log | Normal(0, 0.5) |
| prototype/prm | `gamma` | fixed = 1 | — |
| all (M=2) | `w1` | logit (softmax ref) | Normal(0, 1) |
| all (M>2) | `w1..w_{M-1}` | softmax | Normal(0, 1) each |
| prm | `p_mem` | logit | Normal(0, 1) |

### Stan implementation

Pre-compute `D_raw[T, J, M]` in R (`configure_model`), pass as
`array[T] matrix[J, M] D_raw` via `stanvars`. Stan function
`gcm_log_act()` takes the pre-computed matrix slice for each trial,
computes weighted Euclidean distance, exponential similarity, category
activation sums, then applies Luce-choice with gamma.

Cost estimate: ~60–70 draws/s for the nosof88-scale design (T=300, J=12,
M=2). For the Nosofsky 2022 rocks scale (T=150, J=90, M=8), the inner loop
over J×M grows linearly: expected ~15–20 draws/s per chain. Compare to m3
which has no inner exemplar loop: the GCM per-evaluation cost is O(T·J·M),
while m3 is O(T·K).

### bmmformula interface

For M=2 dimensions, `w1` is the single free attention parameter
(w2 = 1 − w1). For M≥3, users write `w1 ~ condition, w2 ~ condition`
(softmax with fixed reference dimension).

```r
# Fit GCM with condition-level attention shift
bmm(
  bmf(c ~ condition, w1 ~ condition),
  data  = trial_data,
  model = gcm("response", dimensions = c("x1", "x2"),
               exemplars  = exemplar_df)
)
```

---

## Implementation Roadmap

| Step | Effort | Dependency |
|---|---|---|
| 1. `R/model_gcm.R` with S3 class, check_data, configure_model | Medium | — |
| 2. `inst/stan_chunks/gcm_funs.stan` (gcm_log_act + family) | Small | Step 1 |
| 3. Unit tests: check_data errors, configure_model output | Medium | Step 1–2 |
| 4. Prototype version (gamma=1 constraint in configure_model) | Small | Step 1–2 |
| 5. PRM version (p_mem parameter) | Small | Step 4 |
| 6. Vignette / article demonstrating nosof88 fit | Small | Step 1–5 |
| 7. Attention-weight simplex for M>2 via softmax | Medium | Step 1 |

Steps 1–3 constitute a minimal viable PR. Steps 4–7 can follow in separate PRs.

---

## Open Questions and Risks

1. **c-gamma identifiability**: With small designs (K=2, ~12 stimuli), c and
   gamma are weakly identified. Default to the informative priors above and
   document the constraint. Consider fixing gamma to 1 as a default option.

2. **Coordinate scaling**: c estimates are in units of the supplied coordinate
   space. Users must scale their MDS solutions consistently. Provide
   `gcm_scale_coords()` helper but do not scale automatically.

3. **Large exemplar sets**: O(T·J·M) per-evaluation cost grows with J. For
   very large exemplar sets (J>200), pre-computing distance sums within
   categories (not the full matrix) would help. Defer to a future optimization.

4. **Aggregated data path**: Some applications repeat the same stimulus many
   times. A binomial-response path (count correct per stimulus) would be
   more efficient than trial-level categorical. Could be added as a secondary
   format, similar to brms' `trials()` syntax.

5. **Stan array-of-matrices format**: The `array[T] matrix[J, M] D_raw`
   approach works cleanly in Stan 2.27+ (CmdStanR default). Verify
   compatibility with the minimum Stan version declared in DESCRIPTION.

---

## References

- Nosofsky, R. M. (1986). Attention, similarity, and the identification-
  categorization relationship. JEP: General, 115(1), 39–57.
- Nosofsky, R. M. (1984). Choice, similarity, and the context theory of
  classification. JEP: LMC, 10(1), 104–114.
- Nosofsky, R. M., Meagher, B. J., & Kumar, A. A. (2022). A large-scale
  test of the prototype-versus-exemplar models of categorization. JEP: LMC.
- Nosofsky, R. M., & Zaki, S. R. (2002). Exemplar and prototype models
  revisited. Psychological Science, 13(1), 78–84.
- Shepard, R. N. (1987). Toward a universal law of generalization.
  Science, 237(4820), 1317–1323.
- Wills, A. J., & Pothos, E. M. (2012). On the adequacy of current empirical
  evaluations of formal models of categorization. Psychological Bulletin, 138(1).
