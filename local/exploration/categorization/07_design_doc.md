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

**Phase 1 result (superseded)**: N=6, n=15/stim, T=1080, prior
`mu_log_c ~ N(0.5, 0.5)`. Sampling time: 924.7 s. `mean_c` true=0.800
fell outside 90% CI [0.884, 2.028] due to the prior pulling toward
exp(0.5)≈1.65. Problem shrank from N=20→15→6 due to runtime.

**Phase 2a (pending re-run)**: multinomial-aggregated likelihood collapses
N×S×n Stan evaluations to N×S. For N=20, S=12, n=90: 21,600→240
evaluations (≈90× faster per MCMC draw). Prior changed to
`mu_log_c ~ N(0, 1)`. N restored to 20. Expected results pending.

Key implementation decisions confirmed:
- Non-centred parameterisation is stable with adapt_delta=0.90
- Divergences: 0 confirmed in Phase 1
- Per-(subj, stimulus) multinomial aggregation is exact (no approximation)
- Stimulus-indexed D_stim passes 12 matrices, not T matrices

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
| all | `c` | log | Normal(0, 1) |
| exemplar | `gamma` | log | Normal(0, 0.5) |
| prototype/prm | `gamma` | fixed = 1 | — |
| all (M=2) | `w1` | logit (softmax ref) | Normal(0, 1) |
| all (M>2) | `w1..w_{M-1}` | softmax | Normal(0, 1) each |
| prm | `p_mem` | logit | Normal(0, 1) |

**Note on `c` prior**: an earlier draft recommended `Normal(0.5, 0.5)` on
`log(c)`, which pulls the posterior toward `exp(0.5)≈1.65`. Under the
hierarchical Phase 1 PoC (true `mean_c=0.8`) this placed the true value
outside the 90% CI. `Normal(0, 1)` is weakly informative across c∈[0.1, 8]
and should be the default. `Normal(0.5, 0.5)` is appropriate only when the
coordinate space is known to be scaled such that c≈1–2 is expected a priori.

### Stan implementation

Pre-compute `D_stim[S, J, M]` in R (`configure_model`), pass as
`array[S] matrix[J, M] D_stim` via `stanvars` where S is the number of
**unique stimuli** (not total trials). Aggregate trial responses to a
`[S, K]` (single-subject) or `[N, S, K]` (hierarchical) integer count
array and pass as `y_counts`. Stan evaluates `multinomial_lpmf` once per
(stimulus) or per (subject, stimulus) pair.

**Why aggregation matters**: the GCM probability for a given stimulus
depends only on the stimulus coordinates and the model parameters — not on
trial position. Summing `n` categorical log-likelihoods with the same
probability vector is identical to one `multinomial_lpmf` on the count
vector. For a nosof88-scale design with n=25 trials/stim, this collapses
300 categorical evaluations to 12 multinomial evaluations (25× faster per
additional replicate). For the hierarchical case (N subjects) it collapses
N×S×n evaluations to N×S.

Cost estimate: ~60–70 draws/s for the nosof88 prototype (S=12, J=12, M=2,
n=25/stim). The rocks scale (S≈150, J=90, M=8) requires benchmarking;
the O(S·J·M) per-draw cost replaces the previous O(T·J·M), where
T=S×n. Benchmark vs m3 is Phase 2b.

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
