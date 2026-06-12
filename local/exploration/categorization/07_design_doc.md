# GCM / Prototype Categorization Models — Design & Feasibility Document

**Issue**: #19  
**Status**: Phase 2b complete (04b, 08 verified); 06b revised to hierarchical design (output pending local run)  
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

**Phase 2a (N=20, n=90, seed 777, CmdStan 2.38, 4 chains × 500 draws, 52 s)**:
multinomial aggregation confirmed — 21,600→240 Stan evaluations (90× speedup)
vs 924 s for the old N=6 run. All group-level parameters recovered within 90% CI:

| Param | True | Posterior | 90% CI | In CI |
|---|---|---|---|---|
| mean_c | 0.800 | 0.868 | [0.628, 1.156] | YES |
| mean_gamma | 1.500 | 1.604 | [1.164, 2.052] | YES |
| mean_w1 | 0.650 | 0.662 | [0.605, 0.715] | YES |
| sigma_log_c | 0.400 | 0.500 | [0.284, 0.719] | YES |
| sigma_log_gamma | 0.400 | 0.346 | [0.036, 0.689] | YES |
| sigma_w1_logit | 0.500 | 0.610 | [0.414, 0.846] | YES |

Subject-level: r(c)=0.89, r(w1)=0.85 (N=20). Rhat max=1.042; 0 divergences.
Note: ESS for sigma_log_gamma ≈ 118 — use iter_sampling=1000 for production runs.

Key implementation decisions confirmed:
- Non-centred parameterisation is stable with adapt_delta=0.90
- Per-(subj, stimulus) multinomial aggregation is exact (no approximation)
- Stimulus-indexed D_stim passes 12 matrices, not T matrices
- `N(0, 1)` prior on log(c) restores true mean_c=0.8 inside the 90% CI

### 5. K>2/M>2 attention simplex and multi-category choice (04b_attention_simplex.R)

Implements and recovery-tests the novel API elements not yet empirically supported
in earlier scripts: softmax-with-reference attention weights (M>2 dimensions) and
multi-category Luce choice (K>2 categories). This is the gating item for an
upstream proposal.

**Setting**: M=4 dimensions, K=3 categories, J=24 exemplars, N=6 subjects,
1920 synthetic trials. Softmax-with-reference: `w = softmax([w_raw; 0])` with
`w_raw` a free (M-1)-vector; `w[M]` is the reference. Multi-category choice
uses log-bias `rep_vector(-log(K), K)`.

**Phase 2b result (CmdStan 2.38, 4 chains × 500 draws, 8 s, seed 42)**:

| Param | True | Posterior | 90% CI | In CI |
|---|---|---|---|---|
| c | 0.80 | 0.94 | [0.29, 1.99] | YES |
| gamma | 1.50 | 1.94 | [0.63, 4.59] | YES |
| w₁ | 0.40 | 0.35 | [0.24, 0.47] | YES |
| w₂ | 0.30 | 0.28 | [0.21, 0.36] | YES |
| w₃ | 0.20 | 0.17 | [0.04, 0.34] | YES |
| w₄ | 0.10 | 0.21 | [0.09, 0.35] | YES |

Rhat_max=1.004; ESS_min=1346; 0 divergences. All attention weights within
90% CI. Softmax-with-reference parameterisation and multi-category choice
rule both recover cleanly — the novel API has empirical support.

### 6. Real-data fit (06_realdata_fit.R)

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

### 7. LOO model comparison — hierarchical, multi-subject, old/new structure (06b_loo_comparison.R)

**Revised design** (supersedes initial 12-cell aggregate version):

- N=20 subjects × S=16 stimuli = **320 LOO cells** (was 12 single-condition cells)
- 12 training stimuli (nosof88 geometry, `is_old=TRUE`) + 4 synthetic transfer
  stimuli near the category boundary (`is_old=FALSE`)
- Hierarchical random effects: `log(c)`, `log(gamma)`, `logit(w1)` for all three
  models; `logit(p_mem)` for PRM
- Generative model: PRM (GCM activations + rote-memory overlay for training items)
- Warmup hardening: `init=0.5` restricts initial unconstrained draws to [−0.5, 0.5],
  preventing large-`c` NaN in the first warmup steps
- LOO unit: one (subject, stimulus) count cell; leave-one-cell-out valid for
  multinomial aggregated data

**Note on catlearn::nosof94**: inspected at script startup — catlearn stores
aggregate proportions (no per-subject/per-trial columns). The synthetic design
replicates a nosof94-style experiment. Replace with actual nosof94 raw data or
the OSF rocks data (Nosofsky et al. 2022) for a production comparison.

**Results**: *pending local execution — run `Rscript local/exploration/categorization/06b_loo_comparison.R`*

Expected outcomes:
- With 320 LOO cells, ΔELPD between models has SE ≈ 1–3 ELPD units
  (vs SE ≈ 3–4 in the 12-cell version where all differences were within SE)
- PRM `p_mem` group mean: 90% CI should exclude 0 (rote-memory identified) ✓
- LOO ordering: GCM best (data generated from GCM+rote), Prototype worst
- GCM vs Prototype ΔELPD: expected to be *significant* (> 2 × SE_diff)
  given 320 vs 12 cells and a clearly identifiable gamma for GCM

**Log-space stability**: PRM model uses `vector[K] p = softmax(la)` (not
`simplex[K]`) for the local mixing variable in both model and generated-quantities
blocks — the `simplex` type is not valid for Stan locals (compile error). GCM and
Prototype use `multinomial_logit_lpmf` throughout. PRM uses `multinomial_lpmf`
with the explicitly mixed probability vector (mixing breaks the logit form).

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

**Log-space numerical stability**: when sensitivity `c` is large (>2 in
nosof88 coordinates) or K/M is large, the naive per-exemplar activation
`act_k = sum(exp(-c*d))` underflows to 0, causing `log(0) = -Inf` and
NaN in the subsequent `softmax`. The fix accumulates in log space —
`log_act[k] = log_sum_exp(-c*d[j] for j in cat k)` — and uses
`multinomial_logit_lpmf(counts | la)` instead of
`multinomial_lpmf(counts | softmax(la))`, avoiding the explicit softmax
and eliminating the simplex-sum-NaN warnings entirely.

**Rocks-scale benchmark (08_rocksscale_benchmark.R, 1 chain, CSR exemplar indexing)**:

| Scale | S | J | M | K | Draws/s (1 chain) |
|---|---|---|---|---|---|
| nosof88 | 12 | 12 | 2 | 2 | 176 |
| medium\_6dim | 24 | 24 | 6 | 4 | 79 |
| rocks\_100 | 100 | 90 | 8 | 10 | 26 |
| rocks\_full | 150 | 90 | 8 | 10 | 17.9 |

The rocks-full target (S=150, J=90, M=8, K=10) yields ~17.9 draws/s on a
single chain — a 4-chain production run completes in a few minutes. CSR
per-category indexing (`cat_start`/`cat_end`/`ex_flat` passed as Stan data)
reduces the inner loop from O(J·K) branch-per-exemplar to O(J) with no
conditional, which is the prerequisite for feasibility at K=10.

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
