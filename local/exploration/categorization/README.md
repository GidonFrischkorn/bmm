# GCM / Prototype Categorization Models — Exploration

**Issue**: [#19](https://github.com/GidonFrischkorn/bmm/issues/19)  
**Status**: Phase 2b complete — all scripts verified (incl. 06b LOO comparison)

Exploration of Nosofsky's Generalized Context Model (GCM) and its prototype
analogue for integration into bmm's multinomial infrastructure.

## Scripts (run from repository root)

| Script | Purpose | Status |
|---|---|---|
| `01_r_reference.R` | Pure-R likelihood for exemplar GCM, prototype, PRM | ✓ validated vs catlearn (1.11e-16) |
| `02_api_design.md` | API design memo: answers all 6 design questions | ✓ complete |
| `03_stan_prototype.R` | CmdStanR GCM implementation with MCMC | ✓ Rhat<1.02, 0 diverg, r=0.978 |
| `04_parameter_recovery.R` | Recovery study across 8 parameter scenarios | ✓ c-gamma trade-off documented |
| `04b_attention_simplex.R` | M>2 dims + K>2 categories with softmax-with-reference w | ✓ M=4/K=3: all w in 90% CI, Rhat_max=1.004, 8 s |
| `05_hierarchical.R` | Multi-subject random-effects GCM | ✓ N=20, 52s, all params in 90% CI |
| `06_realdata_fit.R` | Bayesian GCM on all 3 nosof88 conditions | ✓ r>0.97 per condition |
| `06b_loo_comparison.R` | GCM vs prototype vs PRM with LOO model comparison | ✓ 0 diverg, Rhat≤1.008; p_mem=0.140 [0.020, 0.279]; LOO: Proto>PRM>GCM (diffs within SE) |
| `07_design_doc.md` | Full design/feasibility document | ✓ Phase 2a verified |
| `08_rocksscale_benchmark.R` | Timing benchmark at rocks scale (S=150, J=90, M=8, K=10) | ✓ rocks_full: 17.9 draws/s (1 chain), CSR indexing |

## Key Findings

1. **R reference validated**: all three model versions match `catlearn::stsimGCM`
   to machine precision (1.11e-16).

2. **Stan implementation works**: GCM fits the nosof88 data with Rhat<1.02,
   0 divergences, and pred-obs r=0.978.

3. **c-gamma identifiability**: sensitivity `c` and response scaling `gamma`
   are partially non-identified in small designs. Recommended defaults:
   `log(c) ~ Normal(0, 1)` and `log(gamma) ~ Normal(0, 0.5)`. The tighter
   `Normal(0.5, 0.5)` on log(c) pulls posteriors toward exp(0.5)≈1.65.
   Prototype version fixes gamma=1 (Nosofsky & Zaki 2002).

4. **w1 (attention weight) recovers well** across all scenarios.

5. **Hierarchical GCM is feasible** (Phase 2a, N=20, 52 s): multinomial
   aggregation (N×S evaluations vs N×S×n) enables N≥20. All group-level
   parameters within 90% CI; r(c)=0.89, r(w1)=0.85.

6. **Log-space numerical stability** (Phase 2b): all Stan models now use
   `log_sum_exp` accumulation for per-category activation and
   `multinomial_logit_lpmf` instead of `multinomial_lpmf(…|softmax(…))`.
   Eliminates simplex-sum-NaN warnings at large c or high K/M.

7. **K>2/M>2 attention simplex verified** (Phase 2b, `04b_attention_simplex.R`):
   softmax-with-reference w (M=4) and multi-category Luce choice (K=3) recover
   cleanly — all weights within 90% CI, Rhat_max=1.004, ESS_min=1346, 8 s.
   Novel API is empirically supported; upstream proposal is unblocked.

8. **Rocks-scale feasible** (Phase 2b, `08_rocksscale_benchmark.R`): CSR
   per-category exemplar indexing yields 17.9 draws/s at rocks_full scale
   (S=150, J=90, M=8, K=10) on 1 chain — a 4-chain production run takes a few
   minutes. Issue #19 task 3 confirmed feasible.

9. **LOO comparison pipeline verified** (Phase 2b, `06b_loo_comparison.R`):
   GCM, prototype, and PRM fit nosof88 condition B (0 divergences, Rhat≤1.008).
   PRM rote-memory component is identified: p_mem=0.140, 90% CI [0.020, 0.279],
   CI excludes 0 (PRM ≠ prototype). LOO ordering Prototype > PRM > GCM with
   differences within SEs — expected on 12-cell aggregate data; treat as a
   machinery demonstration. Trial-level data (nosof94/rocks) needed for a
   conclusive model-selection result.

## Data

Uses `catlearn::nosof88` (Nosofsky 1988, 12 Munsell chips, 3 frequency
conditions) and the 2-D MDS solution. Phase 2b scripts (`04b`, `08`) use
synthetic data; `06b` uses nosof88 condition B with is_old=TRUE for all stimuli.
No external data downloads required.

## Implementation Recommendation

See `07_design_doc.md` for the full design document. TL;DR:

- One constructor `gcm(resp_cat, dimensions, exemplars, version=)`
- Custom Stan family with pre-computed distance array via `stanvars`
- Softmax-with-reference parameterisation for attention weights (M-1 free)
- Informative default priors on c and gamma
- Steps 1–3 (R model file, Stan chunk, unit tests) constitute a minimal PR
