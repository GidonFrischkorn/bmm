# GCM / Prototype Categorization Models — Exploration

**Issue**: [#19](https://github.com/GidonFrischkorn/bmm/issues/19)  
**Status**: Phase 2a in progress — multinomial aggregation + prior recalibration  

Exploration of Nosofsky's Generalized Context Model (GCM) and its prototype
analogue for integration into bmm's multinomial infrastructure.

## Scripts (run from repository root)

| Script | Purpose | Output |
|---|---|---|
| `01_r_reference.R` | Pure-R likelihood for exemplar GCM, prototype, PRM | Validates against catlearn (max dev 1.11e-16) |
| `02_api_design.md` | API design memo: answers all 6 design questions | Design recommendations |
| `03_stan_prototype.R` | CmdStanR GCM implementation with MCMC | Rhat<1.02, 0 diverg, r=0.978 |
| `04_parameter_recovery.R` | Recovery study across 8 parameter scenarios | c-gamma trade-off documented |
| `05_hierarchical.R` | Multi-subject random-effects GCM | Non-centred parameterisation |
| `06_realdata_fit.R` | Bayesian GCM on all 3 nosof88 conditions | Per-condition diagnostics |
| `07_design_doc.md` | Full design/feasibility document | Upstream proposal-ready |

## Key Findings

1. **R reference validated**: all three model versions match `catlearn::stsimGCM`
   to machine precision (1.11e-16).

2. **Stan implementation works**: GCM fits the nosof88 data with Rhat<1.02,
   0 divergences, and pred-obs r=0.978 at ~61 draws/s (T=300, J=12, M=2).

3. **c-gamma identifiability**: sensitivity `c` and response scaling `gamma`
   are partially non-identified in small designs. Recommended defaults:
   `log(c) ~ Normal(0, 1)` and `log(gamma) ~ Normal(0, 0.5)`. The tighter
   `Normal(0.5, 0.5)` on log(c) pulls posteriors toward exp(0.5)≈1.65 and
   should not be used as a default. Prototype version fixes gamma=1
   (Nosofsky & Zaki 2002).

4. **w1 (attention weight) recovers well** across all scenarios (r=0.77).

5. **Hierarchical GCM is feasible** with non-centred parameterisation and
   adapt_delta=0.90. **Phase 2a**: multinomial-aggregated likelihood
   (N×S evaluations vs N×S×n) enables N≥20 at manageable runtime.

6. **Real data**: GCM fits all 3 nosof88 conditions with r>0.97 and 0
   divergences. Condition-specific attention shifts are present but modest.

## Data

Uses `catlearn::nosof88` (Nosofsky 1988, 12 Munsell chips, 3 frequency
conditions) and the 2-D MDS solution hard-coded in `catlearn::nosof88train`.
No additional data files required.

## Implementation Recommendation

See `07_design_doc.md` for the full design document. TL;DR:

- One constructor `gcm(resp_cat, dimensions, exemplars, version=)`
- Custom Stan family with pre-computed distance array via `stanvars`
- Softmax-with-reference parameterisation for attention weights
- Informative default priors on c and gamma
- Steps 1–3 (R model file, Stan chunk, unit tests) constitute a minimal PR
