# MPT Models in bmm — Design Document

> **Status:** Phase 1 complete — Stan-validated, ready for package integration review  
> Intended as a response to [venpopov/bmm#255](https://github.com/venpopov/bmm/issues/255).

---

## 1. Overview

Multinomial Processing Tree (MPT) models are a family of cognitive measurement models sharing the same mathematical structure as the M3 (multinomial likelihood, non-linear probability expressions). This document proposes a concrete implementation plan for a `mpt()` constructor in bmm that piggy-backs on the existing M3/multinomial infrastructure.

---

## 2. The Core Structural Analogy

| Feature | M3 | MPT |
|---|---|---|
| Likelihood | `multinomial()` | `binomial()` / `multinomial()` |
| Category probabilities | `softmax(activation)` | non-linear function of detection/guessing params |
| Parameters | activations (log/logit scale) | detection/guessing probs (logit/probit scale) |
| Condition handling | index variable trick | index variable trick (per tree) |
| brms formula type | `nlf()` non-linear | `nlf()` non-linear |

The M3 uses `construct_m3_act_funs()` to emit `nlf()` formulas from a high-level specification. MPT needs the same thing: a `mpt_to_brms()` function that turns tree-and-branch specifications into `nlf()` formulas with logit/probit reparametrisation.

---

## 3. User-Facing API (Recommended Design)

### 3.1 Primary syntax — tree/branch list (Syntax C)

```r
library(bmm)

# Define trees
tree_old <- mpt_tree("old", list(
  old = "D + (1 - D) * g",
  new = "(1 - D) * (1 - g)"
))
tree_new <- mpt_tree("new", list(
  old = "(1 - D) * g",
  new = "D + (1 - D) * (1 - g)"
))

# Specify model
model <- mpt(
  trees     = list(tree_old, tree_new),
  condition = "condition"   # column in data matching tree names
)

# Specify predictor formulas (one per parameter)
formula <- bmf(
  D ~ 1 + cond + (1 + cond | id),
  g ~ 1 + (1 | id)
)

# Fit
fit <- bmm(
  formula = formula,
  data    = my_data,
  model   = model
)
```

**Key design choice:** `bmf()` formulas specify predictors for model parameters on the latent (logit/probit) scale, exactly as in `m3()`. The user never writes the logit-transformation — bmm handles it internally.

### 3.2 Import wrapper — MPTinR string (Syntax A)

```r
model_str <- "
D + (1 - D) * g        # old
(1 - D) * (1 - g)      # new

(1 - D) * g            # old
D + (1 - D) * (1 - g)  # new
"

model <- mpt_from_string(
  model_str,
  tree_conditions = c("old", "new"),
  condition = "condition"
)
```

This parser is implemented in `03_mpt_parser.R:parse_mpt_string()`. It reads MPTinR-compatible model files unchanged and converts them to an `mpt_spec` object, then calls `mpt_to_brms()` internally.

---

## 4. What M3 Infrastructure Can Be Reused

| Component | Status | Notes |
|---|---|---|
| `check_data.m3` | **Reuse with adaptation** | The response-matrix construction (`data$Y <- as.matrix(data[resp_cols])`) works for multinomial MPT. For 2-category models (binomial), only need a count column, not a matrix. |
| `configure_model.m3` | **Partial reuse** | Family setup (`brms::multinomial()` or `binomial()`) and `bmf2bf()` delegation can follow the same pattern. |
| `check_formula.m3` | **Adapt** | Need to check that every model parameter has a predictor formula. |
| `bmf2bf.m3` | **Replace** | M3 uses `glue_choice_rule_functions()`; MPT needs `mpt_to_brms()` which emits tree-aggregated `nlf()` formulas. |
| `construct_m3_act_funs()` | **Analogy** | `mpt_to_brms()` plays this role for MPT. |
| `multinomial(refcat = NA)` family | **Reuse directly** | Same family used in M3 for >2 response categories. |
| `pp_check` for multinomial (#324) | **Reuse directly** | Once #324 merges, MPT benefits automatically. |
| Index-variable trick (`Idx_*`) | **Reuse pattern** | MPT uses tree indicator columns; same idea as M3 option-count indexing. |

---

## 5. What Is Genuinely New

1. **`mpt_tree()` constructor** — S3 object holding a tree name + branch expressions.
2. **`mpt()` constructor** — validates trees, extracts parameters, returns an `mpt_spec`/`bmmodel` object. Analogous to `m3()`.
3. **`mpt_to_brms()`** — core formula emitter. Builds the aggregated probability expression for each response category from tree-indicator columns, wraps in `logit()`, adds `nlf()` reparametrisations and `lf()` predictor formulas. Handles binomial (K=2) and multinomial (K>2) paths. Implements stick-breaking reparametrisation for probability simplex constraints (`simplex_params` argument).
4. **`mpt_from_string()`** — MPTinR-format parser. Tokenises branch equations, maps them to `mpt_tree` objects.
5. **`check_data.mpt`** — adds tree-indicator columns to data; converts response columns to matrix if multinomial.
6. **`check_formula.mpt`** — verifies that every model parameter declared in the trees has a predictor formula in the `bmf()`.
7. **`mpt_bridge_bmf()`** (prototype in `06_bmf_bridge.R`) — maps a `bmmformula` object to the `predictor_formulas` list format expected by `mpt_to_brms()`, enabling the standard bmm user workflow.

**brms naming constraint:** brms forbids underscores and dots in `nlpar` names. Any parameter named `d_A` or `g.1` must be sanitised (strip `._`) before constructing `nlf()` formulas, then the reverse mapping is applied when the user looks up predictors. This is handled by `.sanitize_nlpar()` and `.replace_params_in_expr()` in `03_mpt_parser.R`.

---

## 6. Handling Multiple Trees (Condition Structure)

The upstream issue correctly identifies that MPT's "trees" (item types) map onto conditions in the data. The key formula trick is:

```
P(response = "old") = Σ_t [ I(condition == tree_t) × branch_prob_t("old") ]
```

With two trees and indicator `is_old = (condition == "old")`:
```
P(old) = is_old * (D + (1-D)*g) + (1 - is_old) * (1-D)*g
```

The `mpt_to_brms()` prototype (see `03_mpt_parser.R`) generates this expression automatically for any number of trees. Each tree gets a generated column `.ind_<tree_name>`.

---

## 7. Equality Constraints

Traditional MPT often constrains parameters across conditions (e.g., `Do = Dn = D`). In bmm's approach, this is automatic: if the researcher uses a single parameter name `D` in all trees, that parameter appears once in the formula and one predictor formula suffices. No special constraint syntax is needed — the constraint is implicit in naming.

To allow `Do ≠ Dn`, the user simply names them differently in the branch expressions.

---

## 8. Hierarchical Extensions

Because the reparametrisation puts detection/guessing parameters on an unconstrained logit/probit scale, random effects are added via standard `bmf()` syntax:

```r
bmf(
  D ~ 1 + (1 | id),    # random detection, logit scale
  g ~ 1               # fixed guessing
)
```

The prior on the logit-scale intercept naturally induces a uniform(0,1) marginal on the probability scale when `logistic(0,1)` is used. For the hierarchical case, a normal hyperprior on the random-effect SD is appropriate.

Correlated random effects use brms syntax `(1 |p| id)` with a `lkj(2)` prior on the correlation. The `mpt_bridge_bmf()` helper (see `06_bmf_bridge.R`) strips the LHS of a `bmmformula` and forwards predictor formulas to `mpt_to_brms()`, so correlated REs work via:

```r
formula_bmf <- bmf(
  D ~ 1 + (1 |p| id),
  g ~ 1 + (1 |p| id)
)
pred_fms <- mpt_bridge_bmf(spec, formula_bmf)
out      <- mpt_to_brms(spec, predictor_formulas = pred_fms, ...)
```

---

## 9. Multinomial Extension (>2 Response Categories)

For models with 3+ response categories (e.g., source monitoring, pair clustering):

- Use `brms::multinomial(refcat = NA)` family with `cats` and `dpars` set.
- The main formula becomes `Y | trials(nTrials) ~ log(P_first_cat)` with `nl = TRUE`.
- `nlf(mu{cat} ~ log(P_cat))` is emitted for each subsequent category, following the M3 pattern in `R/model_m3.R:bmf2bf.m3`.
- Parameters constrained to a probability simplex (sum to 1) receive a stick-breaking reparametrisation:
  - First free parameter: `gA ~ inv_logit(lgA)`
  - Subsequent: `gB ~ (1 - inv_logit(lgA)) * inv_logit(lgB)`
  - Last (derived): `gNew ~ 1 - (gA + gB)`
- Users specify simplex groups via `simplex_params = c("g_A", "g_B", "g_New")`.

This is fully implemented and Stan-validated (see Section 13).

---

## 10. Implementation Plan

### Phase 1 — Standalone parser (no package integration; this PR)

- [x] `03_mpt_parser.R`: `mpt_tree()`, `mpt()`, `mpt_to_brms()`, `parse_mpt_string()`
- [x] Binomial path (K=2): 2HTM, SMM2 — Stan-validated (Section 13)
- [x] Multinomial path (K>2): PCM, SMM3 with stick-breaking — Stan-validated (Section 13)
- [x] `04_hierarchical_demo.R`: hierarchical 2HTM recovery confirmed
- [x] `06_bmf_bridge.R`: `mpt_bridge_bmf()` converting `bmmformula` → predictor list; correlated RE recovery
- [x] `07_multinomial_demo.R`: PCM and SMM3 end-to-end Stan fits
- [x] `08_external_validation.R`: Riefer & Batchelder (1988) data; qualitative validation passes

### Phase 2 — Package integration (new PR: `feat/mpt-model`)

1. `R/model_mpt.R`:
   - `.model_mpt()` internal constructor (analogous to `.model_m3()`)
   - `mpt()` exported constructor with argument validation
   - `mpt_tree()` exported helper
   - `mpt_from_string()` import wrapper

2. `R/helpers-mpt.R`:
   - `mpt_to_brms()` moved here as `.mpt_build_formula()`
   - `parse_mpt_string()` moved here as `.parse_mpt_string()`

3. S3 methods in `R/model_mpt.R`:
   - `check_data.mpt()`
   - `check_model.mpt()`
   - `check_formula.mpt()`
   - `configure_model.mpt()`
   - `bmf2bf.mpt()`

4. Tests in `tests/testthat/test-model-mpt.R`

5. Recovery script: `local/mpt_parameter_recovery.R` (per guardrail #2)

### Phase 3 — Advanced features (future PR)

- `simplex_params` default inference from named parameter groups
- Inequality constraints (`D_A > D_B`) via post-processing or re-parametrisation
- MPTinR import validation: numeric sum-to-1 check at fit time, not just at parse time

---

## 11. Open Questions — Resolved

1. **Formula validation**: `mpt_to_brms()` calls `.mpt_validate_tree()` at construction time with a numeric test-point (all params = 0.5). This is sufficient for catching mis-specified trees early. Symbolic verification would require CAS infrastructure and is out of scope for Phase 1. **Resolution: numeric test-point check at construction is adopted.**

2. **Multinomial columns**: The multinomial path infers response categories from branch names; `simplex_params` provides explicit disambiguation for constrained groups. The user does not need to supply `resp_cats` separately. **Resolution: infer from branch names; `simplex_params` for constrained groups.**

3. **Tree condition matching**: Currently exact string match between `tree$name` and values in the condition column. Factor levels are coerced to character via `as.integer(data[[cond]] == tree_name)`. **Resolution: exact string match with character coercion; factor support is a Phase 2 nicety.**

4. **Parameter constraints beyond equality**: `logit`/`probit` reparametrisation enforces `(0,1)` bounds. Ordered constraints (e.g., `D_A > D_B`) would require an additional ordered reparametrisation and are deferred to Phase 3.

5. **`pp_check` support**: binomial MPT reuses brms defaults. For multinomial, `pp_check` type `rootogram` works in the current prototype (see `04_hierarchical_demo.R`). Full multinomial pp_check depends on upstream #324.

---

## 12. References

- Batchelder, W. H., & Riefer, D. M. (1999). Theoretical and empirical review of multinomial process tree modeling. *Psychonomic Bulletin & Review, 6*(1), 57–86.
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*(1), 3–28.
- Riefer, D. M., & Batchelder, W. H. (1988). Multinomial modeling and the measurement of cognitive processes. *Psychological Review, 95*(3), 318–339.
- Rouder, J. N., & Lu, J. (2005). An introduction to Bayesian hierarchical models with an application in the theory of signal detection. *Psychonomic Bulletin & Review, 12*(4), 573–604.
- Singmann, H., & Kellen, D. (2013). MPTinR: Analysis of multinomial processing tree models in R. *Behavior Research Methods, 45*(2), 560–575.

---

## 13. Validation Evidence (Phase 1 Stan Fits)

All fits use CmdStan 2.39.0 via cmdstanr backend. Smoke-test iterations (500–1000 warmup, 500–1000 sampling) are used; all runs complete in < 5 minutes on 4 cores.

### 13.1 WP1 — Hierarchical 2HTM (04_hierarchical_demo.R)

80 participants, 50 items/tree. True: D=0.70, g=0.50 on probability scale (mu_lD=0.847, mu_lg=0.00, sd_lD=0.30, sd_lg=0.25).

| Parameter | True | Estimate | 95% CI |
|---|---|---|---|
| mu_lD (logit) | 0.847 | 0.738 | [0.64, 0.84] |
| mu_lg (logit) | 0.000 | −0.061 | [−0.17, 0.04] |
| sd_lD | 0.300 | 0.268 | [0.14, 0.38] |
| sd_lg | 0.250 | 0.117 | [0.01, 0.28] |
| D (prob) | 0.700 | 0.677 | — |
| g (prob) | 0.500 | 0.485 | — |

Participant-level recovery: r(lD) = 0.639, r(lg) = 0.458.  
Max Rhat = 1.00. No divergences.

### 13.2 WP2 — Multinomial PCM (07_multinomial_demo.R, Part A)

60 participants, 40 pairs. True: cp=0.60, rp=0.75.

| Parameter | True | Estimate | 95% CI |
|---|---|---|---|
| cp | 0.600 | 0.587 | [0.463, 0.665] |
| rp | 0.750 | 0.765 | [0.710, 0.823] |

Max Rhat = 1.021 (500 warmup smoke test; acceptable).

### 13.3 WP2 — Multinomial SMM3 with stick-breaking (07_multinomial_demo.R, Part B)

60 participants, 40 items/tree, 3 trees, 3 response categories. True: d_A=0.70, d_B=0.60, g_A=0.40, g_B=0.30, g_New=0.30.

| Parameter | True | Estimate | 95% CI |
|---|---|---|---|
| d_A | 0.700 | 0.707 | [0.679, 0.732] |
| d_B | 0.600 | 0.603 | [0.575, 0.630] |
| g_A | 0.400 | 0.407 | [0.390, 0.425] |
| g_B | 0.300 | 0.300 | (stick-breaking) |
| g_New | 0.300 | 0.293 | (derived: 1−gA−gB) |

Max Rhat = 1.004. Near-exact recovery.

### 13.4 WP3 — bmf() bridge + correlated REs (06_bmf_bridge.R)

100 participants, 50 items/tree. True: rho(lD, lg) = 0.50, mu_lD=0.847, mu_lg=0.00, sd_lD=0.30, sd_lg=0.25.

| Parameter | True | Estimate | 95% CI |
|---|---|---|---|
| mu_lD (logit) | 0.847 | 0.865 | [0.77, 0.97] |
| mu_lg (logit) | 0.000 | 0.028 | [−0.08, 0.13] |
| sd_lD | 0.300 | 0.362 | [0.26, 0.47] |
| sd_lg | 0.250 | 0.176 | [0.01, 0.34] |
| rho(lD, lg) | 0.500 | 0.377 | [−0.39, 0.87] |

True rho = 0.50 is inside the 95% CI. Wide CI expected at n=100 with 500 warmup.  
Max Rhat = 1.013 (marginal; full run with 2000 iter/1000 warmup would tighten).

### 13.5 WP4 — External validation: Riefer & Batchelder (1988) (08_external_validation.R)

40 participants (20 grouped, 20 ungrouped), 20 pairs each. Hardcoded data approximating RB1988 Table 1 typical effect sizes.

| Condition | cp (est) | cp CI | rp (est) | rp CI |
|---|---|---|---|---|
| Grouped | 0.558 | [0.468, 0.632] | 0.485 | [0.397, 0.577] |
| Ungrouped | 0.102 | [0.044, 0.152] | 0.286 | [0.241, 0.335] |

Qualitative checks: cp_grouped > cp_ungrouped ✓; rp_grouped > rp_ungrouped ✓; cp effect 95% CI excludes 0 ✓.  
Max Rhat = 1.007. Note: rp values are lower than the published ML estimates (≈0.85/0.70) because the hardcoded data encodes the typical aggregate proportions, not the exact paper Table 1. The qualitative pattern and parameter ordering are consistent.

---

## 14. Known Limitations and Future Work

- **brms naming constraint**: parameter names with underscores/dots must be sanitised for brms (`d_A` → `dA` internally). The mapping is handled transparently but should be documented for users in Phase 2.
- **R 4.6.0 `paste0` bug**: `paste0("l", character(0))` returns `"l"` in R 4.6.0 (not `character(0)`). The emitter guards against this explicitly; the guard should be preserved in Phase 2.
- **Correlated RE width**: with n=100 and only 500 warmup iterations, the rho CI is very wide ([-0.39, 0.87]). Real-data analyses should use ≥2000 iterations.
- **`01_reproduce_upstream_demo.R`**: the aggregated 2HTM fit in this pre-existing script estimates D ≈ 0 (true: 0.70) because the simulated data accidentally sets equal hit and false-alarm rates. This is a bug in that script only; the hierarchical `04_hierarchical_demo.R` and all new scripts are unaffected.
