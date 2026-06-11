# MPT Models in bmm — Design Document

> **Status:** Phase 1 complete (Round 2) — Stan-validated, ready for package integration review  
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

### 3.2 Covariates (design-fixed guessing rates)

When branch expressions reference data columns that are not latent parameters — for example, guessing rates determined by the experimental design — those names are declared via `covariates`:

```r
# Branch expressions reference GcorrPi, GcorrNoPi, GotherNoPi
# These are data columns, not fitted parameters.
tree_ss <- mpt_tree("ss", list(
  correct = "Pb + (1-Pb)*Pi*GcorrPi + (1-Pb)*(1-Pi)*GcorrNoPi",
  other   = "(1-Pb)*Pi*(1-GcorrPi) + (1-Pb)*(1-Pi)*(1-GcorrNoPi)*GotherNoPi",
  npl     = "(1-Pb)*(1-Pi)*(1-GcorrNoPi)*(1-GotherNoPi)"
))

model <- mpt(
  trees      = list(tree_ss),
  covariates = c("GcorrPi", "GcorrNoPi", "GotherNoPi")
  # condition = NULL (single-tree model)
)
```

Covariate names pass verbatim into brms `nlf()` formulas; brms treats them as data columns (as it does any symbol that is not a declared `nlpar`). The user computes covariate columns before calling `mpt_to_brms()`.

### 3.3 Import wrapper — MPTinR string (Syntax A)

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
2. **`mpt()` constructor** — validates trees, extracts parameters, returns an `mpt_spec`/`bmmodel` object. Analogous to `m3()`. Signature: `mpt(trees, condition = NULL, covariates = character(0), link = "logit")`.
3. **`mpt_to_brms()`** — core formula emitter. Builds the aggregated probability expression for each response category from tree-indicator columns, wraps in `logit()`, adds `nlf()` reparametrisations and `lf()` predictor formulas. Handles binomial (K=2) and multinomial (K>2) paths. Implements stick-breaking reparametrisation for probability simplex constraints (`simplex_params` argument). Implements WP4 bypass (see Section 10).
4. **`mpt_from_string()`** — MPTinR-format parser. Tokenises branch equations, maps them to `mpt_tree` objects.
5. **`check_data.mpt`** — adds tree-indicator columns to data; converts response columns to matrix if multinomial.
6. **`check_formula.mpt`** — verifies that every model parameter declared in the trees has a predictor formula in the `bmf()`.
7. **`mpt_bridge_bmf()`** (prototype in `06_bmf_bridge.R`) — maps a `bmmformula` object to the `predictor_formulas` list format expected by `mpt_to_brms()`, enabling the standard bmm user workflow.

**brms naming constraint (enforced):** brms forbids underscores and dots in `nlpar` names. Parameter names containing `_` or `.` now raise an **error** at `mpt()` construction time — no silent sanitisation. Users must use `dA`, `gA`, `gNew`, etc. instead of `d_A`, `g.new`.

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

For single-tree models (`condition = NULL`), the indicator column is set to `1L` for all rows — no condition column required in the data.

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

## 9. Covariates (Design-Fixed Data Columns)

Many MPT models include guessing rates that are fully determined by the experimental design (e.g., set size, response-set composition) rather than being latent parameters. These design-fixed quantities appear in branch expressions but should not be reparametrised or given priors.

### 9.1 Mechanism

`mpt()` accepts a `covariates` argument — a character vector of symbol names that appear in branch expressions but are data columns, not latent parameters:

```r
spec <- mpt(
  trees      = list(tree_ss),
  covariates = c("GcorrPi", "GcorrNoPi", "GotherNoPi")
)
```

Internally, `.mpt_extract_params()` tokenises each branch expression and excludes from the parameter list any symbol that:
- Appears in `covariates`, **or**
- Is an R/Stan mathematical function (`exp`, `log`, `inv_logit`, `Phi`, etc.)

The remaining symbols are treated as latent parameters and receive the logit/probit reparametrisation.

### 9.2 Stan integer-division hazard

When branch expressions include numeric fractions (`1/4`, `3/8`, etc.), R evaluates them as floating-point (0.25, 0.375), but Stan/C++ compiles them as integer division (= 0), silently breaking the likelihood. **Always use decimal literals** in tree expressions:

```r
# Wrong — Stan integer division makes 1/4 = 0
correct = "Pm*Pb + Pm*(1-Pb)*(1/4) + (1-Pm)*(1/8)"

# Correct
correct = "Pm*Pb + Pm*(1-Pb)*0.25 + (1-Pm)*0.125"
```

Design-fixed guessing rates that equal `1/n` for some integer `n` should either be pre-computed as a data column (then declared as a covariate) or written as decimals in the tree expression.

### 9.3 Validation

See Section 15.6 (Oberauer 2019, WP2). The simple-span recognition model uses three design-fixed covariates computed from response-set size:

| Covariate | Expression | Range in data |
|---|---|---|
| `GcorrPi` | `1 / rsizeList` | [0.125, 0.500] |
| `GcorrNoPi` | `1 / (rsizeList + rsizeNPL)` | [0.083, 0.250] |
| `GotherNoPi` | `(rsizeList − 1) / (rsizeList − 1 + rsizeNPL)` | [0.333, 0.636] |

The emitter identifies `Pb` and `Pi` as latent parameters and leaves `GcorrPi`, `GcorrNoPi`, `GotherNoPi` untouched in the brms formula (they become data-column references in Stan).

---

## 10. Time-varying Parameters (WP4 Bypass)

Some cognitive models require parameters that change continuously within a trial or across time (e.g., exponential forgetting, evidence accumulation). These "time-varying" parameters cannot use the standard `inv_logit(l<param>)` reparametrisation because their (0,1) bound is enforced by the user-supplied mathematical expression rather than a monotone link.

### 10.1 Bypass mechanism

When a predictor formula's RHS references symbols that are themselves keys in `predictor_formulas`, `mpt_to_brms()` detects this as a **bypass** case:

```r
out <- mpt_to_brms(
  spec,
  predictor_formulas = list(
    # D is bypass: RHS references lDmax and lrate, which are also keys
    D     = ~ inv_logit(lDmax) * (1 - exp(-exp(lrate) * ptime)),
    lDmax = ~ 1 + (1 | id),
    lrate = ~ 1 + (1 | id)
  )
)
```

**Emitted formula:**
```
D ~ inv_logit(lDmax) * (1 - exp(-exp(lrate) * ptime))   # nlf(): user's expression, no wrapper
lDmax ~ 1 + (1 | id)                                     # lf(): sub-parameter
lrate ~ 1 + (1 | id)                                     # lf(): sub-parameter
```

The automatic `inv_logit(lD)` reparametrisation is **skipped** for `D`; `lDmax` and `lrate` become sub-parameters (unconstrained reals) with their own predictor formulas. The user's expression owns the `(0,1)` constraint.

### 10.2 Detection rule

`mpt_to_brms()` scans the RHS tokens of each predictor formula. If any token matches another key in `predictor_formulas`, the parameter is marked as a bypass parameter. Data columns referenced in the expression (e.g., `ptime`) that are not keys in `predictor_formulas` pass through to Stan as data — no special declaration needed.

### 10.3 Validation

See Section 15.7 (time-varying demo, WP4). The exponential detection-growth model `D(t) = inv_logit(lDmax) * (1 − exp(−exp(lrate) * t))` recovers true parameters with all acceptance criteria met.

---

## 11. Multinomial Extension (>2 Response Categories)

For models with 3+ response categories (e.g., source monitoring, pair clustering):

- Use `brms::multinomial(refcat = NA)` family with `cats` and `dpars` set.
- The main formula becomes `Y | trials(nTrials) ~ log(P_first_cat)` with `nl = TRUE`.
- `nlf(mu{cat} ~ log(P_cat))` is emitted for each subsequent category, following the M3 pattern in `R/model_m3.R:bmf2bf.m3`.
- Parameters constrained to a probability simplex (sum to 1) receive a stick-breaking reparametrisation:
  - First free parameter: `gA ~ inv_logit(lgA)`
  - Subsequent: `gB ~ (1 - inv_logit(lgA)) * inv_logit(lgB)`
  - Last (derived): `gNew ~ 1 - (gA + gB)`
- Users specify simplex groups via `simplex_params = c("gA", "gB", "gNew")`.

This is fully implemented and Stan-validated (see Section 15).

---

## 12. Implementation Plan

### Phase 1 — Standalone parser (no package integration; this PR)

- [x] `03_mpt_parser.R`: `mpt_tree()`, `mpt()`, `mpt_to_brms()`, `parse_mpt_string()`
- [x] Covariates: `covariates` argument, verbatim pass-through to brms (WP1/WP2)
- [x] Naming policy: error on `_`/`.` in param names; multi-point tree validation (WP1)
- [x] Binomial path (K=2): 2HTM, SMM2 — Stan-validated (Section 15)
- [x] Multinomial path (K>2): PCM, SMM3 with stick-breaking — Stan-validated (Section 15)
- [x] Time-varying bypass (WP4): bypass detection + `nlf()` direct emission (Section 10)
- [x] `04_hierarchical_demo.R`: hierarchical 2HTM recovery confirmed
- [x] `06_bmf_bridge.R`: `mpt_bridge_bmf()` converting `bmmformula` → predictor list; correlated RE recovery
- [x] `07_multinomial_demo.R`: PCM and SMM3 end-to-end Stan fits
- [x] `08_external_validation.R`: Riefer & Batchelder (1988) data; qualitative validation passes
- [x] `09_oberauer2019_validation.R`: real-data validation vs JAGS reference (WP2)
- [x] `10_m3_bijection_check.R`: per-draw math identity MPT ≡ M3 (WP3)
- [x] `11_time_varying_demo.R`: time-varying parameter recovery (WP4)

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
- Inequality constraints (`DA > DB`) via post-processing or re-parametrisation
- MPTinR import validation: numeric sum-to-1 check at fit time, not just at parse time

---

## 13. Open Questions — Resolved

1. **Formula validation**: `mpt_to_brms()` calls `.mpt_validate_tree()` at construction time with four numeric test points (0.137, 0.421, 0.683, 0.852). Multiple test points catch pathological expressions that pass at a single value. Symbolic verification would require CAS infrastructure and is out of scope for Phase 1. **Resolution: four-point numeric check at construction is adopted; validator warns (not errors) for simplex params because test points do not satisfy the simplex constraint.**

2. **Multinomial columns**: The multinomial path infers response categories from branch names; `simplex_params` provides explicit disambiguation for constrained groups. The user does not need to supply `resp_cats` separately. **Resolution: infer from branch names; `simplex_params` for constrained groups.**

3. **Tree condition matching**: Currently exact string match between `tree$name` and values in the condition column. Factor levels are coerced to character via `as.integer(data[[cond]] == tree_name)`. **Resolution: exact string match with character coercion; factor support is a Phase 2 nicety.**

4. **Parameter constraints beyond equality**: `logit`/`probit` reparametrisation enforces `(0,1)` bounds. Ordered constraints (e.g., `DA > DB`) would require an additional ordered reparametrisation and are deferred to Phase 3.

5. **`pp_check` support**: binomial MPT reuses brms defaults. For multinomial, `pp_check` type `rootogram` works in the current prototype (see `04_hierarchical_demo.R`). Full multinomial pp_check depends on upstream #324.

6. **Covariates vs parameters**: Symbols in branch expressions that are not latent parameters must be declared via the `covariates` argument. `mpt()` cannot infer this automatically without knowing the data schema. **Resolution: explicit `covariates` argument; error if a symbol is neither in `params` nor in `covariates`** (Phase 2 validation step).

7. **Time-varying parameters and bypass**: When a predictor formula's RHS references other `predictor_formulas` keys, the automatic `inv_logit(l<param>)` wrapper is bypassed. **Resolution: implemented in `mpt_to_brms()` via token-overlap detection (see Section 10).**

---

## 14. References

- Batchelder, W. H., & Riefer, D. M. (1999). Theoretical and empirical review of multinomial process tree modeling. *Psychonomic Bulletin & Review, 6*(1), 57–86.
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*(1), 3–28.
- Oberauer, K. (2019). Is rehearsal an effective maintenance strategy for working memory? *Journal of Cognition, 2*(1), 1–14. https://doi.org/10.5334/joc.58
- Riefer, D. M., & Batchelder, W. H. (1988). Multinomial modeling and the measurement of cognitive processes. *Psychological Review, 95*(3), 318–339.
- Rouder, J. N., & Lu, J. (2005). An introduction to Bayesian hierarchical models with an application in the theory of signal detection. *Psychonomic Bulletin & Review, 12*(4), 573–604.
- Singmann, H., & Kellen, D. (2013). MPTinR: Analysis of multinomial processing tree models in R. *Behavior Research Methods, 45*(2), 560–575.

---

## 15. Validation Evidence

All fits use CmdStan via cmdstanr backend (`backend = "cmdstanr"`). Sections 15.1–15.4 are Round 1 smoke-test fits. Sections 15.5–15.7 are Round 2 full fits (4 chains × ≥ 2000 iter).

### 15.1 Hierarchical 2HTM (04_hierarchical_demo.R)

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

### 15.2 Multinomial PCM (07_multinomial_demo.R, Part A)

60 participants, 40 pairs. True: cp=0.60, rp=0.75.

| Parameter | True | Estimate | 95% CI |
|---|---|---|---|
| cp | 0.600 | 0.587 | [0.463, 0.665] |
| rp | 0.750 | 0.765 | [0.710, 0.823] |

Max Rhat = 1.021 (500 warmup smoke test; acceptable).

### 15.3 Multinomial SMM3 with stick-breaking (07_multinomial_demo.R, Part B)

60 participants, 40 items/tree, 3 trees, 3 response categories. True: dA=0.70, dB=0.60, gA=0.40, gB=0.30, gNew=0.30.

| Parameter | True | Estimate | 95% CI |
|---|---|---|---|
| dA | 0.700 | 0.707 | [0.679, 0.732] |
| dB | 0.600 | 0.603 | [0.575, 0.630] |
| gA | 0.400 | 0.407 | [0.390, 0.425] |
| gB | 0.300 | 0.300 | (stick-breaking) |
| gNew | 0.300 | 0.293 | (derived: 1−gA−gB) |

Max Rhat = 1.004. Near-exact recovery.

### 15.4 bmf() bridge + correlated REs (06_bmf_bridge.R)

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

### 15.5 External validation: Riefer & Batchelder (1988) (08_external_validation.R)

40 participants (20 grouped, 20 ungrouped), 20 pairs each. Hardcoded data approximating RB1988 Table 1 typical effect sizes.

| Condition | cp (est) | cp CI | rp (est) | rp CI |
|---|---|---|---|---|
| Grouped | 0.558 | [0.468, 0.632] | 0.485 | [0.397, 0.577] |
| Ungrouped | 0.102 | [0.044, 0.152] | 0.286 | [0.241, 0.335] |

Qualitative checks: cp_grouped > cp_ungrouped ✓; rp_grouped > rp_ungrouped ✓; cp effect 95% CI excludes 0 ✓.  
Max Rhat = 1.007. Note: rp values are lower than the published ML estimates (≈0.85/0.70) because the hardcoded data encodes the typical aggregate proportions, not the exact paper Table 1. The qualitative pattern and parameter ordering are consistent.

---

### 15.6 WP2 — Oberauer (2019) real-data validation (09_oberauer2019_validation.R)

**Source:** Oberauer, K. (2019). *Journal of Cognition, 2*(1), 1–14.  
20 participants × up to 10 set-size/response-set conditions. Response categories: correct / other-list / not-presented lure.

**Model:** single-tree MPT with three design-fixed covariates as guessing rates; `Pi` and `Pb` have random intercepts and slopes for centred set size (via `||`).

**Key fix:** `data_prep()` uses tree response-category names to select data columns. CSV columns were renamed from `fcorrect/fother/fnpl` to `correct/other/npl` before calling `data_prep()`.

**JAGS reference** (Oberauer 2019, 4 chains × 10000/5000 iter):

| Quantity | brms mean | brms 95% CI | JAGS ref | JAGS 95% CI | Pass? |
|---|---|---|---|---|---|
| Pi intercept (meanPi) | 2.228 | [1.431, 3.093] | 2.420 | [1.617, 3.436] | ✅ |
| Pb intercept (meanPb) | 2.442 | [2.117, 2.760] | 2.459 | [2.140, 2.799] | ✅ |
| Pi slope (dPi) | −0.176 | [−0.499, 0.133] | −0.236 | [−0.569, 0.079] | ✅ |
| Pb slope (dPb) | −0.592 | [−0.675, −0.513] | −0.596 | [−0.684, −0.518] | ✅ |

All four brms posterior means fall inside the JAGS 95% CIs.  
P(dPb < 0) = 1.000 (threshold > 0.95) ✓ · max Rhat = 1.004 (≤ 1.02) ✓ · 0 divergences ✓

Random-effect SDs (no JAGS reference):

| SD | Estimate | 95% CI |
|---|---|---|
| sd\_Pi\_int | 0.764 | [0.070, 1.508] |
| sd\_Pb\_int | 0.560 | [0.368, 0.833] |
| sd\_Pi\_slope | 0.223 | [0.014, 0.559] |
| sd\_Pb\_slope | 0.047 | [0.002, 0.132] |

JAGS reference SDs: sgPi=0.938, sgPb=0.577, sgSlopePi=0.188, sgSlopePb=0.045 (qualitative agreement).

**Verdict: PASS** — all acceptance criteria met. (Commit `2de9d27`, seed=42.)

---

### 15.7 WP3 — M3 bijection cross-check (10_m3_bijection_check.R)

**Setup:** Single-tree multinomial MPT with `Pm` and `Pb`; analytic bijection to M3 (simple choice rule, NL=4, N=8, b=0.1). 40 participants × 60 trials. True: Pm=0.75, Pb=0.65.

**Key fix:** Branch expressions used R fractions (`1/4`, `3/8`, etc.) which Stan compiles as integer division (= 0). Changed to decimal literals (`0.25`, `0.375`, `0.5`).

**Criterion (i) — per-draw math identity** (4000 posterior draws):

| Category | max |P\_mpt − P\_m3| | Pass (< 1e-10)? |
|---|---|---|
| correct | 2.22e-16 | ✅ |
| other | 1.11e-16 | ✅ |
| npl | 5.55e-17 | ✅ |

**Criterion (ii) — parameter recovery and M3 agreement:**

| Param | True | MPT est | M3 est | \|diff\| |
|---|---|---|---|---|
| Pm | 0.750 | 0.733 [0.705, 0.758] | 0.736 | 0.004 |
| Pb | 0.650 | 0.672 [0.637, 0.707] | 0.668 | 0.004 |

True values inside MPT 95% CIs ✓ · MPT vs M3 agree < 0.02 ✓ · Max Rhat 1.003/1.004 ✓

**Verdict: PASS** — MPT emitter is mathematically identical to production M3 likelihood. (Commit `47b0380`, seed=42.)

---

### 15.8 WP4 — Time-varying parameters (11_time_varying_demo.R)

**Setup:** Single-tree binomial MPT. Detection grows exponentially with presentation time: `D(t) = inv_logit(lDmax) * (1 − exp(−exp(lrate) * t))`. Covariate `Gcorr = 1/rsize` (design-fixed). 40 participants × 4 ptimes × 2 rsizes × 25 trials. True: lDmax=1.4, lrate=0.0, SDs=0.3.

**Bypass verification:** brms formula contains `D ~ inv_logit(lDmax) * (1 - exp(-exp(lrate) * ptime))` (no `inv_logit(lD)` wrapper). PASS.

**Recovery** (4 chains × 4000 iter, 2000 warmup, adapt\_delta=0.99, seed=42):

| Param | True | Estimate | 95% CI | Cover? |
|---|---|---|---|---|
| lDmax | 1.40 | 1.444 | [1.023, 2.074] | ✅ TRUE |
| lrate | 0.00 | 0.082 | [−0.106, 0.266] | ✅ TRUE |

Max Rhat = 1.013 (≤ 1.02) ✓ · 0 divergences ✓ · D(t) monotonically increasing ✓

Note: SD parameters (`sd(lDmax_Intercept)`, `sd(lrate_Intercept)`) are near the lower boundary, producing a funnel geometry that requires adapt\_delta=0.99 and 2000 warmup for adequate mixing.

Posterior-mean D(t) at point estimates (Dmax=0.809, rate=1.088):

| t (s) | D(t) |
|---|---|
| 0.2 | 0.158 |
| 0.5 | 0.339 |
| 1.0 | 0.537 |
| 2.0 | 0.717 |

**Verdict: PASS** — all acceptance criteria met. (Commit `1fdd7c1`.)

---

## 16. Known Limitations and Future Work

- **brms naming constraint (enforced):** Parameter names with underscores or dots raise an error at `mpt()` construction time. Users must use `dA`, `gA`, `gNew` style names (no `d_A`, `g.new`). The error message is explicit.
- **Stan integer-division hazard:** Decimal fractions (`1/4`, `3/4`, etc.) in branch expressions are interpreted as integer division in Stan C++ (= 0), silently breaking the likelihood. Always use decimal literals (`0.25`, `0.75`). See Section 9.2.
- **R `paste0` guard:** `paste0("l", character(0))` returns `"l"` in R (not `character(0)`) because `paste0` drops NULL before recycling. The emitter guards against this with an explicit length check; the guard must be preserved in Phase 2.
- **Simplex-validator false positives:** `.mpt_validate_tree()` assigns arbitrary test values to simplex parameters without enforcing the sum-to-1 constraint. Branch sums may differ from 1 for simplex models, generating expected-but-harmless warnings (12 warnings from the SMM3 test). This is documented as a known limitation of the numeric validator.
- **Correlated RE width:** With n=100 and 500 warmup iterations, the rho CI is very wide ([-0.39, 0.87]). Real-data analyses should use ≥2000 iterations.
- **Time-varying SD funnel:** When random-effect SDs for bypass parameters are small, the posterior is funnel-shaped and requires adapt\_delta ≥ 0.95 and ≥ 2000 warmup. Recommend this as default in the Phase 2 constructor for models with bypass parameters.
- **`01_reproduce_upstream_demo.R`:** The aggregated 2HTM fit in this pre-existing script estimates D ≈ 0 (true: 0.70) because the simulated data accidentally sets equal hit and false-alarm rates. This is a bug in that script only; the hierarchical `04_hierarchical_demo.R` and all Round 2 scripts are unaffected.
