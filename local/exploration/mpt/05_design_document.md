# MPT Models in bmm — Design Document

> **Status:** Prototype (exploration branch)  
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
3. **`mpt_to_brms()`** — core formula emitter. Builds the aggregated probability expression for each response category from tree-indicator columns, wraps in `logit()`, adds `nlf()` reparametrisations and `lf()` predictor formulas.
4. **`mpt_from_string()`** — MPTinR-format parser. Tokenises branch equations, maps them to `mpt_tree` objects.
5. **`check_data.mpt`** — adds tree-indicator columns to data; converts response columns to matrix if multinomial.
6. **`check_formula.mpt`** — verifies that every model parameter declared in the trees has a predictor formula in the `bmf()`.

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

The hierarchical demo (`04_hierarchical_demo.R`) confirms recovery of population means and SDs with 80 simulated participants.

---

## 9. Multinomial Extension (>2 Response Categories)

For models with 3+ response categories (e.g., source monitoring: responded A / B / New):

- Use `brms::multinomial(refcat = NA)` family.
- The main formula becomes `Y | trials(nTrials) ~ ...` with `Y` as a response matrix.
- The `mpt_to_brms()` function must emit formulas for **each non-reference category**:
  ```
  mu_A   ~ logit(Σ_t I_t * P_t(A))
  mu_B   ~ logit(Σ_t I_t * P_t(B))
  ```
- The current prototype handles binomial (2 categories); multinomial extension is a straightforward loop.

---

## 10. Implementation Plan

### Phase 1 — Standalone parser (no package integration; this PR)

- [x] `03_mpt_parser.R`: `mpt_tree()`, `mpt()`, `mpt_to_brms()`, `parse_mpt_string()`  
- [x] Tests on 2HTM, Source Monitoring, Pair-Clustering  
- [x] `04_hierarchical_demo.R`: recovery confirmed with 80 participants  

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

### Phase 3 — Multinomial MPT (separate PR)

- Extend `mpt_to_brms()` to emit per-category `nlf()` formulas  
- Switch from `binomial()` to `multinomial(refcat = NA)` when `length(resp_cats) > 2`  
- Add source monitoring demo  

---

## 11. Open Questions

1. **Formula validation**: How should bmm warn when branch probabilities don't sum to 1? Currently a numeric check at a test point; symbolic verification would be stronger but hard to implement generally.

2. **Multinomial columns**: Should the user pass `resp_cats = c("A", "B", "New")` (like M3's `resp_cats`) or let the parser infer them from branch names? Inference is more ergonomic but less explicit.

3. **Tree condition matching**: Currently exact string match. Should we support factor levels or numeric codes?

4. **Parameter constraints beyond equality**: Some MPT models have inequality constraints (`0 < D < 1`) — these are satisfied by the logit reparametrisation. But ordered-parameter constraints (e.g., `D_A > D_B`) require more thought.

5. **`pp_check` support**: binomial MPT can reuse brms defaults. For multinomial, we need the multinomial pp_check from upstream #324.

---

## 12. References

- Batchelder, W. H., & Riefer, D. M. (1999). Theoretical and empirical review of multinomial process tree modeling. *Psychonomic Bulletin & Review, 6*(1), 57–86.
- Johnson, M. K., Hashtroudi, S., & Lindsay, D. S. (1993). Source monitoring. *Psychological Bulletin, 114*(1), 3–28.
- Rouder, J. N., & Lu, J. (2005). An introduction to Bayesian hierarchical models with an application in the theory of signal detection. *Psychonomic Bulletin & Review, 12*(4), 573–604.
- Singmann, H., & Kellen, D. (2013). MPTinR: Analysis of multinomial processing tree models in R. *Behavior Research Methods, 45*(2), 560–575.
- Oberauer, K., & Lewandowsky, S. (2019). Simple measurement models for complex working-memory tasks. *Psychological Review, 126*.
