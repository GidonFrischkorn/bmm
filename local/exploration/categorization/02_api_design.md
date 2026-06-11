# GCM / Prototype Categorization Models — API Design Memo

**Issue**: #19  
**Date**: 2026-06-11  
**Status**: Proposal — not yet implemented in `R/` or `inst/`

---

## 1. Constructor design

**Recommendation: one constructor `gcm()` with a `version` argument.**

```r
gcm(
  resp_cat,                   # column name: observed category response (factor)
  dimensions,                 # character vector: MDS coordinate column names
  exemplars,                  # data.frame: exemplar set (see §2)
  similarity  = "exponential",# "exponential" | "gaussian"
  metric      = "euclidean",  # "euclidean" | "cityblock"
  version     = "exemplar",   # "exemplar" | "prototype" | "prm"
  links       = NULL,
  version     = NULL,
  ...
)
```

Class stack (following bmm convention):

```r
c("bmmodel", "categorization", "gcm", "gcm_exemplar")   # exemplar
c("bmmodel", "categorization", "gcm", "gcm_prototype")  # prototype
c("bmmodel", "categorization", "gcm", "gcm_prm")        # PRM
```

**Rationale for one constructor**: The three versions share the same data
interface (stimulus coordinates + exemplar reference set + categorical
response), the same attention simplex, and the same bmmformula predictor
variables. The primary difference is whether distances are computed to
individual exemplars or category prototypes, and whether a rote-memory
mixture is added. A single `gcm()` call with `version=` mirrors the existing
`m3()` and `sdt()` pattern in bmm.

**Against separate constructors**: The m3 has separate `version=` levels for
"ss"/"cs"/"custom"; SDT uses `sdt_binary()`, `sdt_rating()`, etc. only
because the data layout (binary vs. multi-row rating) changes the response
variable type substantially. Here the response layout is identical across
versions, making one constructor preferable.

**Parameters by version:**

| Parameter | exemplar | prototype | PRM | Link |
|---|---|---|---|---|
| `c` (sensitivity) | ✔ | ✔ | ✔ | log |
| `gamma` (response scaling) | ✔ | fixed = 1 | fixed = 1 | log |
| `w_1..w_{M-1}` (attention) | ✔ | ✔ | ✔ | softmax (see §4) |
| `b_1..b_{K-1}` (bias, optional) | optional | optional | optional | softmax |
| `p_mem` (rote-memory weight) | — | — | ✔ | logit |

---

## 2. Exemplar set passing

**Recommendation: `exemplars` argument as a plain `data.frame`.**

```r
# Minimal exemplar data.frame
exemplars <- data.frame(
  x1 = c(-2.543,  0.943, -1.092, ...),
  x2 = c( 2.641,  4.341,  1.848, ...),
  cat = c(1, 2, 2, ...)           # integer or factor, same levels as resp_cat
)

fit <- bmm(
  bmf(c ~ 1, w1 ~ condition),
  data    = trial_data,
  model   = gcm("response", dimensions = c("x1", "x2"),
                exemplars = exemplars)
)
```

**check_data responsibilities:**

1. Verify that every name in `dimensions` exists as a column in the trial data.
2. Verify that the exemplar `data.frame` has columns matching `dimensions` plus
   a category column with the same levels as `resp_cat`.
3. Verify category levels are consistent between `exemplars$cat` and
   `data[[resp_cat]]`.
4. Convert exemplar data.frame → a flat numeric matrix passed to Stan via
   `stanvars` as a data block array.
5. Warn if any trial-level `dimensions` column is missing (common mistake).

**Alternative rejected: deriving exemplars from a `training == TRUE` flag.**  
Adding a flag column to the trial data mixes the reference set (which is
constant across the experiment) with per-trial data, complicating both the
data interface and Stan data passing. A separate argument is cleaner and
mirrors the precedent in `imm()` (`nt_features` is a separate argument, not
a reshaping of the response column).

---

## 3. Response / data structure

**Recommendation: trial-level categorical response + optional aggregation helper.**

The primary interface is one row per trial with a factor response column:

```r
trial_data
# subject trial x1    x2    response
# 1       1     -2.54 2.64  A
# 1       2      0.94 4.34  B
# ...
```

**Rationale**: GCM coordinates are typically unique per stimulus (MDS
solutions), so exact stimulus repetition across trials is uncommon. Aggregated
formats (like m3's binomial counts per stimulus) save computation only when the
same stimulus repeats many times with identical coordinates. Providing trial-
level data as primary keeps `check_data` simple and avoids ambiguity.

**Aggregation case**: If the design uses a small fixed stimulus set (e.g.
nosof88's 12 chips repeated), an aggregated path can be supported by accepting
a `trials` column (count of presentations per row), analogous to `trials()` in
brms. Document this as optional.

---

## 4. Attention simplex in bmmformula

**Recommendation: softmax-with-reference parameterization, one `w` parameter
per non-reference dimension.**

For M = 3 dimensions, bmm internally creates parameters `w1`, `w2`; the third
dimension's weight is derived as `1 - softmax(w1, w2)[1] - softmax(w1, w2)[2]`.
On the identity/log-odds scale this is the standard softmax-with-reference:

```r
# User writes (M=3 example):
bmf(c ~ 1, gamma ~ 1, w1 ~ condition, w2 ~ condition)
```

Users predict the unconstrained `w` parameters directly. Under the hood:

```
softmax([w1_tilde, w2_tilde, 0]) = [exp(w1_tilde), exp(w2_tilde), 1] / sum(...)
```

where `w_M_tilde = 0` is the fixed reference (the last dimension). This
mirrors how the m3 handles activations and avoids the `Dirichlet`-in-brms
complications.

The default prior on intercepts is `Normal(0, 1)` on the log-odds scale
(weakly informative: roughly uniform over [0.05, 0.95] for each weight in a
2-D space).

**Documentation convention**: The user manual must state which dimension is the
reference (the last dimension, index M) and explain that the prior is on the
softmax-transformed scale. This is non-obvious and the precedent is in the m3
vignette.

---

## 5. Coordinate scaling

**Recommendation: pass coordinates as-is; document the dependency on units;
provide a utility function `gcm_scale_coords()`.**

GCM sensitivity `c` is scale-dependent (Wills & Pothos, 2012). Normalizing
internally would make `c` interpretable only as "sensitivity given our
normalization scheme," not as the parameter in the published literature. The
preferred approach is:

1. Users supply coordinates in their natural scale (MDS solution units, or raw
   feature values).
2. The model documentation warns that `c` is in units of "per unit of
   coordinate distance."
3. A helper `gcm_scale_coords(x, center = TRUE, scale = TRUE)` is provided for
   convenience; it is not called automatically.

This matches how brms handles continuous predictors: scaling is the analyst's
responsibility, and the documentation explains the implications.

---

## 6. Stan implementation route

**Recommendation: custom Stan function over pre-computed distance array.**

Two options were evaluated:

| Option | Approach | Pro | Con |
|---|---|---|---|
| A | Precompute `D[T, J, M] = |x_tm - e_jm|^r` as a data array; smooth gradient over `w` and `c` via `sum(w .* D[t, j, ])^(1/r)` | Clean separation of data preprocessing from model | Large data array for many exemplars × dimensions |
| B | brms non-linear formula (`nlf`) calling a user Stan function | Consistent with existing bmm nlf pattern | Requires custom brms family; nlf cannot loop over exemplars without a custom function anyway |

**Option A (custom family)** is preferred. The distance array `D[T, J, M]` can
be computed in R's `configure_model` step (or in a `data {}` block), and the
Stan `model {}` block implements only the similarity and Luce-choice
aggregation. This is analogous to how the m3 pre-aggregates response counts.

The log-likelihood on the log scale:

```stan
// In generated code, for each test item t:
for (t in 1:T) {
  vector[K] log_act;
  for (k in 1:K) {
    real sum_sim = 0;
    for (j in 1:J_k[k]) {
      real d = pow(dot_product(w, D[t, ex_of_cat_k[k, j]]), inv_r);
      sum_sim += exp(-c * pow(d, p_sim));
    }
    log_act[k] = log(sum_sim);
  }
  // Luce choice with gamma and optional bias
  log_act = gamma * log_act + log(bias);
  target += log_act[y[t]] - log_sum_exp(log_act);
}
```

**Benchmark scope** (task #3): compare per-evaluation cost vs m3 for
~90 exemplars × 8 dims × 150 test items (Nosofsky 2022 rocks scale).

---

## 7. `check_data` / `configure_model` sketch

```r
check_data.gcm <- function(model, data, formula, ...) {
  dims   <- model$other_vars$dimensions
  ex_df  <- model$other_vars$exemplars
  resp   <- model$resp_vars$resp_cat

  # 1. response column present and factor
  stopif(!resp %in% names(data),
         "Response column '{resp}' not found in data.")
  data[[resp]] <- as.factor(data[[resp]])

  # 2. dimension columns present in data
  missing_dims <- setdiff(dims, names(data))
  stopif(length(missing_dims) > 0,
         "Dimension column(s) {collapse_comma(missing_dims)} not found in data.")

  # 3. exemplar category levels match response levels
  resp_levels <- levels(data[[resp]])
  ex_cats <- levels(as.factor(ex_df$cat))
  stopif(!setequal(resp_levels, ex_cats),
         "Category levels differ between data and exemplars.")

  # 4. integer encode response and exemplar categories
  data[[resp]] <- as.integer(data[[resp]])
  attr(data, "gcm_exemplars_mat") <- as.matrix(ex_df[, dims])
  attr(data, "gcm_exemplar_cats") <- as.integer(as.factor(ex_df$cat))
  attr(data, "gcm_K")             <- length(resp_levels)
  attr(data, "gcm_J")             <- nrow(ex_df)
  attr(data, "gcm_M")             <- length(dims)
  data
}

configure_model.gcm <- function(model, data, formula, ...) {
  # Extract precomputed attributes from check_data
  ex_mat  <- attr(data, "gcm_exemplars_mat")
  ex_cats <- attr(data, "gcm_exemplar_cats")
  K <- attr(data, "gcm_K")
  J <- attr(data, "gcm_J")
  M <- attr(data, "gcm_M")

  test_mat <- as.matrix(data[, model$other_vars$dimensions])
  T_items  <- nrow(test_mat)
  r_metric <- if (model$other_vars$metric == "euclidean") 2 else 1

  # Pre-compute |x_tm - e_jm|^r (data block in Stan)
  D_array <- array(0, dim = c(T_items, J, M))
  for (m in seq_len(M)) {
    D_array[, , m] <- abs(outer(test_mat[, m], ex_mat[, m], `-`))^r_metric
  }

  stanvars <- brms::stanvar(D_array, name = "D_arr", scode = "data",
                             block = "data") +
              brms::stanvar(ex_cats, name = "ex_cat", scode = "int[J]",
                             block = "data") +
              brms::stanvar(as.integer(K), name = "K", block = "data") +
              brms::stanvar(as.integer(J), name = "J", block = "data") +
              brms::stanvar(as.integer(M), name = "M", block = "data")

  # ... (formula, family, prior construction follow bmm pattern)
  nlist(formula, data, family, prior, stanvars)
}
```

---

## 8. Open questions for the upstream proposal

1. **Multi-subject hierarchical GCM**: `c` and `gamma` on log links are
   straightforward random effects. Attention weights `w` require a multivariate
   softmax normal — is the Dirichlet-multinomial an acceptable alternative for
   the population-level prior?
2. **PRM with unknown old/new status**: The rote-memory component requires
   identifying which test items appeared in training. If this flag is not
   available, the PRM reduces to the prototype. Should a `training_flag` column
   be required or optional?
3. **Stimulus coordinates from data vs. MDS**: Many GCM applications use
   coordinates from a separately-run MDS analysis. Should bmm support
   loading coordinates from a fitted MDS object (e.g., `smacof::smacofSym`),
   or is the current "columns in data" interface sufficient?
4. **Category prototype re-use across subjects**: In multi-subject designs,
   the prototype is the same for all subjects (fixed from the exemplar set).
   Make this explicit in `check_data` by computing prototypes once and storing
   in `stanvars`, not per-row in data.
