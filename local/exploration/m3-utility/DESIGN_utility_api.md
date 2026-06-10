# Design Memo: Utility-Theory API for bmm M3

**Date:** 2026-06-10  
**Branch:** feat/issue-7-m3-utility-exploration  
**Author:** thom-more (agent run, round 3)  
**Status:** Exploration — not yet merged to R/

---

## Background

The WP5 spike (`10_bmm_api_example.R`) and the WP2 wrapper prototype
(`11_utility_wrapper.R`) demonstrate that an EU-weighted M3 is achievable
with the current `m3()` API. Three sharp edges were identified and resolved:

1. **`model$links` mutation** — `m3()` already accepts `links = list(...)` directly
   (R/model_m3.R line 90). No post-construction mutation needed.
2. **No default prior for gamma** — must be user-supplied; wrapper adds sensible defaults.
3. **Predictor-vs-activation trap** — adding `V_corr` via `a ~ 1 + V_corr` gives the
   same slope for all categories sharing `a`, which is not the EU model. Correct
   approach is explicit per-category activation formulas via `version = "custom"`.

This memo compares three integration paths and recommends one.

---

## Three Integration Options

### Option A — Exported wrapper constructor (`m3_utility()`)

**What it is:** A user-facing function `m3_utility()` in `R/model_m3.R` (or a new
`R/model_m3_utility.R`) that builds and returns a correctly configured m3 object with
default priors and activation-formula helpers.

```r
model <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",   # or "power"
  weighting   = "none"      # or "prelec"
)
formula <- do.call(bmf, c(
  m3_utility_formula(model),
  list(a ~ 1 + (1|subj), c ~ 1 + (1|subj), gamma ~ 1 + (1|subj))
))
```

**Pros:**
- Discoverability: `?m3_utility` is searchable; users never need to read the source
- Removes all three sharp edges at the point of entry
- Can validate inputs at construction time (e.g., warn if value_cols is NULL but
  utility != "none")
- Natural place for the RUM/utility vocabulary from WP3 (argument names match
  econometric conventions)

**Cons:**
- New exported function → must be documented, tested, listed in `_pkgdown.yml`
- Maintenance cost: if `.m3_version_table` changes, wrapper may need updating
- Two entry points (`m3()` and `m3_utility()`) for what is structurally the same model
- "Custom" formulas still required from user; formula generation is additive complexity

**Maintenance cost:** Medium. Thin wrapper over `m3()` — less than 60 lines of R.
If `m3()` internals change, the wrapper needs one review pass.

---

### Option B — New version entry (`version = "utility"`)

**What it is:** Add a new entry to `.m3_version_table` in `R/model_m3.R`:

```r
.m3_version_table[["utility"]] <- list(
  parameters = list(
    a     = "General activation ...",
    c     = "Context activation ...",
    gamma = "Value-weighting slope (EU activation)"
  ),
  links = list(
    softmax = list(a = "identity", c = "identity", gamma = "identity")
  ),
  priors = list(
    softmax = list(
      a     = list(main = "normal(2,1)",  effects = "normal(0,0.5)"),
      c     = list(main = "normal(3,1)",  effects = "normal(0,2)"),
      gamma = list(main = "normal(0,1)",  effects = "normal(0,0.3)")
    )
  )
)
```

Usage:

```r
model <- m3(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c("n_corr", "n_other", "n_npl"),
  choice_rule = "softmax",
  version     = "utility"
)
```

**Pros:**
- Minimal new surface area: `version = "utility"` is discoverable via `?m3`
- Consistent with existing `version = "ss"` and `version = "cs"` pattern
- Default priors and links are set once in the version table
- No new exported functions

**Cons:**
- `version` conflates "model structure" (ss/cs) with "estimation approach"
  (utility) — semantically muddled. A user asking "which version should I use?"
  is not necessarily asking about utility theory.
- Power utility (rho) and Prelec (alpha) would require separate version entries
  (`version = "utility_power"`, `version = "utility_prelec"`), inflating the table.
- Formula generation for per-category value terms is still manual (the version table
  specifies parameters, not activation formulas).
- Activation formula trap (shared slope via `a ~` vs per-category) is not resolved —
  users still need `version = "custom"` for the EU formula structure.

**Maintenance cost:** Low for the table entry itself. Medium overall because
the activation formula issue is unresolved and will surface as user confusion.

---

### Option C — Documentation-only (vignette recipe)

**What it is:** Add a section to the m3 vignette showing the manual recipe:

```r
# EU-weighted VDR analysis (no new functions needed)
model <- m3(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c("n_corr", "n_other", "n_npl"),
  choice_rule = "softmax",
  version     = "custom",
  links       = list(a = "identity", c = "identity", gamma = "identity"),
  default_priors = list(gamma = list(main = "normal(0,1)", effects = "normal(0,0.3)"))
)
formula <- bmf(
  corr  ~ b + a + c + gamma * V_corr,
  other ~ b + a     + gamma * V_other,
  npl   ~ b,
  a     ~ 1 + (1 | subj),
  c     ~ 1 + (1 | subj),
  gamma ~ 1 + (1 | subj)
)
```

**Pros:**
- Zero maintenance cost: no new code
- Forces users to understand the activation formula structure
- Compatible with all current and future `m3()` versions
- Script `10_bmm_api_example.R` is already the seed for this vignette

**Cons:**
- Activation formula trap is not prevented; it is only documented
- Users who skip the vignette will make the shared-slope mistake silently
- No default priors unless user reads closely
- Discoverability: only findable via vignette, not `?m3_utility`

**Maintenance cost:** Near-zero. Risk: documentation diverges from code if m3
API changes.

---

## Recommendation: Option A (exported wrapper)

**Verdict: Option A, with a thin implementation.**

**Reasoning:**

1. **The activation formula trap is the most dangerous sharp edge.** It produces
   silently wrong models (shared slope instead of per-category EU activation).
   Option C documents it; Options A and B resolve it. Option A resolves it at
   the entry point, before the user writes a formula.

2. **Option B does not resolve the formula trap.** The version table specifies
   parameters but not activation formulas. A `version = "utility"` user still
   needs to write the `version = "custom"` formula structure, making the version
   entry misleading.

3. **Maintenance cost of Option A is acceptably low.** The wrapper delegates all
   model construction to `m3()` (confirmed in `11_utility_wrapper.R`) and is fewer
   than 100 lines. The activation formula generator (`m3_utility_formula()`) is
   deterministic from inputs; it does not depend on `m3()` internals.

4. **Discoverability favors A.** `?m3_utility` is the natural entry point for a
   user who has read about value-directed recall and wants a Bayesian utility model.
   `version = "utility"` requires knowing to look at the `m3()` help page first.

5. **Option A composes with Options B and C.** The wrapper can be documented in the
   vignette (Option C), and its internals can use a future `version = "utility"` table
   entry (Option B) once the formula generation question is resolved.

**Deferred:** Power utility (rho) and Prelec (alpha) variants are implemented in
the `11_utility_wrapper.R` prototype but should not be exported until their
identifiability conditions are documented in the help page (≥3 value levels for rho;
variable set sizes for alpha).

---

## WP4 Structural Decision: Binary Utility Models

### Question

In M3, utility comes from **memory parameters** (activation `a`, context `c`).
In canonical lottery choice tasks, utility comes from **option attributes**
(outcome $x$ and probability $p$ per option). These are structurally different:

- **M3 utility**: `gamma * V_i` is added to the *category's* activation, which
  already encodes how well items of that type are remembered.
- **Binary PT**: `u(x) = x^rho` is applied to the *option's outcome attribute*,
  independent of any memory process.

### Decision

**Binary utility models are a sibling family, not an extension of m3_utility.**

Reasons:

1. **Different generative processes.** M3 assumes a memory retrieval process
   (activation → choice). Binary PT assumes direct preference (utility → choice).
   Folding binary PT into `m3_utility()` would conflate the two processes.

2. **Different formula structure.** M3 utility adds `gamma * V` to the activation
   formula shared across categories. Binary PT requires per-option utility formulas
   (`u_A ~ rho * log(x_A)`, `u_B ~ rho * log(x_B)`) that are category-specific.
   The wrapper's formula generator would need to be redesigned for this case.

3. **The choice-rule layer is shared but not the constructor.** Both M3 and binary
   PT use softmax/logit choice rules. They can share the choice-rule layer
   (`choice_rule = "softmax"` in `m3()`), but the model constructor and formula
   logic are separate.

4. **Practical scope.** bmm's primary domain is working memory (WM) measurement.
   Binary lottery choice is a separate literature (JDM/behavioral economics). A
   `binary_pt()` model constructor — if added to bmm — would live alongside `m3()`,
   not inside it.

**Implication for m3_utility():** The wrapper handles M3-style utility only.
Per-option utility formulas (for binary or multi-attribute choice) are out of scope.
Script `12_binary_choice_pt.R` demonstrates binary PT as an independent prototype,
not as an m3_utility() variant.

---

## Summary

| Option | Formula trap resolved? | Discoverability | Maintenance | Recommended |
|---|---|---|---|---|
| A — `m3_utility()` wrapper | ✅ Yes | ✅ High | Medium | **YES** |
| B — `version = "utility"` | ❌ No | Medium | Low | No |
| C — Documentation only | ❌ No | Low | Minimal | No |

**Binary PT:** Sibling family (`binary_pt()`), not an m3_utility() variant. Shares
the choice-rule layer; has its own model constructor.

**Next step:** Move `m3_utility()` and `m3_utility_formula()` from
`local/exploration/m3-utility/11_utility_wrapper.R` to `R/model_m3_utility.R`,
add roxygen documentation, tests in `tests/testthat/test-model_m3_utility.R`,
and a vignette section. This is a separate PR from the exploration.
