# Design Memo: Utility-Theory API for bmm M3

**Date:** 2026-06-11 (revised round 7: R9/R10 corrected; round 6 original)  
**Branch:** feat/issue-7-m3-utility-exploration  
**Author:** thom-more (agent run, rounds 3, 5, and 7)  
**Status:** Exploration — not yet merged to R/

---

## 1. Background

The WP5 spike (`10_bmm_api_example.R`) and the WP2 wrapper prototype
(`11_utility_wrapper.R`) demonstrate that an EU-weighted M3 is achievable
with the current `m3()` API. Three sharp edges were identified and resolved
in round 3:

1. **`model$links` mutation** — `m3()` already accepts `links = list(...)` directly
   (R/model_m3.R line 90). No post-construction mutation needed.
2. **No default prior for gamma** — must be user-supplied; wrapper adds sensible defaults.
3. **Predictor-vs-activation trap** — adding `V_corr` via `a ~ 1 + V_corr` gives the
   same slope for all categories sharing `a`, which is not the EU model. Correct
   approach is explicit per-category activation formulas via `version = "custom"`.

Round 5 added a requirements inventory (`14_requirements_inventory.md`) and user-surface
analysis (`15_user_surface.R`, `16_identifiability_guards.R`) that sharpen the
architecture comparison. The round 3 Option A recommendation is revised below.

---

## 2. Three Integration Options

### Option A — Exported wrapper constructor (`m3_utility()`)

**What it is:** A user-facing function `m3_utility()` in `R/model_m3_utility.R`
that builds and returns a correctly configured m3 object with default priors and
activation-formula helpers.

```r
model <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "none"
)
formula <- do.call(bmf, c(
  m3_utility_formula(model),
  list(a ~ 1 + (1|subj), c ~ 1 + (1|subj), gamma ~ 1 + (1|subj))
))
```

**Round 5 addition:** The exploration wrapper (local/exploration/) returns an object of
class `c("bmmodel", "m3", "m3_custom")`. In **production** (in R/), `m3_utility()` must
return `c("bmmodel", "m3", "m3_custom", "m3_utility")` and define `check_data.m3_utility`
and `check_formula.m3_utility` S3 methods to enable auto-dispatched identifiability guards.
Without the subclass, guards fire only via manual `check_utility_design(model, data)` calls.

**Pros:**
- Resolves activation formula trap (R1) at entry point via `m3_utility_formula()`
- Resolves default prior issue (R2) and link issue (R3) at construction
- With subclass in production: resolves identifiability guards (R4) automatically
- Discoverability: `?m3_utility` is searchable
- Natural place for RUM/utility vocabulary (argument names)
- Thin: delegates all model construction to `m3()`; fewer than 120 lines of R

**Cons:**
- Still requires user to call `do.call(bmf, c(m3_utility_formula(model), ...))`
  — activation formulas are generated but not assembled automatically
- Binary PT (R5) is out of scope: type = "memory" only
- New exported function → must be documented, tested

---

### Option B — New version entry (`version = "utility"`)

**What it is:** Add a new entry to `.m3_version_table` in `R/model_m3.R`.

```r
model <- m3(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c("n_corr", "n_other", "n_npl"),
  choice_rule = "softmax",
  version     = "utility"   # does not exist — runtime error today
)
```

**Round 5 finding:** This option is MORE invasive than it appears.
`construct_m3_act_funs()` (R/model_m3.R line 462+) only generates activation formulas
for versions "ss" and "cs" and explicitly stops for any other version:
```r
stopif(
  !inherits(model, "m3") || !model$version %in% c("ss", "cs"),
  'Activation functions can only be generated for "m3" models "ss" and "cs"'
)
```
Adding `version = "utility"` to the version table requires also updating
`construct_m3_act_funs()` — otherwise the user still needs `version = "custom"` and
the activation formula trap is not resolved. Option B therefore has the same formula
generation problem as the documentation-only Option C.

**Pros:**
- Minimal new surface area (consistent with existing ss/cs pattern)
- Default priors and links from table

**Cons:**
- Activation formula trap is NOT resolved unless `construct_m3_act_funs()` is updated
- Power and Prelec variants would need separate version entries
- Identifiability guards require the same subclassing work as Option A

---

### Option C — Documentation-only (vignette recipe)

**What it is:** A section in the m3 vignette showing the manual recipe
(scripts 10–12 as seed).

**Pros:** Zero maintenance cost; zero new code.

**Cons:** Activation formula trap and identifiability failures are documented but
not prevented. Both are silent-wrong in typical use.

---

## 3. Requirements × Architecture Coverage Matrix

Based on the requirements in `14_requirements_inventory.md` (R8 from round 6; R9/R10
corrected in round 7 — see §3 note below).

**Severity codes:** SW = silent-wrong (worst), RE = runtime error, F = friction, ✅ = resolved

| Requirement | Severity | Option A (exploration) | Option A (production) | Option B | Option C |
|---|---|---|---|---|---|
| R1 Per-category formula trap | SW | ✅ via `m3_utility_formula()` | ✅ via `m3_utility_formula()` | ❌ not resolved¹ | ❌ documented only |
| R2 Default priors for γ/ρ/α | F | ✅ via `default_priors=` arg | ✅ | ✅ via version table | ❌ user-supplied |
| R3 Identity links (undocumented) | RE | ✅ via `links=` arg (wrapped) | ✅ | ✅ via version table | F: documented |
| R4 Identifiability guards | SW | ❌ manual call only² | ✅ `check_data.m3_utility` | ~ requires subclass | ❌ documented only |
| R5 Binary PT (attribute data) | RE | ❌ out of scope | ❌ out of scope | ❌ out of scope | ❌ raw brms only |
| R6 Utility × weighting composition | F | ✅ `utility=` + `weighting=` args | ✅ | ❌ requires multiple version entries | ❌ manual formula |
| R7 Parameter labelling | F | F: brms NL names | ~ `postprocess_brm.m3_utility` | F: brms NL names | F: brms NL names |
| R8 Fixed payoff coefficients (welfare) | SW | ❌ out of scope | ❌ out of scope | ❌ out of scope | ❌ raw brms only |
| R9 Positivity hazard (identity link + simple rule) | RE | ✅⁴ lower-bounded priors | ✅⁴ + guard G4 | ✅⁴ | ✅⁴ |
| R10 Numeraire fixing (docs gap only) | F | ✅ `fixed_parameters$b` works (helpers-prior.R:147) | ✅ auto-set by constructor | ✅ via version table | F: documented |

¹ Updating `construct_m3_act_funs()` for "utility" version would be required;
  this is as much work as Option A production.
² `check_utility_design(model, data)` from `16_identifiability_guards.R` works
  as a manual pre-flight validator but is not auto-dispatched.
⁴ R9 is resolved in ALL architectures by lower-bounded priors (`T[0,]` or `lb=0`).
  In production, guard G4 (`16_identifiability_guards.R`) makes the required bound
  explicit. R9 is NOT a structural barrier — see §3 note below.

**Round 7 correction to R9 and R10.**  Round 6 mischaracterized both requirements.
- **R9 (corrected):** The claim "Luce/simple rule with non-linear utility requires raw brms"
  was wrong. `glue_choice_rule_functions()` (R/model_m3.R:399–404) generates
  `log({cat} * n_options)` for the simple rule, which is natively the Luce ratio rule
  for n_options=1. `22_welfare_bmm_simple.R` confirms wi=1.2, wo=0.8 recovery using
  only `m3(choice_rule="simple")` — no raw brms. The real hazard is positivity:
  identity-linked parameters can produce negative activations → `log(negative)` in Stan.
  Guard G4 in `16_identifiability_guards.R` addresses this.
- **R10 (corrected):** The claim that `fixed_parameters$b = 1.0` "conflicts semantically
  with M3's background noise" was factually wrong. `b` in `fixed_parameters` IS the
  formula parameter `b` that appears in activation formulas. Setting it to 1.0 is the
  correct and supported numeraire mechanism: `fixed_pars_priors()` (R/helpers-prior.R:147)
  converts it to `constant(1)` automatically; `update_model_fixed_parameters()`
  (R/helpers-model.R:165) syncs it at check_model time. R10 is a documentation gap,
  not a correctness problem. Severity downgraded from SW to F.

**Key finding from round 5:** The critical distinction between Option A exploration and
Option A production is whether `m3_utility()` adds `"m3_utility"` to the class vector
and whether `check_data.m3_utility` is in the bmm namespace. This is a small but
structurally important difference — adding ~30 lines of code to R/model_m3_utility.R
turns the exploration prototype into a production-grade guard.

**What Option C (first-class sibling) gets for free that A cannot:**
- R5: binary PT as a first-class model family with correct data shape
- Unified formula generation (user writes only hyperparameter formulas; activation
  formulas are generated internally by `check_formula.utility`)
- Full postprocessing for clean parameter labels

**Cost of Option C:** Full S3 method set (~7 methods), test coverage, documentation,
CRAN surface expansion, binary PT data shape diverges from M3 pipeline. Estimate:
~400–600 lines of R plus tests.

---

## 4. User-Surface Comparison

Full code in `15_user_surface.R`. Summary:

| Use case | Architecture A lines | Architecture C lines | Footgun A | Footgun C |
|---|---|---|---|---|
| UC1: EU VDR (gamma) | 18 (model+formula+call) | 12 (model+call; formula internal) | R4 silent-wrong (exploration); resolved in production | Guard fires |
| UC2: Composite PT | 22 | 15 | R4 silent-wrong (exploration); resolved in production | Guard fires |
| UC3: Binary PT | not expressible | 15 (`type="attribute"`) | runtime error | guard fires |

**Architecture A footgun (power utility + 2 value levels):**
- Exploration: `m3_utility()` succeeds; `bmm()` runs; rho has flat posterior — **silent-wrong**
- Production (with check_data.m3_utility): `bmm()` errors with clear message at `check_data`
- Architecture C: `utility()` errors at `check_data.utility_memory` — same quality

**Formula generation:**
- Architecture A: user calls `do.call(bmf, c(m3_utility_formula(model), list(...)))`
- Architecture C: user writes only `bmf(a ~ 1 + (1|subj), c ~ 1 + (1|subj), gamma ~ 1 + (1|subj))`
  — activation formulas are generated internally by `check_formula.utility`

Architecture C reduces user code by ~25% and eliminates the `do.call(bmf, c(...))` pattern,
which is a minor footgun (easy to forget the activation formulas entirely).

---

## 5. Guard Placement Analysis

From `16_identifiability_guards.R`:

Three identifiability guards were prototyped and tested:
- G1: Power utility (rho) needs ≥ 3 distinct value levels → `check_power_utility_ident()`
- G2: Prelec weighting (alpha) needs variable set size → `check_prelec_ident()`
- G3: EU slope (gamma) needs value variation → `check_gamma_ident()`

All three guards fire correctly on deficient datasets and pass on adequate datasets
(confirmed in `16_identifiability_guards.R` Sections 2–3).

**Where the guards live:**

| Architecture | Guard placement | Auto-dispatched? |
|---|---|---|
| A (exploration) | Standalone `check_utility_design(model, data)` | No — manual call |
| A (production) | `check_data.m3_utility()` in `R/model_m3_utility.R` | Yes, via S3 dispatch |
| B | Same as A-production (requires subclass addition) | Yes (if subclass added) |
| C | `check_data.utility_memory()` in `R/model_utility.R` | Yes, built-in |

**Critical structural finding:** Guards that require access to `data` (which is only
available at the `check_data` stage) CANNOT fire automatically in any architecture
unless the model object's class includes a custom subclass dispatched in that method.
The exploration wrapper returns an `m3_custom` object — `check_data.m3_utility` would
never be dispatched unless:
1. `m3_utility()` adds `"m3_utility"` to the class vector, AND
2. `check_data.m3_utility` is exported from the bmm package.

This is the single most important design change needed to move from exploration to
production quality.

---

## 6. Revised Recommendation

### Decision

**Revised verdict: Option A production (staged) — with `"m3_utility"` subclass.**

The round 3 recommendation (Option A, thin wrapper) was based on the assumption that
the wrapper alone resolves the critical issues. Round 5 shows this is only half true:

- **Construction-time issues** (R1, R2, R3, R6): ✅ resolved by wrapper
- **Data-time issues** (R4 identifiability guards): ❌ NOT resolved by exploration wrapper;
  require production subclass

The revised recommendation is a **staged path**:

**Stage 1 (near-term, M3 utility extension):**
Move `m3_utility()` to `R/model_m3_utility.R` with:
- Class: `c("bmmodel", "m3", "m3_custom", "m3_utility")`
- `check_data.m3_utility`: identifiability guards G1–G4 (from `16_identifiability_guards.R`)
  - G1: range criterion for power utility (max(V)/min(V) ≥ 3), warning not error
  - G2: variable set sizes required for Prelec, error
  - G3: value variation required for EU slope, error
  - G4: positivity hazard for identity-linked utility with simple rule, warning
- `m3_utility_formula()`: per-category activation formula generator
- Default priors: lower-bounded (`lb=0`) for identity-linked utility params
- `fixed_parameters$b` auto-set (numeraire) — no user action needed

This resolves R1–R4, R6, R7, R9 (via G4). R10 resolved by auto-setting `fixed_parameters$b`.
R8 remains out of scope for Stage 1. Estimated: ~180 lines of R + tests + 1 vignette section.

**Stage 2 (later, binary PT extension):**
Add `binary_pt()` as a separate sibling constructor in `R/model_binary_pt.R`:
- Class: `c("bmmodel", "binary_pt")`
- `check_data.binary_pt`: validates per-option attribute data
- `configure_model.binary_pt`: Bernoulli/logit family

This resolves R5. Estimated: ~200 lines of R + tests + 1 vignette section.

### What would change the recommendation

**Toward full sibling (Option C):**
- If user research shows the `do.call(bmf, c(m3_utility_formula(model), list(...)))`
  pattern is a significant barrier (more than 20% of utility-model issues are activation
  formula mistakes that `m3_utility_formula()` doesn't prevent)
- If binary PT and M3 utility need to share postprocessing, comparison tools, or
  vignette structure that benefits from a common class hierarchy

**Away from Stage 1 entirely:**
- If it turns out that ≥5 users have successfully used the vignette recipe (Option C)
  without hitting the formula trap, Option C may suffice
- Decision point: when the first utility-model use case appears in an upstream PR

---

## 7. Upstream Implementation Sketch (Stage 1)

### File list

```
R/
  model_m3_utility.R     # m3_utility(), m3_utility_formula(), S3 methods
tests/
  testthat/
    test-model_m3_utility.R  # unit tests
vignettes/
  bmm_m3_utility.Rmd     # vignette: "Value-directed recall with EU-weighted M3"
man/
  m3_utility.Rd          # auto-generated from roxygen
  m3_utility_formula.Rd
```

### Exported functions

```r
m3_utility(resp_cats, num_options, value_cols = NULL,
           utility = c("linear", "power"), weighting = c("none", "prelec"),
           choice_rule = "softmax", ...)
m3_utility_formula(model)
check_utility_design(model, data)  # manual pre-flight; may be un-exported
```

### S3 methods (unexported)

```r
check_model.m3_utility(model, data, formula)
check_data.m3_utility(model, data, formula)   # G1–G3 guards
check_formula.m3_utility(model, data, formula)
```

Note: `check_model.m3_custom`, `check_data.m3`, `configure_model.m3` are inherited via
`NextMethod()` and require no duplication.

### Test cases

| Test | Expected result |
|---|---|
| `m3_utility()` with no `value_cols`: links and priors for a, c only | pass |
| `m3_utility()` with `value_cols`: gamma link = "identity", prior = normal(0,1) | pass |
| `m3_utility()` with `utility="power"`: rho link = "log", prior = lognormal(0,0.5) | pass |
| `m3_utility_formula()` on linear model: correct per-category formula terms | pass |
| `check_data.m3_utility` with power utility + 2-level V: informative error | error |
| `check_data.m3_utility` with prelec + constant set size: informative error | error |
| `bmm()` with `m3_utility()` model on EU VDR data: samples gamma | pass |

### Vignette structure

```
1. Overview: EU activation as a utility model
2. Design requirements (citing WP1 identifiability conditions)
3. Example: VDR with 3 value levels (UC1 from 15_user_surface.R)
4. Power utility (UC1 variant with rho)
5. Prelec weighting (UC2 from 15_user_surface.R)
6. Interpreting the output: RUM vocabulary (from README.qmd §12)
7. What m3_utility() does NOT do: binary PT, attribute utility
```

---

## 8. Summary

| Option | R1 formula trap | R4 identifiability | R5 binary PT | Maintenance | Recommended |
|---|---|---|---|---|---|
| A (exploration) | ✅ | ❌ manual only | ❌ | Low | No — incomplete |
| A (production, Stage 1) | ✅ | ✅ | ❌ | Medium | **YES (Stage 1)** | Resolves R1–R4, R6, R7, R9 (G4), R10 |
| B — version="utility" | ❌¹ | ~ | ❌ | Medium | No |
| C — sibling utility() | ✅ | ✅ | ✅ | High | No (premature) |
| Binary PT separately (Stage 2) | N/A | N/A | ✅ | Medium | **YES (Stage 2, later)** |

¹ Unless `construct_m3_act_funs()` is also updated, which is equivalent effort to Stage 1.

**Next step:** Move Stage 1 files to R/ in a separate upstream PR. The round 6–7 empirical
evidence is now complete:
- WP1 (power utility recovery): both designs cover (gamma, rho) at N=30×90; rho CI ratio=1.2× → guard G1 needs range criterion (max(V)/min(V) ≥ 3), demoted from error to warning
- WP2 (hierarchical Prelec): clean recovery, 0 divergences; no false positives at alpha=1.0 → guard G2 validated
- WP3 (composite PT): ALL params covered with adapt_delta=0.95, N=30×120; cor(c,alpha)=0.906 — confound confirmed but not fatal with adequate sampling
- WP4 (stage confusion): |ELPD diff| < 0.2 — E and D indistinguishable at fixed-effects level → design manipulation required
- WP5 (welfare weights): E_D ≈ Model D (Δlog-lik = 0.000); u_EA non-covered at near-indifference (R7 note); R8 genuine new requirement; R9 and R10 corrected (see §3 note)
- Round 7 WP1: 22_welfare_bmm_simple.R confirms m3(choice_rule="simple") + identity links + fixed_parameters$b=1.0 → wi=1.2, wo=0.8 recovery without raw brms

---

## Appendix: WP4 Structural Decision (unchanged from round 3)

**Binary utility models are a sibling family, not an extension of m3_utility.**

Reasons:

1. **Different generative processes.** M3 assumes a memory retrieval process
   (activation → choice). Binary PT assumes direct preference (utility → choice).

2. **Different data shape.** M3 utility adds `gamma * V` to an activation formula for
   category counts. Binary PT requires per-option attributes (`x_A`, `p_A`, `x_B`, `p_B`)
   and a Bernoulli/logit family — incompatible with `check_data.m3`.

3. **The choice-rule layer is shared but not the constructor.** Both use softmax/logit;
   both can potentially share a `choice_rule = "softmax"` argument. But the model
   constructor and formula logic are separate.

4. **Practical scope.** Binary lottery choice is a JDM/behavioral-economics literature.
   A `binary_pt()` model constructor would live alongside `m3()`, not inside it.

Script `12_binary_choice_pt.R` demonstrates binary PT as a raw brms prototype.
The Stage 2 sketch above would make it a first-class bmm model.
