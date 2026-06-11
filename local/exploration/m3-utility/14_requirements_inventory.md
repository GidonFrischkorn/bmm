# Requirements Inventory: Utility Models vs Vanilla `m3()`

**Date:** 2026-06-11 (revised round 7: R9 corrected, R10 demoted; round 5 original)  
**Branch:** feat/issue-7-m3-utility-exploration  
**Author:** thom-more (agent run, rounds 5–7)  
**Evidence base:** rounds 1–7 of this exploration (R8 added round 6; R9–R10 corrected round 7)

---

## Overview

This document enumerates every requirement that utility-theory M3 models impose
that vanilla `m3()` (or any of the named versions "ss", "cs", "custom") does not
natively satisfy. For each requirement the table records:

- **Where it bites** — which pipeline stage (constructor / check_model /
  check_data / check_formula / configure_model / default_priors / postprocess)
- **Current workaround** — what the user must do today
- **Severity** — ranked: *silent-wrong* > *runtime error* > *friction*

"Silent-wrong" means the model runs to completion and produces samples, but the
samples are either answering the wrong question (R1) or the posterior is
effectively flat on the parameter of interest (R4). These are the most dangerous
failure modes because users receive no error signal.

---

## Requirements Table

| # | Requirement | Pipeline stage | Current workaround | Severity |
|---|---|---|---|---|
| R1 | Per-category activation formula (shared-slope trap) | check_formula | Manual per-category `bmf()` | **silent-wrong** |
| R2 | Default priors for utility parameters (γ, ρ, α) | default_priors | `default_priors = list(gamma=...)` arg, or user-specified `prior=` in `bmm()` | friction |
| R3 | Identity links for utility parameters (γ, ρ) | constructor / check_model | `m3(..., links = list(gamma="identity"))` — works, but undocumented for utility use | runtime error |
| R4 | Design-dependent identifiability guards | check_data | None — user must know design requirements from paper | **silent-wrong** |
| R5 | Per-option attribute data (binary PT outcomes, probabilities) | constructor + check_data | Bypass m3 pipeline entirely; use raw `brms` (script 12) | runtime error |
| R6 | Utility function × weighting composition (linear/power × none/prelec) | constructor + check_formula | Manual activation formula construction | friction |
| R7 | Parameter labelling in `summary()` output | postprocess | Know brms NL naming: `b_gamma_Intercept`, `r_subj__gamma`, etc. | friction |
| R8 | Fixed numeric payoff coefficients in activation formula | constructor + check_formula | Bypass m3 pipeline; write raw `brms::bf()` with numeric literals | **silent-wrong** |
| R9 | Positivity hazard: identity-linked utility with simple rule | check_model / prior spec | Use lower-bounded priors `T[0,]` or `lb=0`; `configure_model.m3` sets `init=0` (model_m3.R:425–428) | runtime error |
| R10 | (demoted to friction) Numeraire fixing not documented for utility use | constructor docs | `fixed_parameters$b <- 1.0` IS the supported mechanism; `fixed_pars_priors()` (helpers-prior.R:147) converts it to `constant(1)` automatically | friction |

---

## Requirement Details

### R1 — Per-category activation formula (shared-slope trap)

**Problem.** When a user writes:
```r
bmf(a ~ 1 + V_corr + (1|subj), ...)
```
the slope for `V_corr` is estimated *once* for `a` and applied identically to
every response category that includes `a` (both `corr` and `other` in the
standard M3). This is **not** the EU model. In the EU model, `gamma * V_corr`
applies only where the value of the *correct* item matters, and `gamma * V_other`
applies where the value of the *other* item matters — these can differ across
trials.

The correct specification requires explicit per-category activation formulas:
```r
bmf(
  corr  ~ b + a + c + gamma * V_corr,
  other ~ b + a     + gamma * V_other,
  npl   ~ b,
  ...
)
```
using `version = "custom"`. This is non-obvious. The `m3()` documentation does
not mention this trap.

**Evidence.** Script `10_bmm_api_example.R` header box documents this trap and
shows the wrong vs. correct specification. Script `11_utility_wrapper.R` resolves
it via `m3_utility_formula()`.

**Why silent-wrong.** The misspecified model fits without error. The estimated
slope from `a ~ 1 + V_corr` is the *average* EU effect pooled across categories,
not the per-category effect. For symmetric VDR designs (V_corr = V_other), the
two specifications are numerically equivalent and neither is wrong. For asymmetric
designs, they differ. The user has no signal that the model answered a different
question.

---

### R2 — Default priors for utility parameters

**Problem.** With `version = "custom"`, `check_model.m3_custom` fills in default
priors for parameters with identity or log links using generic rules:
```r
identity = list(main = "normal(1, 1)", effects = "normal(0, 1)")
```
For `gamma` (EU slope, range roughly −2 to +2) a `normal(1, 1)` intercept prior
is miscentred — it places substantial prior mass on gamma > 2, which is
unrealistic (it would mean high-value items are *much* more likely to be recalled
than low-value items). A `normal(0, 1)` prior centred at zero (no value effect)
is more appropriate.

For `rho` (power utility exponent, range 0.3–1.5) the log-link default
`normal(0, 1)` on the log scale implies `rho ~ lognormal(0,1)`, which is
reasonable but not communicated to the user.

**Evidence.** Script `11_utility_wrapper.R` attaches better defaults:
`gamma ~ normal(0, 1)` (main), `normal(0, 0.3)` (random effects);
`rho ~ lognormal(0, 0.5)`.

**Why friction, not worse.** The model runs correctly regardless of which prior
is used. The wrong prior affects inference quality but not model validity.
`check_model.m3_custom` warns the user that default priors are being applied
automatically.

---

### R3 — Identity links for utility parameters

**Problem.** `check_model.m3_custom` errors if any parameter has no link:
```
Please provide link functions for all model parameters via the `link` argument
of m3() to ensure proper identification of your model.
```
Without `links = list(gamma = "identity")`, the model fails at the check stage.

**Status: Partially resolved.** `m3(..., links = list(gamma = "identity"))`
works directly (R/model_m3.R line 90: `out$links[names(links)] <- links`). This
was confirmed in script `11_utility_wrapper.R`. However:

1. The `m3()` help page does not mention `links` as an argument to the user-facing
   alias (line 189: `m3 <- function(resp_cats, num_options, choice_rule, version, ...)`).
   The `links` argument passes through `...` to `.model_m3()` but is undocumented.
2. Passing a link for `rho` (log link, `rho > 0`) via the utility wrapper requires
   knowing the link before writing the formula — a discoverability problem.

**Why runtime error.** The `check_model.m3_custom` error fires loudly. At least
the user knows something is wrong.

---

### R4 — Design-dependent identifiability guards

**Problem.** Three utility parameters have design-dependent identifiability
conditions that vanilla `m3()` does not check:

| Parameter | Required design feature | What happens without it |
|---|---|---|
| γ (EU slope) | ≥ 2 distinct value levels in `V_corr` | Flat posterior on γ; γ is unidentified |
| ρ (power utility) | ≥ 3 distinct value levels | Flat posterior on ρ; (γ, ρ) non-identifiable from 2-level data |
| α (Prelec) | Variable set size across trials | Flat posterior on α; α is unidentified |

The conditions are documented in scripts `07_power_utility.R` (rho), `03_prototype_b_prelec.R` / `09_composite_pt.R` (alpha), and `06_realistic_recovery.R` / `13_recovery_one_cell.R` (gamma).

**Why silent-wrong.** The model runs to completion. The Stan sampler does not
diverge; it just explores the prior for the unidentified parameter. The user may
interpret the wide posterior as "the parameter is uncertain in this dataset" rather
than "the parameter is structurally unidentified by this design". There is no
warning from `m3()`, `check_model`, or `check_data`.

**Where it belongs.** These are *data*-level checks — the condition depends on
the actual distribution of values/set-sizes in `data`, not on the model
specification. The natural home is `check_data`. The current `check_data.m3`
(R/model_m3.R lines 268–313) checks for missing variables and zero option counts,
but has no utility-specific identifiability logic.

---

### R5 — Per-option attribute data (binary PT)

**Problem.** The M3 data pipeline assumes the response is a vector of *category
counts* (e.g., how many times each response type was chosen). Binary PT requires
*per-option attributes*: outcome `x_A`, probability `p_A`, outcome `x_B`,
probability `p_B` for each trial. These attributes drive the utility computation
(`u(x) = x^rho`, `w(p) = exp(-(-ln p)^alpha)`) and are not category counts.

The `check_data.m3` method:
1. Expects column names matching `resp_cats` (count variables)
2. Computes `nTrials = rowSums(resp_matrix)` — wrong for binary choice (each row
   IS one trial)
3. Wraps the response in `data$Y <- resp_matrix` — incompatible with brms
   `bernoulli` or `binomial` families

Binary PT must bypass the `m3()` pipeline entirely (script `12_binary_choice_pt.R`
uses raw `brms::bf()` with a custom `bernoulli` family). This is the structural
reason binary PT is a *sibling family*, not an m3_utility variant.

**Why runtime error.** If a user tries to force binary PT data through `m3()`,
they get either an error in `check_data.m3` or a misspecified brms family.

---

### R6 — Utility function × weighting composition

**Problem.** Users wanting composite prospect-theory M3 (both EU activation and
Prelec weighting) must manually construct activation formulas that include:
- The `gamma * V_i` term (EU activation)
- The `-((-log(p_i))^alpha)` term (Prelec offset)

combined correctly for each response category. This formula is subtle: the Prelec
term replaces `log(n_i)` in the softmax normalisation, not an additive activation
term. Script `11_utility_wrapper.R` (`m3_utility_formula()`) shows the correct
formula, but the derivation is non-obvious.

**Why friction.** Users can get the right formula from the documentation/vignette
(Option C). `m3_utility_formula()` resolves this at entry (Option A). The formula
is deterministic from `utility` and `weighting` arguments; a generator can produce
it without user intervention.

---

### R7 — Parameter labelling in `summary()` output

**Problem.** In brms NL models, parameters are named by their linear-predictor
sub-model. The EU slope `gamma` appears in `summary()` output as:
- Fixed effects: `b_gamma_Intercept`
- Random effects: `r_subj__gamma_Intercept` 

Users expecting `gamma` (or `gamma_mu`, `sd_gamma`) find the output confusing.
This is a general brms NL naming issue, not specific to m3. Postprocessing could
rename the parameters for cleaner output.

**Why friction.** Results are numerically correct. Users familiar with brms NL
models recognize the naming convention. A postprocess step could relabel for
clean printing but is not essential for correctness.

---

### R8 — Fixed numeric payoff coefficients in activation formula

**Problem.** The welfare-weight use case (Gross et al. 2025) requires activation
formulas with *fixed numeric coefficients* from the payoff matrix:
```r
universal ~ log(0.3 + 0.6 * wi + 0.9 * wo)
```
where `0.3`, `0.6`, `0.9` are known constants from the experimental design (not
per-trial data columns). Stage-1 `m3_utility()` generates activations of the form
`gamma * V_corr` where `V_corr` must be a column in the data frame. There is no
mechanism to embed fixed numeric coefficients in the NLF formula.

**Evidence.** Script `21_welfare_weight.R` demonstrates this directly: the E_A
formula uses `nlf(corr ~ log(0.3 + 0.6 * c + u))` with hardcoded payoff constants.

**Why silent-wrong.** A user who tries to map welfare weights onto `gamma * V_corr`
would need to pre-compute `V_corr = 0.6 * wi_init` as a constant column — which
works numerically only for the linear case and fails for the log-utility form.

---

### R9 — Positivity hazard: identity-linked utility with simple rule

**Problem.** When `choice_rule = "simple"` and utility parameters use `identity`
links, the activation for each category is a linear combination of parameters
(e.g., `ningroup = 0.5*b + wi`). `glue_choice_rule_functions()` (R/model_m3.R:399–404)
wraps this as `log(ningroup * n_options)`. If any activation goes non-positive
(e.g., `0.5*b + wi <= 0` at some posterior draw), Stan evaluates `log(negative)`,
which returns NaN and the trajectory is rejected as a divergence.

**Source citation.**  
- `R/model_m3.R:399–404`: `simple = glue("log({cat} * {options_vars[cat]})")` — the
  Luce-rule formula generation. Identity-linked parameters enter here directly.
- `R/model_m3.R:425–428`: `configure_model.m3` sets `init = 0` when `choice_rule ==
  "simple" && any(model$links == "identity")` — a partial guard, but does not prevent
  negative draws during sampling.

**What this is NOT.** The round-6 claim that "Luce/simple rule with non-linear utility
requires raw brms" is **incorrect**. `m3(choice_rule = "simple")` natively expresses
welfare-weight models — `22_welfare_bmm_simple.R` confirms wi=1.2, wo=0.8 recovery
at N=30×100 without any raw brms workaround. The formula-generation half of R9 belongs
in R8 (fixed payoff coefficients, which do require raw brms or a new Stage-1 feature).

**Correct framing.** R9 is a runtime hazard, not a structural limitation:
- If the prior permits non-positive activations (e.g., `normal(0, 1)` on wi with
  `identity` link), sampling will encounter `log(negative)` = divergences.
- Fix: lower-bounded priors (`T[0,]` or `lb=0`) ensure all activations remain positive.
- Guard G4 (added in `16_identifiability_guards.R`) checks this at configuration time.

**Severity: runtime error.** The model fails noisily (divergences, NaN warnings) rather
than silently. The fix (lower-bounded priors) is straightforward once the user knows about it.

---

### R10 — Numeraire fixing (documentation gap only)

**CORRECTED from round 6.** Round 6 incorrectly claimed R10 was a semantic mismatch
("silent-wrong"). This was **wrong**. The mechanism works correctly.

**Mechanism (verified against source).**  
- `R/model_m3.R:79–81`: `fixed_parameters = list(b = if (choice_rule == "softmax") 0 else 0.1)`.
  The key point: `b` in `fixed_parameters` IS the formula parameter `b` that appears in
  activation formulas (e.g., `nkeep ~ b`). It is not a separate "background noise" detached
  from the formula.
- `R/helpers-prior.R:147–168` (`fixed_pars_priors()`): converts every `fixed_parameters`
  entry to a `constant(value)` prior automatically. Setting `model$fixed_parameters$b <- 1.0`
  produces `constant(1)` for the formula parameter `b`.
- `R/helpers-model.R:165–180` (`update_model_fixed_parameters()`): syncs the list with the
  user formula at `check_model` time.

**Conclusion.** `model$fixed_parameters$b <- 1.0` is the correct, supported numeraire
mechanism. `22_welfare_bmm_simple.R` uses this and confirms correct recovery.

**Why friction, not silent-wrong.** The mechanism is not documented as a utility-model
pattern — users would not discover it from `?m3` or `?bmm`. `m3_utility()` (Stage 1)
should set `fixed_parameters$b` automatically so users never need to know about this.
Severity is friction (documentation gap), not a correctness problem.

---

## Priority Order

By impact on correctness:
1. **R4 identifiability guards** (silent-wrong, no current workaround)
2. **R1 activation formula trap** (silent-wrong, workaround exists but undiscovered)
3. **R8 fixed payoff coefficients** (silent-wrong, from WP5 round 6; requires Stage-2)
4. **R5 binary PT data** (runtime error / pipeline bypass required)
5. **R3 links documentation** (runtime error but loud; workaround exists)
6. **R9 positivity hazard** (runtime error — Stan rejection on negative utility; guard-able via G4)
7. **R2 default priors** (friction; workaround exists)
8. **R6 composition formula** (friction; workaround in 11_utility_wrapper.R)
9. **R7 parameter labelling** (friction; cosmetic)
10. **R10 numeraire fixing** (friction; `fixed_parameters$b` works; documentation gap only)

---

## Implications for Architecture Choice

The severity ordering drives the architecture question:

- **R1 and R4** (the two silent-wrong failures) require *pipeline integration*:
  - R1 needs formula generation in `check_formula` (or an equivalent formula
    generator applied before `bmm()` is called)
  - R4 needs data-level guards in `check_data`

- A **thin wrapper** in `local/exploration/` cannot auto-dispatch `check_data`
  guards because it returns an `m3_custom` object — `check_data.m3_utility` would
  need to be in the bmm namespace to fire.

- An **`m3_utility` subclass** (model object has class `c(..., "m3_utility")`) plus
  exported `check_data.m3_utility` in `R/` would resolve both R1 and R4 without a
  full sibling family.

- A **first-class sibling** (`utility()`) resolves all requirements and adds R5
  (binary PT) coverage through a companion `binary_pt()` constructor.

Stage-1 scope: R1–R4, R6, R7, R9 (guard G4). R8 requires Stage-2. R10 is a
documentation gap resolved by `m3_utility()` setting `fixed_parameters$b` automatically.

The coverage analysis is in `DESIGN_utility_api.md` §3 (Requirements × Architecture
Coverage Matrix).
