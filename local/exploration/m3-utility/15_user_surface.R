# 15_user_surface.R
# WP3 — User-surface comparison: three use cases x three architectures
#
# Compares the literal end-user code for three canonical utility-model use cases
# under three API architectures. Also identifies footguns (misuse paths) and what
# each architecture says (or doesn't say) about them.
#
# Three use cases:
#   UC1: EU activation in a VDR design (gamma, linear utility)
#   UC2: Composite PT (gamma + Prelec weighting alpha, variable set sizes)
#   UC3: Hierarchical binary lottery PT (Nilsson, Rieskamp & Wagenmakers 2011)
#
# Three architectures:
#   A: m3_utility() thin exported wrapper [IMPLEMENTED in 11_utility_wrapper.R]
#   B: version = "utility" entry in .m3_version_table [PSEUDOCODE — not in table]
#   C: first-class sibling constructor utility() [PSEUDOCODE — not implemented]
#
# Smoke fits (1 chain, warmup=500, iter=1000) for use cases expressible by A:
#   Arch A, UC1: EU VDR — REAL FIT -> results/15_smoke_A_UC1.csv
#   Arch A, UC2: Composite PT — REAL FIT -> results/15_smoke_A_UC2.csv
#   Arch A, UC3: binary PT — NOT expressible by wrapper (sibling family required)
#   Arch B, C:   not implemented -> no fits
#
# Run from repo root:
#   source("local/exploration/m3-utility/15_user_surface.R")

library(brms)
if (requireNamespace("bmm", quietly = TRUE)) {
  library(bmm)
} else {
  suppressMessages(devtools::load_all("."))
}

# Load m3_utility() and m3_utility_formula() from prototype
# (In production: library(bmm) after the feature is merged)
invisible(capture.output(
  source("local/exploration/m3-utility/11_utility_wrapper.R")
))
cat("m3_utility() and m3_utility_formula() loaded from 11_utility_wrapper.R\n\n")


# ==============================================================================
# PART 0 — Shared data simulation helpers
# ==============================================================================

# EU activation data (VDR design, 3 response categories, variable value)
sim_eu_vdr <- function(N_subj = 20, N_trials = 60,
                       a_mu = 2.0, c_mu = 3.0, gamma_mu = 0.8,
                       sigma_a = 0.3, sigma_c = 0.3, sigma_gamma = 0.2,
                       V_levels = c(1, 3, 5),
                       n_corr = 1L, n_other = 4L, n_npl = 5L,
                       seed = 42) {
  set.seed(seed)
  a_subj     <- rnorm(N_subj, a_mu, sigma_a)
  c_subj     <- rnorm(N_subj, c_mu, sigma_c)
  gamma_subj <- rnorm(N_subj, gamma_mu, sigma_gamma)

  rows <- vector("list", N_subj)
  for (i in seq_len(N_subj)) {
    V <- sample(V_levels, N_trials, replace = TRUE)
    act_corr  <- a_subj[i] + c_subj[i] + gamma_subj[i] * V
    act_other <- a_subj[i]              + gamma_subj[i] * V
    act_npl   <- rep(0, N_trials)
    denom <- n_corr * exp(act_corr) + n_other * exp(act_other) + n_npl * exp(act_npl)
    p <- cbind(n_corr * exp(act_corr) / denom,
               n_other * exp(act_other) / denom,
               n_npl   * exp(act_npl)   / denom)
    resp <- t(apply(p, 1, function(pr) rmultinom(1, 1, pr)))
    rows[[i]] <- data.frame(
      subj    = i,
      V_corr  = V,
      V_other = V,
      n_corr  = n_corr,
      n_other = n_other,
      n_npl   = n_npl,
      corr    = resp[, 1L],
      other   = resp[, 2L],
      npl     = resp[, 3L]
    )
  }
  do.call(rbind, rows)
}

# Composite PT data (EU + Prelec, variable set sizes)
sim_composite_pt <- function(N_subj = 20, N_trials = 80,
                             a_mu = 2.0, c_mu = 3.0, gamma_mu = 0.6, alpha_mu = 0.7,
                             sigma_a = 0.3, sigma_c = 0.3, sigma_gamma = 0.2, sigma_alpha = 0.15,
                             V_levels = c(1, 3, 5),
                             n_other_levels = c(2L, 4L, 6L),
                             n_corr = 1L, n_npl = 5L,
                             seed = 43) {
  set.seed(seed)
  a_subj     <- rnorm(N_subj, a_mu, sigma_a)
  c_subj     <- rnorm(N_subj, c_mu, sigma_c)
  gamma_subj <- rnorm(N_subj, gamma_mu, sigma_gamma)
  alpha_subj <- rnorm(N_subj, alpha_mu, sigma_alpha)

  rows <- vector("list", N_subj)
  for (i in seq_len(N_subj)) {
    V       <- sample(V_levels, N_trials, replace = TRUE)
    n_other <- sample(n_other_levels, N_trials, replace = TRUE)
    n_total <- n_corr + n_other + n_npl

    # Prelec weighting for each category
    p_corr_raw  <- n_corr  / n_total
    p_other_raw <- n_other / n_total
    p_npl_raw   <- n_npl   / n_total

    ai  <- alpha_subj[i]
    w_corr  <- exp(-((-log(p_corr_raw))^ai))
    w_other <- exp(-((-log(p_other_raw))^ai))
    w_npl   <- exp(-((-log(p_npl_raw))^ai))

    act_corr  <- a_subj[i] + c_subj[i] + gamma_subj[i] * V + log(w_corr)
    act_other <- a_subj[i]              + gamma_subj[i] * V + log(w_other)
    act_npl   <- log(w_npl)

    denom <- exp(act_corr) + exp(act_other) + exp(act_npl)
    p <- cbind(exp(act_corr) / denom, exp(act_other) / denom, exp(act_npl) / denom)
    resp <- t(apply(p, 1, function(pr) rmultinom(1, 1, pr)))

    rows[[i]] <- data.frame(
      subj    = i,
      V_corr  = V,
      V_other = V,
      n_corr  = n_corr,
      n_other = n_other,
      n_npl   = n_npl,
      p_corr  = p_corr_raw,
      p_other = p_other_raw,
      p_npl   = p_npl_raw,
      corr    = resp[, 1L],
      other   = resp[, 2L],
      npl     = resp[, 3L]
    )
  }
  do.call(rbind, rows)
}


# ==============================================================================
# PART I — Architecture A: m3_utility() wrapper
# ==============================================================================
#
# Architecture A expresses UC1 and UC2. UC3 (binary PT) is out of scope
# for the wrapper (requires a sibling family; see 12_binary_choice_pt.R).

cat("==========================================================================\n")
cat("ARCHITECTURE A — m3_utility() wrapper\n")
cat("==========================================================================\n\n")

# --------------------------------------------------------------------
# UC1: EU activation in a VDR design
# Lines of user code: 18 (model + formula + fit call)
# Footgun status: resolved — m3_utility_formula() generates per-category formulas
# --------------------------------------------------------------------

cat("--- UC1: EU activation (Architecture A) ---\n\n")

# User code begins here (after: library(bmm); m3_utility and m3_utility_formula
# are exported functions in the production package)

# Step 1: Build model
model_A_UC1 <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "none"
)

# Step 2: Build formula using auto-generated activation formulas
formula_A_UC1 <- do.call(bmf, c(
  m3_utility_formula(model_A_UC1),
  list(
    a     ~ 1 + (1 | subj),
    c     ~ 1 + (1 | subj),
    gamma ~ 1 + (1 | subj)
  )
))

cat("Generated activation formulas (no manual per-category writing required):\n")
for (nm in names(m3_utility_formula(model_A_UC1))) {
  cat(sprintf("  %s\n", deparse(m3_utility_formula(model_A_UC1)[[nm]])))
}
cat("\nLinks:", paste(names(model_A_UC1$links), "=",
                      unlist(model_A_UC1$links), collapse=", "), "\n")
cat("Default priors: gamma main =", model_A_UC1$default_priors$gamma$main, "\n\n")

# Step 3: fit call (shown; run in Part IV below for smoke fit)
# fit_A_UC1 <- bmm(formula = formula_A_UC1, data = data_eu, model = model_A_UC1,
#                  chains = 1, warmup = 500, iter = 1000, backend = "cmdstanr")


# --------------------------------------------------------------------
# UC2: Composite PT (EU + Prelec, variable set sizes)
# Lines of user code: 22 (model + formula + fit call)
# Footgun status: partially resolved — formula generated, but user must supply
#   p_<cat> columns in data and ensure set-size variation
# --------------------------------------------------------------------

cat("--- UC2: Composite PT (Architecture A) ---\n\n")

model_A_UC2 <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "prelec"
)

formula_A_UC2 <- do.call(bmf, c(
  m3_utility_formula(model_A_UC2),
  list(
    a     ~ 1 + (1 | subj),
    c     ~ 1 + (1 | subj),
    gamma ~ 1 + (1 | subj),
    alpha ~ 1 + (1 | subj)
  )
))

cat("Generated activation formulas for composite PT:\n")
for (nm in names(m3_utility_formula(model_A_UC2))) {
  cat(sprintf("  %s\n", deparse(m3_utility_formula(model_A_UC2)[[nm]])))
}
cat("\nLinks:", paste(names(model_A_UC2$links), "=",
                      unlist(model_A_UC2$links), collapse=", "), "\n\n")

# NOTE: Footgun remaining in Arch A:
# If user provides fixed-set-size data, alpha is unidentifiable — no guard fires.
# Architecture A (exploration-only) cannot add check_data.m3_utility guard.


# --------------------------------------------------------------------
# UC3: Binary lottery PT (Nilsson et al. 2011)
# Architecture A: NOT expressible
# m3_utility() is M3-style (category count data). Binary PT requires per-option
# attribute data (x_A, p_A, x_B, p_B) and a Bernoulli/logit family.
# The m3 data pipeline rejects this at check_data.m3.
# See 12_binary_choice_pt.R for the raw brms implementation.
# --------------------------------------------------------------------

cat("--- UC3: Binary lottery PT (Architecture A) ---\n")
cat("NOT EXPRESSIBLE: m3_utility() handles category-count data only.\n")
cat("Binary PT requires per-option attribute data and a Bernoulli family.\n")
cat("Architecture A verdict: UC3 requires a sibling family (binary_pt()).\n\n")


# --------------------------------------------------------------------
# FOOTGUN DEMO (Architecture A): power utility with 2 value levels
# --------------------------------------------------------------------

cat("--- Footgun demo: power utility with 2 value levels (Architecture A) ---\n\n")

model_A_footgun <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "power"
)

cat("Arch A result: m3_utility() SUCCEEDS (no construction-time guard for 2-level V).\n")
cat("Links:", paste(names(model_A_footgun$links), "=",
                    unlist(model_A_footgun$links), collapse=", "), "\n")
cat("With 2 value levels (V in {1,2}):\n")
cat("  -> bmm() runs without error\n")
cat("  -> Posterior for rho is flat (unidentified) — SILENT-WRONG\n")
cat("  -> User sees wide CI for rho and may conclude 'rho is uncertain in my data'\n")
cat("  -> No warning is issued\n\n")

cat("Arch C (sibling) result: check_data.utility would fire:\n")
cat("  Error: Power utility (rho) requires >= 3 distinct value levels.\n")
cat("  Column 'V_corr' has only 2 distinct values: 1, 2.\n\n")


# ==============================================================================
# PART II — Architecture B: version = "utility" (PSEUDOCODE)
# ==============================================================================
#
# Architecture B would add a new entry to .m3_version_table in R/model_m3.R.
# It does NOT exist in the current codebase.
#
# Key finding: version="utility" does NOT auto-generate activation formulas.
# construct_m3_act_funs() only handles versions "ss" and "cs" (line 473 in
# R/model_m3.R). Adding version="utility" to the table requires ALSO adding
# formula generation logic — making this MORE invasive than it appears.

cat("==========================================================================\n")
cat("ARCHITECTURE B — version = 'utility' [PSEUDOCODE — NOT IMPLEMENTED]\n")
cat("==========================================================================\n\n")

cat("--- UC1: EU activation (Architecture B, pseudocode) ---\n\n")

cat('
# If version="utility" were added to .m3_version_table with gamma parameter,
# user code would be:

model_B_UC1 <- m3(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c("n_corr", "n_other", "n_npl"),
  choice_rule = "softmax",
  version     = "utility"     # <-- does not exist in .m3_version_table
)
# -> Error: Unknown version: "utility". It should be one of "ss", "cs" or "custom"

# Even if added: construct_m3_act_funs() would need to be updated for "utility".
# Currently it stops with an error for any version not in c("ss","cs").
# The version table specifies PARAMETERS, not activation formulas.
# User still needs to write per-category formulas manually.

# Footgun status: UNRESOLVED (same as architecture C documentation-only)
# The activation formula trap (shared slope) is not prevented by version="utility"
# because formula generation is NOT driven by the version table.
')

cat("--- UC2: Composite PT (Architecture B, pseudocode) ---\n\n")
cat('
# No version entry covers composite PT (gamma + Prelec alpha).
# Would require a separate version="utility_prelec" or version="composite_pt".
# This inflates the version table; the DESIGN_utility_api.md §2 Option B
# documents this as a maintenance problem.
')

cat("--- UC3: Binary PT (Architecture B, pseudocode) ---\n\n")
cat('
# version="utility" is M3-based (category counts). Binary PT is not expressible.
# Same conclusion as Architecture A.
')

cat("--- Footgun: power utility 2-level (Architecture B) ---\n\n")
cat('
# m3() with version="utility" (if implemented) would have the same issue:
# no check_data guard for insufficient value levels.
# Severity: SILENT-WRONG — same as Architecture A.
')


# ==============================================================================
# PART III — Architecture C: first-class sibling utility() (PSEUDOCODE)
# ==============================================================================
#
# Architecture C is a completely new model family in a new file R/model_utility.R.
# It does NOT exist in the current codebase.
#
# Key capabilities:
# - utility() constructor: type = "memory" (M3-style) or "attribute" (binary PT)
# - check_data.utility: fires identifiability guards (R4), validates data shape (R5)
# - check_formula.utility: generates per-category activation formulas automatically
# - configure_model.utility: builds correct brms formula for memory vs attribute case
# - Default priors: embedded in model object, not table lookup

cat("==========================================================================\n")
cat("ARCHITECTURE C — first-class sibling utility() [PSEUDOCODE — NOT IMPLEMENTED]\n")
cat("==========================================================================\n\n")

cat("--- UC1: EU activation (Architecture C, pseudocode) ---\n\n")

cat('
# User code (after: library(bmm)):

model_C_UC1 <- utility(
  type        = "memory",        # M3-style: activation parameters carry utility
  resp_cats   = c("corr", "other", "npl"),
  num_options = c("n_corr", "n_other", "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility_fn  = "linear",        # gamma * V_i
  weighting   = "none"
)
# -> Returns bmmodel object of class c("bmmodel", "utility", "utility_memory")
# -> Links, priors, activation formulas all set internally
# -> check_formula.utility generates correct per-category formulas

# Formula: user specifies only hyperparameter regressions
formula_C_UC1 <- bmf(
  a     ~ 1 + (1 | subj),
  c     ~ 1 + (1 | subj),
  gamma ~ 1 + (1 | subj)
)
# NOTE: activation formulas (corr ~ b+a+c+gamma*V_corr, etc.) are
# generated internally by check_formula.utility — user does NOT write them.
# This is different from Arch A where do.call(bmf, c(m3_utility_formula(model), ...))
# is still required.

fit_C_UC1 <- bmm(formula = formula_C_UC1, data = data_eu, model = model_C_UC1,
                 chains = 1, warmup = 500, iter = 1000, backend = "cmdstanr")
# Parameters in summary():  gamma (cleaned label via postprocess_brm.utility)
')

cat("Lines of user code: 12 (fewer than Arch A's 18; activation formulas internal)\n\n")

cat("--- UC2: Composite PT (Architecture C, pseudocode) ---\n\n")

cat('
model_C_UC2 <- utility(
  type        = "memory",
  resp_cats   = c("corr", "other", "npl"),
  num_options = c("n_corr", "n_other", "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility_fn  = "linear",
  weighting   = "prelec"
)
formula_C_UC2 <- bmf(
  a     ~ 1 + (1 | subj),
  c     ~ 1 + (1 | subj),
  gamma ~ 1 + (1 | subj),
  alpha ~ 1 + (1 | subj)
)
fit_C_UC2 <- bmm(formula = formula_C_UC2, data = data_cpt, model = model_C_UC2, ...)
# check_data.utility fires if set sizes are constant:
#   Error: Prelec alpha requires variable set size. n_other has 1 unique value.
')

cat("--- UC3: Binary PT (Architecture C, pseudocode) ---\n\n")

cat('
# Architecture C can express binary PT via type = "attribute"

model_C_UC3 <- utility(
  type        = "attribute",     # option-attribute utility (not memory activation)
  resp_cats   = c("A", "B"),
  utility_fn  = "power",         # u(x) = x^rho
  weighting   = "prelec",        # w(p) = exp(-(-ln p)^alpha)
  outcome_cols = c(A = "x_A", B = "x_B"),
  prob_cols    = c(A = "p_A",  B = "p_B"),
  sensitivity = TRUE             # lambda parameter
)
formula_C_UC3 <- bmf(
  rho       ~ 1 + (1 | subj),
  alpha     ~ 1 + (1 | subj),
  lambda    ~ 1 + (1 | subj)
)
fit_C_UC3 <- bmm(formula = formula_C_UC3, data = data_binary, model = model_C_UC3, ...)
# check_data.utility validates per-option attribute columns exist
# check_model.utility selects Bernoulli/logit family for type="attribute"
')

cat("--- Footgun: power utility 2-level (Architecture C) ---\n\n")
cat('
# check_data.utility would fire early:
#   Error: Power utility (rho) requires >= 3 distinct value levels.
#   Column "V_corr" has only 2 distinct values (1, 2).
#   Add a third value level or switch to utility_fn = "linear".
#
# This is the key quality-of-life advantage over Architectures A and B:
# silent non-identifiability becomes a loud, early failure.
')

cat("==========================================================================\n")
cat("Summary: Lines of code and footguns by architecture and use case\n")
cat("==========================================================================\n\n")

summary_tbl <- data.frame(
  Architecture = c("A (wrapper)", "A (wrapper)", "A (wrapper)",
                   "B (version)", "B (version)", "B (version)",
                   "C (sibling)", "C (sibling)", "C (sibling)"),
  Use_case = rep(c("UC1: EU VDR", "UC2: Composite PT", "UC3: Binary PT"), 3),
  Expressible = c("YES", "YES", "NO (sibling required)",
                  "NO (not impl.)", "NO (not impl.)", "NO (sibling required)",
                  "YES", "YES", "YES (type='attribute')"),
  LoC_model = c(5, 6, NA, NA, NA, NA, 4, 5, 5),
  LoC_formula = c(4, 5, NA, NA, NA, NA, 4, 5, 4),
  Footgun_rho_2levels = c("silent-wrong", "silent-wrong", NA,
                           "silent-wrong", "silent-wrong", NA,
                           "loud error", "loud error", "loud error"),
  stringsAsFactors = FALSE
)

print(summary_tbl, row.names = FALSE)
cat("\nLoC = lines of code for model spec + formula; NA = not applicable/expressible\n\n")


# ==============================================================================
# PART IV — Smoke fits (Architecture A)
# ==============================================================================
#
# Two smoke fits:
#   Fit 1: Arch A, UC1 (EU VDR, N=20x60, 1 chain x 1000 iter)
#   Fit 2: Arch A, UC2 (Composite PT, N=20x80, 1 chain x 1000 iter)
#
# Goal: verify (a) samples are drawn, (b) parameters are labelled correctly,
#        (c) no sampling errors (Rhat, divergences)

cat("==========================================================================\n")
cat("PART IV — Smoke fits (Architecture A)\n")
cat("==========================================================================\n\n")

dir.create("local/exploration/m3-utility/results", showWarnings = FALSE)

# ---- Smoke Fit 1: Arch A, UC1 (EU VDR) ----------------------------------------

cat("--- Smoke Fit 1: Arch A, UC1 (EU VDR) ---\n\n")
cat("Simulating data: N=20 subjects x 60 trials, gamma_true=0.8\n")
data_eu_smoke <- sim_eu_vdr(N_subj = 20, N_trials = 60)
cat(sprintf("Data: %d rows, %d subjects, %d trials per subject\n",
            nrow(data_eu_smoke),
            length(unique(data_eu_smoke$subj)),
            sum(data_eu_smoke$subj == 1)), "\n")

model_smoke1 <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "none"
)
formula_smoke1 <- do.call(bmf, c(
  m3_utility_formula(model_smoke1),
  list(
    a     ~ 1 + (1 | subj),
    c     ~ 1 + (1 | subj),
    gamma ~ 1 + (1 | subj)
  )
))

cat("Running smoke fit 1 (1 chain, warmup=500, iter=1000)...\n")
t0 <- proc.time()["elapsed"]
fit_smoke1 <- bmm(
  formula = formula_smoke1,
  data    = data_eu_smoke,
  model   = model_smoke1,
  chains  = 1,
  warmup  = 500,
  iter    = 1000,
  backend = "cmdstanr",
  silent  = 2
)
t1 <- proc.time()["elapsed"]
cat(sprintf("Fit completed in %.1f seconds\n\n", t1 - t0))

# Extract key parameters
s1 <- summary(fit_smoke1)$fixed
gamma_row <- s1[grepl("gamma_Intercept$", rownames(s1)), ]
a_row     <- s1[grepl("a_Intercept$",     rownames(s1)), ]
c_row     <- s1[grepl("c_Intercept$",     rownames(s1)), ]

cat("Fixed-effects parameter labels (verifying correct labelling):\n")
print(round(s1[, c("Estimate", "l-95% CI", "u-95% CI", "Rhat", "Bulk_ESS")], 3))
cat("\n")

max_rhat <- max(s1[, "Rhat"], na.rm = TRUE)
divs     <- sum(nuts_params(fit_smoke1)$Value[nuts_params(fit_smoke1)$Parameter == "divergent__"])
cat(sprintf("Max R-hat: %.3f | Divergences: %d\n\n", max_rhat, divs))

cat("Gamma recovery:\n")
cat(sprintf("  True gamma_mu = 0.80 | Estimated = %.3f | 95%% CI = [%.3f, %.3f]\n",
            gamma_row$Estimate, gamma_row$`l-95% CI`, gamma_row$`u-95% CI`))
covered1 <- gamma_row$`l-95% CI` <= 0.80 & 0.80 <= gamma_row$`u-95% CI`
cat(sprintf("  CI covers true value: %s\n\n", covered1))

# Save results
res1 <- data.frame(
  fit = "smoke_A_UC1",
  param = c("a", "c", "gamma"),
  true_value = c(2.0, 3.0, 0.8),
  estimate = c(a_row$Estimate, c_row$Estimate, gamma_row$Estimate),
  ci_lower = c(a_row$`l-95% CI`, c_row$`l-95% CI`, gamma_row$`l-95% CI`),
  ci_upper = c(a_row$`u-95% CI`, c_row$`u-95% CI`, gamma_row$`u-95% CI`),
  covered  = c(a_row$`l-95% CI` <= 2.0 & 2.0 <= a_row$`u-95% CI`,
               c_row$`l-95% CI` <= 3.0 & 3.0 <= c_row$`u-95% CI`,
               covered1),
  max_rhat = max_rhat,
  divergences = divs,
  elapsed_sec = round(t1 - t0, 1)
)
write.csv(res1, "local/exploration/m3-utility/results/15_smoke_A_UC1.csv",
          row.names = FALSE)
cat("Results saved to results/15_smoke_A_UC1.csv\n\n")


# ---- Smoke Fit 2: Arch A, UC2 (Composite PT) ----------------------------------

cat("--- Smoke Fit 2: Arch A, UC2 (Composite PT) ---\n\n")
cat("Simulating data: N=20 subjects x 80 trials, gamma_true=0.6, alpha_true=0.7\n")
data_cpt_smoke <- sim_composite_pt(N_subj = 20, N_trials = 80)
cat(sprintf("Data: %d rows, n_other ∈ {%s} (set-size varies: identifiability met)\n",
            nrow(data_cpt_smoke),
            paste(sort(unique(data_cpt_smoke$n_other)), collapse=",")), "\n")

model_smoke2 <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "prelec"
)
formula_smoke2 <- do.call(bmf, c(
  m3_utility_formula(model_smoke2),
  list(
    a     ~ 1 + (1 | subj),
    c     ~ 1 + (1 | subj),
    gamma ~ 1 + (1 | subj),
    alpha ~ 1 + (1 | subj)
  )
))

cat("Running smoke fit 2 (1 chain, warmup=500, iter=1000)...\n")
t0b <- proc.time()["elapsed"]
fit_smoke2 <- bmm(
  formula = formula_smoke2,
  data    = data_cpt_smoke,
  model   = model_smoke2,
  chains  = 1,
  warmup  = 500,
  iter    = 1000,
  backend = "cmdstanr",
  silent  = 2
)
t1b <- proc.time()["elapsed"]
cat(sprintf("Fit completed in %.1f seconds\n\n", t1b - t0b))

s2 <- summary(fit_smoke2)$fixed
cat("Fixed-effects parameter labels:\n")
print(round(s2[, c("Estimate", "l-95% CI", "u-95% CI", "Rhat", "Bulk_ESS")], 3))
cat("\n")

gamma_row2 <- s2[grepl("gamma_Intercept$", rownames(s2)), ]
alpha_row2 <- s2[grepl("alpha_Intercept$", rownames(s2)), ]
max_rhat2  <- max(s2[, "Rhat"], na.rm = TRUE)
divs2      <- sum(nuts_params(fit_smoke2)$Value[nuts_params(fit_smoke2)$Parameter == "divergent__"])

cat(sprintf("Max R-hat: %.3f | Divergences: %d\n", max_rhat2, divs2))
cat(sprintf("Gamma: true=0.6 | est=%.3f | 95%% CI=[%.3f, %.3f]\n",
            gamma_row2$Estimate, gamma_row2$`l-95% CI`, gamma_row2$`u-95% CI`))
cat(sprintf("Alpha: true=0.7 | est=%.3f | 95%% CI=[%.3f, %.3f]\n\n",
            alpha_row2$Estimate, alpha_row2$`l-95% CI`, alpha_row2$`u-95% CI`))

a_row2 <- s2[grepl("a_Intercept$", rownames(s2)), ]
c_row2 <- s2[grepl("c_Intercept$", rownames(s2)), ]

res2 <- data.frame(
  fit = "smoke_A_UC2",
  param = c("a", "c", "gamma", "alpha"),
  true_value = c(2.0, 3.0, 0.6, 0.7),
  estimate = c(a_row2$Estimate, c_row2$Estimate,
               gamma_row2$Estimate, alpha_row2$Estimate),
  ci_lower = c(a_row2$`l-95% CI`, c_row2$`l-95% CI`,
               gamma_row2$`l-95% CI`, alpha_row2$`l-95% CI`),
  ci_upper = c(a_row2$`u-95% CI`, c_row2$`u-95% CI`,
               gamma_row2$`u-95% CI`, alpha_row2$`u-95% CI`),
  covered  = c(a_row2$`l-95% CI` <= 2.0 & 2.0 <= a_row2$`u-95% CI`,
               c_row2$`l-95% CI` <= 3.0 & 3.0 <= c_row2$`u-95% CI`,
               gamma_row2$`l-95% CI` <= 0.6 & 0.6 <= gamma_row2$`u-95% CI`,
               alpha_row2$`l-95% CI` <= 0.7 & 0.7 <= alpha_row2$`u-95% CI`),
  max_rhat = max_rhat2,
  divergences = divs2,
  elapsed_sec = round(t1b - t0b, 1)
)
write.csv(res2, "local/exploration/m3-utility/results/15_smoke_A_UC2.csv",
          row.names = FALSE)
cat("Results saved to results/15_smoke_A_UC2.csv\n\n")

cat("==========================================================================\n")
cat("15_user_surface.R complete.\n")
cat("Smoke fit 1 (Arch A, UC1): MEASURED — see results/15_smoke_A_UC1.csv\n")
cat("Smoke fit 2 (Arch A, UC2): MEASURED — see results/15_smoke_A_UC2.csv\n")
cat("Arch B, C: pseudocode only (not implemented)\n")
cat("Binary PT (UC3): sibling family required for all architectures\n")
cat("==========================================================================\n")
