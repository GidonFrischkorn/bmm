# 22_welfare_bmm_simple.R — Round 7 (WP1)
#
# Execution-backed proof that the welfare-weight case runs in today's m3.
#
# Uses m3(choice_rule = "simple") with identity links for welfare weights,
# with parameters set well away from indifference (wi = 1.2, wo = 0.8):
#
#   U(keep)      = b              = 1.0  (numeraire)
#   U(ingroup)   = 0.5*b + wi    = 1.70  (wi = 1.2)
#   U(universal) = 0.3*b + 0.6*wi + 0.9*wo = 1.74  (wi = 1.2, wo = 0.8)
#
# Luce choice rule: P(k) = U(k) / sum_j U(j)
#
# This single fit simultaneously proves:
#   (a) The simple rule (glue_choice_rule_functions, model_m3.R:399-404) generates
#       log({cat} * n_options), which equals log(U_k) for n_k=1 — i.e., the Luce
#       rule IS native to m3(choice_rule = "simple"). R9 corrected.
#   (b) fixed_parameters$b <- 1.0 is the supported numeraire mechanism via
#       fixed_pars_priors() (helpers-prior.R:147) which converts it to constant(1).
#       R10 corrected.
#   (c) Identity-linked welfare weights recover correctly when utilities are
#       well-separated (avoiding the near-indifference problem in round 6 WP5).
#   (d) Guard G4 positivity condition is satisfied: all activations > 0 given
#       lower-bounded priors T[0,]. See 16_identifiability_guards.R.
#
# Design:
#   N = 30 subjects x 100 trials (subject-level aggregated counts)
#   wi_true = 1.2, wo_true = 0.8, b = 1.0 (numeraire)
#   2 chains x 1000 iter (500 warmup)
#
# Results saved to: results/22_welfare_bmm_simple.csv
#
# Run from repo root:
#   Rscript local/exploration/m3-utility/22_welfare_bmm_simple.R

suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
})
if (requireNamespace("bmm", quietly = TRUE)) {
  library(bmm)
} else {
  suppressMessages(devtools::load_all(".", quiet = TRUE))
}

dir.create("local/exploration/m3-utility/results", showWarnings = FALSE, recursive = TRUE)

# ==============================================================================
# PARAMETERS
# ==============================================================================
N_SUBJ   <- 30L
N_TRIALS <- 100L
WI_TRUE  <- 1.2    # ingroup welfare weight (> 0.5: prefer ingroup over keep)
WO_TRUE  <- 0.8    # outgroup welfare weight (> 0: some outgroup concern)
B_FORM   <- 1.0    # numeraire

# Derived utilities (all positive — no positivity hazard)
U_KEEP      <- B_FORM                                    # 1.00
U_INGROUP   <- 0.5 * B_FORM + WI_TRUE                   # 1.70
U_UNIVERSAL <- 0.3 * B_FORM + 0.6 * WI_TRUE + 0.9 * WO_TRUE  # 1.74

cat("=== 22_welfare_bmm_simple.R (Round 7 WP1) ===\n\n")
cat(sprintf("True parameters: wi=%.2f, wo=%.2f (b=%.1f)\n", WI_TRUE, WO_TRUE, B_FORM))
cat(sprintf("Utilities: U(keep)=%.2f, U(ingroup)=%.2f, U(universal)=%.2f\n",
            U_KEEP, U_INGROUP, U_UNIVERSAL))
cat(sprintf("Choice probs: P(keep)=%.3f, P(ingroup)=%.3f, P(universal)=%.3f\n\n",
            U_KEEP / (U_KEEP + U_INGROUP + U_UNIVERSAL),
            U_INGROUP / (U_KEEP + U_INGROUP + U_UNIVERSAL),
            U_UNIVERSAL / (U_KEEP + U_INGROUP + U_UNIVERSAL)))

# ==============================================================================
# SIMULATE DATA
# ==============================================================================
set.seed(2207L)

# Subject-level aggregated counts (standard M3 input format)
sim_counts <- function(n_subj, n_trials, u_keep, u_ingroup, u_universal) {
  total_u <- u_keep + u_ingroup + u_universal
  probs   <- c(u_keep, u_ingroup, u_universal) / total_u

  counts <- t(sapply(seq_len(n_subj), function(s) {
    rmultinom(1, n_trials, probs)
  }))
  colnames(counts) <- c("nkeep", "ningroup", "nuniversal")
  data.frame(
    subj      = seq_len(n_subj),
    nkeep     = counts[, "nkeep"],
    ningroup  = counts[, "ningroup"],
    nuniversal = counts[, "nuniversal"],
    n_opt_keep = 1L,
    n_opt_ingroup = 1L,
    n_opt_universal = 1L,
    nTrials   = n_trials
  )
}

dat <- sim_counts(N_SUBJ, N_TRIALS, U_KEEP, U_INGROUP, U_UNIVERSAL)
cat(sprintf("Data: N=%d subjects x %d trials\n", N_SUBJ, N_TRIALS))
cat(sprintf("Mean counts: keep=%.1f, ingroup=%.1f, universal=%.1f\n\n",
            mean(dat$nkeep), mean(dat$ningroup), mean(dat$nuniversal)))

# ==============================================================================
# MODEL SPECIFICATION
# ==============================================================================

# m3(choice_rule = "simple"):
#   glue_choice_rule_functions() generates:
#     mu_nkeep      <- log(nkeep      * n_opt_keep)      = log(nkeep)
#     mu_ningroup   <- log(ningroup   * n_opt_ingroup)   = log(ningroup)
#     mu_nuniversal <- log(nuniversal * n_opt_universal) = log(nuniversal)
#   The multinomial probability P(k) = exp(mu_k) / sum_j exp(mu_j) = U_k / sum U_j
#   This IS the Luce ratio rule. (Proves R9 correction.)
#
# fixed_parameters$b <- 1.0:
#   fixed_pars_priors() (helpers-prior.R:147) converts to constant(1) prior.
#   b is the formula parameter for background utility, equal to the own-payoff
#   numeraire. (Proves R10 correction.)

model <- m3(
  resp_cats   = c("nkeep", "ningroup", "nuniversal"),
  num_options = c("n_opt_keep", "n_opt_ingroup", "n_opt_universal"),
  choice_rule = "simple"
)

# Set identity links for welfare weights (positive on real line)
# and fixed b = 1.0 (numeraire) via supported mechanism
model$links <- list(wi = "identity", wo = "identity")
model$fixed_parameters$b <- 1.0   # numeraire: supported via fixed_pars_priors()

# Welfare-weight activation formulas (payoff matrix as fixed numeric coefficients)
#   nkeep      = b                              = 1.0
#   ningroup   = 0.5 * b + wi                  = 0.5 + 1.2 = 1.7
#   nuniversal = 0.3 * b + 0.6 * wi + 0.9 * wo = 0.3 + 0.72 + 0.72 = 1.74
formula <- bmf(
  nkeep      ~ b,
  ningroup   ~ 0.5 * b + wi,
  nuniversal ~ 0.3 * b + 0.6 * wi + 0.9 * wo,
  wi ~ 1,
  wo ~ 1
)

# Lower-bounded priors: ensure positivity of activations (guard G4 condition)
# wi > 0 required: ingroup utility = 0.5 + wi > 0 always when wi > 0
# wo > 0 required: keeps universal utility > 0 for positive wo given wi > 0
prior_spec <- c(
  brms::set_prior("normal(1, 0.5)", class = "b", nlpar = "wi", lb = 0),
  brms::set_prior("normal(0.5, 0.5)", class = "b", nlpar = "wo", lb = 0)
)

cat("Model and formula constructed.\n")
cat("  choice_rule = 'simple' -> Luce rule (P(k) proportional to U(k))\n")
cat("  fixed_parameters$b = 1.0 -> numeraire via constant() prior\n")
cat("  wi, wo: identity links with lower bounds at 0 (positivity)\n\n")

# ==============================================================================
# FIT
# ==============================================================================
cat("--- Fitting model (2 chains x 1000 iter) ---\n")

fit <- bmm(
  formula  = formula,
  data     = dat,
  model    = model,
  prior    = prior_spec,
  chains   = 2L,
  iter     = 1000L,
  warmup   = 500L,
  seed     = 2207L,
  backend  = "cmdstanr",
  refresh  = 100
)

# ==============================================================================
# RESULTS
# ==============================================================================
cat("\n--- Diagnostics ---\n")
rhat_max  <- max(brms::rhat(fit),       na.rm = TRUE)
ess_min   <- min(brms::neff_ratio(fit), na.rm = TRUE)
n_divs    <- sum(brms::nuts_params(fit, pars = "divergent__")$Value)
cat(sprintf("  Max R-hat:     %.3f\n", rhat_max))
cat(sprintf("  Min ESS ratio: %.3f\n", ess_min))
cat(sprintf("  Divergences:   %d\n", n_divs))

post <- posterior_summary(fit)

extract_row <- function(par, true_val) {
  rows <- rownames(post)[grepl(paste0("^b_", par, "_Intercept$"), rownames(post))]
  if (length(rows) == 0) {
    rows <- rownames(post)[grepl(par, rownames(post))][1]
  }
  r     <- post[rows[1], , drop = FALSE]
  est   <- r[1, "Estimate"]
  lo    <- r[1, "Q2.5"]
  hi    <- r[1, "Q97.5"]
  covered <- (lo <= true_val && true_val <= hi)
  data.frame(
    param      = par,
    true_value = true_val,
    estimate   = round(est,  4),
    ci_lower   = round(lo,   4),
    ci_upper   = round(hi,   4),
    covered    = covered,
    max_rhat   = round(rhat_max, 4),
    divergences = n_divs
  )
}

res <- rbind(
  extract_row("wi", WI_TRUE),
  extract_row("wo", WO_TRUE)
)

cat("\n--- Recovery results ---\n")
cat(sprintf("  wi: true=%.2f  est=%.3f  95%%CI=[%.3f, %.3f]  covered=%s\n",
            WI_TRUE, res$estimate[1], res$ci_lower[1], res$ci_upper[1], res$covered[1]))
cat(sprintf("  wo: true=%.2f  est=%.3f  95%%CI=[%.3f, %.3f]  covered=%s\n",
            WO_TRUE, res$estimate[2], res$ci_lower[2], res$ci_upper[2], res$covered[2]))

cat("\n--- Interpretation ---\n")
cat("R9 (corrected): m3(choice_rule='simple') + identity links = native Luce rule.\n")
cat("  log(nkeep * 1) = log(U_keep); multinomial softmax over logs = Luce rule.\n")
cat("  No raw brms needed for welfare-weight models with positive utilities.\n\n")
cat("R10 (corrected): fixed_parameters$b = 1.0 is the supported numeraire mechanism.\n")
cat("  fixed_pars_priors() (helpers-prior.R:147) converts to constant(1) prior.\n")
cat("  b appears in formulas (nkeep ~ b, etc.) as the utility numeraire.\n\n")

# Save results
out_file <- "local/exploration/m3-utility/results/22_welfare_bmm_simple.csv"
write.csv(res, out_file, row.names = FALSE)
cat(sprintf("Results saved to %s\n", out_file))
