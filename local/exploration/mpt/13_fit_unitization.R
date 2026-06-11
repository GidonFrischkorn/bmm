# 13_fit_unitization.R
#
# Fit the Unitization model from Al Hadhrami, Bartsch & Oberauer (2025) to
# Experiment 1 response frequencies using the MPT prototype.
#
# Source: Al Hadhrami, A., Bartsch, L. M., & Oberauer, K. (2025). A
# multinomial model-based analysis of bindings in working memory.
# Psychological Review. OSF: https://osf.io/nu4sb/
#
# Run from the repository root:
#   Rscript local/exploration/mpt/13_fit_unitization.R

suppressPackageStartupMessages({
  library(brms)
  library(cmdstanr)
})

source("local/exploration/mpt/12_eqn_converter.R")

set.seed(2025)
options(mc.cores = 4)

# ---------------------------------------------------------------------------
# Build model spec and data
# ---------------------------------------------------------------------------
eqn_dir <- "local/exploration/mpt/binding_models"
spec     <- build_binding_spec(file.path(eqn_dir, "unitization.eqn"), constants)
long     <- build_long_data(file.path(eqn_dir, "exp1_response_frequencies.csv"))

cat("Unitization model spec:\n")
print(spec)

# ---------------------------------------------------------------------------
# Predictor formulas: all 12 free params with full correlated RE (|p| id)
# ---------------------------------------------------------------------------
pred_forms <- setNames(
  lapply(spec$params, function(p) ~ 1 + (1 | p | id)),
  spec$params
)

out <- mpt_to_brms(
  spec,
  predictor_formulas = pred_forms,
  response_col       = spec$resp_cats[1],
  trials_col         = "n"
)

data_fit <- out$data_prep(long)
cat("\nData dimensions:", nrow(data_fit), "rows,", ncol(data_fit), "cols\n")

cat("\nbrms formula:\n")
print(out$brms_formula)

# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
# normal(0,1) on probit-scale intercepts (matches paper's mu ~ dnorm(0,1))
# student_t(3,0,1) on SDs; lkj(1) on correlation matrix
l_params <- as.character(out$link_params)  # values = l-param names
priors <- do.call(c, lapply(l_params, function(lp) {
  brms::prior_string("normal(0, 1)", nlpar = lp, class = "b", coef = "Intercept")
}))
# SD priors per nlpar (required for NL models)
for (lp in l_params) {
  priors <- priors +
    brms::prior_string("student_t(3, 0, 1)", class = "sd", nlpar = lp)
}
# Correlation matrix prior for the shared |p| correlation group
priors <- priors +
  brms::prior_string("lkj(1)", class = "cor", group = "id")

cat("\nPriors:\n")
print(priors)

# ---------------------------------------------------------------------------
# Fit (commit + push happens before this line)
# ---------------------------------------------------------------------------
cat("\nFitting Unitization model (4 chains × 2000 iter, 1000 warmup) ...\n")
t0 <- proc.time()
fit_unit <- brm(
  formula  = out$brms_formula,
  data     = data_fit,
  family   = out$family_obj,
  prior    = priors,
  chains   = 4,
  iter     = 2000,
  warmup   = 1000,
  adapt_delta = 0.95,
  cores    = 4,
  backend  = "cmdstanr",
  seed     = 2025
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Sampling time: %.1f s\n", elapsed))

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
rhats    <- brms::rhat(fit_unit)
max_rhat <- max(rhats, na.rm = TRUE)
div      <- sum(nuts_params(fit_unit, pars = "divergent__")$Value)
cat(sprintf("\nMax Rhat: %.3f | Divergent transitions: %d\n", max_rhat, div))

# WAIC
waic_unit <- brms::waic(fit_unit)
cat("\nWAIC (Unitization):\n")
print(waic_unit)

# ---------------------------------------------------------------------------
# Posterior summary
# ---------------------------------------------------------------------------
smry <- as.data.frame(fixef(fit_unit))
cat("\nGroup-level intercepts (probit scale):\n")
print(round(smry, 3))

# Save results
results_dir <- "local/exploration/mpt/results"
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(smry,
  file.path(results_dir, "unitization_posterior_summary.csv"))
saveRDS(waic_unit,
  file.path(results_dir, "unitization_waic.rds"))

# Convert intercepts to probability scale for reporting
cat("\nGroup-level parameters on probability scale P = Phi(mu):\n")
prob_smry <- smry[grep("Intercept", rownames(smry)), , drop = FALSE]
prob_smry[, c("Estimate", "Q2.5", "Q97.5")] <-
  pnorm(prob_smry[, c("Estimate", "Q2.5", "Q97.5")])
print(round(prob_smry, 3))

cat("\nDone. Results saved to", results_dir, "\n")
