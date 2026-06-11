# WP4 (Round 2): Time-varying parameters — auto-link bypass demo
#
# Demonstrates the WP4 bypass implemented in mpt_to_brms():
#   When a predictor formula's RHS references symbols that are themselves
#   keys in predictor_formulas, the parameter is treated as "time-varying":
#   nlf(D ~ user_rhs) is emitted directly, skipping the automatic
#   inv_logit(lD) reparameterisation.  The user's expression owns the (0,1)
#   constraint.
#
# Model: one-tree binomial MPT with exponential detection growth
#   Tree: correct = D + (1-D)*Gcorr;  incorrect = (1-D)*(1-Gcorr)
#   D(t) = inv_logit(lDmax) * (1 - exp(-exp(lrate) * ptime))
#   Gcorr = 1/rsize  [design-fixed; declared as covariate]
#   ptime             [presentation time, passes through as data column in brms]
#   lDmax ~ N(logit(0.80), 0.5) on log scale — log-odds of asymptote
#   lrate ~ N(0, 0.5) on log scale — log of decay rate
#
# Simulation:
#   40 participants x presentation times {0.2, 0.5, 1, 2} s
#               x response-set sizes {2, 4}  (Gcorr = 1/rsize)
#               x 25 trials per cell
#   True: lDmax = 1.4, lrate = 0.0, random-intercept SDs = 0.3 on both
#
# Acceptance: group-mean lDmax and lrate inside 95% CIs; max Rhat <= 1.02;
#   posterior-mean D(t) monotonically increasing in t at both rsize values.
#
# Run from repo root:  Rscript local/exploration/mpt/11_time_varying_demo.R

suppressMessages(library(brms))
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))

source("local/exploration/mpt/03_mpt_parser.R", local = TRUE)

# =============================================================================
# 1. True parameters and simulation
# =============================================================================
lDmax_true <- 1.4
lrate_true <- 0.0
sd_lDmax   <- 0.3
sd_lrate   <- 0.3

ptimes <- c(0.2, 0.5, 1.0, 2.0)
rsizes <- c(2L, 4L)
n_subj  <- 40L
n_trials <- 25L

cat("=== Time-varying MPT (exponential detection growth) ===\n\n")
cat(sprintf("True: lDmax = %.2f, lrate = %.2f\n", lDmax_true, lrate_true))
cat(sprintf("      D(ptime) = inv_logit(%.2f) * (1 - exp(-exp(%.2f) * ptime))\n",
            lDmax_true, lrate_true))
cat(sprintf("Gcorr = 1/rsize; response-set sizes: %s\n", paste(rsizes, collapse = ", ")))
cat(sprintf("Presentation times: %s s\n", paste(ptimes, collapse = ", ")))

set.seed(42)
lDmax_i <- rnorm(n_subj, lDmax_true, sd_lDmax)
lrate_i <- rnorm(n_subj, lrate_true, sd_lrate)
Dmax_i  <- plogis(lDmax_i)

# Compute true D(t, i) = Dmax_i * (1 - exp(-exp(lrate_i) * t))
D_fun <- function(t, Dmax, lrate) Dmax * (1 - exp(-exp(lrate) * t))

# Verify D(t) is monotonically increasing in t for all participants and rsizes
for (i in seq_len(n_subj)) {
  Dt <- D_fun(ptimes, Dmax_i[i], lrate_i[i])
  stopifnot(all(diff(Dt) >= 0))
}
cat("True D(t) is monotonically increasing for all participants.\n")

sim_dat <- expand.grid(
  id    = seq_len(n_subj),
  ptime = ptimes,
  rsize = rsizes
) |>
  as_tibble() |>
  mutate(
    id    = factor(id),
    Gcorr = 1 / rsize,
    Dmax  = Dmax_i[as.integer(id)],
    lrate_val = lrate_i[as.integer(id)],
    D     = D_fun(ptime, Dmax, lrate_val),
    # Binomial success probability for "correct" category
    p_correct = D + (1 - D) * Gcorr,
    correct   = rbinom(n(), n_trials, p_correct),
    n         = n_trials
  ) |>
  select(-Dmax, -lrate_val, -D, -p_correct)

cat(sprintf("\nSimulated data: %d rows (%d subj x %d ptimes x %d rsizes)\n",
            nrow(sim_dat), n_subj, length(ptimes), length(rsizes)))
cat("First 6 rows:\n")
print(head(sim_dat, 6))

# =============================================================================
# 2. Build MPT spec with covariate Gcorr
#    D is a bypass parameter (time-varying): predictor formula RHS references
#    lDmax and lrate, which are also keys in predictor_formulas.
# =============================================================================
tree_tv <- mpt_tree("main", list(
  correct   = "D + (1 - D) * Gcorr",
  incorrect = "(1 - D) * (1 - Gcorr)"
))
spec_tv <- mpt(
  trees      = list(tree_tv),
  covariates = c("Gcorr")
  # condition = NULL: single tree, no condition column
)
print(spec_tv)
stopifnot(setequal(spec_tv$params, "D"))
stopifnot(setequal(spec_tv$covariates, "Gcorr"))

# Predictor formulas:
#   D     ~ non-linear compound (bypass): references lDmax and lrate
#   lDmax ~ 1 + (1|id)   sub-parameter for D's formula
#   lrate ~ 1 + (1|id)   sub-parameter for D's formula
# ptime is a data column referenced in D's formula but NOT in predictor_formulas;
# it is treated by brms as a data covariate in the nlf() formula for D.
out_tv <- mpt_to_brms(
  spec_tv,
  predictor_formulas = list(
    D     = ~ inv_logit(lDmax) * (1 - exp(-exp(lrate) * ptime)),
    lDmax = ~ 1 + (1 | id),
    lrate = ~ 1 + (1 | id)
  ),
  response_col = "correct",
  trials_col   = "n"
)

cat("\nGenerated brms formula (D is bypassed — no inv_logit(lD) wrapper):\n")
print(out_tv$brms_formula)

# Verify bypass: 'D' must appear as nlf(D ~ inv_logit(lDmax) * ...)
# NOT as nlf(D ~ inv_logit(lD)) + lf(lD ~ ...)
tv_str <- paste(capture.output(print(out_tv$brms_formula)), collapse = " ")
stopifnot(!grepl("inv_logit\\(lD\\)", tv_str))
stopifnot(grepl("lDmax", tv_str))
stopifnot(grepl("lrate",  tv_str))
cat("\nBypass verification: PASS — D formula contains lDmax/lrate, no inv_logit(lD).\n")

dat_prep <- out_tv$data_prep(sim_dat)
stopifnot(all(dat_prep$.ind_main == 1L))

# =============================================================================
# 3. Priors — logistic(0,1) on lDmax intercept; normal(0,1) on lrate intercept;
#    half-normal on SDs
# =============================================================================
priors_tv <- c(
  prior("logistic(0, 1)", nlpar = "lDmax", class = "b", coef = "Intercept"),
  prior("normal(0, 1)",   nlpar = "lrate", class = "b", coef = "Intercept"),
  prior("normal(0, 1)",   nlpar = "lDmax", class = "sd"),
  prior("normal(0, 1)",   nlpar = "lrate", class = "sd")
)

cat("\nPriors:\n")
print(priors_tv)

# =============================================================================
# 4. Fit — 4 chains x 2000 iter (1000 warmup)
# =============================================================================
cat("\nFitting time-varying MPT (4 chains x 2000 iter, 1000 warmup)...\n")
fit_tv <- brm(
  formula  = out_tv$brms_formula,
  data     = dat_prep,
  prior    = priors_tv,
  family   = binomial(),
  backend  = "cmdstanr",
  chains   = 4,
  iter     = 2000,
  warmup   = 1000,
  cores    = 4,
  seed     = 42,
  silent   = 2,
  refresh  = 0
)

cat("\nModel summary:\n")
print(summary(fit_tv))

# =============================================================================
# 5. Recovery check
# =============================================================================
draws <- as_draws_df(fit_tv)

lDmax_hat <- mean(draws$b_lDmax_Intercept)
lrate_hat <- mean(draws$b_lrate_Intercept)
lDmax_ci  <- quantile(draws$b_lDmax_Intercept, c(0.025, 0.975))
lrate_ci  <- quantile(draws$b_lrate_Intercept, c(0.025, 0.975))
max_rhat  <- max(brms::rhat(fit_tv), na.rm = TRUE)

cat("\n=== Recovery Summary ===\n")
cat(sprintf("%-10s  %-6s  %-8s  %-18s  %s\n",
            "Param", "True", "Estimate", "95% CI", "Cover?"))
cat(strrep("-", 60), "\n")
cat(sprintf("%-10s  %-6.2f  %-8.4f  [%-6.4f, %-6.4f]  %s\n",
            "lDmax", lDmax_true, lDmax_hat, lDmax_ci[1], lDmax_ci[2],
            lDmax_true >= lDmax_ci[1] & lDmax_true <= lDmax_ci[2]))
cat(sprintf("%-10s  %-6.2f  %-8.4f  [%-6.4f, %-6.4f]  %s\n",
            "lrate", lrate_true, lrate_hat, lrate_ci[1], lrate_ci[2],
            lrate_true >= lrate_ci[1] & lrate_true <= lrate_ci[2]))
cat(strrep("-", 60), "\n")
cat(sprintf("Max Rhat: %.4f (target: <= 1.02)\n", max_rhat))

# =============================================================================
# 6. Posterior-mean D(t) monotonicity check
# =============================================================================
Dmax_pm <- plogis(lDmax_hat)
rate_pm  <- exp(lrate_hat)

D_pm_by_time <- tibble(
  ptime   = ptimes,
  rsize2  = Dmax_pm * (1 - exp(-rate_pm * ptimes)),
  rsize4  = Dmax_pm * (1 - exp(-rate_pm * ptimes))
)

cat("\nPosterior-mean D(t) at posterior-mean parameters:\n")
cat(sprintf("  Dmax = inv_logit(%.4f) = %.4f;  rate = exp(%.4f) = %.4f\n",
            lDmax_hat, Dmax_pm, lrate_hat, rate_pm))
for (pt in ptimes) {
  Dt <- Dmax_pm * (1 - exp(-rate_pm * pt))
  cat(sprintf("  D(t=%.1f s) = %.4f\n", pt, Dt))
}

D_curve <- Dmax_pm * (1 - exp(-rate_pm * ptimes))
mono_increasing <- all(diff(D_curve) > 0)
cat(sprintf("\nD(t) monotonically increasing: %s\n", mono_increasing))

# Overall verdict
all_pass <- (lDmax_true >= lDmax_ci[1]) & (lDmax_true <= lDmax_ci[2]) &
            (lrate_true >= lrate_ci[1]) & (lrate_true <= lrate_ci[2]) &
            (max_rhat <= 1.02) &
            mono_increasing
cat(sprintf("\n=== Overall verdict: %s ===\n",
            ifelse(all_pass, "PASS — all acceptance criteria met",
                   "PARTIAL — check individual criteria above")))
cat("\nDone.\n")
