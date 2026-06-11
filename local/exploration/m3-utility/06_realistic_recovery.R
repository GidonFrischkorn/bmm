# 06_realistic_recovery.R
# WP1 — Realistic hierarchical recovery
#
# Previous recovery (05_parameter_recovery.R) used pooled fixed-effects on
# aggregated counts (10 responses/row; subject unused). This script redoes
# recovery under realistic conditions:
#
#   - Single categorical response per trial
#   - N = 30 subjects × {50, 100, 200} trials
#   - Hierarchical truth AND model: random effects on a, c, gamma (SDs 0.3/0.3/0.2)
#   - gamma grid: {0, 0.2, 0.5, 0.8} with ≥5 replicates per cell
#     → False-positive rates at gamma = 0 are the primary diagnostic
#   - Same treatment for Prelec alpha ∈ {0.4, 0.6, 1.0, 1.5}
#
# Outputs a summary data.frame with: true value, N_trials, replicate, bias,
# RMSE, CI coverage, and (for gamma/alpha) proportion of 95% CI excluding 0.
#
# Run time: ~30 min on 4 cores (all fits 2 chains x 1000 iter).
#   source("local/exploration/m3-utility/06_realistic_recovery.R")

library(brms)
library(dplyr)

set.seed(42)

# ---- shared parameters -------------------------------------------------------

N_SUBJ     <- 30L
N_CHAINS   <- 2L
N_ITER     <- 1000L
N_WARMUP   <- 500L
N_CORES    <- 2L

MU_A_TRUE  <- 1.5
MU_C_TRUE  <- 2.0
MU_B_TRUE  <- 0.0
SD_A_TRUE  <- 0.3
SD_C_TRUE  <- 0.3

# ---- helper: softmax M3 log-normaliser ---------------------------------------

log_Z_m3 <- function(act, n) {
  lv <- act + log(n)
  lv[1] + log(sum(exp(lv - lv[1])))   # stable log-sum-exp
}

# ==============================================================================
# PROTOTYPE A: Hierarchical EU recovery
# ==============================================================================

cat("============================================================\n")
cat("PROTOTYPE A — Hierarchical EU activation (gamma) recovery\n")
cat("============================================================\n\n")

# Grid -------------------------------------------------------------------------
gamma_grid   <- c(0, 0.2, 0.5, 0.8)
trial_grid   <- c(50L, 100L, 200L)
n_replicates <- 5L

# Priors for hierarchical model ------------------------------------------------
# Group means as fixed effects; per-subject deviations as random effects.
# b fixed at 0 (scale identification).
hier_priors_eu <- c(
  prior(constant(0),   nlpar = "b",      class = "b"),
  prior(normal(2, 1),  nlpar = "a",      class = "b"),
  prior(normal(3, 1),  nlpar = "c",      class = "b"),
  prior(normal(0, 1),  nlpar = "gamma",  class = "b"),
  prior(normal(0, 0.5), nlpar = "a",     class = "sd"),
  prior(normal(0, 0.5), nlpar = "c",     class = "sd"),
  prior(normal(0, 0.3), nlpar = "gamma", class = "sd")
)

# Hierarchical brms formula ----------------------------------------------------
# (1 | subj) on a, c, gamma; b fixed.
eu_hier_formula <- bf(
  resp | trials(1) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + gamma * V_corr),
  nlf(other ~ b + a     + gamma * V_other),
  nlf(npl   ~ b),
  b     ~          1,
  a     ~          1 + (1 | subj),
  c     ~          1 + (1 | subj),
  gamma ~          1 + (1 | subj),
  nl = TRUE
)

# Simulate function for Prototype A hierarchical data --------------------------
simulate_hier_eu <- function(n_trials, gamma_true, sd_gamma = 0.2,
                              seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  # Draw subject-level parameters
  a_subj     <- rnorm(N_SUBJ, MU_A_TRUE, SD_A_TRUE)
  c_subj     <- rnorm(N_SUBJ, MU_C_TRUE, SD_C_TRUE)
  gamma_subj <- rnorm(N_SUBJ, gamma_true, sd_gamma)

  N <- N_SUBJ * n_trials

  subj_idx <- rep(seq_len(N_SUBJ), each = n_trials)
  V_corr   <- rep(c(2, 1), length.out = N)
  V_other  <- 3 - V_corr

  n_corr  <- rep(1L, N)
  n_other <- rep(4L, N)
  n_npl   <- rep(5L, N)

  resp <- integer(N)
  for (i in seq_len(N)) {
    s   <- subj_idx[i]
    act <- c(
      MU_B_TRUE + a_subj[s] + c_subj[s] + gamma_subj[s] * V_corr[i],
      MU_B_TRUE + a_subj[s]             + gamma_subj[s] * V_other[i],
      MU_B_TRUE
    )
    n_i  <- c(n_corr[i], n_other[i], n_npl[i])
    lZ   <- log_Z_m3(act, n_i)
    p    <- exp(act + log(n_i) - lZ)
    resp[i] <- sample.int(3L, 1L, prob = p)
  }

  # Encode as 3-category count matrix with nTrials = 1
  Y <- matrix(0L, N, 3L)
  for (i in seq_len(N)) Y[i, resp[i]] <- 1L
  colnames(Y) <- c("corr", "other", "npl")

  data.frame(
    subj    = subj_idx,
    V_corr  = V_corr,
    V_other = V_other,
    n_corr  = n_corr,
    n_other = n_other,
    n_npl   = n_npl,
    Idx_corr  = 1L,
    Idx_other = 1L,
    Idx_npl   = 1L,
    resp    = I(Y)   # 1-trial multinomial (single categorical response)
  )
}

# Run recovery grid for Prototype A -------------------------------------------
results_a <- list()
idx <- 1L

for (gam in gamma_grid) {
  for (nt in trial_grid) {
    for (rep_i in seq_len(n_replicates)) {
      cat(sprintf(
        "  gamma=%.1f | N_trials=%d | rep=%d ...\n", gam, nt, rep_i
      ))

      # Simulate data
      seed_i <- 1000L * which(gamma_grid == gam) +
                 10L * which(trial_grid == nt) + rep_i
      dat <- simulate_hier_eu(n_trials = nt, gamma_true = gam,
                               sd_gamma = 0.2, seed = seed_i)
      dat$resp <- as.matrix(dat$resp)

      # Add nTrials = 1 (needed by multinomial family)
      dat$nTrials <- 1L

      # Fit
      fit <- suppressWarnings(brm(
        eu_hier_formula,
        data    = dat,
        family  = multinomial(refcat = NA),
        prior   = hier_priors_eu,
        chains  = N_CHAINS,
        iter    = N_ITER,
        warmup  = N_WARMUP,
        cores   = N_CORES,
        refresh = 0,
        backend = "cmdstanr",
        silent  = 2
      ))

      fe    <- fixef(fit)
      gamma_row <- fe["gamma_Intercept", ]
      a_row     <- fe["a_Intercept",     ]
      c_row     <- fe["c_Intercept",     ]

      # Extract SD of gamma random effect
      re_summary  <- as.data.frame(VarCorr(fit, summary = TRUE)$subj$sd)
      gamma_sd_est <- re_summary["gamma_Intercept", "Estimate"]

      covered_gamma <- gam >= gamma_row["Q2.5"] & gam <= gamma_row["Q97.5"]
      excl_zero     <- gamma_row["Q2.5"] > 0 | gamma_row["Q97.5"] < 0

      results_a[[idx]] <- data.frame(
        param        = "gamma",
        true_value   = gam,
        n_trials     = nt,
        replicate    = rep_i,
        estimate     = gamma_row["Estimate"],
        q2.5         = gamma_row["Q2.5"],
        q97.5        = gamma_row["Q97.5"],
        bias         = gamma_row["Estimate"] - gam,
        covered      = covered_gamma,
        excl_zero    = excl_zero,
        max_rhat     = max(rhat(fit), na.rm = TRUE)
      )
      idx <- idx + 1L
    }
  }
}

res_a <- do.call(rbind, results_a)

cat("\n--- Prototype A: Hierarchical Recovery Summary ---\n\n")
cat("Coverage and false-positive rates by gamma value:\n\n")

sum_a <- res_a |>
  group_by(true_value, n_trials) |>
  summarise(
    mean_bias    = mean(bias),
    rmse         = sqrt(mean(bias^2)),
    coverage     = mean(covered),
    false_pos    = mean(excl_zero[true_value == 0]),
    tp_rate      = mean(excl_zero[true_value != 0]),
    mean_rhat    = mean(max_rhat),
    .groups      = "drop"
  )
print(sum_a, n = Inf)

cat("\nKey diagnostics (gamma = 0 false-positive rates):\n")
fp <- res_a |>
  filter(true_value == 0) |>
  group_by(n_trials) |>
  summarise(fp_rate = mean(excl_zero), .groups = "drop")
print(fp)

# ==============================================================================
# PROTOTYPE B: Hierarchical Prelec recovery
# ==============================================================================

cat("\n============================================================\n")
cat("PROTOTYPE B — Hierarchical Prelec weighting (alpha) recovery\n")
cat("============================================================\n\n")

alpha_grid   <- c(0.4, 0.6, 1.0, 1.5)
SD_ALPHA_TRUE <- 0.2

hier_priors_prelec <- c(
  prior(constant(0),         nlpar = "b",     class = "b"),
  prior(normal(2, 1),        nlpar = "a",     class = "b"),
  prior(normal(3, 1),        nlpar = "c",     class = "b"),
  prior(lognormal(0, 0.5),   nlpar = "alpha", class = "b",  lb = 0.01),
  prior(normal(0, 0.5),      nlpar = "a",     class = "sd"),
  prior(normal(0, 0.5),      nlpar = "c",     class = "sd"),
  prior(normal(0, 0.2),      nlpar = "alpha", class = "sd", lb = 0)
)

prelec_hier_formula <- bf(
  resp | trials(1) ~
    Idx_corr  * (corr  + (-((-log(p_corr))^alpha)))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + (-((-log(p_other))^alpha))) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + (-((-log(p_npl))^alpha)))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c),
  nlf(other ~ b + a),
  nlf(npl   ~ b),
  b     ~ 1,
  a     ~ 1 + (1 | subj),
  c     ~ 1 + (1 | subj),
  alpha ~ 1 + (1 | subj),
  nl = TRUE
)

softmax_prelec_log <- function(a, n, alpha) {
  N_tot <- sum(n)
  p     <- n / N_tot
  log_w <- -((-log(p))^alpha)
  lv    <- a + log_w
  lv - (lv[1] + log(sum(exp(lv - lv[1]))))   # log-prob
}

simulate_hier_prelec <- function(n_trials, alpha_true,
                                  sd_alpha = SD_ALPHA_TRUE, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)

  a_subj     <- rnorm(N_SUBJ, MU_A_TRUE, SD_A_TRUE)
  c_subj     <- rnorm(N_SUBJ, MU_C_TRUE, SD_C_TRUE)
  alpha_subj <- pmax(0.05, rnorm(N_SUBJ, alpha_true, sd_alpha))

  N        <- N_SUBJ * n_trials
  subj_idx <- rep(seq_len(N_SUBJ), each = n_trials)

  n_corr  <- rep(1L, N)
  n_other <- rep(c(2L, 6L), length.out = N)
  n_npl   <- rep(c(2L, 6L), length.out = N)

  N_total  <- n_corr + n_other + n_npl
  p_corr   <- n_corr  / N_total
  p_other  <- n_other / N_total
  p_npl    <- n_npl   / N_total

  resp <- integer(N)
  for (i in seq_len(N)) {
    s   <- subj_idx[i]
    act <- c(MU_B_TRUE + a_subj[s] + c_subj[s],
             MU_B_TRUE + a_subj[s],
             MU_B_TRUE)
    n_i  <- c(n_corr[i], n_other[i], n_npl[i])
    lp   <- softmax_prelec_log(act, n_i, alpha_subj[s])
    resp[i] <- sample.int(3L, 1L, prob = exp(lp))
  }

  Y <- matrix(0L, N, 3L)
  for (i in seq_len(N)) Y[i, resp[i]] <- 1L
  colnames(Y) <- c("corr", "other", "npl")

  data.frame(
    subj    = subj_idx,
    n_corr  = n_corr,
    n_other = n_other,
    n_npl   = n_npl,
    N_total = N_total,
    p_corr  = p_corr,
    p_other = p_other,
    p_npl   = p_npl,
    Idx_corr  = 1L,
    Idx_other = 1L,
    Idx_npl   = 1L,
    resp    = I(Y)
  )
}

results_b <- list()
idx <- 1L

for (alp in alpha_grid) {
  for (nt in trial_grid) {
    for (rep_i in seq_len(n_replicates)) {
      cat(sprintf(
        "  alpha=%.1f | N_trials=%d | rep=%d ...\n", alp, nt, rep_i
      ))

      seed_i <- 2000L * which(alpha_grid == alp) +
                 10L  * which(trial_grid == nt) + rep_i
      dat <- simulate_hier_prelec(n_trials = nt, alpha_true = alp,
                                   sd_alpha = SD_ALPHA_TRUE, seed = seed_i)
      dat$resp    <- as.matrix(dat$resp)
      dat$nTrials <- 1L

      fit <- suppressWarnings(brm(
        prelec_hier_formula,
        data    = dat,
        family  = multinomial(refcat = NA),
        prior   = hier_priors_prelec,
        chains  = N_CHAINS,
        iter    = N_ITER,
        warmup  = N_WARMUP,
        cores   = N_CORES,
        refresh = 0,
        backend = "cmdstanr",
        silent  = 2
      ))

      fe        <- fixef(fit)
      alpha_row <- fe["alpha_Intercept", ]

      # alpha=1 is the null (standard softmax); check whether 95% CI excludes 1
      excl_one <- alpha_row["Q2.5"] > 1 | alpha_row["Q97.5"] < 1
      covered  <- alp >= alpha_row["Q2.5"] & alp <= alpha_row["Q97.5"]

      results_b[[idx]] <- data.frame(
        param      = "alpha",
        true_value = alp,
        n_trials   = nt,
        replicate  = rep_i,
        estimate   = alpha_row["Estimate"],
        q2.5       = alpha_row["Q2.5"],
        q97.5      = alpha_row["Q97.5"],
        bias       = alpha_row["Estimate"] - alp,
        covered    = covered,
        excl_one   = excl_one,
        max_rhat   = max(rhat(fit), na.rm = TRUE)
      )
      idx <- idx + 1L
    }
  }
}

res_b <- do.call(rbind, results_b)

cat("\n--- Prototype B: Hierarchical Recovery Summary ---\n\n")
cat("Coverage and false-positive rates by alpha value:\n\n")

sum_b <- res_b |>
  group_by(true_value, n_trials) |>
  summarise(
    mean_bias   = mean(bias),
    rmse        = sqrt(mean(bias^2)),
    coverage    = mean(covered),
    excl_one_rate = mean(excl_one),
    mean_rhat   = mean(max_rhat),
    .groups     = "drop"
  )
print(sum_b, n = Inf)

cat("\nKey diagnostics (alpha = 1 false-positive rate; 'excl_one' should be ~0.05):\n")
fp_b <- res_b |>
  filter(true_value == 1.0) |>
  group_by(n_trials) |>
  summarise(fp_rate = mean(excl_one), .groups = "drop")
print(fp_b)

# ==============================================================================
# Summary
# ==============================================================================

cat("\n============================================================\n")
cat("OVERALL SUMMARY\n")
cat("============================================================\n\n")

cat("Prototype A (EU / gamma):\n")
cat("  gamma=0 false-positive rates across N_trials:\n")
print(fp, row.names = FALSE)
cat("\n")

cat("  Coverage at gamma={0.2, 0.5, 0.8} (should be ~0.95):\n")
cov_a <- res_a |>
  filter(true_value != 0) |>
  group_by(true_value, n_trials) |>
  summarise(coverage = mean(covered), bias = mean(bias), .groups = "drop")
print(cov_a, n = Inf)

cat("\nPrototype B (Prelec / alpha):\n")
cat("  alpha=1 false-positive rates across N_trials:\n")
print(fp_b, row.names = FALSE)
cat("\n")

cat("  Coverage at alpha={0.4, 0.6, 1.5} (should be ~0.95):\n")
cov_b <- res_b |>
  filter(true_value != 1.0) |>
  group_by(true_value, n_trials) |>
  summarise(coverage = mean(covered), bias = mean(bias), .groups = "drop")
print(cov_b, n = Inf)

cat("\nDone.\n")
