# 20_stage_confusion_loo.R — WP4 (Round 6)
#
# Stage confusion: 2x2 LOO cross-tabulation.
# Generate data from each model (E = encoding-stage; D = decision-stage),
# fit both models to each dataset, compute LOO, tabulate winners.
#
# Mathematical background (from 08_stage_confusion.R):
#   Model E: gamma * V enters activation (encoding-stage value effect)
#     corr ~ b + a + c + gamma * V_corr
#   Model D: delta * V enters identically (decision-stage; same fixed-effects form)
#     corr ~ b + a + c + delta * V_corr
#   In fixed-effects models, E and D are MATHEMATICALLY IDENTICAL.
#   The LOO comparison must return near-zero ELPD difference and random winners.
#
# 2 reps per generating model = 4 datasets, 8 fits + 4 LOO comparisons.
# N = 30 subjects x 60 trials per rep (smaller N for tractability with 8 fits).
#
# Results saved to:
#   results/20_stage_confusion.csv
#
# Run from repo root:
#   Rscript local/exploration/m3-utility/20_stage_confusion_loo.R

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

N_SUBJ    <- 30L
N_TRIALS  <- 60L
TRUE_B    <- 0.0; TRUE_A <- 1.5; TRUE_C <- 2.0
TRUE_GAM  <- 0.5  # encoding (gamma) or decision (delta) parameter
TRUE_DELTA <- 0.5

V_corr  <- rep(c(2, 1), length.out = N_TRIALS)
V_other <- 3 - V_corr

log_Z_fn <- function(act, n) {
  lv <- act + log(n); lv[1L] + log(sum(exp(lv - lv[1L])))
}

# ---- Simulate from Model E (gamma correlated with c across subjects) ----
sim_E <- function(seed) {
  set.seed(seed)
  a_s <- rnorm(N_SUBJ, TRUE_A, 0.3)
  c_s <- rnorm(N_SUBJ, TRUE_C, 0.3)
  gamma_s <- pmax(-2, TRUE_GAM + 0.5 * (c_s - TRUE_C) + rnorm(N_SUBJ, 0, 0.15))
  gamma_s <- gamma_s + (TRUE_GAM - mean(gamma_s))

  N_tot <- N_SUBJ * N_TRIALS
  subj  <- rep(seq_len(N_SUBJ), each = N_TRIALS)
  Vc    <- rep(V_corr, N_SUBJ)
  Vo    <- rep(V_other, N_SUBJ)

  resp <- integer(N_tot)
  for (i in seq_len(N_tot)) {
    s   <- subj[i]
    act <- c(TRUE_B + a_s[s] + c_s[s] + gamma_s[s] * Vc[i],
             TRUE_B + a_s[s]            + gamma_s[s] * Vo[i],
             TRUE_B)
    n_i <- c(1L, 4L, 5L)
    resp[i] <- sample.int(3L, 1L, prob = exp(act + log(n_i) - log_Z_fn(act, n_i)))
  }
  Y <- matrix(0L, N_tot, 3L)
  for (i in seq_len(N_tot)) Y[i, resp[i]] <- 1L
  colnames(Y) <- c("corr", "other", "npl")
  data.frame(subj = subj, V_corr = Vc, V_other = Vo,
             n_corr = 1L, n_other = 4L, n_npl = 5L,
             Idx_corr = 1L, Idx_other = 1L, Idx_npl = 1L,
             nTrials = 1L, Y = I(Y))
}

# ---- Simulate from Model D (delta independent of c) ----
sim_D <- function(seed) {
  set.seed(seed)
  a_s     <- rnorm(N_SUBJ, TRUE_A, 0.3)
  c_s     <- rnorm(N_SUBJ, TRUE_C, 0.3)
  delta_s <- rnorm(N_SUBJ, TRUE_DELTA, 0.15)   # uncorrelated with c

  N_tot <- N_SUBJ * N_TRIALS
  subj  <- rep(seq_len(N_SUBJ), each = N_TRIALS)
  Vc    <- rep(V_corr, N_SUBJ)
  Vo    <- rep(V_other, N_SUBJ)

  resp <- integer(N_tot)
  for (i in seq_len(N_tot)) {
    s   <- subj[i]
    act <- c(TRUE_B + a_s[s] + c_s[s] + delta_s[s] * Vc[i],
             TRUE_B + a_s[s]            + delta_s[s] * Vo[i],
             TRUE_B)
    n_i <- c(1L, 4L, 5L)
    resp[i] <- sample.int(3L, 1L, prob = exp(act + log(n_i) - log_Z_fn(act, n_i)))
  }
  Y <- matrix(0L, N_tot, 3L)
  for (i in seq_len(N_tot)) Y[i, resp[i]] <- 1L
  colnames(Y) <- c("corr", "other", "npl")
  data.frame(subj = subj, V_corr = Vc, V_other = Vo,
             n_corr = 1L, n_other = 4L, n_npl = 5L,
             Idx_corr = 1L, Idx_other = 1L, Idx_npl = 1L,
             nTrials = 1L, Y = I(Y))
}

# ---- Fixed-effects formulas (E and D have identical functional form) ----
fe_form <- function(param_name) {
  bf(
    Y | trials(nTrials) ~
      Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
    nlf(muother ~
      Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
    nlf(munpl ~
      Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
    nlf(corr  ~ paste0("b + a + c + ", param_name, " * V_corr")),
    nlf(other ~ paste0("b + a     + ", param_name, " * V_other")),
    nlf(npl   ~ "b"),
    as.formula(paste0("b + a + c + ", param_name, " ~ 1")),
    nl = TRUE
  )
}

form_E <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~ Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl   ~ Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + gamma * V_corr),
  nlf(other ~ b + a     + gamma * V_other),
  nlf(npl   ~ b),
  b + a + c + gamma ~ 1,
  nl = TRUE
)

form_D <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~ Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl   ~ Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + delta * V_corr),
  nlf(other ~ b + a     + delta * V_other),
  nlf(npl   ~ b),
  b + a + c + delta ~ 1,
  nl = TRUE
)

priors_E <- c(
  prior(constant(0), nlpar = "b",     class = "b"),
  prior(normal(2, 1), nlpar = "a",    class = "b"),
  prior(normal(3, 1), nlpar = "c",    class = "b"),
  prior(normal(0, 1), nlpar = "gamma", class = "b")
)
priors_D <- c(
  prior(constant(0), nlpar = "b",     class = "b"),
  prior(normal(2, 1), nlpar = "a",    class = "b"),
  prior(normal(3, 1), nlpar = "c",    class = "b"),
  prior(normal(0, 1), nlpar = "delta", class = "b")
)

fit_and_loo <- function(form, priors, dat, label) {
  cat(sprintf("  Fitting %s ...\n", label))
  fit <- suppressWarnings(brm(
    form, data = dat, family = multinomial(refcat = NA),
    prior = priors, chains = 2L, iter = 1000L, warmup = 500L,
    cores = 2L, refresh = 0, backend = "cmdstanr", silent = 2
  ))
  loo_obj <- add_criterion(fit, "loo")$criteria$loo
  loo_obj
}

# ---- Run confusion matrix ----
cat("=== WP4: Stage confusion LOO cross-tabulation ===\n\n")
cat(sprintf("2 reps per model, N=%d subj x %d trials, 2 chains x 1000 iter\n\n",
            N_SUBJ, N_TRIALS))

confusion_rows <- list()

for (rep_i in 1:2) {
  cat(sprintf("--- Replicate %d/2 ---\n", rep_i))

  # Data from Model E
  cat("Model E data:\n")
  dat_E <- sim_E(seed = 1000L + rep_i)
  dat_E$Y <- as.matrix(dat_E$Y)
  loo_EE <- fit_and_loo(form_E, priors_E, dat_E, "Model_E_on_E_data")
  loo_DE <- fit_and_loo(form_D, priors_D, dat_E, "Model_D_on_E_data")
  cmp_E  <- loo_compare(loo_EE, loo_DE)
  winner_E <- rownames(cmp_E)[1]   # "model1" = E, "model2" = D
  winner_E_label <- ifelse(winner_E == "model1", "E", "D")
  elpd_E <- cmp_E[2, "elpd_diff"]
  cat(sprintf("  LOO winner (E data): %s  ELPD diff=%.2f\n", winner_E_label, elpd_E))
  confusion_rows[[length(confusion_rows)+1]] <- data.frame(
    rep = rep_i, gen_model = "E", loo_winner = winner_E_label,
    elpd_diff = elpd_E, stringsAsFactors = FALSE)

  # Data from Model D
  cat("Model D data:\n")
  dat_D <- sim_D(seed = 2000L + rep_i)
  dat_D$Y <- as.matrix(dat_D$Y)
  loo_ED <- fit_and_loo(form_E, priors_E, dat_D, "Model_E_on_D_data")
  loo_DD <- fit_and_loo(form_D, priors_D, dat_D, "Model_D_on_D_data")
  cmp_D  <- loo_compare(loo_ED, loo_DD)
  winner_D <- rownames(cmp_D)[1]
  winner_D_label <- ifelse(winner_D == "model1", "E", "D")
  elpd_D <- cmp_D[2, "elpd_diff"]
  cat(sprintf("  LOO winner (D data): %s  ELPD diff=%.2f\n", winner_D_label, elpd_D))
  confusion_rows[[length(confusion_rows)+1]] <- data.frame(
    rep = rep_i, gen_model = "D", loo_winner = winner_D_label,
    elpd_diff = elpd_D, stringsAsFactors = FALSE)
}

confusion <- do.call(rbind, confusion_rows)
write.csv(confusion, "local/exploration/m3-utility/results/20_stage_confusion.csv",
          row.names = FALSE)
cat("\n  Saved -> results/20_stage_confusion.csv\n")

# ---- Summary ----
cat("\n=== WP4: Confusion matrix ===\n\n")
print(confusion)

tab <- table(gen = confusion$gen_model, winner = confusion$loo_winner)
cat("\n2x2 cross-tab:\n")
print(tab)

indistinguishable <- all(abs(confusion$elpd_diff) < 2)
cat(sprintf("\nMedian |ELPD diff|: %.2f\n", median(abs(confusion$elpd_diff))))
if (indistinguishable) {
  cat("RESULT: E and D are INDISTINGUISHABLE in fixed-effects models (|ELPD diff| < 2).\n")
  cat("  This confirms the mathematical equivalence shown in 08_stage_confusion.R.\n")
  cat("  Design manipulation (encoding-time vs retrieval-time value cues) is needed.\n")
} else {
  cat("RESULT: LOO finds some difference between E and D in this design.\n")
  cat("  Examine: are there subject-level differences driving this?\n")
}
cat("\nDone.\n")
