# 18_prelec_hierarchical.R — WP2 (Round 6)
#
# Hierarchical Prelec recovery: alpha in {0.6, 1.0} (true effect / null),
# 2 reps each (4 fits total), variable set sizes (guard G2 design).
#
# Model: Prelec probability-weighting with Gumbel noise on activations.
#   log P(i) proportional to (b + a + c for corr; b + a for other; b for npl)
#                            + log(w(p_i))
#   where log(w(p)) = -((-log p)^alpha)  [Prelec 1998]
#   and p_i = n_i / (n_corr + n_other + n_npl)
#
# Design: n_other in {2, 4, 6} cycling — variable set sizes to satisfy guard G2.
# Population model (no per-subject random effects on alpha) for tractability.
#
# alpha = 0.6: genuine probability-weighting (< 1 = overweighting rare,
#              underweighting common categories)
# alpha = 1.0: null (standard softmax / no probability weighting)
#
# Results saved to:
#   results/18_prelec_alpha06_rep1.csv
#   results/18_prelec_alpha06_rep2.csv
#   results/18_prelec_alpha10_rep1.csv
#   results/18_prelec_alpha10_rep2.csv
#
# Run from repo root:
#   Rscript local/exploration/m3-utility/18_prelec_hierarchical.R

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

N_SUBJ   <- 30L
N_TRIALS <- 90L   # 30 per set-size level (3 levels)
TRUE_B   <- 0.0; TRUE_A <- 1.5; TRUE_C <- 2.0

# Variable set sizes: n_other cycles through {2, 4, 6}
# n_corr = 1 always; n_npl = n_other
N_SS_LEVELS  <- 3L
N_PER_SS     <- N_TRIALS %/% N_SS_LEVELS    # = 30 trials per set-size level

log_Z_fn <- function(act, n) {
  lv <- act + log(n); lv[1L] + log(sum(exp(lv - lv[1L])))
}

log_prelec <- function(p, alpha) -((-log(p))^alpha)

# ---- Simulate Prelec data ----
sim_prelec <- function(alpha_true, seed) {
  set.seed(seed)
  n_other_s  <- rep(c(2L, 4L, 6L), each = N_PER_SS)   # per-subject
  n_corr_s   <- rep(1L, N_TRIALS)
  n_npl_s    <- n_other_s   # symmetrical

  N_tot  <- N_SUBJ * N_TRIALS
  n_corr <- rep(n_corr_s, N_SUBJ)
  n_other <- rep(n_other_s, N_SUBJ)
  n_npl   <- rep(n_npl_s, N_SUBJ)
  subj    <- rep(seq_len(N_SUBJ), each = N_TRIALS)

  resp <- integer(N_tot)
  for (i in seq_len(N_tot)) {
    N_total <- n_corr[i] + n_other[i] + n_npl[i]
    p_c <- n_corr[i]  / N_total
    p_o <- n_other[i] / N_total
    p_n <- n_npl[i]   / N_total

    act <- c(
      TRUE_B + TRUE_A + TRUE_C + log_prelec(p_c, alpha_true),
      TRUE_B + TRUE_A           + log_prelec(p_o, alpha_true),
      TRUE_B                    + log_prelec(p_n, alpha_true)
    )
    # Normalize (Prelec replaces log(n_i) in softmax)
    resp[i] <- sample.int(3L, 1L, prob = exp(act - act[1L] - log(sum(exp(act - act[1L])))))
  }
  Y <- matrix(0L, N_tot, 3L)
  for (i in seq_len(N_tot)) Y[i, resp[i]] <- 1L
  colnames(Y) <- c("corr", "other", "npl")

  N_total_v <- n_corr + n_other + n_npl
  data.frame(
    subj     = subj,
    n_corr   = n_corr, n_other = n_other, n_npl = n_npl,
    p_corr   = n_corr  / N_total_v,
    p_other  = n_other / N_total_v,
    p_npl    = n_npl   / N_total_v,
    Idx_corr = 1L, Idx_other = 1L, Idx_npl = 1L,
    nTrials  = 1L,
    Y        = I(Y)
  )
}

# ---- brms Prelec formula ----
# Prelec term replaces log(n_i); no log(n_i) in main formula.
# alpha is estimated as a positive parameter (lognormal prior, no log link).
prelec_form <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + (-((-log(p_corr))^alpha)))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + (-((-log(p_other))^alpha))) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + (-((-log(p_npl))^alpha)))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c),
  nlf(other ~ b + a),
  nlf(npl   ~ b),
  b + a + c + alpha ~ 1,
  nl = TRUE
)

prelec_priors <- c(
  prior(constant(0),       nlpar = "b",     class = "b"),
  prior(normal(2, 1),      nlpar = "a",     class = "b"),
  prior(normal(3, 1),      nlpar = "c",     class = "b"),
  prior(lognormal(0, 0.5), nlpar = "alpha", class = "b", lb = 0.05)
)

# ---- Run fit and save ----
run_prelec_fit <- function(d, alpha_true, label, outfile) {
  cat(sprintf("\nFitting Prelec: %s (alpha_true=%.2f)...\n", label, alpha_true))
  fit <- brm(
    prelec_form, data = d,
    family  = multinomial(refcat = NA),
    prior   = prelec_priors,
    chains  = 2L, iter = 1000L, warmup = 500L, cores = 2L,
    refresh = 0, backend = "cmdstanr", silent = 2
  )

  fe  <- fixef(fit)
  rh  <- max(rhat(fit), na.rm = TRUE)
  np  <- nuts_params(fit)
  div <- sum(np$Value[np$Parameter == "divergent__"])

  params <- list(
    a     = list(name = "a",     true = TRUE_A),
    c     = list(name = "c",     true = TRUE_C),
    alpha = list(name = "alpha", true = alpha_true)
  )
  rows <- lapply(params, function(p) {
    rn <- paste0(p$name, "_Intercept")
    if (!rn %in% rownames(fe)) return(NULL)
    lo <- fe[rn, "Q2.5"]; hi <- fe[rn, "Q97.5"]; est <- fe[rn, "Estimate"]
    data.frame(fit = label, param = p$name, true_value = p$true,
               estimate = est, ci_lower = lo, ci_upper = hi,
               covered = (p$true >= lo & p$true <= hi),
               ci_width = hi - lo,
               excl_one = (1.0 < lo | 1.0 > hi),  # for alpha=1.0 null: is 1.0 outside CI?
               max_rhat = rh, divergences = div,
               stringsAsFactors = FALSE)
  })
  res <- do.call(rbind, Filter(Negate(is.null), rows))
  write.csv(res, outfile, row.names = FALSE)
  cat(sprintf("  Saved -> %s\n", outfile))
  cat(sprintf("  Max R-hat: %.3f  Divergences: %d\n", rh, div))

  alpha_row <- res[res$param == "alpha", ]
  cat(sprintf("  alpha: est=%.3f [%.3f, %.3f]  %s\n",
      alpha_row$estimate, alpha_row$ci_lower, alpha_row$ci_upper,
      ifelse(alpha_row$covered, "COVERED", "MISSED")))
  if (alpha_true == 1.0) {
    cat(sprintf("  alpha=1.0 (null): CI %s 1.0  [false positive: %s]\n",
        ifelse(alpha_row$excl_one, "EXCLUDES", "includes"),
        ifelse(alpha_row$excl_one, "YES", "no")))
  }
  res
}

cat("=== WP2: Hierarchical Prelec recovery ===\n")
cat(sprintf("N=%d subj x %d trials, n_other in {2,4,6}, 2 reps per alpha value\n\n",
            N_SUBJ, N_TRIALS))

# alpha = 0.6: genuine Prelec weighting
r1 <- run_prelec_fit(sim_prelec(0.6, 3001L), 0.6, "prelec_alpha06_rep1",
                     "local/exploration/m3-utility/results/18_prelec_alpha06_rep1.csv")
r2 <- run_prelec_fit(sim_prelec(0.6, 3002L), 0.6, "prelec_alpha06_rep2",
                     "local/exploration/m3-utility/results/18_prelec_alpha06_rep2.csv")

# alpha = 1.0: null (standard softmax)
r3 <- run_prelec_fit(sim_prelec(1.0, 3003L), 1.0, "prelec_alpha10_rep1",
                     "local/exploration/m3-utility/results/18_prelec_alpha10_rep1.csv")
r4 <- run_prelec_fit(sim_prelec(1.0, 3004L), 1.0, "prelec_alpha10_rep2",
                     "local/exploration/m3-utility/results/18_prelec_alpha10_rep2.csv")

# ---- Summary ----
cat("\n=== WP2 Summary ===\n")
all_alpha <- rbind(r1, r2, r3, r4)
all_alpha <- all_alpha[all_alpha$param == "alpha", ]
cat("\nalpha recovery results:\n")
print(all_alpha[, c("fit", "true_value", "estimate", "ci_lower", "ci_upper",
                    "covered", "excl_one", "max_rhat")])

fp_count <- sum(all_alpha$true_value == 1.0 & all_alpha$excl_one)
cat(sprintf("\nFalse positives at alpha=1.0 (null): %d/2\n", fp_count))
tp_count <- sum(all_alpha$true_value == 0.6 & all_alpha$estimate < 0.9 & all_alpha$ci_lower < 0.8)
cat(sprintf("alpha=0.6 recovery (CI<0.8): %d/2\n", tp_count))
cat("\nDone.\n")
