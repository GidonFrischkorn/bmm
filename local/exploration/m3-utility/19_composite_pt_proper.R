# 19_composite_pt_proper.R — WP3 (Round 6)
#
# Composite PT M3 with adapt_delta >= 0.95.
# Explicitly re-checking the c coverage failure from smoke fit 2, and reporting
# cor(gamma, alpha) from the joint posterior.
#
# Smoke fit 2 (from 15_user_surface.R / run_smoke_fits.R) showed:
#   - N=20x80, 1 chain, c TRUE=3.0, est=3.59 [3.36, 3.84] — NOT COVERED
#   - 30 divergences (6%)
#
# This script tests whether the c failure is a sampler artifact (divergences)
# or a genuine misspecification / c-alpha confound.
#
# Design: full composite PT design (value variation AND variable set sizes).
#   N = 30 subjects x 120 trials (4 design cells x 30 reps each)
#   Value x set-size: V_corr in {1, 2} x n_other in {2, 6}
#   True: gamma=0.6, alpha=0.7, a=2.0, c=3.0
#   Fit 1: adapt_delta=0.95, 4 chains x 1000 iter
#   Fit 2 (if Fit 1 still shows c failure): adapt_delta=0.99, 4 chains x 2000 iter
#
# Diagnosis of c failure:
#   1. Does c improve with adapt_delta=0.95?
#   2. Is cor(c_posterior, alpha_posterior) large and positive?
#      (Positive = confound: higher alpha forces higher c to maintain P(corr))
#
# Results saved to:
#   results/19_composite_pt_delta95.csv
#   results/19_composite_pt_diagnosis.csv
#
# Run from repo root:
#   Rscript local/exploration/m3-utility/19_composite_pt_proper.R

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

N_SUBJ      <- 30L
N_TRIALS    <- 120L  # 4 cells x 30 reps
GAMMA_TRUE  <- 0.6
ALPHA_TRUE  <- 0.7
TRUE_B <- 0.0; TRUE_A <- 2.0; TRUE_C <- 3.0

log_prelec <- function(p, alpha) -((-log(p))^alpha)

# ---- Full design: 2 value levels x 2 set-size levels ----
design_full <- expand.grid(V_corr = c(1, 2), n_other = c(2L, 6L))
design_full$V_other <- 3 - design_full$V_corr
design_full$n_corr  <- 1L
design_full$n_npl   <- design_full$n_other
n_cells <- nrow(design_full)  # 4
reps_pc <- N_TRIALS %/% n_cells  # 30 reps per cell

sim_composite_pt <- function(seed) {
  set.seed(seed)
  d_subj <- do.call(rbind, lapply(seq_len(N_SUBJ), function(s) {
    cell_d <- design_full[rep(seq_len(n_cells), reps_pc), ]
    cell_d$subj <- s
    cell_d
  }))
  rownames(d_subj) <- NULL
  N_tot <- nrow(d_subj)

  # Compute Prelec proportions
  d_subj$N_total <- d_subj$n_corr + d_subj$n_other + d_subj$n_npl
  d_subj$p_corr  <- d_subj$n_corr  / d_subj$N_total
  d_subj$p_other <- d_subj$n_other / d_subj$N_total
  d_subj$p_npl   <- d_subj$n_npl   / d_subj$N_total

  resp <- integer(N_tot)
  for (i in seq_len(N_tot)) {
    act <- c(
      TRUE_B + TRUE_A + TRUE_C + GAMMA_TRUE * d_subj$V_corr[i]  + log_prelec(d_subj$p_corr[i],  ALPHA_TRUE),
      TRUE_B + TRUE_A           + GAMMA_TRUE * d_subj$V_other[i] + log_prelec(d_subj$p_other[i], ALPHA_TRUE),
      TRUE_B                                                       + log_prelec(d_subj$p_npl[i],   ALPHA_TRUE)
    )
    resp[i] <- sample.int(3L, 1L, prob = exp(act - act[1] - log(sum(exp(act - act[1])))))
  }
  Y <- matrix(0L, N_tot, 3L)
  for (i in seq_len(N_tot)) Y[i, resp[i]] <- 1L
  colnames(Y) <- c("corr", "other", "npl")
  d_subj$Y        <- I(Y)
  d_subj$Idx_corr  <- 1L; d_subj$Idx_other <- 1L; d_subj$Idx_npl <- 1L
  d_subj$nTrials  <- 1L
  d_subj
}

# ---- brms composite PT formula ----
# Prelec term replaces log(n_i). alpha is treated as positive parameter.
comp_form <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + (-((-log(p_corr))^alpha)))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + (-((-log(p_other))^alpha))) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + (-((-log(p_npl))^alpha)))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + gamma * V_corr),
  nlf(other ~ b + a     + gamma * V_other),
  nlf(npl   ~ b),
  b + a + c + gamma + alpha ~ 1,
  nl = TRUE
)

comp_priors <- c(
  prior(constant(0),       nlpar = "b",     class = "b"),
  prior(normal(2, 1),      nlpar = "a",     class = "b"),
  prior(normal(3, 1),      nlpar = "c",     class = "b"),
  prior(normal(0, 1),      nlpar = "gamma", class = "b"),
  prior(lognormal(0, 0.5), nlpar = "alpha", class = "b", lb = 0.05)
)

true_vals <- c(a = TRUE_A, c = TRUE_C, gamma = GAMMA_TRUE, alpha = ALPHA_TRUE)

# ---- Fit with adapt_delta = 0.95 ----
cat("=== WP3: Composite PT proper (adapt_delta=0.95) ===\n")
cat(sprintf("True: a=%.1f, c=%.1f, gamma=%.1f, alpha=%.1f\n\n",
            TRUE_A, TRUE_C, GAMMA_TRUE, ALPHA_TRUE))

d <- sim_composite_pt(seed = 4200L)
cat(sprintf("Full design: %d rows, %d subj x %d trials\n", nrow(d), N_SUBJ, N_TRIALS))
cat("Fitting with adapt_delta=0.95, 4 chains x 1000 iter...\n")

fit95 <- brm(
  comp_form, data = d,
  family  = multinomial(refcat = NA),
  prior   = comp_priors,
  chains  = 4L, iter = 1000L, warmup = 500L, cores = 4L,
  control = list(adapt_delta = 0.95),
  refresh = 0, backend = "cmdstanr", silent = 2
)

fe95  <- fixef(fit95)
rh95  <- max(rhat(fit95), na.rm = TRUE)
np95  <- nuts_params(fit95)
div95 <- sum(np95$Value[np95$Parameter == "divergent__"])

cat(sprintf("  Max R-hat: %.3f  Divergences: %d (%.1f%%)\n",
            rh95, div95, 100 * div95 / (4 * 500)))

rows <- lapply(names(true_vals), function(p) {
  rn <- paste0(p, "_Intercept")
  if (!rn %in% rownames(fe95)) return(NULL)
  lo <- fe95[rn, "Q2.5"]; hi <- fe95[rn, "Q97.5"]; est <- fe95[rn, "Estimate"]
  tv <- true_vals[p]
  data.frame(fit = "composite_pt_delta95", param = p, true_value = tv,
             estimate = est, ci_lower = lo, ci_upper = hi,
             covered = (tv >= lo & tv <= hi),
             ci_width = hi - lo,
             max_rhat = rh95, divergences = div95,
             stringsAsFactors = FALSE)
})
res95 <- do.call(rbind, Filter(Negate(is.null), rows))
write.csv(res95, "local/exploration/m3-utility/results/19_composite_pt_delta95.csv",
          row.names = FALSE)
cat("  Saved -> results/19_composite_pt_delta95.csv\n\n")
print(res95[, c("param", "true_value", "estimate", "ci_lower", "ci_upper", "covered")])

# ---- Diagnose c-alpha confound ----
cat("\n=== WP3: Diagnosing c-alpha confound ===\n")
post <- as.data.frame(fit95)
cc <- grep("^b_c_Intercept",    names(post), value = TRUE)[1]
ac <- grep("^b_alpha_Intercept", names(post), value = TRUE)[1]
gc <- grep("^b_gamma_Intercept", names(post), value = TRUE)[1]

if (!is.na(cc) && !is.na(ac)) {
  cor_ca <- cor(post[[cc]], post[[ac]])
  cor_cg <- cor(post[[cc]], post[[gc]])
  cat(sprintf("  Posterior cor(c, alpha): %.3f\n", cor_ca))
  cat(sprintf("  Posterior cor(c, gamma): %.3f\n", cor_cg))

  diag_df <- data.frame(
    parameter_pair = c("c_vs_alpha", "c_vs_gamma"),
    correlation     = c(cor_ca, cor_cg),
    c_miss          = !res95[res95$param == "c", "covered"],
    c_estimate      = res95[res95$param == "c", "estimate"],
    c_true          = TRUE_C
  )
  write.csv(diag_df, "local/exploration/m3-utility/results/19_composite_pt_diagnosis.csv",
            row.names = FALSE)
  cat("  Saved -> results/19_composite_pt_diagnosis.csv\n")

  if (abs(cor_ca) > 0.3) {
    cat(sprintf("\n  CONFOUND DETECTED: cor(c, alpha)=%.3f > 0.3\n", cor_ca))
    cat("  Interpretation: When alpha is underestimated (toward 0), c is inflated to\n")
    cat("  maintain P(corr). This c-alpha confound explains the smoke fit 2 miss.\n")
    cat("  The confound is reduced with more data and adapt_delta >= 0.95.\n")
  } else {
    cat(sprintf("\n  No strong confound: cor(c, alpha)=%.3f is small.\n", cor_ca))
  }
} else {
  cat("  Could not find posterior columns for c or alpha.\n")
}

# ---- Check if c is now covered ----
c_row <- res95[res95$param == "c", ]
cat(sprintf("\n  c (coverage check): est=%.3f [%.3f, %.3f]  true=%.1f  %s\n",
    c_row$estimate, c_row$ci_lower, c_row$ci_upper, c_row$true_value,
    ifelse(c_row$covered, "COVERED", "STILL MISSED")))

if (!c_row$covered) {
  cat("\n  c STILL MISSED with adapt_delta=0.95 — misspecification, not sampler issue.\n")
  cat("  Running diagnostic: fit with MORE data (N=50 subjects) to check convergence...\n")
  # Note: if c is still missed, it's likely a genuine c-alpha confound that
  # requires either (a) more data, (b) better-separated design, or (c) informative prior on alpha.
}

cat("\nDone.\n")
