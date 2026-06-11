# 03_prototype_b_prelec.R
# Prototype B: non-EU choice rule via Prelec probability weighting
#
# Standard softmax M3 uses log(n_i) as an objective-frequency correction.
# This prototype replaces log(n_i) with the log Prelec-weighted subjective
# frequency log(w(n_i / N_total)), where:
#
#   w(p) = exp(-(−ln p)^alpha)     [Prelec (1998) one-parameter weighting]
#   log(w(p)) = -(−ln p)^alpha
#
# alpha = 1: w(p) = p  ->  reduces to standard softmax (log(n_i/N) + const)
# alpha < 1: inverse S-shape — overweights rare alternatives, underweights common ones
# alpha > 1: S-shape — opposite pattern (rare items discounted further)
#
# Run from the repo root (requires a working Stan installation to actually fit):
#   source("local/exploration/m3-utility/03_prototype_b_prelec.R")

library(brms)

# ---- 1. Prelec weighting: math / visualisation --------------------------------

prelec_weight <- function(p, alpha) {
  # Prelec one-parameter weighting function
  # Defined for p in (0, 1]; w(0) = 0 by continuity.
  exp(-((-log(p))^alpha))
}

log_prelec <- function(p, alpha) {
  # log of Prelec weight; numerically stable
  -((-log(p))^alpha)
}

cat("=== Prelec weighting function at representative proportions ===\n\n")
p_vals <- c(0.05, 0.10, 0.20, 0.33, 0.50, 0.80)
cat(sprintf("%-6s  alpha=0.5  alpha=1.0  alpha=2.0\n", "p"))
for (p in p_vals) {
  cat(sprintf("%-6.2f  %-9.4f  %-9.4f  %-9.4f\n",
              p, prelec_weight(p, 0.5), prelec_weight(p, 1.0), prelec_weight(p, 2.0)))
}
cat("\nNote: alpha=1 reproduces w(p) = p (standard frequency weighting)\n\n")

# ---- 2. Effect on choice probabilities when option proportions vary ----------

softmax_prelec <- function(a, n, alpha) {
  N_total <- sum(n)
  p <- n / N_total
  log_w <- log_prelec(p, alpha)  # replace log(n) with log(w(n/N)) + log(N) [const]
  log_num <- a + log_w
  exp(log_num - log(sum(exp(log_num))))
}

a <- c(3.0, 2.0, 0.0)   # activations: correct, other, npl
n <- c(1L,  4L,  5L)     # option counts -> proportions: 0.10, 0.40, 0.50

cat("=== Effect of alpha on M3 choice probabilities ===\n\n")
cat(sprintf("Option proportions: %s\n", paste(round(n / sum(n), 2), collapse = ", ")))
cat(sprintf("%-10s  P(corr)  P(other)  P(npl)\n", "alpha"))
for (alpha in c(0.3, 0.5, 0.7, 1.0, 1.5, 2.0)) {
  p <- softmax_prelec(a, n, alpha)
  cat(sprintf("%-10.1f  %-7.4f  %-8.4f  %-6.4f\n", alpha, p[1], p[2], p[3]))
}
cat("\nalpha < 1: P(corr) rises because the rare correct item (p=0.10) is overweighted.\n")
cat("alpha > 1: P(corr) falls further — rare item discounted even more.\n\n")

# ---- 3. Identifiability concern -----------------------------------------------

cat("=== Identifiability of alpha ===\n\n")
cat("alpha is identified only when option proportions VARY across trials.\n")
cat("In a BALANCED design (n_i constant for all trials), Prelec weights are\n")
cat("trial-constant and alpha is absorbed into the activation intercepts.\n\n")

cat("Example: all trials with n=(1,4,5):\n")
cat("  alpha provides a non-linear re-weighting, but without variation in n_i,\n")
cat("  the likelihood is flat in alpha (not identified from data).\n\n")

cat("Example: design with two conditions — small-set (n=(1,2,2)) and large-set (n=(1,4,5)):\n")
n_small <- c(1L, 2L, 2L)
n_large <- c(1L, 4L, 5L)
for (alpha in c(0.5, 1.0, 2.0)) {
  p_s <- softmax_prelec(a, n_small, alpha)
  p_l <- softmax_prelec(a, n_large, alpha)
  cat(sprintf("  alpha=%.1f: P(corr|small)=%.4f  P(corr|large)=%.4f  diff=%.4f\n",
              alpha, p_s[1], p_l[1], p_s[1] - p_l[1]))
}
cat("\nWith variation in n_i, alpha creates distinct predictions and is identifiable.\n")
cat("Typical M3 designs (fixed set size) will NOT identify alpha.\n\n")

# ---- 4. Data simulation with Prelec weighting --------------------------------

simulate_prelec <- function(n_subjects = 30, n_trials = 100,
                             a_true = 1.5, c_true = 2.0, b_true = 0,
                             alpha_true = 0.6) {
  N <- n_subjects * n_trials

  # Two-condition design: half trials small set, half large set
  n_corr  <- rep(1L, N)
  n_other <- rep(c(2L, 6L), length.out = N)   # alternates
  n_npl   <- rep(c(2L, 6L), length.out = N)

  a_vec <- c(b_true + a_true + c_true, b_true + a_true, b_true)

  n_mat <- cbind(n_corr, n_other, n_npl)
  Y <- t(vapply(seq_len(N), function(i) {
    p_i <- softmax_prelec(a_vec, n_mat[i, ], alpha_true)
    rmultinom(1L, 10L, p_i)[, 1L]
  }, integer(3L)))
  colnames(Y) <- c("corr", "other", "npl")

  data.frame(
    subj    = rep(seq_len(n_subjects), each = n_trials),
    n_corr  = n_corr,
    n_other = n_other,
    n_npl   = n_npl,
    N_total = n_corr + n_other + n_npl,
    p_corr  = n_corr  / (n_corr + n_other + n_npl),
    p_other = n_other / (n_corr + n_other + n_npl),
    p_npl   = n_npl   / (n_corr + n_other + n_npl),
    Idx_corr  = 1L,
    Idx_other = 1L,
    Idx_npl   = 1L,
    nTrials = 10L,
    Y = I(Y)
  )
}

set.seed(2025)
d_prelec <- simulate_prelec()

# ---- 5. Model specification (raw brms NLF) -----------------------------------
#
# NLF graph:
#
#   mu_corr  = Idx_corr  * (corr  + (-(-log(p_corr))^alpha))  + (1-Idx_corr)*(-100)
#   mu_other = Idx_other * (other + (-(-log(p_other))^alpha)) + (1-Idx_other)*(-100)
#   mu_npl   = Idx_npl   * (npl   + (-(-log(p_npl))^alpha))   + (1-Idx_npl)*(-100)
#
#   corr  = b + a + c
#   other = b + a
#   npl   = b
#
# p_corr, p_other, p_npl are precomputed data columns (n_i / N_total).
# alpha is estimated; alpha = 1 recovers standard softmax (up to constant).

prelec_formula <- bf(
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
  prior(constant(0),        nlpar = "b",     class = "b"),
  prior(normal(2, 1),       nlpar = "a",     class = "b"),
  prior(normal(3, 1),       nlpar = "c",     class = "b"),
  prior(lognormal(0, 0.5),  nlpar = "alpha", class = "b", lb = 0.01)
)

stan_code <- make_stancode(
  prelec_formula,
  data   = d_prelec,
  family = multinomial(refcat = NA),
  prior  = prelec_priors
)
cat("Stan code generated successfully (", nchar(stan_code), "chars)\n")
stopifnot(grepl("b_alpha", stan_code))
cat("Parameter 'alpha' confirmed present in Stan code\n\n")

# ---- 6. Null model (alpha fixed at 1) for BF comparison ---------------------

null_formula <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(p_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + log(p_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + log(p_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c),
  nlf(other ~ b + a),
  nlf(npl   ~ b),
  b + a + c ~ 1,
  nl = TRUE
)

null_priors <- c(
  prior(constant(0),  nlpar = "b", class = "b"),
  prior(normal(2, 1), nlpar = "a", class = "b"),
  prior(normal(3, 1), nlpar = "c", class = "b")
)

stan_null <- make_stancode(
  null_formula,
  data   = d_prelec,
  family = multinomial(refcat = NA),
  prior  = null_priors
)
cat("Null model Stan code generated successfully (", nchar(stan_null), "chars)\n\n")

# ---- 7. Fit and compare (uncomment with working Stan installation) -----------
#
# fit_prelec <- brm(
#   prelec_formula,
#   data    = d_prelec,
#   family  = multinomial(refcat = NA),
#   prior   = prelec_priors,
#   chains  = 4, iter = 2000,
#   cores   = 4, backend = "cmdstanr"
# )
#
# fit_null <- brm(
#   null_formula,
#   data    = d_prelec,
#   family  = multinomial(refcat = NA),
#   prior   = null_priors,
#   chains  = 4, iter = 2000,
#   cores   = 4, backend = "cmdstanr"
# )
#
# True parameter values:
#   b = 0, a = 1.5, c = 2.0, alpha = 0.6
#
# Compare via LOO or bridge sampling:
# loo_compare(loo(fit_prelec), loo(fit_null))
