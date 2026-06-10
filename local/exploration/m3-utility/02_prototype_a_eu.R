# 02_prototype_a_eu.R
# Prototype A: payoff-modulated activation (Expected-Utility extension of M3)
#
# Design: Value-Directed Remembering (VDR) — items carry point values.
# Extension: activation(i) += gamma * V(i), where V(i) is the point value.
# This is an additive expected-utility term inside the softmax exponent.
#
# Key insight: this is just adding a predictor to the activation sub-formula —
# NO structural change to the choice rule is needed.  The bmm API already
# supports this via the bmmformula; the prototype uses raw brms NLF to make
# the math fully explicit.
#
# Run from the repo root (requires a working Stan installation to actually fit):
#   source("local/exploration/m3-utility/02_prototype_a_eu.R")

library(brms)

# ---- 1. Data simulation -------------------------------------------------------

simulate_vdr <- function(n_subjects = 30, n_trials = 80,
                          a_true = 1.5, c_true = 2.0, b_true = 0,
                          gamma_true = 0.8) {
  N <- n_subjects * n_trials

  # VDR design: value of correct item alternates between high (2) and low (1)
  V_corr  <- rep(c(2, 1), length.out = N)
  V_other <- 3 - V_corr          # opposite value so total value is constant

  # activations (identity link for softmax)
  act_corr  <- b_true + a_true + c_true + gamma_true * V_corr
  act_other <- b_true + a_true           + gamma_true * V_other
  act_npl   <- b_true

  n_corr <- rep(1L, N)
  n_other <- rep(4L, N)
  n_npl   <- rep(5L, N)

  log_Z <- log(exp(act_corr  + log(n_corr)) +
               exp(act_other + log(n_other)) +
               exp(act_npl   + log(n_npl)))
  p_corr  <- exp(act_corr  + log(n_corr)  - log_Z)
  p_other <- exp(act_other + log(n_other) - log_Z)
  p_npl   <- exp(act_npl   + log(n_npl)   - log_Z)

  n_resp <- 10L  # total responses per trial (e.g. set-size 10)
  Y <- t(vapply(seq_len(N), function(i) {
    rmultinom(1L, n_resp, c(p_corr[i], p_other[i], p_npl[i]))[, 1L]
  }, integer(3L)))
  colnames(Y) <- c("corr", "other", "npl")

  data.frame(
    subj    = rep(seq_len(n_subjects), each = n_trials),
    V_corr  = V_corr,
    V_other = V_other,
    n_corr  = n_corr,
    n_other = n_other,
    n_npl   = n_npl,
    Idx_corr  = 1L,
    Idx_other = 1L,
    Idx_npl   = 1L,
    nTrials = n_resp,
    Y = I(Y)
  )
}

set.seed(2024)
d_vdr <- simulate_vdr()

# ---- 2. Model specification (raw brms NLF) -----------------------------------
#
# NLF graph:
#
#   mu_corr  = Idx_corr  * (corr  + log(n_corr))  + (1-Idx_corr)  * (-100)
#   mu_other = Idx_other * (other + log(n_other)) + (1-Idx_other) * (-100)
#   mu_npl   = Idx_npl   * (npl   + log(n_npl))   + (1-Idx_npl)   * (-100)
#
#   corr  = b + a + c + gamma * V_corr    <- EU term added here
#   other = b + a     + gamma * V_other
#   npl   = b
#
# Parameters estimated: a, c, gamma  (b fixed at 0 for scale identification)

eu_formula <- bf(
  Y | trials(nTrials) ~
    Idx_corr * (corr + log(n_corr)) + (1 - Idx_corr) * (-100),
  nlf(muother ~
    Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl * (npl + log(n_npl)) + (1 - Idx_npl) * (-100)),
  nlf(corr  ~ b + a + c + gamma * V_corr),
  nlf(other ~ b + a     + gamma * V_other),
  nlf(npl   ~ b),
  b + a + c + gamma ~ 1,
  nl = TRUE
)

eu_priors <- c(
  prior(constant(0),   nlpar = "b",     class = "b"),
  prior(normal(2, 1),  nlpar = "a",     class = "b"),
  prior(normal(3, 1),  nlpar = "c",     class = "b"),
  prior(normal(0, 1),  nlpar = "gamma", class = "b")
)

# Verify: make_stancode() parses without error (does not require Stan compilation)
stan_code <- make_stancode(
  eu_formula,
  data   = d_vdr,
  family = multinomial(refcat = NA),
  prior  = eu_priors
)
cat("Stan code generated successfully (", nchar(stan_code), "chars)\n")

# Check that gamma appears as an estimated parameter
stopifnot(grepl("b_gamma", stan_code))
cat("Parameter 'gamma' confirmed present in Stan code\n\n")

# ---- 3. Identifiability note --------------------------------------------------
#
# gamma is identified only when V_corr varies ACROSS trials.  If all trials
# have the same value (e.g. always high-value correct item), gamma * V_corr
# is absorbed into the intercept of c (or a) and is not separately identified.
#
# Minimum requirement: at least two distinct values of V_corr in the data.
#
cat("Value variation in simulated data:\n")
cat("  V_corr unique:", sort(unique(d_vdr$V_corr)), "\n")
cat("  V_other unique:", sort(unique(d_vdr$V_other)), "\n\n")

# ---- 4. Parameter recovery (run locally with working Stan) -------------------
#
# Uncomment to actually fit; requires a working Stan/cmdstanr installation.
#
# fit_eu <- brm(
#   eu_formula,
#   data    = d_vdr,
#   family  = multinomial(refcat = NA),
#   prior   = eu_priors,
#   chains  = 4, iter = 2000, warmup = 1000,
#   cores   = 4, refresh = 200,
#   backend = "cmdstanr"
# )
#
# True parameter values:
#   b = 0, a = 1.5, c = 2.0, gamma = 0.8
#
# Expected: 90% credible intervals should cover true values.
# fixef(fit_eu)

# ---- 5. Predicted probability difference as a function of gamma --------------
#
# Show how gamma shifts choice probabilities in the VDR design:
pred_vdr <- function(gamma, a = 1.5, c = 2.0, b = 0) {
  V_vals <- c(2, 1)
  act_c <- b + a + c + gamma * V_vals
  act_o <- b + a     + gamma * rev(V_vals)
  act_n <- b
  n     <- c(1, 4, 5)
  log_Z <- log(exp(act_c + log(n[1])) + exp(act_o + log(n[2])) + exp(act_n + log(n[3])))
  p_c   <- exp(act_c + log(n[1]) - log_Z)
  data.frame(
    V_corr   = V_vals,
    p_correct = p_c,
    gamma     = gamma
  )
}

cat("Effect of gamma on P(correct) for high- vs low-value trials:\n")
results <- do.call(rbind, lapply(c(0, 0.5, 1.0, 2.0), pred_vdr))
print(results, digits = 3, row.names = FALSE)
cat("\nAt gamma=0: P(correct) is identical for high and low value trials (no EU effect)\n")
cat("As gamma increases: high-value correct items attract higher recall probability\n")
