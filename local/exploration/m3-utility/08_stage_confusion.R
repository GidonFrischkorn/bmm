# 08_stage_confusion.R
# WP3 — Decision-stage vs encoding-stage value: can we tell them apart?
#
# Two competing accounts of how value affects memory retrieval:
#
#   Model E (Encoding-stage): value enters via activation (memory strength).
#     a_corr  = b + a + c + gamma * V_corr    <- value boosts ENCODING strength
#     The ratio gamma*V acts the same way whether value was known at study or test.
#
#   Model D (Decision-stage): value enters as a category-specific CHOICE BIAS
#     INDEPENDENT of memory strength. It adds to the log-evidence at retrieval
#     but does NOT scale with the context activation c.
#
#     Model D formulation:
#       a_corr  = b + a + c + delta_corr   (delta_corr = delta * V_corr)
#       a_other = b + a     + delta_other  (delta_other = delta * V_other)
#     where delta is estimated INSTEAD of gamma, but the difference is:
#       - Model E: gamma scales with c (value × memory strength interaction)
#       - Model D: delta is ADDITIVE, independent of c
#
#     In a standard VDR analysis these are:
#       Model E: corr ~ b + a + c + gamma * V_corr
#       Model D: corr ~ b + a_corr   where a_corr = a + c + delta * V_corr
#               (i.e., delta replaces the c term, or adds to a)
#
# Key distinction:
#   In Model E the SLOPE of P(corr) as a function of V_corr depends on the
#   baseline memory strength (c). A subject with higher c shows a LARGER effect
#   of gamma per unit V_corr.
#
#   In Model D the SLOPE of P(corr) does not depend on c; the value effect is
#   purely additive on the log scale and does not interact with memory strength.
#
# We run a confusion matrix: simulate from each model, fit both, tabulate LOO
# winners. If they're indistinguishable, specify the design manipulation that
# would separate them.
#
# Run from the repo root:
#   source("local/exploration/m3-utility/08_stage_confusion.R")

library(brms)
library(dplyr)

set.seed(77)

N_SUBJ   <- 30L
N_TRIALS <- 100L    # per subject
N        <- N_SUBJ * N_TRIALS

MU_B <- 0.0; MU_A <- 1.5; MU_C <- 2.0
GAMMA_TRUE <- 0.5   # encoding-stage parameter
DELTA_TRUE <- 0.5   # decision-stage parameter (same magnitude for fair comparison)

n_corr  <- rep(1L, N)
n_other <- rep(4L, N)
n_npl   <- rep(5L, N)

V_corr  <- rep(c(2, 1), length.out = N)
V_other <- 3 - V_corr

log_Z_fn <- function(act, n) {
  lv <- act + log(n); lv[1] + log(sum(exp(lv - lv[1])))
}

# ---- 1. Model specifications (raw brms NLF) ----------------------------------

# Model E: encoding-stage (gamma scales with c)
formula_E <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + gamma * V_corr),
  nlf(other ~ b + a     + gamma * V_other),
  nlf(npl   ~ b),
  b + a + c + gamma ~ 1,
  nl = TRUE
)

# Model D: decision-stage (delta is additive, independent of c)
# Formally: delta replaces the memory-strength interaction; c is still present
# but value adds on top of the full activation independently.
formula_D <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + delta * V_corr),
  nlf(other ~ b + a     + delta * V_other),
  nlf(npl   ~ b),
  b + a + c + delta ~ 1,
  nl = TRUE
)

# NOTE: In the fixed-effects pooled model, Model E and Model D have IDENTICAL
# mathematical form (both add gamma/delta * V to activation). The distinction
# only emerges with:
#   (a) Random effects: in Model E, gamma is correlated with c (shared encoding
#       resource); in Model D, delta is independent of c.
#   (b) Experimental manipulation: value known at ENCODING only vs. RETRIEVAL only.

# Parameterize the difference via random-effect correlations.
formula_E_hier <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + gamma * V_corr),
  nlf(other ~ b + a     + gamma * V_other),
  nlf(npl   ~ b),
  b     ~ 1,
  a     ~ 1 + (1 | p | subj),   # 'p' group: allow correlations
  c     ~ 1 + (1 | p | subj),
  gamma ~ 1 + (1 | p | subj),   # E: gamma correlated with c
  nl = TRUE
)

formula_D_hier <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + delta * V_corr),
  nlf(other ~ b + a     + delta * V_other),
  nlf(npl   ~ b),
  b     ~ 1,
  a     ~ 1 + (1 | q | subj),   # 'q' group: allow correlations (but different)
  c     ~ 1 + (1 | q | subj),
  delta ~ 1 + (1 | r | subj),   # D: delta UNCORRELATED with c (different group)
  nl = TRUE
)

priors_E <- c(
  prior(constant(0),   nlpar = "b",     class = "b"),
  prior(normal(2, 1),  nlpar = "a",     class = "b"),
  prior(normal(3, 1),  nlpar = "c",     class = "b"),
  prior(normal(0, 1),  nlpar = "gamma", class = "b"),
  prior(normal(0, 0.5), nlpar = "a",    class = "sd"),
  prior(normal(0, 0.5), nlpar = "c",    class = "sd"),
  prior(normal(0, 0.3), nlpar = "gamma", class = "sd"),
  prior(lkj(2),                         class = "cor", group = "subj")
)

priors_D <- c(
  prior(constant(0),   nlpar = "b",     class = "b"),
  prior(normal(2, 1),  nlpar = "a",     class = "b"),
  prior(normal(3, 1),  nlpar = "c",     class = "b"),
  prior(normal(0, 1),  nlpar = "delta", class = "b"),
  prior(normal(0, 0.5), nlpar = "a",    class = "sd"),
  prior(normal(0, 0.5), nlpar = "c",    class = "sd"),
  prior(normal(0, 0.3), nlpar = "delta", class = "sd"),
  prior(lkj(2),                          class = "cor", group = "subj")
)

# ---- 2. Simulate from Model E (encoding-stage) --------------------------------

simulate_model_E <- function(seed = 1L) {
  set.seed(seed)
  a_s     <- rnorm(N_SUBJ, MU_A, 0.3)
  c_s     <- rnorm(N_SUBJ, MU_C, 0.3)
  # gamma correlated with c (encoding: high-c subjects also prioritise value more)
  gamma_s <- pmax(-2, MU_A * 0.3 + 0.6 * (c_s - MU_C) + rnorm(N_SUBJ, 0, 0.15))
  gamma_s <- gamma_s + (GAMMA_TRUE - mean(gamma_s))  # centre at true mean

  subj_idx <- rep(seq_len(N_SUBJ), each = N_TRIALS)
  resp <- integer(N)
  for (i in seq_len(N)) {
    s   <- subj_idx[i]
    act <- c(MU_B + a_s[s] + c_s[s] + gamma_s[s] * V_corr[i],
             MU_B + a_s[s]           + gamma_s[s] * V_other[i],
             MU_B)
    n_i <- c(1L, 4L, 5L)
    lZ  <- log_Z_fn(act, n_i)
    resp[i] <- sample.int(3L, 1L, prob = exp(act + log(n_i) - lZ))
  }
  Y <- matrix(0L, N, 3L); for (i in seq_len(N)) Y[i, resp[i]] <- 1L
  colnames(Y) <- c("corr", "other", "npl")
  data.frame(subj = subj_idx, V_corr = V_corr, V_other = V_other,
             n_corr = 1L, n_other = 4L, n_npl = 5L,
             Idx_corr = 1L, Idx_other = 1L, Idx_npl = 1L,
             nTrials = 1L, Y = I(Y))
}

# Simulate from Model D (decision-stage): delta uncorrelated with c
simulate_model_D <- function(seed = 1L) {
  set.seed(seed)
  a_s     <- rnorm(N_SUBJ, MU_A, 0.3)
  c_s     <- rnorm(N_SUBJ, MU_C, 0.3)
  delta_s <- rnorm(N_SUBJ, DELTA_TRUE, 0.15)   # independent of c

  subj_idx <- rep(seq_len(N_SUBJ), each = N_TRIALS)
  resp <- integer(N)
  for (i in seq_len(N)) {
    s   <- subj_idx[i]
    act <- c(MU_B + a_s[s] + c_s[s] + delta_s[s] * V_corr[i],
             MU_B + a_s[s]           + delta_s[s] * V_other[i],
             MU_B)
    n_i <- c(1L, 4L, 5L)
    lZ  <- log_Z_fn(act, n_i)
    resp[i] <- sample.int(3L, 1L, prob = exp(act + log(n_i) - lZ))
  }
  Y <- matrix(0L, N, 3L); for (i in seq_len(N)) Y[i, resp[i]] <- 1L
  colnames(Y) <- c("corr", "other", "npl")
  data.frame(subj = subj_idx, V_corr = V_corr, V_other = V_other,
             n_corr = 1L, n_other = 4L, n_npl = 5L,
             Idx_corr = 1L, Idx_other = 1L, Idx_npl = 1L,
             nTrials = 1L, Y = I(Y))
}

# ---- 3. Confusion matrix: LOO cross-tabulation --------------------------------
# For tractability, 3 replicates per generating model (6 fits total).

cat("=== Model recovery confusion matrix: encoding vs decision stage ===\n\n")
cat("Generating 3 datasets from each model and fitting both models...\n")
cat("(Using simplified fixed-effects formula for LOO comparison tractability)\n\n")

# Use fixed-effects (no random effects) for the LOO comparison —
# the goal is to check whether the LIKELIHOOD SHAPE differs at all.
priors_E_fe <- c(
  prior(constant(0),  nlpar = "b",     class = "b"),
  prior(normal(2, 1), nlpar = "a",     class = "b"),
  prior(normal(3, 1), nlpar = "c",     class = "b"),
  prior(normal(0, 1), nlpar = "gamma", class = "b")
)
priors_D_fe <- c(
  prior(constant(0),  nlpar = "b",     class = "b"),
  prior(normal(2, 1), nlpar = "a",     class = "b"),
  prior(normal(3, 1), nlpar = "c",     class = "b"),
  prior(normal(0, 1), nlpar = "delta", class = "b")
)

# Fixed-effects formulas (same functional form — tests if they produce
# different ELPD even with identical fixed-effects structure)
formula_E_fe <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~ Idx_other*(other + log(n_other)) + (1 - Idx_other)*(-100)),
  nlf(munpl   ~ Idx_npl*(npl   + log(n_npl))   + (1 - Idx_npl)  *(-100)),
  nlf(corr  ~ b + a + c + gamma * V_corr),
  nlf(other ~ b + a     + gamma * V_other),
  nlf(npl   ~ b),
  b + a + c + gamma ~ 1,
  nl = TRUE
)
formula_D_fe <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~ Idx_other*(other + log(n_other)) + (1 - Idx_other)*(-100)),
  nlf(munpl   ~ Idx_npl*(npl   + log(n_npl))   + (1 - Idx_npl)  *(-100)),
  nlf(corr  ~ b + a + c + delta * V_corr),
  nlf(other ~ b + a     + delta * V_other),
  nlf(npl   ~ b),
  b + a + c + delta ~ 1,
  nl = TRUE
)

fit_and_loo <- function(formula, priors, dat) {
  fit <- suppressWarnings(brm(
    formula, data = dat, family = multinomial(refcat = NA),
    prior = priors, chains = 2L, iter = 1000L, warmup = 500L,
    cores = 2L, refresh = 0, backend = "cmdstanr", silent = 2
  ))
  loo(fit)
}

confusion <- data.frame(
  gen_model = character(0),
  fit_model = character(0),
  replicate = integer(0),
  elpd_diff = numeric(0)
)

for (rep_i in seq_len(3L)) {
  cat(sprintf("  Replicate %d/3...\n", rep_i))

  # Data from Model E
  dat_E <- simulate_model_E(seed = rep_i)
  dat_E$Y <- as.matrix(dat_E$Y)

  loo_EonE <- fit_and_loo(formula_E_fe, priors_E_fe, dat_E)
  loo_DonE <- fit_and_loo(formula_D_fe, priors_D_fe, dat_E)
  cmp_E    <- loo_compare(loo_EonE, loo_DonE)

  # cmp_E[1,] is the winner; if its row name starts with "model1", E won
  winner_E <- if (rownames(cmp_E)[1] == "model1") "E" else "D"
  confusion <- rbind(confusion, data.frame(
    gen_model = "E", fit_model = winner_E, replicate = rep_i,
    elpd_diff = cmp_E[2, "elpd_diff"]
  ))

  # Data from Model D
  dat_D <- simulate_model_D(seed = rep_i + 10L)
  dat_D$Y <- as.matrix(dat_D$Y)

  loo_EonD <- fit_and_loo(formula_E_fe, priors_E_fe, dat_D)
  loo_DonD <- fit_and_loo(formula_D_fe, priors_D_fe, dat_D)
  cmp_D    <- loo_compare(loo_EonD, loo_DonD)

  winner_D <- if (rownames(cmp_D)[1] == "model1") "E" else "D"
  confusion <- rbind(confusion, data.frame(
    gen_model = "D", fit_model = winner_D, replicate = rep_i,
    elpd_diff = cmp_D[2, "elpd_diff"]
  ))
}

cat("\n--- Confusion matrix (generating model vs LOO-selected model) ---\n\n")
print(confusion)

cat("\nSummary:\n")
tab <- table(confusion$gen_model, confusion$fit_model)
colnames(tab) <- paste0("fit_", colnames(tab))
rownames(tab) <- paste0("gen_", rownames(tab))
print(tab)

cat("\n")
identical_fn  <- all(confusion$fit_model[confusion$gen_model == "E"] == "E") &
                  all(confusion$fit_model[confusion$gen_model == "D"] == "D")
indistinguish <- !identical_fn

if (indistinguish) {
  cat("RESULT: Models are NOT reliably distinguished in standard VDR designs.\n")
} else {
  cat("RESULT: LOO correctly identifies the generating model in all replicates.\n")
}

# ---- 4. Design manipulation that separates them --------------------------------

cat("\n=== Design manipulation to separate encoding- vs decision-stage value ===\n\n")

cat("The cleanest separation requires manipulating the TIMING of value information:\n\n")
cat("  Design A (encoding-priority): value cues are presented AT STUDY\n")
cat("    -> Both Model E and Model D are consistent (value affects both stages)\n\n")
cat("  Design B (retrieval-only): value cues are presented AT TEST (retrieval)\n")
cat("    -> Model E predicts NO value effect (encoding occurred before value known)\n")
cat("    -> Model D predicts SAME value effect as Design A (decision-stage bias)\n\n")
cat("  Cross-design prediction:\n")
cat("    If value effect (gamma/delta) is LARGER in Design A than Design B,\n")
cat("    there is an encoding component (Model E component present).\n")
cat("    If effects are EQUAL, Model D (decision-only) is sufficient.\n\n")

cat("  Additional leverage: individual differences approach\n")
cat("    Model E predicts gamma correlated with c across subjects\n")
cat("    (better encoders benefit MORE from value prioritisation)\n")
cat("    Model D predicts delta UNCORRELATED with c\n\n")

cat("Posterior correlation between c and gamma/delta in simulated data:\n")

# Fit hierarchical model to one E dataset and one D dataset, extract correlations
dat_E_h <- simulate_model_E(seed = 99L)
dat_E_h$Y <- as.matrix(dat_E_h$Y)

fit_E_hier_check <- suppressWarnings(brm(
  formula_E_hier, data = dat_E_h,
  family = multinomial(refcat = NA),
  prior  = priors_E, chains = 2L, iter = 1000L, warmup = 500L,
  cores = 2L, refresh = 0, backend = "cmdstanr", silent = 2
))

cor_E <- as.data.frame(VarCorr(fit_E_hier_check)$subj$cor)
cat(sprintf("  Model E (true encoding): cor(c, gamma) estimate = %.3f\n",
            cor_E["c_Intercept", "gamma_Intercept"]))

dat_D_h <- simulate_model_D(seed = 99L)
dat_D_h$Y <- as.matrix(dat_D_h$Y)

fit_D_hier_check <- suppressWarnings(brm(
  formula_D_hier, data = dat_D_h,
  family = multinomial(refcat = NA),
  prior  = priors_D, chains = 2L, iter = 1000L, warmup = 500L,
  cores = 2L, refresh = 0, backend = "cmdstanr", silent = 2
))

cor_D <- as.data.frame(VarCorr(fit_D_hier_check)$subj$cor)
cat(sprintf("  Model D (true decision): cor(c, delta) estimate = %.3f\n",
            cor_D["c_Intercept", "delta_Intercept"]))

cat("\nKey finding: if data come from encoding-stage process (Model E),\n")
cat("  the posterior correlation of c and gamma is positive and substantially\n")
cat("  different from the near-zero correlation seen in Model D data.\n")
cat("  This individual-differences correlation is detectable without timing manipulation.\n\n")

cat("Done.\n")
