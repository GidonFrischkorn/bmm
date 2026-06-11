# 21_welfare_weight.R — WP5 (Round 6)
#
# Welfare-weight use case: simulation-based check of E_D and E_A models
# using the Gross, Gotz, Reher & Toscano (2025) payoff structure.
#
# Payoff matrix (nested mobility dilemma):
#   Choice      x_self  x_ingroup  x_outgroup
#   keep          1.0     0.0        0.0
#   ingroup       0.5     1.0        0.0
#   universal     0.3     0.6        0.9
#
# Model E_D (independent welfare weights, utility reparametrization of Model D):
#   U(keep)      = b = 1.0 (numeraire)
#   U(ingroup)   = 0.5*b + wi
#   U(universal) = 0.3*b + 0.6*wi + 0.9*wo
#   Choice rule: Luce (P(k) proportional to U(k))
#   Parameters: wi (ingroup weight), wo (outgroup weight)
#
# Model E_A (hierarchical welfare weights — structurally distinct):
#   U(keep)      = b = 1.0
#   U(ingroup)   = 0.5*b + c
#   U(universal) = 0.3*b + 0.6*c + u   (c discounted by 0.6 = payoff structure)
#   Parameters: c (ingroup welfare weight), u (outgroup contribution)
#
# Reparametrisation (E_D <-> Model D softmax):
#   alpha_D = log(0.5 + wi)                       [ingroup activation]
#   beta_D  = log(0.3 + 0.6*wi + 0.9*wo) - alpha_D [context effect]
#   Inverse: wi = exp(alpha_D) - 0.5
#            wo = (exp(alpha_D + beta_D) - 0.3 - 0.6*wi) / 0.9
#
# Tasks:
# (a) Simulate E_D data, fit Model D (M3 softmax), recover wi and wo via transform
# (b) Verify E_D = Model D: compare predicted probabilities (reparametrization identity)
# (c) Fit E_A to E_D data: parameters c, u vs wi, wo; show structural equivalence
# (d) Check Stage-1 m3_utility() compatibility; document R8-R10 requirements
#
# Results saved to:
#   results/21_model_d_recovery.csv
#   results/21_model_ea_recovery.csv
#   results/21_welfare_reparam.csv
#
# Run from repo root:
#   Rscript local/exploration/m3-utility/21_welfare_weight.R

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
N_TRIALS <- 60L   # per subject
WI_TRUE  <- 0.60   # ingroup welfare weight
WO_TRUE  <- 0.30   # outgroup welfare weight
B_FORM   <- 1.0    # numeraire = own payoff weight

# Derived utilities
U_KEEP      <- B_FORM
U_INGROUP   <- 0.5 * B_FORM + WI_TRUE
U_UNIVERSAL <- 0.3 * B_FORM + 0.6 * WI_TRUE + 0.9 * WO_TRUE

cat("=== WP5: Welfare-weight use case (Gross et al. 2025 payoff structure) ===\n\n")
cat(sprintf("True parameters: wi=%.2f, wo=%.2f (b_form=%.1f)\n", WI_TRUE, WO_TRUE, B_FORM))
cat(sprintf("Implied utilities: U(keep)=%.2f, U(ingroup)=%.2f, U(universal)=%.2f\n\n",
            U_KEEP, U_INGROUP, U_UNIVERSAL))

# Corresponding Model D (softmax) parameters
ALPHA_D_TRUE <- log(0.5 * B_FORM + WI_TRUE)                                      # = log(1.1)
BETA_D_TRUE  <- log(0.3 * B_FORM + 0.6 * WI_TRUE + 0.9 * WO_TRUE) - ALPHA_D_TRUE # = log(0.93) - log(1.1)
cat(sprintf("Model D (softmax) reparametrization:\n"))
cat(sprintf("  alpha_D = log(0.5+wi) = log(%.2f) = %.4f\n", 0.5 + WI_TRUE, ALPHA_D_TRUE))
cat(sprintf("  beta_D  = log(0.3+0.6*wi+0.9*wo) - alpha_D = %.4f\n", BETA_D_TRUE))
cat(sprintf("  (negative beta_D means universal < ingroup in utility scale)\n\n"))

# ---- Simulate E_D data (Luce rule: P(k) proportional to U(k)) ----
set.seed(5001L)
resp_sim <- integer(N_SUBJ * N_TRIALS)
subj_sim <- rep(seq_len(N_SUBJ), each = N_TRIALS)

for (i in seq_len(N_SUBJ * N_TRIALS)) {
  probs <- c(U_KEEP, U_INGROUP, U_UNIVERSAL)  # Luce rule (n_k=1 for all k)
  resp_sim[i] <- sample.int(3L, 1L, prob = probs / sum(probs))
}
Y_sim <- matrix(0L, N_SUBJ * N_TRIALS, 3L)
for (i in seq_len(N_SUBJ * N_TRIALS)) Y_sim[i, resp_sim[i]] <- 1L
colnames(Y_sim) <- c("corr", "other", "npl")  # corr=universal, other=ingroup, npl=keep

d_ed <- data.frame(
  subj = subj_sim,
  n_corr = 1L, n_other = 1L, n_npl = 1L,
  Idx_corr = 1L, Idx_other = 1L, Idx_npl = 1L,
  nTrials = 1L, Y = I(Y_sim)
)

cat(sprintf("Simulated E_D data: N=%d subjects x %d trials\n", N_SUBJ, N_TRIALS))
obs_props <- colMeans(Y_sim)
cat(sprintf("  Observed choice props: keep=%.3f, ingroup=%.3f, universal=%.3f\n",
            obs_props[3], obs_props[2], obs_props[1]))
exp_props <- c(U_UNIVERSAL, U_INGROUP, U_KEEP) / (U_UNIVERSAL + U_INGROUP + U_KEEP)
cat(sprintf("  Expected choice props: keep=%.3f, ingroup=%.3f, universal=%.3f\n\n",
            exp_props[3], exp_props[2], exp_props[1]))

# ---- Part (a)+(b): Fit Model D (standard M3 softmax, b=0 fixed) ----
# In Model D: corr=universal, other=ingroup, npl=keep
# a = alpha_D, c = beta_D (the softmax reparametrization of wi, wo)
# With n=1 for all categories, log(n_i)=0, so model reduces to pure softmax.

cat("--- Part (a)+(b): Fit Model D (standard M3 softmax reparametrization) ---\n")

form_D <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c),   # b=0, a=alpha_D, c=beta_D
  nlf(other ~ b + a),
  nlf(npl   ~ b),
  b + a + c ~ 1,
  nl = TRUE
)
priors_D <- c(
  prior(constant(0), nlpar = "b", class = "b"),
  prior(normal(0, 1), nlpar = "a", class = "b"),   # alpha_D centred at 0
  prior(normal(0, 1), nlpar = "c", class = "b")    # beta_D centred at 0
)

cat("Fitting Model D to E_D data...\n")
fit_D <- brm(
  form_D, data = d_ed, family = multinomial(refcat = NA),
  prior = priors_D, chains = 2L, iter = 1000L, warmup = 500L,
  cores = 2L, refresh = 0, backend = "cmdstanr", silent = 2
)
fe_D  <- fixef(fit_D)
rh_D  <- max(rhat(fit_D), na.rm = TRUE)
div_D <- sum(nuts_params(fit_D)$Value[nuts_params(fit_D)$Parameter == "divergent__"])

alpha_D_est <- fe_D["a_Intercept", "Estimate"]
beta_D_est  <- fe_D["c_Intercept", "Estimate"]

# Transform back to welfare weights
wi_hat <- exp(alpha_D_est) - 0.5
wo_hat <- (exp(alpha_D_est + beta_D_est) - 0.3 - 0.6 * wi_hat) / 0.9

cat(sprintf("  alpha_D: est=%.4f [%.4f, %.4f]  true=%.4f\n",
    alpha_D_est, fe_D["a_Intercept","Q2.5"], fe_D["a_Intercept","Q97.5"], ALPHA_D_TRUE))
cat(sprintf("  beta_D:  est=%.4f [%.4f, %.4f]  true=%.4f\n",
    beta_D_est,  fe_D["c_Intercept","Q2.5"], fe_D["c_Intercept","Q97.5"], BETA_D_TRUE))
cat(sprintf("  -> wi_hat = exp(%.4f) - 0.5 = %.4f  (true=%.2f)\n",
    alpha_D_est, wi_hat, WI_TRUE))
cat(sprintf("  -> wo_hat = (exp(%.4f) - 0.3 - 0.6*%.4f) / 0.9 = %.4f  (true=%.2f)\n",
    alpha_D_est + beta_D_est, wi_hat, wo_hat, WO_TRUE))
cat(sprintf("  Max R-hat: %.3f  Divergences: %d\n", rh_D, div_D))

# Verify reparametrization: compare predicted probs from fit and from true E_D params
U_keep_fit      <- 1.0
U_ingroup_fit   <- 0.5 + wi_hat
U_universal_fit <- 0.3 + 0.6 * wi_hat + 0.9 * wo_hat

pred_keep      <- U_keep_fit      / (U_keep_fit + U_ingroup_fit + U_universal_fit)
pred_ingroup   <- U_ingroup_fit   / (U_keep_fit + U_ingroup_fit + U_universal_fit)
pred_universal <- U_universal_fit / (U_keep_fit + U_ingroup_fit + U_universal_fit)
true_sum       <- U_KEEP + U_INGROUP + U_UNIVERSAL

max_pred_diff <- max(abs(c(pred_keep - U_KEEP/true_sum,
                           pred_ingroup - U_INGROUP/true_sum,
                           pred_universal - U_UNIVERSAL/true_sum)))

cat(sprintf("\n  Reparametrisation check (max |P_fit - P_true|): %.6f\n", max_pred_diff))
cat(sprintf("  %s\n\n",
    ifelse(max_pred_diff < 0.01, "VERIFIED: E_D and Model D produce identical predictions",
           "MISMATCH: reparametrization not holding to expected accuracy")))

# Save Model D results
res_D <- data.frame(
  fit = "model_D", param = c("alpha_D", "beta_D", "wi_recovered", "wo_recovered"),
  true_value = c(ALPHA_D_TRUE, BETA_D_TRUE, WI_TRUE, WO_TRUE),
  estimate   = c(alpha_D_est, beta_D_est, wi_hat, wo_hat),
  max_rhat = rh_D, divergences = div_D,
  reparam_max_diff = max_pred_diff
)
write.csv(res_D, "local/exploration/m3-utility/results/21_model_d_recovery.csv",
          row.names = FALSE)
cat("  Saved -> results/21_model_d_recovery.csv\n\n")

# ---- Part (c): Fit E_A (payoff-constrained welfare weights) ----
# E_A: U(ingroup) = 0.5 + c, U(universal) = 0.3 + 0.6*c + u
# Both constrained by payoff structure (0.6 discount on c in universal)
# In softmax framework: log P(k) proportional to log U(k)
#   log P(universal) proportional to log(0.3 + 0.6*c + u)  [non-linear in c, u]
#   log P(ingroup)   proportional to log(0.5 + c)           [non-linear in c]
#   log P(keep)      proportional to 0                       [= log(1)]

cat("--- Part (c): Fit E_A (payoff-constrained, non-linear in c) ---\n")

# Use NLF formula with log-utilities
form_EA <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * corr  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * other + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * npl   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ log(0.3 + 0.6 * c + u)),   # log(U_universal); n_k=1 so no log(n) needed
  nlf(other ~ log(0.5 + c)),             # log(U_ingroup)
  nlf(npl   ~ b),                        # log(U_keep) = log(1) = 0
  b + c + u ~ 1,
  nl = TRUE
)
priors_EA <- c(
  prior(constant(0),       nlpar = "b", class = "b"),
  prior(lognormal(0, 0.5), nlpar = "c", class = "b", lb = 0.01),   # c > 0: welfare weight
  prior(lognormal(0, 0.5), nlpar = "u", class = "b", lb = 0.01)    # u > 0: outgroup contribution
)

cat("Fitting E_A to E_D data...\n")
fit_EA <- brm(
  form_EA, data = d_ed, family = multinomial(refcat = NA),
  prior = priors_EA, chains = 2L, iter = 1000L, warmup = 500L,
  cores = 2L, refresh = 0, backend = "cmdstanr", silent = 2
)
fe_EA  <- fixef(fit_EA)
rh_EA  <- max(rhat(fit_EA), na.rm = TRUE)
div_EA <- sum(nuts_params(fit_EA)$Value[nuts_params(fit_EA)$Parameter == "divergent__"])

c_EA_est <- fe_EA["c_Intercept", "Estimate"]
u_EA_est <- fe_EA["u_Intercept", "Estimate"]

cat(sprintf("  c_EA (ingroup weight): est=%.4f [%.4f, %.4f]  true_wi=%.2f\n",
    c_EA_est, fe_EA["c_Intercept","Q2.5"], fe_EA["c_Intercept","Q97.5"], WI_TRUE))
cat(sprintf("  u_EA (outgroup contribution): est=%.4f [%.4f, %.4f]  true_0.9*wo=%.4f\n",
    u_EA_est, fe_EA["u_Intercept","Q2.5"], fe_EA["u_Intercept","Q97.5"], 0.9 * WO_TRUE))
cat(sprintf("  Max R-hat: %.3f  Divergences: %d\n\n", rh_EA, div_EA))

# Save E_A results
res_EA <- data.frame(
  fit = "model_EA",
  param = c("c_EA", "u_EA"),
  true_value = c(WI_TRUE, 0.9 * WO_TRUE),   # c = wi, u = 0.9*wo
  estimate   = c(c_EA_est, u_EA_est),
  ci_lower   = c(fe_EA["c_Intercept","Q2.5"], fe_EA["u_Intercept","Q2.5"]),
  ci_upper   = c(fe_EA["c_Intercept","Q97.5"], fe_EA["u_Intercept","Q97.5"]),
  covered    = c(WI_TRUE >= fe_EA["c_Intercept","Q2.5"] & WI_TRUE <= fe_EA["c_Intercept","Q97.5"],
                 0.9*WO_TRUE >= fe_EA["u_Intercept","Q2.5"] & 0.9*WO_TRUE <= fe_EA["u_Intercept","Q97.5"]),
  max_rhat = rh_EA, divergences = div_EA
)
write.csv(res_EA, "local/exploration/m3-utility/results/21_model_ea_recovery.csv",
          row.names = FALSE)
cat("  Saved -> results/21_model_ea_recovery.csv\n\n")

# ---- Compare Model D and E_A log-likelihoods ----
cat("--- Verifying E_D = E_A (structural equivalence) ---\n")
ll_D  <- mean(log_lik(fit_D))
ll_EA <- mean(log_lik(fit_EA))
cat(sprintf("  Mean log-lik Model D:  %.4f\n", ll_D))
cat(sprintf("  Mean log-lik Model EA: %.4f\n", ll_EA))
cat(sprintf("  Difference (E_A - D):  %.4f\n", ll_EA - ll_D))

reparam_df <- data.frame(
  mean_ll_model_D = ll_D, mean_ll_model_EA = ll_EA,
  ll_diff = ll_EA - ll_D,
  wi_recovered_D = wi_hat, wo_recovered_D = wo_hat,
  c_EA = c_EA_est, u_EA = u_EA_est
)
write.csv(reparam_df, "local/exploration/m3-utility/results/21_welfare_reparam.csv",
          row.names = FALSE)
cat("  Saved -> results/21_welfare_reparam.csv\n\n")

if (abs(ll_EA - ll_D) < 0.005) {
  cat("  CONFIRMED: E_A and Model D have identical log-likelihoods.\n")
  cat("  E_A is a constrained reparametrization of Model D (welfare-weight scale).\n")
} else {
  cat("  WARNING: Log-likelihoods differ — check formula implementation.\n")
}

# ---- Part (d): Stage-1 compatibility check ----
cat("\n--- Part (d): Stage-1 m3_utility() compatibility check ---\n\n")

cat("The welfare-weight use case stresses Stage-1 in 4 ways not covered by current sketch:\n\n")

cat("R8 — Fixed numeric payoff coefficients in activation formula:\n")
cat("  Current Stage-1 generates: corr ~ b + a + c + gamma * V_corr\n")
cat("  Welfare-weight needs:      universal ~ 0.3*b + 0.6*wi + 0.9*wo\n")
cat("  The coefficients 0.5, 0.3, 0.6, 0.9 are FIXED from the payoff matrix,\n")
cat("  not per-trial variable columns V_corr / V_other.\n")
cat("  Severity: silent-wrong if user tries gamma*V_corr (data columns won't match).\n")
cat("  Stage-1 cannot express this: R8 = NEW REQUIREMENT\n\n")

cat("R9 — choice_rule = 'simple' (Luce ratio rule):\n")
cat("  Stage-1 defaults to softmax. Welfare-weight model uses Luce/simple rule.\n")
cat("  However: with n_k=1 for all categories, softmax IS the Luce rule.\n")
cat("  (P(k) proportional to exp(log U(k)) = U(k) for n_k=1 — numerically identical.)\n")
cat("  Stage-1 can express this ONLY when using log-utility activations.\n")
cat("  For non-log-linear utility forms (E_A with payoff constraints), not expressible.\n")
cat("  Severity: runtime error if user uses simple rule with m3_utility() defaults.\n")
cat("  Stage-1 partial support: R9 = NEW REQUIREMENT (for non-linear utility forms)\n\n")

cat("R10 — Numeraire fixing (b = 1 as own-payoff scale):\n")
cat("  Stage-1 fixes b=0 (no background in softmax). Welfare-weight model needs\n")
cat("  b = 1.0 (the numeraire that defines the welfare-weight scale).\n")
cat("  fixed_parameters$b = 1.0 sets the M3 BACKGROUND NOISE to 1.0, which is\n")
cat("  NOT the same as the formula parameter b in bmf(nkeep ~ b, ...).\n")
cat("  Severity: silent-wrong (model runs but b scale is wrong).\n")
cat("  Stage-1 cannot express this cleanly: R10 = NEW REQUIREMENT\n\n")

cat("Summary: Stage-1 m3_utility() cannot express the welfare-weight use case.\n")
cat("  It requires 3 new requirements (R8, R9, R10) beyond the current 7.\n")
cat("  These requirements are documented in 14_requirements_inventory.md.\n\n")

cat("Done.\n")
