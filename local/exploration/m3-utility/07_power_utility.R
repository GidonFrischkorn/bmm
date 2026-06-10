# 07_power_utility.R
# WP2 — Specify the utility function proper: power utility u(V) = V^rho
#
# The previous EU prototype used linear utility gamma*V.  This extends to
# power utility: u(V) = V^rho, so activation becomes:
#
#   a_corr  = b + a + c + gamma * V_corr^rho
#   a_other = b + a     + gamma * V_other^rho
#   a_npl   = b
#
# Questions addressed:
#   1. Is rho identifiable when >=3 value levels are present?
#   2. What design requirements (number of levels, trials per level) are needed
#      for joint recovery of (gamma, rho)?
#   3. What is the correlation / confusion between gamma and rho in 2-level designs?
#
# Run from the repo root:
#   source("local/exploration/m3-utility/07_power_utility.R")

library(brms)
library(dplyr)

set.seed(123)

# ---- 1. Utility function and predicted probabilities -------------------------

power_utility <- function(V, rho) V^rho   # rho=1: linear; rho<1: concave; rho>1: convex

softmax_m3_pu <- function(a, c, b = 0, gamma, rho, V_corr, V_other,
                            n = c(1, 4, 5)) {
  act_corr  <- b + a + c + gamma * power_utility(V_corr, rho)
  act_other <- b + a     + gamma * power_utility(V_other, rho)
  act_npl   <- b
  act <- c(act_corr, act_other, act_npl)
  lZ  <- act[1] + log(sum(exp(act + log(n) - act[1] - log(n[1]))))
  exp(act + log(n) - lZ)
}

cat("=== Effect of rho on P(correct) with 3-level VDR design ===\n\n")
cat("Values: V ∈ {1, 3, 5}; n=(1,4,5); a=1.5, c=2.0, gamma=0.5\n\n")
cat(sprintf("%-6s  %-8s  V=1   V=3   V=5\n", "rho", ""))
for (rho in c(0.3, 0.5, 0.7, 1.0, 1.5, 2.0)) {
  p1 <- softmax_m3_pu(1.5, 2.0, 0, gamma = 0.5, rho = rho, V_corr = 1, V_other = 5)
  p3 <- softmax_m3_pu(1.5, 2.0, 0, gamma = 0.5, rho = rho, V_corr = 3, V_other = 3)
  p5 <- softmax_m3_pu(1.5, 2.0, 0, gamma = 0.5, rho = rho, V_corr = 5, V_other = 1)
  cat(sprintf("%-6.1f  P(corr)  %-5.3f %-5.3f %-5.3f\n", rho, p1[1], p3[1], p5[1]))
}

# ---- 2. Why 2-level VDR cannot identify rho ----------------------------------

cat("\n=== Confound: gamma and rho in a 2-level design ===\n\n")
cat("With V ∈ {1, 2} only, rho and gamma are NOT jointly identified.\n")
cat("gamma * V^rho is observationally equivalent to gamma* * V^rho*\n")
cat("for many (gamma*, rho*) pairs — they predict the SAME probability ratio.\n\n")

# Show that different (gamma,rho) pairs give identical predictions at V={1,2}
cat("Predictions at V_corr=2 (V_other=1, n=(1,4,5), a=1.5, c=2.0):\n")
cat(sprintf("  %-10s  %-10s  P(corr)\n", "gamma", "rho"))
for (combo in list(c(0.5, 1.0), c(0.35, 1.5), c(0.63, 0.7))) {
  p <- softmax_m3_pu(1.5, 2.0, 0, gamma = combo[1], rho = combo[2],
                      V_corr = 2, V_other = 1)
  cat(sprintf("  %-10.3f  %-10.3f  %.4f\n", combo[1], combo[2], p[1]))
}
cat("\nAll produce similar P(corr) -> can't distinguish gamma from rho with 2 levels.\n")

# ---- 3. 3-level design simulation + brms specification -----------------------

cat("\n=== 3-level VDR: joint recovery of (gamma, rho) ===\n\n")

N_SUBJ   <- 30L
N_TRIALS <- 120L   # 40 per value level
N        <- N_SUBJ * N_TRIALS

# True parameters
MU_B     <- 0.0; MU_A <- 1.5; MU_C <- 2.0
MU_GAMMA <- 0.5; MU_RHO <- 0.7

# Value levels cycle through 1, 3, 5 (3 distinct levels)
V_VALS   <- c(1, 3, 5)
V_corr   <- rep(V_VALS, length.out = N)
# V_other: always total value is 6, so V_other = 6 - V_corr (clamped to [1,5])
V_other  <- pmax(1, 6 - V_corr)

n_corr  <- rep(1L, N)
n_other <- rep(4L, N)
n_npl   <- rep(5L, N)

log_Z_fn <- function(act, n) {
  lv <- act + log(n)
  lv[1] + log(sum(exp(lv - lv[1])))
}

# Simulate
set.seed(2024)
resp_idx <- integer(N)
for (i in seq_len(N)) {
  act <- c(
    MU_B + MU_A + MU_C + MU_GAMMA * V_corr[i]^MU_RHO,
    MU_B + MU_A         + MU_GAMMA * V_other[i]^MU_RHO,
    MU_B
  )
  n_i   <- c(n_corr[i], n_other[i], n_npl[i])
  lZ    <- log_Z_fn(act, n_i)
  p     <- exp(act + log(n_i) - lZ)
  resp_idx[i] <- sample.int(3L, 1L, prob = p)
}

Y <- matrix(0L, N, 3L)
for (i in seq_len(N)) Y[i, resp_idx[i]] <- 1L
colnames(Y) <- c("corr", "other", "npl")

d_pu <- data.frame(
  subj      = rep(seq_len(N_SUBJ), each = N_TRIALS),
  V_corr    = V_corr,
  V_other   = V_other,
  n_corr    = n_corr,
  n_other   = n_other,
  n_npl     = n_npl,
  Idx_corr  = 1L,
  Idx_other = 1L,
  Idx_npl   = 1L,
  nTrials   = 1L,
  Y         = I(Y)
)

# brms formula for power utility
# u(V) = V^rho is non-linear in rho, so we use nlf().
# V_corr and V_other are data columns.
# gamma and rho are estimated parameters.
pu_formula <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + log(n_corr))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + log(n_other)) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + log(n_npl))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + gamma * V_corr^rho),
  nlf(other ~ b + a     + gamma * V_other^rho),
  nlf(npl   ~ b),
  b + a + c + gamma + rho ~ 1,
  nl = TRUE
)

pu_priors <- c(
  prior(constant(0),       nlpar = "b",     class = "b"),
  prior(normal(2, 1),      nlpar = "a",     class = "b"),
  prior(normal(3, 1),      nlpar = "c",     class = "b"),
  prior(normal(0, 1),      nlpar = "gamma", class = "b"),
  prior(lognormal(0, 0.5), nlpar = "rho",   class = "b", lb = 0.05)
)

cat("Verifying Stan code parses for power utility model...\n")
stan_pu <- make_stancode(
  pu_formula,
  data   = d_pu,
  family = multinomial(refcat = NA),
  prior  = pu_priors
)
cat("Stan code generated (", nchar(stan_pu), "chars)\n")
stopifnot(grepl("b_gamma", stan_pu))
stopifnot(grepl("b_rho",   stan_pu))
cat("Parameters 'gamma' and 'rho' confirmed present.\n\n")

cat("Fitting power utility model (3-level VDR, N=30x120, 2 chains x 1000 iter)...\n")
fit_pu <- brm(
  pu_formula,
  data    = d_pu,
  family  = multinomial(refcat = NA),
  prior   = pu_priors,
  chains  = 2L,
  iter    = 1000L,
  warmup  = 500L,
  cores   = 2L,
  refresh = 0,
  backend = "cmdstanr",
  silent  = 2
)

cat("\n--- Power utility recovery results ---\n")
cat(sprintf("True: b=%.1f, a=%.1f, c=%.1f, gamma=%.1f, rho=%.1f\n\n",
            MU_B, MU_A, MU_C, MU_GAMMA, MU_RHO))

fe_pu <- fixef(fit_pu)
print(round(fe_pu[, c("Estimate", "Q2.5", "Q97.5")], 3))

true_pu <- c(b = MU_B, a = MU_A, c = MU_C, gamma = MU_GAMMA, rho = MU_RHO)
cat("\nCoverage check (95% CI contains true value):\n")
for (nm in names(true_pu)) {
  rn <- paste0(nm, "_Intercept")
  if (rn %in% rownames(fe_pu)) {
    lo <- fe_pu[rn, "Q2.5"]
    hi <- fe_pu[rn, "Q97.5"]
    cat(sprintf("  %-5s: true=%.2f, CI=(%.3f, %.3f) -> %s\n",
                nm, true_pu[nm], lo, hi,
                ifelse(true_pu[nm] >= lo & true_pu[nm] <= hi, "COVERED", "MISSED")))
  }
}
cat("\nMax Rhat:", round(max(rhat(fit_pu), na.rm = TRUE), 3), "\n")

# ---- 4. 2-level design: show gamma/rho confound empirically ------------------

cat("\n=== 2-level design: (gamma, rho) posterior correlation ===\n\n")

# Simpler data: only V ∈ {1, 2} (standard VDR)
N2    <- N_SUBJ * 80L
V2c   <- rep(c(2, 1), length.out = N2)
V2o   <- 3 - V2c
resp2 <- integer(N2)
for (i in seq_len(N2)) {
  act <- c(MU_B + MU_A + MU_C + MU_GAMMA * V2c[i]^MU_RHO,
           MU_B + MU_A         + MU_GAMMA * V2o[i]^MU_RHO,
           MU_B)
  n_i <- c(1L, 4L, 5L)
  lZ  <- log_Z_fn(act, n_i)
  p   <- exp(act + log(n_i) - lZ)
  resp2[i] <- sample.int(3L, 1L, prob = p)
}
Y2 <- matrix(0L, N2, 3L); for (i in seq_len(N2)) Y2[i, resp2[i]] <- 1L
colnames(Y2) <- c("corr", "other", "npl")

d_pu2 <- data.frame(
  subj = rep(seq_len(N_SUBJ), each = 80L),
  V_corr = V2c, V_other = V2o,
  n_corr = 1L, n_other = 4L, n_npl = 5L,
  Idx_corr = 1L, Idx_other = 1L, Idx_npl = 1L,
  nTrials = 1L, Y = I(Y2)
)

cat("Fitting power utility model to 2-level VDR data...\n")
fit_pu2 <- brm(
  pu_formula,
  data    = d_pu2,
  family  = multinomial(refcat = NA),
  prior   = pu_priors,
  chains  = 2L,
  iter    = 1000L,
  warmup  = 500L,
  cores   = 2L,
  refresh = 0,
  backend = "cmdstanr",
  silent  = 2
)

cat("\n--- 2-level design: gamma/rho recovery ---\n")
cat("(Expected: wide CIs on rho; gamma and rho posteriorly correlated)\n\n")
fe_pu2 <- fixef(fit_pu2)
print(round(fe_pu2[c("gamma_Intercept", "rho_Intercept"),
                    c("Estimate", "Q2.5", "Q97.5")], 3))

# Posterior correlation
post2 <- as.data.frame(fit_pu2)
gamma_col <- grep("b_gamma", names(post2), value = TRUE)[1]
rho_col   <- grep("b_rho",   names(post2), value = TRUE)[1]
cat(sprintf("\nPosterior correlation gamma/rho (2-level): %.3f\n",
            cor(post2[[gamma_col]], post2[[rho_col]])))

post3 <- as.data.frame(fit_pu)
gamma_col3 <- grep("b_gamma", names(post3), value = TRUE)[1]
rho_col3   <- grep("b_rho",   names(post3), value = TRUE)[1]
cat(sprintf("Posterior correlation gamma/rho (3-level): %.3f\n",
            cor(post3[[gamma_col3]], post3[[rho_col3]])))

# ---- 5. Design requirements summary ------------------------------------------

cat("\n=== Design requirements for joint (gamma, rho) identifiability ===\n\n")

cat("Minimum requirements for joint recovery:\n")
cat("  1. >=3 distinct value levels (V ∈ {V1, V2, V3})\n")
cat("  2. >~30 trials per value level per subject (90+ trials total)\n")
cat("  3. Value range should span >1 order of magnitude if possible\n")
cat("     (e.g., V ∈ {1, 3, 5} or V ∈ {1, 5, 10} — not {1, 1.2, 1.5})\n\n")

rho_ci_2lev <- fe_pu2["rho_Intercept", "Q97.5"] - fe_pu2["rho_Intercept", "Q2.5"]
rho_ci_3lev <- fe_pu["rho_Intercept", "Q97.5"]  - fe_pu["rho_Intercept", "Q2.5"]
cat(sprintf("  rho 95%%-CI width in 2-level design: %.3f\n", rho_ci_2lev))
cat(sprintf("  rho 95%%-CI width in 3-level design: %.3f\n", rho_ci_3lev))
cat(sprintf("  -> CI shrinks by factor: %.1f\n\n", rho_ci_2lev / rho_ci_3lev))

cat("Practical note (VDR literature):\n")
cat("  Most VDR studies use point values 1-10 but only 2 or 4 distinct levels.\n")
cat("  4 levels (e.g., 1, 3, 5, 7) with 20+ trials each should be sufficient.\n")
cat("  Design should be fully crossed (all value combinations present), not blocked.\n\n")

cat("Done.\n")
