# 13_recovery_one_cell.R
# WP1 — Run one cell of the reduced hierarchical EU recovery grid.
#
# Usage: Rscript 13_recovery_one_cell.R <gamma_true> <rep_i>
# Example: Rscript 13_recovery_one_cell.R 0.0 1
#
# Writes result to: results/wp1_gamma<gamma>_rep<rep>.csv
# Called by the bash loop in the round-3 redo; run each cell and push immediately.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("Usage: Rscript 13_recovery_one_cell.R <gamma_true> <rep_i>")

gamma_true <- as.numeric(args[1])
rep_i      <- as.integer(args[2])

cat(sprintf("=== WP1 cell: gamma=%.1f rep=%d ===\n\n", gamma_true, rep_i))

library(brms)
library(dplyr)

# ---- Shared parameters (match 06_realistic_recovery.R) ----------------------
N_SUBJ   <- 30L
N_TRIALS <- 100L
N        <- N_SUBJ * N_TRIALS

MU_A_TRUE <- 1.5; MU_C_TRUE <- 2.0; MU_B_TRUE <- 0.0
SD_A_TRUE <- 0.3; SD_C_TRUE <- 0.3; SD_GAMMA_TRUE <- 0.2

# ---- Helper: stable log-sum-exp softmax M3 -----------------------------------
log_Z_m3 <- function(act, n) {
  lv <- act + log(n)
  lv[1] + log(sum(exp(lv - lv[1])))
}

# ---- Formula and priors (match 06_realistic_recovery.R) ----------------------
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

hier_priors_eu <- c(
  prior(constant(0),    nlpar = "b",     class = "b"),
  prior(normal(2, 1),   nlpar = "a",     class = "b"),
  prior(normal(3, 1),   nlpar = "c",     class = "b"),
  prior(normal(0, 1),   nlpar = "gamma", class = "b"),
  prior(normal(0, 0.5), nlpar = "a",     class = "sd"),
  prior(normal(0, 0.5), nlpar = "c",     class = "sd"),
  prior(normal(0, 0.3), nlpar = "gamma", class = "sd")
)

# ---- Simulate data -----------------------------------------------------------
# Seed: deterministic from (gamma, rep) so results are exactly reproducible
seed_i <- 100L * round(gamma_true * 10) + rep_i
set.seed(seed_i)

a_subj     <- rnorm(N_SUBJ, MU_A_TRUE, SD_A_TRUE)
c_subj     <- rnorm(N_SUBJ, MU_C_TRUE, SD_C_TRUE)
gamma_subj <- rnorm(N_SUBJ, gamma_true, SD_GAMMA_TRUE)

subj_idx <- rep(seq_len(N_SUBJ), each = N_TRIALS)
V_corr   <- rep(c(2, 1), length.out = N)
V_other  <- 3 - V_corr
n_corr   <- rep(1L, N); n_other <- rep(4L, N); n_npl <- rep(5L, N)

resp <- integer(N)
for (i in seq_len(N)) {
  s   <- subj_idx[i]
  act <- c(
    MU_B_TRUE + a_subj[s] + c_subj[s] + gamma_subj[s] * V_corr[i],
    MU_B_TRUE + a_subj[s]             + gamma_subj[s] * V_other[i],
    MU_B_TRUE
  )
  n_i <- c(n_corr[i], n_other[i], n_npl[i])
  lZ  <- log_Z_m3(act, n_i)
  p   <- exp(act + log(n_i) - lZ)
  resp[i] <- sample.int(3L, 1L, prob = p)
}

Y <- matrix(0L, N, 3L)
for (i in seq_len(N)) Y[i, resp[i]] <- 1L
colnames(Y) <- c("corr", "other", "npl")

dat <- data.frame(
  subj    = subj_idx,
  V_corr  = V_corr,
  V_other = V_other,
  n_corr  = n_corr,
  n_other = n_other,
  n_npl   = n_npl,
  Idx_corr  = 1L, Idx_other = 1L, Idx_npl = 1L,
  nTrials   = 1L,
  resp      = I(Y)
)
dat$resp <- as.matrix(dat$resp)

# ---- Fit ---------------------------------------------------------------------
cat(sprintf("Fitting: gamma=%.1f rep=%d (N=%d subjects x %d trials, seed=%d)\n",
            gamma_true, rep_i, N_SUBJ, N_TRIALS, seed_i))

fit <- suppressWarnings(brm(
  eu_hier_formula,
  data    = dat,
  family  = multinomial(refcat = NA),
  prior   = hier_priors_eu,
  chains  = 2L,
  iter    = 1000L,
  warmup  = 500L,
  cores   = 2L,
  refresh = 200,
  backend = "cmdstanr",
  silent  = 0
))

# ---- Extract results ---------------------------------------------------------
fe         <- fixef(fit)
gamma_row  <- fe["gamma_Intercept", ]

covered   <- gamma_true >= gamma_row["Q2.5"] & gamma_true <= gamma_row["Q97.5"]
excl_zero <- gamma_row["Q2.5"] > 0 | gamma_row["Q97.5"] < 0
max_rhat  <- round(max(rhat(fit), na.rm = TRUE), 3)

result <- data.frame(
  gamma_true = gamma_true,
  rep        = rep_i,
  estimate   = round(gamma_row["Estimate"], 4),
  q2.5       = round(gamma_row["Q2.5"],     4),
  q97.5      = round(gamma_row["Q97.5"],    4),
  covered    = covered,
  excl_zero  = excl_zero,
  max_rhat   = max_rhat
)

cat("\n--- Result ---\n")
print(result)
cat(sprintf("  Covered: %s | Excl zero: %s | Max R-hat: %.3f\n\n",
            covered, excl_zero, max_rhat))

# ---- Write CSV ---------------------------------------------------------------
dir.create("local/exploration/m3-utility/results", showWarnings = FALSE)
fname <- sprintf("local/exploration/m3-utility/results/wp1_gamma%s_rep%d.csv",
                 gsub("\\.", "", sprintf("%.1f", gamma_true)), rep_i)
write.csv(result, fname, row.names = FALSE)
cat(sprintf("Result written to: %s\n", fname))
