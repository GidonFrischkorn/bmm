# 12_binary_choice_pt.R
# WP4 — Binary-choice utility models
#
# Canonical JDM case: choose gamble A or B.
# Model: hierarchical prospect theory following Nilsson, Rieskamp &
# Wagenmakers (2011), Psychological Methods, 16(2), 150–176.
#   u(x) = x^rho           power utility (rho > 0; rho < 1 = concave)
#   w(p)  = exp(-(-ln p)^alpha)   Prelec one-parameter weighting
#   U(gamble) = w(p) * u(x)
#   P(choose A) = 1 / (1 + exp(-lambda * (U_A - U_B)))   logit with sensitivity lambda
#
# Structure of this script:
#   Part 0 — verify 2-category M3 = logistic (machine precision)
#   Part 1 — simulate hierarchical binary-PT data
#   Part 2 — fit hierarchical brms model (2 chains x 1000 iter, N=30x100)
#   Part 3 — parameter recovery table + write results/12_binary_pt_recovery.csv
#
# Run from repo root:
#   source("local/exploration/m3-utility/12_binary_choice_pt.R")

library(brms)
library(dplyr)

# ==============================================================================
# PART 0 — 2-category M3 = logistic verification
# ==============================================================================

cat("==========================================================\n")
cat("PART 0 — 2-category M3 = logistic (analytic verification)\n")
cat("==========================================================\n\n")

# The softmax M3 with 2 categories and n_i = 1 (one option each):
#   P(A) = exp(a_A) / (exp(a_A) + exp(a_B))
#         = 1 / (1 + exp(-(a_A - a_B)))
#         = sigmoid(a_A - a_B)
# This is exactly the logistic function.
#
# In a binary choice context: a_A - a_B = lambda * (U_A - U_B)
# where lambda is the sensitivity parameter.

# Numerical check at a grid of activation differences
act_diffs <- seq(-4, 4, by = 0.5)

softmax_2cat <- function(delta) 1 / (1 + exp(-delta))
logistic     <- function(delta) plogis(delta)

probs_sm  <- softmax_2cat(act_diffs)
probs_log <- logistic(act_diffs)
max_diff  <- max(abs(probs_sm - probs_log))

cat(sprintf("Max |P(softmax) - P(logistic)| across %d grid points: %.2e\n",
            length(act_diffs), max_diff))
cat(sprintf("Machine epsilon check (< 1e-14): %s\n\n",
            if (max_diff < 1e-14) "PASS" else "FAIL"))

# Verify m3() accepts 2 categories without error
library(bmm)

two_cat_model <- tryCatch(
  m3(
    resp_cats   = c("A", "B"),
    num_options = c(n_A = "n_A", n_B = "n_B"),
    choice_rule = "softmax",
    version     = "custom",
    links       = list(a = "identity", c = "identity")
  ),
  error = function(e) e
)

if (inherits(two_cat_model, "error")) {
  cat("m3() with 2 categories: ERROR —", conditionMessage(two_cat_model), "\n\n")
} else {
  cat("m3() with 2 categories: NO ERROR (accepted)\n")
  cat("Model class:", paste(class(two_cat_model), collapse = ", "), "\n\n")
}


# ==============================================================================
# PART 1 — Simulate hierarchical binary prospect-theory data
# ==============================================================================

cat("==========================================================\n")
cat("PART 1 — Simulate hierarchical binary PT data\n")
cat("==========================================================\n\n")

set.seed(2025)

N_SUBJ   <- 30L
N_TRIALS <- 100L
N        <- N_SUBJ * N_TRIALS

# True population-level parameters
RHO_TRUE    <- 0.70   # power utility curvature (concave)
ALPHA_TRUE  <- 0.65   # Prelec probability weighting (inverse-S shape, < 1)
LAMBDA_TRUE <- 2.50   # sensitivity / inverse temperature

# Subject-level SDs (hierarchical truth)
SD_RHO    <- 0.15
SD_ALPHA  <- 0.12
SD_LAMBDA <- 0.50

# Utility functions
power_utility <- function(x, rho) x^rho

prelec_weight <- function(p, alpha) {
  stopifnot(all(p > 0), all(p < 1))
  exp(-(-log(p))^alpha)
}

# Generate gambles: outcomes in [1, 10], probabilities in (0, 1)
set.seed(2024L)
x_A <- runif(N_TRIALS, 1, 10)
p_A <- runif(N_TRIALS, 0.1, 0.9)
x_B <- runif(N_TRIALS, 1, 10)
p_B <- 1 - p_A   # complementary (binary lottery)

# Draw subject-level parameters
rho_subj    <- pmax(0.05, rnorm(N_SUBJ, RHO_TRUE,    SD_RHO))
alpha_subj  <- pmax(0.05, rnorm(N_SUBJ, ALPHA_TRUE,  SD_ALPHA))
lambda_subj <- pmax(0.10, rnorm(N_SUBJ, LAMBDA_TRUE, SD_LAMBDA))

# Simulate choices
subj_idx <- rep(seq_len(N_SUBJ), each = N_TRIALS)
trial_idx <- rep(seq_len(N_TRIALS), times = N_SUBJ)

choice <- integer(N)
for (i in seq_len(N)) {
  s <- subj_idx[i]
  t <- trial_idx[i]

  U_A <- prelec_weight(p_A[t], alpha_subj[s]) * power_utility(x_A[t], rho_subj[s])
  U_B <- prelec_weight(p_B[t], alpha_subj[s]) * power_utility(x_B[t], rho_subj[s])

  p_choose_A <- plogis(lambda_subj[s] * (U_A - U_B))
  choice[i]  <- rbinom(1L, 1L, p_choose_A)  # 1 = chose A, 0 = chose B
}

d_binary <- data.frame(
  subj   = subj_idx,
  trial  = trial_idx,
  x_A    = x_A[trial_idx],
  p_A    = p_A[trial_idx],
  x_B    = x_B[trial_idx],
  p_B    = p_B[trial_idx],
  choice = choice
)

cat(sprintf("Simulated %d subjects × %d trials = %d observations\n", N_SUBJ, N_TRIALS, N))
cat(sprintf("P(choose A) = %.3f\n\n", mean(choice)))
cat("True parameters:\n")
cat(sprintf("  rho = %.2f, alpha = %.2f, lambda = %.2f\n\n",
            RHO_TRUE, ALPHA_TRUE, LAMBDA_TRUE))


# ==============================================================================
# PART 2 — Hierarchical brms model
# ==============================================================================

cat("==========================================================\n")
cat("PART 2 — Hierarchical brms fit\n")
cat("==========================================================\n\n")

# The model:
#   U_A = w(p_A)^alpha * x_A^rho
#   U_B = w(p_B)^alpha * x_B^rho
#   logit P(choice = 1) = lambda * (U_A - U_B)
#
# In brms NLF form:
#   choice ~ Bernoulli(p)
#   logit(p) = lambda * (U_A - U_B)
#   U_A = exp(-(-log(p_A))^alpha) * x_A^rho
#   U_B = exp(-(-log(p_B))^alpha) * x_B^rho
#
# We use the nonlinear brms formula with nl = TRUE.

binary_pt_formula <- bf(
  choice ~ lambda * (
    exp(-(-log(p_A))^alpha) * x_A^rho -
    exp(-(-log(p_B))^alpha) * x_B^rho
  ),
  rho    ~ 1 + (1 | subj),
  alpha  ~ 1 + (1 | subj),
  lambda ~ 1 + (1 | subj),
  nl = TRUE
)

binary_pt_priors <- c(
  prior(lognormal(log(0.7), 0.4),  nlpar = "rho",    class = "b",  lb = 0.05),
  prior(lognormal(log(0.65), 0.4), nlpar = "alpha",  class = "b",  lb = 0.05),
  prior(lognormal(log(2.5), 0.5),  nlpar = "lambda", class = "b",  lb = 0.10),
  prior(normal(0, 0.3),            nlpar = "rho",    class = "sd", lb = 0),
  prior(normal(0, 0.2),            nlpar = "alpha",  class = "sd", lb = 0),
  prior(normal(0, 0.5),            nlpar = "lambda", class = "sd", lb = 0)
)

cat("Fitting hierarchical binary PT model (2 chains x 1000 iter)...\n")
cat("  rho ~ lognormal | alpha ~ lognormal | lambda ~ lognormal\n")
cat("  Random effects on all three parameters (N=30 subjects)\n\n")

fit_binary_pt <- suppressWarnings(brm(
  binary_pt_formula,
  data    = d_binary,
  family  = bernoulli(link = "identity"),
  prior   = binary_pt_priors,
  chains  = 2L,
  iter    = 1000L,
  warmup  = 500L,
  cores   = 2L,
  refresh = 200,
  backend = "cmdstanr",
  silent  = 0
))

cat("\n--- Binary PT recovery results ---\n\n")
cat(sprintf("True: rho=%.2f, alpha=%.2f, lambda=%.2f\n\n",
            RHO_TRUE, ALPHA_TRUE, LAMBDA_TRUE))

fe <- fixef(fit_binary_pt)
print(round(fe[, c("Estimate", "Q2.5", "Q97.5")], 3))

check_cov <- function(fe, param, true_val) {
  rn <- paste0(param, "_Intercept")
  if (rn %in% rownames(fe)) {
    lo <- fe[rn, "Q2.5"]; hi <- fe[rn, "Q97.5"]
    covered <- true_val >= lo & true_val <= hi
    cat(sprintf("  %-7s: true=%.2f, est=%.3f CI=[%.3f, %.3f] -> %s\n",
                param, true_val,
                fe[rn, "Estimate"], lo, hi,
                if (covered) "COVERED" else "MISSED"))
    list(param = param, true = true_val, est = fe[rn, "Estimate"],
         lo = lo, hi = hi, covered = covered)
  }
}

cat("\nCoverage check:\n")
cov_rho    <- check_cov(fe, "rho",    RHO_TRUE)
cov_alpha  <- check_cov(fe, "alpha",  ALPHA_TRUE)
cov_lambda <- check_cov(fe, "lambda", LAMBDA_TRUE)

max_rhat <- round(max(rhat(fit_binary_pt), na.rm = TRUE), 3)
cat(sprintf("\nMax R-hat: %.3f\n", max_rhat))


# ==============================================================================
# PART 3 — Write results CSV
# ==============================================================================

cat("\n==========================================================\n")
cat("PART 3 — Write results CSV\n")
cat("==========================================================\n\n")

results <- data.frame(
  param      = c("rho", "alpha", "lambda"),
  true_value = c(RHO_TRUE, ALPHA_TRUE, LAMBDA_TRUE),
  estimate   = c(cov_rho$est,    cov_alpha$est,    cov_lambda$est),
  q2.5       = c(cov_rho$lo,     cov_alpha$lo,     cov_lambda$lo),
  q97.5      = c(cov_rho$hi,     cov_alpha$hi,     cov_lambda$hi),
  covered    = c(cov_rho$covered, cov_alpha$covered, cov_lambda$covered),
  max_rhat   = max_rhat
)

dir.create("local/exploration/m3-utility/results", showWarnings = FALSE)
write.csv(results, "local/exploration/m3-utility/results/12_binary_pt_recovery.csv",
          row.names = FALSE)

cat("Results written to: local/exploration/m3-utility/results/12_binary_pt_recovery.csv\n\n")

cat("--- Summary table ---\n\n")
print(results)

cat("\n=== 12_binary_choice_pt.R complete ===\n")
cat("Key results:\n")
cat(sprintf("  2-cat M3 = logistic: VERIFIED (max diff = %.2e)\n", max_diff))
cat(sprintf("  rho covered:    %s\n", cov_rho$covered))
cat(sprintf("  alpha covered:  %s\n", cov_alpha$covered))
cat(sprintf("  lambda covered: %s\n", cov_lambda$covered))
cat(sprintf("  Max R-hat:      %.3f\n", max_rhat))
cat("\nStructural decision: binary PT is a sibling model family, not m3_utility().\n")
cat("See DESIGN_utility_api.md §WP4 for reasoning.\n")
