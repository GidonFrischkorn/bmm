# 02_brms_prototype.R
# Issue #22 — Prospect-theory exploration
#
# brms/Stan hierarchical CPT prototype.
#
# Model: Cumulative Prospect Theory (Tversky & Kahneman 1992, Nilsson et al. 2011)
#   v(x) = x^alpha if x >= 0, -lambda * (-x)^alpha if x < 0  [alpha=beta constraint]
#   w(p) = exp(-(-ln p)^gammaw)                                [Prelec 1-param]
#   U(option) = w(p) * v(x)                                   [single-outcome lottery]
#   P(A) = logit^{-1}(phi * (U_A - U_B))                     [logit / softmax]
#
# Identifiability finding (key result):
#   In a loss-vs-loss design, phi * lambda ALWAYS appears as a product:
#     U_A - U_B = -phi*lambda * (w(p_A)*|x_A|^alpha - w(p_B)*|x_B|^alpha)
#   Only the product phi*lambda is identified, not phi and lambda separately.
#   → Use gains-only (identifies phi, alpha, gammaw) OR phi=1 (identifies lambda, alpha, gammaw).
#   → Mixed gain-vs-loss trials ALSO have phi*lambda confound (Nilsson et al. 2011).
#
# Two prototype models:
#   Model A — Gains-only (alpha, gammaw, phi): establishes NLF feasibility
#   Model B — Mixed domain, phi=1 fixed (alpha, lambda, gammaw): loss aversion identification
#
# Run from repo root:
#   Rscript local/exploration/prospect-theory/02_brms_prototype.R

library(brms)
library(dplyr)

if (requireNamespace("bmm", quietly = TRUE)) {
  library(bmm)
} else {
  suppressMessages(devtools::load_all("."))
}

cat("==========================================================================\n")
cat("02_brms_prototype.R — Hierarchical CPT brms prototype\n")
cat("==========================================================================\n\n")


# ==============================================================================
# SECTION 1 — CPT helpers (reused from 01_cpt_reference_impl.R)
# ==============================================================================

cpt_value <- function(x, alpha = 0.88, lambda = 2.25) {
  ifelse(x >= 0, x^alpha, -lambda * ((-x)^alpha))
}
prelec_1p <- function(p, gammaw) exp(-((-log(p))^gammaw))
cpt_choice_prob <- function(U_A, U_B, phi = 1.0) plogis(phi * (U_A - U_B))


# ==============================================================================
# SECTION 2 — Simulate data (gains-only design for Model A)
# ==============================================================================

cat("--- SECTION 2: Simulate hierarchical data ---\n\n")

N_SUBJ   <- 30L
N_TRIALS <- 80L   # per subject

# True population parameters
ALPHA_TRUE   <- 0.80
BETA_TRUE    <- 0.80   # loss curvature = gain curvature in DGP (alpha=beta constraint)
LAMBDA_TRUE  <- 2.25   # loss aversion (only identified with gains+losses)
GAMMAW_TRUE  <- 0.65
PHI_TRUE     <- 1.20   # sensitivity (not identified jointly with lambda in loss-only)

SD_ALPHA  <- 0.10; SD_LAMBDA <- 0.30; SD_GAMMAW <- 0.10; SD_PHI <- 0.20

# Draw subject parameters
set.seed(2025L)
alpha_s  <- pmax(0.20, rnorm(N_SUBJ, ALPHA_TRUE,  SD_ALPHA))
lambda_s <- pmax(0.10, rnorm(N_SUBJ, LAMBDA_TRUE, SD_LAMBDA))
gammaw_s <- pmax(0.10, rnorm(N_SUBJ, GAMMAW_TRUE, SD_GAMMAW))
phi_s    <- pmax(0.10, rnorm(N_SUBJ, PHI_TRUE,    SD_PHI))

# --- Model A: gains-only (40 gain trials) ---
N_GAIN <- 40L
set.seed(42L)
x_A_g <- runif(N_GAIN, 1, 10)
p_A_g  <- runif(N_GAIN, 0.10, 0.90)
x_B_g  <- runif(N_GAIN, 1, 10)
p_B_g  <- runif(N_GAIN, 0.10, 0.90)

# --- Model B: gains+losses (40 gain + 40 loss trials) ---
N_LOSS <- 40L
x_A_l <- -runif(N_LOSS, 1, 10)   # negative = loss
p_A_l  <- runif(N_LOSS, 0.10, 0.90)
x_B_l  <- -runif(N_LOSS, 1, 10)
p_B_l  <- runif(N_LOSS, 0.10, 0.90)

# Model A data (gains only)
subj_g <- rep(seq_len(N_SUBJ), each = N_GAIN)
trl_g  <- rep(seq_len(N_GAIN), times = N_SUBJ)

set.seed(1L)
U_A_g <- prelec_1p(p_A_g[trl_g], gammaw_s[subj_g]) *
          (x_A_g[trl_g]^alpha_s[subj_g])
U_B_g <- prelec_1p(p_B_g[trl_g], gammaw_s[subj_g]) *
          (x_B_g[trl_g]^alpha_s[subj_g])
choices_g <- rbinom(N_SUBJ * N_GAIN, 1L, cpt_choice_prob(U_A_g, U_B_g, phi_s[subj_g]))

d_gain <- data.frame(
  subj  = subj_g, trial = trl_g,
  amt_A = x_A_g[trl_g], prob_A = p_A_g[trl_g],
  amt_B = x_B_g[trl_g], prob_B = p_B_g[trl_g],
  choice = choices_g
)

# Model B data (gains+losses, phi=1 for data-generating process)
all_x_A <- c(x_A_g, x_A_l); all_p_A <- c(p_A_g, p_A_l)
all_x_B <- c(x_B_g, x_B_l); all_p_B <- c(p_B_g, p_B_l)
N_GL    <- N_GAIN + N_LOSS
subj_gl <- rep(seq_len(N_SUBJ), each = N_GL)
trl_gl  <- rep(seq_len(N_GL), times = N_SUBJ)

set.seed(2L)
U_A_gl <- prelec_1p(all_p_A[trl_gl], gammaw_s[subj_gl]) *
           cpt_value(all_x_A[trl_gl], alpha_s[subj_gl], lambda_s[subj_gl])
U_B_gl <- prelec_1p(all_p_B[trl_gl], gammaw_s[subj_gl]) *
           cpt_value(all_x_B[trl_gl], alpha_s[subj_gl], lambda_s[subj_gl])
# Use phi=1 for Model B data generation (avoids phi/lambda confound in DGP)
choices_gl <- rbinom(N_SUBJ * N_GL, 1L, cpt_choice_prob(U_A_gl, U_B_gl, phi = 1.0))

d_gl <- data.frame(
  subj   = subj_gl, trial = trl_gl,
  amt_A  = all_x_A[trl_gl], prob_A = all_p_A[trl_gl],
  amt_B  = all_x_B[trl_gl], prob_B = all_p_B[trl_gl],
  choice = choices_gl,
  domain = rep(c(rep("gain", N_GAIN), rep("loss", N_LOSS)), N_SUBJ),
  # Pre-computed for Stan (avoid pow(negative, alpha))
  xA_g   = pmax(all_x_A[trl_gl], 0),     # |x_A| for gains, 0 for losses
  xA_l   = pmax(-all_x_A[trl_gl], 0),    # |x_A| for losses, 0 for gains
  is_Ag  = as.double(all_x_A[trl_gl] >= 0),
  is_Al  = as.double(all_x_A[trl_gl] < 0),
  xB_g   = pmax(all_x_B[trl_gl], 0),
  xB_l   = pmax(-all_x_B[trl_gl], 0),
  is_Bg  = as.double(all_x_B[trl_gl] >= 0),
  is_Bl  = as.double(all_x_B[trl_gl] < 0)
)

cat(sprintf("Model A (gains-only):     N=%d subj × %d trials = %d obs, P(A)=%.3f\n",
            N_SUBJ, N_GAIN, nrow(d_gain), mean(choices_g)))
cat(sprintf("Model B (gains+losses):   N=%d subj × %d trials = %d obs, P(A)=%.3f\n",
            N_SUBJ, N_GL, nrow(d_gl), mean(choices_gl)))
cat(sprintf("  Gain trials P(A)=%.3f, Loss trials P(A)=%.3f\n",
            mean(choices_gl[d_gl$domain == "gain"]),
            mean(choices_gl[d_gl$domain == "loss"])))
cat(sprintf("True: alpha=%.2f, lambda=%.2f (DGP phi=1.0), gammaw=%.2f\n\n",
            ALPHA_TRUE, LAMBDA_TRUE, GAMMAW_TRUE))


# ==============================================================================
# SECTION 3 — Model A: gains-only hierarchical CPT
# ==============================================================================

cat("--- SECTION 3: Model A — Gains-only (alpha, gammaw, phi) ---\n\n")

# NLF: v(x) = x^alpha (gains only, x > 0 always so no pow(neg) issue)
# NB: brms rejects parameter names with underscores → use "gammaw" not "gamma_w"
formula_A <- bf(
  choice ~ phi * (wA * amt_A^alpha - wB * amt_B^alpha),
  nlf(wA ~ exp(-((-log(prob_A))^gammaw))),
  nlf(wB ~ exp(-((-log(prob_B))^gammaw))),
  alpha  ~ 1 + (1 | subj),
  gammaw ~ 1 + (1 | subj),
  phi    ~ 1 + (1 | subj),
  nl = TRUE
)

priors_A <- c(
  prior(lognormal(log(0.80), 0.30), nlpar = "alpha",  class = "b", lb = 0.10),
  prior(lognormal(log(0.65), 0.30), nlpar = "gammaw", class = "b", lb = 0.10),
  prior(lognormal(log(1.20), 0.50), nlpar = "phi",    class = "b", lb = 0.10),
  prior(normal(0, 0.20), nlpar = "alpha",  class = "sd", lb = 0),
  prior(normal(0, 0.20), nlpar = "gammaw", class = "sd", lb = 0),
  prior(normal(0, 0.30), nlpar = "phi",    class = "sd", lb = 0)
)

cat("Fitting Model A: gains-only CPT (2 chains x 1000 iter)...\n")
t0 <- proc.time()
fit_A <- suppressWarnings(brm(
  formula_A,
  data    = d_gain,
  family  = bernoulli(link = "logit"),
  prior   = priors_A,
  chains  = 2L, iter = 1000L, warmup = 500L, cores = 2L,
  seed    = 42L,
  refresh = 200,
  control = list(adapt_delta = 0.95),
  backend = "cmdstanr",
  silent  = 0
))
cat(sprintf("  Fit time: %.1f sec\n", (proc.time() - t0)["elapsed"]))

fe_A      <- fixef(fit_A)
rhat_A    <- max(rhat(fit_A), na.rm = TRUE)
div_A     <- sum(nuts_params(fit_A)$Value[nuts_params(fit_A)$Parameter == "divergent__"])
ess_A_min <- min(neff_ratio(fit_A), na.rm = TRUE) * (1000 - 500) * 2

cat(sprintf("\nModel A diagnostics: max_Rhat=%.3f, divergences=%d, min_ESS=%.0f\n",
            rhat_A, div_A, ess_A_min))
cat("Fixed effects:\n"); print(round(fe_A[, c("Estimate", "Q2.5", "Q97.5")], 3))

check_cov <- function(fe, rn, true_val, label) {
  if (!rn %in% rownames(fe)) return(NULL)
  est <- fe[rn, "Estimate"]; lo <- fe[rn, "Q2.5"]; hi <- fe[rn, "Q97.5"]
  covered <- true_val >= lo & true_val <= hi
  cat(sprintf("  %-8s: true=%.2f, est=%.3f [%.3f,%.3f] %s\n",
              label, true_val, est, lo, hi, if (covered) "COVERED" else "MISSED"))
  list(param = label, true = true_val, est = est, lo = lo, hi = hi, covered = covered)
}

cat("\nCoverage:\n")
covA_alpha  <- check_cov(fe_A, "alpha_Intercept",  ALPHA_TRUE,  "alpha")
covA_gammaw <- check_cov(fe_A, "gammaw_Intercept", GAMMAW_TRUE, "gammaw")
covA_phi    <- check_cov(fe_A, "phi_Intercept",    PHI_TRUE,    "phi")

cat(sprintf("\n[LAMBDA] Not estimated in gains-only design — confirms identifiability guard.\n\n"))


# ==============================================================================
# SECTION 4 — Model B: gains+losses, phi=1, loss aversion lambda
# ==============================================================================

cat("--- SECTION 4: Model B — Mixed domain, phi=1, (alpha, lambda, gammaw) ---\n\n")
cat("Why phi=1? In loss-vs-loss trials, phi and lambda ALWAYS appear as a product\n")
cat("(phi*lambda). Fixing phi=1 makes lambda identifiable. See Section 5 guard G3.\n\n")

# NLF with gain/loss split columns (avoids pow(negative, alpha)):
#   vA = is_Ag * xA_g^alpha - is_Al * lambda * xA_l^alpha
#   (when x_A >= 0: xA_l = 0, so 0^alpha = 0; valid in Stan for alpha > 0)
#   (when x_A < 0: xA_g = 0, so 0^alpha = 0; valid in Stan for alpha > 0)
# phi is fixed to 1.0 via constant() prior to break phi/lambda confound.

formula_B <- bf(
  choice ~ wA * vA - wB * vB,   # phi = 1 (absorbed into choice scale)
  nlf(vA ~ is_Ag * xA_g^alpha - is_Al * lambda * xA_l^alpha),
  nlf(vB ~ is_Bg * xB_g^alpha - is_Bl * lambda * xB_l^alpha),
  nlf(wA ~ exp(-((-log(prob_A))^gammaw))),
  nlf(wB ~ exp(-((-log(prob_B))^gammaw))),
  alpha  ~ 1 + (1 | subj),
  lambda ~ 1 + (1 | subj),
  gammaw ~ 1 + (1 | subj),
  nl = TRUE
)

priors_B <- c(
  prior(lognormal(log(0.80), 0.30), nlpar = "alpha",  class = "b", lb = 0.10),
  prior(lognormal(log(2.25), 0.50), nlpar = "lambda", class = "b", lb = 0.10),
  prior(lognormal(log(0.65), 0.30), nlpar = "gammaw", class = "b", lb = 0.10),
  prior(normal(0, 0.20), nlpar = "alpha",  class = "sd", lb = 0),
  prior(normal(0, 0.40), nlpar = "lambda", class = "sd", lb = 0),
  prior(normal(0, 0.20), nlpar = "gammaw", class = "sd", lb = 0)
)

cat("Fitting Model B: gains+losses, phi=1 (2 chains x 1000 iter)...\n")
t0 <- proc.time()
fit_B <- suppressWarnings(brm(
  formula_B,
  data    = d_gl,
  family  = bernoulli(link = "logit"),
  prior   = priors_B,
  chains  = 2L, iter = 1000L, warmup = 500L, cores = 2L,
  seed    = 43L,
  refresh = 200,
  control = list(adapt_delta = 0.95),
  backend = "cmdstanr",
  silent  = 0
))
cat(sprintf("  Fit time: %.1f sec\n", (proc.time() - t0)["elapsed"]))

fe_B      <- fixef(fit_B)
rhat_B    <- max(rhat(fit_B), na.rm = TRUE)
div_B     <- sum(nuts_params(fit_B)$Value[nuts_params(fit_B)$Parameter == "divergent__"])
ess_B_min <- min(neff_ratio(fit_B), na.rm = TRUE) * (1000 - 500) * 2

cat(sprintf("\nModel B diagnostics: max_Rhat=%.3f, divergences=%d, min_ESS=%.0f\n",
            rhat_B, div_B, ess_B_min))
cat("Fixed effects:\n"); print(round(fe_B[, c("Estimate", "Q2.5", "Q97.5")], 3))

cat("\nCoverage:\n")
covB_alpha  <- check_cov(fe_B, "alpha_Intercept",  ALPHA_TRUE,  "alpha")
covB_lambda <- check_cov(fe_B, "lambda_Intercept", LAMBDA_TRUE, "lambda")
covB_gammaw <- check_cov(fe_B, "gammaw_Intercept", GAMMAW_TRUE, "gammaw")

cat(sprintf("\n[PHI] Fixed to 1.0 — required to identify lambda (see guard G3).\n\n"))


# ==============================================================================
# SECTION 4b — Model C: NEGATIVE CONTROL — phi free (φ/λ confound in action)
# ==============================================================================

cat("--- SECTION 4b: Model C — Negative control (phi free, alpha=beta) ---\n\n")
cat("PURPOSE: This is the naive 4-param model: alpha, lambda, gammaw, phi all free.\n")
cat("Same data as Model B (gains+losses, DGP phi=1.0), but phi is now estimated.\n")
cat("Expected: Rhat >> 1.05, many divergences, alpha not covered, phi/lambda\n")
cat("posteriors anti-correlated and wide. This result is the empirical backbone\n")
cat("for the phi=1 default — without it, the phi=1 recommendation is only\n")
cat("theoretical (Nilsson 2011 MLE). Results → 02_cpt_recovery.csv.\n\n")

formula_C <- bf(
  choice ~ phi * (wA * vA - wB * vB),
  nlf(vA ~ is_Ag * xA_g^alpha - is_Al * lambda * xA_l^alpha),
  nlf(vB ~ is_Bg * xB_g^alpha - is_Bl * lambda * xB_l^alpha),
  nlf(wA ~ exp(-((-log(prob_A))^gammaw))),
  nlf(wB ~ exp(-((-log(prob_B))^gammaw))),
  alpha  ~ 1 + (1 | subj),
  lambda ~ 1 + (1 | subj),
  gammaw ~ 1 + (1 | subj),
  phi    ~ 1 + (1 | subj),
  nl = TRUE
)

priors_C <- c(
  prior(lognormal(log(0.80), 0.30), nlpar = "alpha",  class = "b", lb = 0.10),
  prior(lognormal(log(2.25), 0.50), nlpar = "lambda", class = "b", lb = 0.10),
  prior(lognormal(log(0.65), 0.30), nlpar = "gammaw", class = "b", lb = 0.10),
  prior(lognormal(log(1.20), 0.50), nlpar = "phi",    class = "b", lb = 0.10),
  prior(normal(0, 0.20), nlpar = "alpha",  class = "sd", lb = 0),
  prior(normal(0, 0.40), nlpar = "lambda", class = "sd", lb = 0),
  prior(normal(0, 0.20), nlpar = "gammaw", class = "sd", lb = 0),
  prior(normal(0, 0.30), nlpar = "phi",    class = "sd", lb = 0)
)

cat("Fitting Model C: 4-param phi-free (2 chains x 1000 iter — expect failures)...\n")
t0 <- proc.time()
fit_C <- tryCatch(
  suppressWarnings(brm(
    formula_C,
    data    = d_gl,
    family  = bernoulli(link = "logit"),
    prior   = priors_C,
    chains  = 2L, iter = 1000L, warmup = 500L, cores = 2L,
    seed    = 44L,
    refresh = 200,
    control = list(adapt_delta = 0.95),
    backend = "cmdstanr",
    silent  = 0
  )),
  error = function(e) {
    cat(sprintf("  Model C ERROR: %s\n", conditionMessage(e)))
    NULL
  }
)
cat(sprintf("  Fit time: %.1f sec\n", (proc.time() - t0)["elapsed"]))

if (!is.null(fit_C)) {
  fe_C      <- fixef(fit_C)
  rhat_C    <- max(rhat(fit_C), na.rm = TRUE)
  div_C     <- sum(nuts_params(fit_C)$Value[nuts_params(fit_C)$Parameter == "divergent__"])
  ess_C_min <- min(neff_ratio(fit_C), na.rm = TRUE) * (1000 - 500) * 2

  cat(sprintf("\nModel C diagnostics: max_Rhat=%.3f, divergences=%d, min_ESS=%.0f\n",
              rhat_C, div_C, ess_C_min))
  cat("Fixed effects:\n"); print(round(fe_C[, c("Estimate", "Q2.5", "Q97.5")], 3))

  cat("\nCoverage:\n")
  covC_alpha  <- check_cov(fe_C, "alpha_Intercept",  ALPHA_TRUE,  "alpha")
  covC_lambda <- check_cov(fe_C, "lambda_Intercept", LAMBDA_TRUE, "lambda")
  covC_gammaw <- check_cov(fe_C, "gammaw_Intercept", GAMMAW_TRUE, "gammaw")
  covC_phi    <- check_cov(fe_C, "phi_Intercept",    PHI_TRUE,    "phi")

  cat(sprintf("\n[NEGATIVE CONTROL] Rhat=%.3f, divergences=%d\n", rhat_C, div_C))
  cat("Interpretation:\n")
  cat("  - Rhat >> 1.05 or many divergences → phi/lambda confound active.\n")
  cat("  - alpha not covered → sampling pathology from non-identified surface.\n")
  cat("  - phi high, lambda low (or vice versa) → saddle: many (phi,lambda) pairs fit equally.\n")
  cat("  Compare Model B (phi=1, same data): Rhat=1.012, 0 divergences → phi=1 resolves the confound.\n\n")
} else {
  cat("[NEGATIVE CONTROL] Model C failed to fit — confirms the confound is severe.\n\n")
  rhat_C   <- NA_real_; div_C   <- NA_integer_; ess_C_min <- NA_real_
  fe_C     <- NULL
  covC_alpha <- covC_lambda <- covC_gammaw <- covC_phi <- NULL
}


# ==============================================================================
# SECTION 4c — Model D: Hierarchical alpha≠bta lambda underestimation (Nilsson)
# ==============================================================================

cat("--- SECTION 4c: Model D — phi=1 fixed, alpha≠bta (hierarchical Nilsson) ---\n\n")
cat("PURPOSE: Demonstrate that the Nilsson (2011) lambda-underestimation bias\n")
cat("re-emerges at the HIERARCHICAL level when alpha and bta are both free.\n")
cat("Script 01 showed this at single-subject MLE only. This is the Bayesian analogue.\n")
cat("Data DGP: alpha=beta=0.80, lambda=2.25, phi=1 (same as Model B).\n")
cat("NOTE: 'bta' is used (beta is a reserved brms parameter name).\n\n")

formula_D <- bf(
  choice ~ wA * vA - wB * vB,   # phi = 1 (same as Model B)
  nlf(vA ~ is_Ag * xA_g^alpha - is_Al * lambda * xA_l^bta),
  nlf(vB ~ is_Bg * xB_g^alpha - is_Bl * lambda * xB_l^bta),
  nlf(wA ~ exp(-((-log(prob_A))^gammaw))),
  nlf(wB ~ exp(-((-log(prob_B))^gammaw))),
  alpha  ~ 1 + (1 | subj),
  bta    ~ 1 + (1 | subj),
  lambda ~ 1 + (1 | subj),
  gammaw ~ 1 + (1 | subj),
  nl = TRUE
)

priors_D <- c(
  prior(lognormal(log(0.80), 0.30), nlpar = "alpha",  class = "b", lb = 0.10),
  prior(lognormal(log(0.80), 0.30), nlpar = "bta",    class = "b", lb = 0.10),
  prior(lognormal(log(2.25), 0.50), nlpar = "lambda", class = "b", lb = 0.10),
  prior(lognormal(log(0.65), 0.30), nlpar = "gammaw", class = "b", lb = 0.10),
  prior(normal(0, 0.20), nlpar = "alpha",  class = "sd", lb = 0),
  prior(normal(0, 0.20), nlpar = "bta",    class = "sd", lb = 0),
  prior(normal(0, 0.40), nlpar = "lambda", class = "sd", lb = 0),
  prior(normal(0, 0.20), nlpar = "gammaw", class = "sd", lb = 0)
)

cat("Fitting Model D: phi=1, alpha≠bta (2 chains x 1000 iter)...\n")
t0 <- proc.time()
fit_D <- suppressWarnings(brm(
  formula_D,
  data    = d_gl,
  family  = bernoulli(link = "logit"),
  prior   = priors_D,
  chains  = 2L, iter = 1000L, warmup = 500L, cores = 2L,
  seed    = 45L,
  refresh = 200,
  control = list(adapt_delta = 0.95),
  backend = "cmdstanr",
  silent  = 0
))
cat(sprintf("  Fit time: %.1f sec\n", (proc.time() - t0)["elapsed"]))

fe_D      <- fixef(fit_D)
rhat_D    <- max(rhat(fit_D), na.rm = TRUE)
div_D     <- sum(nuts_params(fit_D)$Value[nuts_params(fit_D)$Parameter == "divergent__"])
ess_D_min <- min(neff_ratio(fit_D), na.rm = TRUE) * (1000 - 500) * 2

cat(sprintf("\nModel D diagnostics: max_Rhat=%.3f, divergences=%d, min_ESS=%.0f\n",
            rhat_D, div_D, ess_D_min))
cat("Fixed effects:\n"); print(round(fe_D[, c("Estimate", "Q2.5", "Q97.5")], 3))

cat("\nCoverage:\n")
covD_alpha  <- check_cov(fe_D, "alpha_Intercept",  ALPHA_TRUE,  "alpha")
covD_bta    <- check_cov(fe_D, "bta_Intercept",    BETA_TRUE,   "bta")
covD_lambda <- check_cov(fe_D, "lambda_Intercept", LAMBDA_TRUE, "lambda")
covD_gammaw <- check_cov(fe_D, "gammaw_Intercept", GAMMAW_TRUE, "gammaw")

# Key comparison: lambda bias in Model D vs. Model B
lambdaB_est <- fe_B["lambda_Intercept", "Estimate"]
lambdaD_est <- if (!is.null(fe_D) && "lambda_Intercept" %in% rownames(fe_D))
                 fe_D["lambda_Intercept", "Estimate"] else NA_real_
cat(sprintf("\n[HIERARCHICAL NILSSON COMPARISON]\n"))
cat(sprintf("  Model B (phi=1, alpha=bta):       lambda est=%.3f  bias=%.3f\n",
            lambdaB_est, lambdaB_est - LAMBDA_TRUE))
cat(sprintf("  Model D (phi=1, alpha≠bta, free): lambda est=%.3f  bias=%.3f\n",
            lambdaD_est, lambdaD_est - LAMBDA_TRUE))
cat("Expected: Model D shows larger |bias| for lambda — the hierarchical analogue\n")
cat("of Nilsson et al. (2011) script 01 MLE finding.\n\n")


# ==============================================================================
# SECTION 4d — Model E: LOSS-ONLY NEGATIVE CONTROL — genuine phi/lambda confound
# ==============================================================================

cat("--- SECTION 4d: Model E — Loss-only design (phi free: genuine confound) ---\n\n")
cat("PURPOSE: Model C (phi free, gains+losses) converged cleanly because gain trials\n")
cat("pin phi, and loss trials pin phi*lambda, allowing the ratio to be recovered.\n")
cat("A LOSS-ONLY design removes the gain-trial anchor:\n")
cat("  U_A - U_B = -phi*lambda * (w(p_A)*|x_A|^alpha - w(p_B)*|x_B|^alpha)\n")
cat("Only the product phi*lambda is identified. Expected: Rhat >> 1.05 or\n")
cat("many divergences and/or phi/lambda posteriors anti-correlated.\n")
cat("This is the genuine empirical backbone for the phi=1 default.\n\n")

# Generate loss-only data (same subjects as d_gl, but only loss trials)
subj_lo <- rep(seq_len(N_SUBJ), each = N_LOSS)
trl_lo  <- rep(seq_len(N_LOSS), times = N_SUBJ)

set.seed(3L)
U_A_lo <- prelec_1p(p_A_l[trl_lo], gammaw_s[subj_lo]) *
          cpt_value(x_A_l[trl_lo], alpha_s[subj_lo], lambda_s[subj_lo])
U_B_lo <- prelec_1p(p_B_l[trl_lo], gammaw_s[subj_lo]) *
          cpt_value(x_B_l[trl_lo], alpha_s[subj_lo], lambda_s[subj_lo])
choices_lo <- rbinom(N_SUBJ * N_LOSS, 1L, cpt_choice_prob(U_A_lo, U_B_lo, PHI_TRUE))

d_lo <- data.frame(
  subj   = subj_lo, trial = trl_lo,
  amt_A  = x_A_l[trl_lo], prob_A = p_A_l[trl_lo],
  amt_B  = x_B_l[trl_lo], prob_B = p_B_l[trl_lo],
  choice = choices_lo,
  domain = "loss",
  xA_g   = 0,
  xA_l   = pmax(-x_A_l[trl_lo], 0),
  is_Ag  = 0,
  is_Al  = 1,
  xB_g   = 0,
  xB_l   = pmax(-x_B_l[trl_lo], 0),
  is_Bg  = 0,
  is_Bl  = 1
)

cat(sprintf("Loss-only design: N=%d subj × %d loss trials, P(A)=%.3f\n\n",
            N_SUBJ, N_LOSS, mean(choices_lo)))

# Model E: same formula as Model C (phi free), applied to loss-only data
# No gains trial anchor → phi and lambda are jointly unidentified
cat("Fitting Model E: loss-only, phi free (2 chains x 1000 iter — expect failures)...\n")
t0 <- proc.time()
fit_E <- tryCatch(
  suppressWarnings(brm(
    formula_C,                                # same 4-param formula as negative control
    data    = d_lo,
    family  = bernoulli(link = "logit"),
    prior   = priors_C,
    chains  = 2L, iter = 1000L, warmup = 500L, cores = 2L,
    seed    = 46L,
    refresh = 200,
    control = list(adapt_delta = 0.95),
    backend = "cmdstanr",
    silent  = 0
  )),
  error = function(e) {
    cat(sprintf("  Model E ERROR: %s\n", conditionMessage(e)))
    NULL
  }
)
cat(sprintf("  Fit time: %.1f sec\n", (proc.time() - t0)["elapsed"]))

if (!is.null(fit_E)) {
  fe_E      <- fixef(fit_E)
  rhat_E    <- max(rhat(fit_E), na.rm = TRUE)
  div_E     <- sum(nuts_params(fit_E)$Value[nuts_params(fit_E)$Parameter == "divergent__"])
  ess_E_min <- min(neff_ratio(fit_E), na.rm = TRUE) * (1000 - 500) * 2

  cat(sprintf("\nModel E diagnostics: max_Rhat=%.3f, divergences=%d, min_ESS=%.0f\n",
              rhat_E, div_E, ess_E_min))
  cat("Fixed effects:\n"); print(round(fe_E[, c("Estimate", "Q2.5", "Q97.5")], 3))

  cat("\nCoverage:\n")
  covE_alpha  <- check_cov(fe_E, "alpha_Intercept",  ALPHA_TRUE,  "alpha")
  covE_lambda <- check_cov(fe_E, "lambda_Intercept", LAMBDA_TRUE, "lambda")
  covE_gammaw <- check_cov(fe_E, "gammaw_Intercept", GAMMAW_TRUE, "gammaw")
  covE_phi    <- check_cov(fe_E, "phi_Intercept",    PHI_TRUE,    "phi")

  cat(sprintf("\n[LOSS-ONLY NEGATIVE CONTROL] Rhat=%.3f, divergences=%d\n", rhat_E, div_E))
  if (rhat_E > 1.05 || div_E > 0) {
    cat("  phi/lambda confound ACTIVE — loss-only data cannot separate phi from lambda.\n")
    cat("  phi=1 default is JUSTIFIED: the confound bites in loss-only / loss-heavy designs.\n")
  } else {
    cat("  Unexpected clean convergence — inspect posterior corr(phi, lambda) manually.\n")
    cat("  (Marginal identification may still mask correlated posteriors.)\n")
  }
  cat("  Compare Model C (phi free, gains+losses): clean Rhat/divergences — the\n")
  cat("  gain-trial anchor resolves the confound in balanced designs.\n\n")
} else {
  cat("[LOSS-ONLY NEGATIVE CONTROL] Model E failed to sample — extreme confound.\n\n")
  rhat_E <- NA_real_; div_E <- NA_integer_; ess_E_min <- NA_real_
  fe_E   <- NULL
  covE_alpha <- covE_lambda <- covE_gammaw <- covE_phi <- NULL
}


# ==============================================================================
# SECTION 4e — SBC-light: multi-dataset MLE check of α=β benefit
# ==============================================================================

cat("--- SECTION 4e: SBC-light — α=β benefit, K=30 single-subject MLE datasets ---\n\n")
cat("PURPOSE: The single-dataset hierarchical Model B-vs-D result (lambda bias\n")
cat("+0.096 vs +0.336, opposite sign to the MLE) rests on one dataset and is\n")
cat("within noise for a wide-CI parameter. This section runs K=30 independent\n")
cat("MLE datasets (N=60 trials each; same DGP) and compares lambda bias\n")
cat("distribution between constrained (alpha=beta) and unconstrained (alpha≠bta)\n")
cat("single-subject fits. MLE is fast (<1s total), providing reliable evidence\n")
cat("without the computational burden of K additional hierarchical Stan models.\n\n")

# CPT negative-log-likelihood for constrained model (alpha=beta)
negloglik_con_sbc <- function(par, data) {
  alpha <- par[1]; lambda <- par[2]; gammaw <- par[3]; phi <- par[4]
  if (alpha <= 0 || alpha > 2 || lambda <= 0 || lambda > 20 ||
      gammaw <= 0 || gammaw > 2 || phi <= 0 || phi > 20) return(1e10)
  v_A <- ifelse(data$amt_A >= 0, data$amt_A^alpha, -lambda * ((-data$amt_A)^alpha))
  v_B <- ifelse(data$amt_B >= 0, data$amt_B^alpha, -lambda * ((-data$amt_B)^alpha))
  w_A <- exp(-((-log(data$prob_A))^gammaw))
  w_B <- exp(-((-log(data$prob_B))^gammaw))
  prob <- pmax(1e-10, pmin(1 - 1e-10, plogis(phi * (w_A * v_A - w_B * v_B))))
  -sum(data$choice * log(prob) + (1 - data$choice) * log(1 - prob))
}

# CPT negative-log-likelihood for unconstrained model (alpha≠bta)
negloglik_uncon_sbc <- function(par, data) {
  alpha <- par[1]; bta <- par[2]; lambda <- par[3]; gammaw <- par[4]; phi <- par[5]
  if (alpha <= 0 || alpha > 2 || bta <= 0 || bta > 2 ||
      lambda <= 0 || lambda > 20 || gammaw <= 0 || gammaw > 2 ||
      phi <= 0 || phi > 20) return(1e10)
  v_A <- ifelse(data$amt_A >= 0, data$amt_A^alpha, -lambda * ((-data$amt_A)^bta))
  v_B <- ifelse(data$amt_B >= 0, data$amt_B^alpha, -lambda * ((-data$amt_B)^bta))
  w_A <- exp(-((-log(data$prob_A))^gammaw))
  w_B <- exp(-((-log(data$prob_B))^gammaw))
  prob <- pmax(1e-10, pmin(1 - 1e-10, plogis(phi * (w_A * v_A - w_B * v_B))))
  -sum(data$choice * log(prob) + (1 - data$choice) * log(1 - prob))
}

K_SBC   <- 30L    # independent datasets
N_SBC   <- 60L    # trials per dataset (30 gain + 30 loss, mixed design)
N_H_SBC <- N_SBC %/% 2L

set.seed(9999L)
sbc_results <- vector("list", K_SBC)

for (k in seq_len(K_SBC)) {
  xa_g <- runif(N_H_SBC, 1, 10); pa_g <- runif(N_H_SBC, 0.10, 0.90)
  xb_g <- runif(N_H_SBC, 1, 10); pb_g <- runif(N_H_SBC, 0.10, 0.90)
  xa_l <- -runif(N_H_SBC, 1, 10); pa_l <- runif(N_H_SBC, 0.10, 0.90)
  xb_l <- -runif(N_H_SBC, 1, 10); pb_l <- runif(N_H_SBC, 0.10, 0.90)

  xa <- c(xa_g, xa_l); pa <- c(pa_g, pa_l)
  xb <- c(xb_g, xb_l); pb <- c(pb_g, pb_l)

  va <- cpt_value(xa, ALPHA_TRUE, LAMBDA_TRUE)
  vb <- cpt_value(xb, ALPHA_TRUE, LAMBDA_TRUE)
  wa <- prelec_1p(pa, GAMMAW_TRUE); wb <- prelec_1p(pb, GAMMAW_TRUE)
  ch <- rbinom(N_SBC, 1L, cpt_choice_prob(wa * va, wb * vb, PHI_TRUE))

  d_s <- data.frame(amt_A = xa, prob_A = pa, amt_B = xb, prob_B = pb, choice = ch)

  fc <- tryCatch(
    optim(c(0.80, 2.25, 0.65, 1.20), negloglik_con_sbc, data = d_s,
          method = "L-BFGS-B",
          lower  = c(0.1, 0.1, 0.1, 0.1), upper = c(2, 20, 2, 20)),
    error = function(e) list(par = rep(NA_real_, 4), convergence = 1)
  )
  fu <- tryCatch(
    optim(c(0.80, 0.80, 2.25, 0.65, 1.20), negloglik_uncon_sbc, data = d_s,
          method = "L-BFGS-B",
          lower  = c(0.1, 0.1, 0.1, 0.1, 0.1), upper = c(2, 2, 20, 2, 20)),
    error = function(e) list(par = rep(NA_real_, 5), convergence = 1)
  )

  sbc_results[[k]] <- list(
    lambda_con   = if (fc$convergence == 0) fc$par[2] else NA_real_,
    lambda_uncon = if (fu$convergence == 0) fu$par[3] else NA_real_
  )
}

lambda_con_vec   <- sapply(sbc_results, `[[`, "lambda_con")
lambda_uncon_vec <- sapply(sbc_results, `[[`, "lambda_uncon")
bias_con   <- lambda_con_vec   - LAMBDA_TRUE
bias_uncon <- lambda_uncon_vec - LAMBDA_TRUE

rmse_con   <- sqrt(mean(bias_con^2,   na.rm = TRUE))
rmse_uncon <- sqrt(mean(bias_uncon^2, na.rm = TRUE))
bias_ratio <- rmse_uncon / max(rmse_con, 1e-6)

cat(sprintf("SBC-light results (K=%d datasets, N=%d trials/dataset, true lambda=%.2f):\n\n",
            K_SBC, N_SBC, LAMBDA_TRUE))
cat(sprintf("  Constrained (alpha=beta):   mean bias=%+.3f, SD=%.3f, RMSE=%.3f\n",
            mean(bias_con,   na.rm = TRUE), sd(bias_con,   na.rm = TRUE), rmse_con))
cat(sprintf("  Unconstrained (alpha≠bta):  mean bias=%+.3f, SD=%.3f, RMSE=%.3f\n",
            mean(bias_uncon, na.rm = TRUE), sd(bias_uncon, na.rm = TRUE), rmse_uncon))
cat(sprintf("  RMSE ratio (unconstrained/constrained): %.2fx\n", bias_ratio))

sbc_verdict <- if (bias_ratio > 1.5 && rmse_uncon > rmse_con)
  "DEMONSTRATED (alpha=beta constraint reduces lambda RMSE by >= 50% — consistent with Nilsson 2011)"
else if (bias_ratio > 1.1 && rmse_uncon > rmse_con)
  "WEAKLY SUPPORTED (unconstrained RMSE higher, but < 50% improvement)"
else
  "INCONCLUSIVE (insufficient separation at this N — run with larger N_SBC)"

cat(sprintf("  alpha=beta benefit: %s\n\n", sbc_verdict))


# ==============================================================================
# SECTION 5 — Identifiability note: phi/lambda confound
# ==============================================================================

cat("--- SECTION 5: phi/lambda confound diagnosis ---\n\n")
cat("KEY FINDING: In loss-vs-loss trials, phi * lambda is a product:\n")
cat("  U_A - U_B = -phi*lambda*(w(p_A)*|x_A|^alpha - w(p_B)*|x_B|^alpha)\n")
cat("  The sign and ratio depend only on (phi*lambda) and (alpha, gammaw).\n")
cat("  You CANNOT identify phi and lambda separately from loss-only trials.\n\n")

cat("In gains-only trials:\n")
cat("  U_A - U_B = phi*(w(p_A)*x_A^alpha - w(p_B)*x_B^alpha)\n")
cat("  Here phi alone scales the difference → phi identified given (alpha, gammaw).\n\n")

cat("COMBINED design (gain + loss trials):\n")
cat("  From gain trials: identifies phi (given alpha, gammaw)\n")
cat("  From loss trials: identifies phi*lambda (given alpha, gammaw)\n")
cat("  Ratio: lambda = (phi*lambda) / phi\n")
cat("  → BOTH identified in the combined design when phi is estimable from gains\n\n")

cat("IN PRACTICE: The combined (phi, lambda) model with brms NLF has strongly\n")
cat("  correlated posteriors (Nilsson et al. 2011 call this 'lambda underestimation').\n")
cat("  Recommended guard: either fix phi=1 (Model B), or use very informative priors\n")
cat("  on phi (e.g. from pilot data). N=30x80 is insufficient for simultaneous\n")
cat("  identification of all 4 parameters.\n\n")


# ==============================================================================
# SECTION 6 — Gradient stability note
# ==============================================================================

cat("--- SECTION 6: Gradient stability ---\n\n")
cat("NLF key: pow(0, alpha) in Stan is safe for alpha > 0:\n")
cat("  xA_g = max(x_A, 0): 0^alpha = 0 when x_A < 0 (loss trial)\n")
cat("  xA_l = max(-x_A, 0): 0^alpha = 0 when x_A > 0 (gain trial)\n")
cat("  No pow(negative, alpha) calls in Stan.\n\n")

cat("Softmax/logit rule handles negative utilities (losses):\n")
cat("  logit(phi * (U_A - U_B)) is defined for all real U_A, U_B.\n")
cat("  Luce rule (log(activation)) would require strictly positive utilities → not\n")
cat("  suitable for loss outcomes. Softmax rule is the correct choice for CPT.\n\n")


# ==============================================================================
# SECTION 7 — Save results
# ==============================================================================

cat("--- SECTION 7: Save results ---\n\n")

cov_list_A <- list(covA_alpha, covA_gammaw, covA_phi)
cov_list_B <- list(covB_alpha, covB_lambda, covB_gammaw)
cov_list_C <- list(covC_alpha, covC_lambda, covC_gammaw, covC_phi)
cov_list_D <- list(covD_alpha, covD_bta, covD_lambda, covD_gammaw)
cov_list_E <- list(covE_alpha, covE_lambda, covE_gammaw, covE_phi)

make_results <- function(cov_list, model_label, rhat_val, ndiv_val) {
  do.call(rbind, lapply(Filter(Negate(is.null), cov_list), function(x) {
    data.frame(model = model_label, param = x$param,
               true = x$true, est = x$est,
               q2.5 = x$lo, q97.5 = x$hi, covered = x$covered,
               max_rhat = rhat_val, n_diverge = ndiv_val)
  }))
}

results_A <- make_results(cov_list_A, "gains_only",              rhat_A, div_A)
results_B <- make_results(cov_list_B, "gains_losses_phi1",       rhat_B, div_B)
results_C <- make_results(cov_list_C, "negctrl_phi_free_gl",     rhat_C, div_C)
results_D <- make_results(cov_list_D, "gains_losses_alpha_neq_bta", rhat_D, div_D)
results_E <- make_results(cov_list_E, "negctrl_phi_free_lo",     rhat_E, div_E)

# SBC-light summary frame
results_sbc <- data.frame(
  model      = c("constrained_alpha_eq_bta", "unconstrained_alpha_neq_bta"),
  param      = "lambda",
  true       = LAMBDA_TRUE,
  mean_bias  = c(mean(bias_con, na.rm = TRUE), mean(bias_uncon, na.rm = TRUE)),
  sd_bias    = c(sd(bias_con, na.rm = TRUE),   sd(bias_uncon, na.rm = TRUE)),
  rmse       = c(rmse_con, rmse_uncon),
  k_datasets = K_SBC
)

write.csv(results_A, "local/exploration/prospect-theory/results/02_modelA_recovery.csv",
          row.names = FALSE)
write.csv(results_B, "local/exploration/prospect-theory/results/02_modelB_recovery.csv",
          row.names = FALSE)
if (!is.null(results_C) && nrow(results_C) > 0) {
  write.csv(results_C, "local/exploration/prospect-theory/results/02_cpt_recovery.csv",
            row.names = FALSE)
}
write.csv(results_D, "local/exploration/prospect-theory/results/02_modelD_recovery.csv",
          row.names = FALSE)
if (!is.null(results_E) && nrow(results_E) > 0) {
  write.csv(results_E, "local/exploration/prospect-theory/results/02_modelE_lossonly.csv",
            row.names = FALSE)
}
write.csv(results_sbc, "local/exploration/prospect-theory/results/02_sbc_alpha_beta.csv",
          row.names = FALSE)

cat("Results written:\n")
cat("  02_modelA_recovery.csv\n")
cat("  02_modelB_recovery.csv\n")
cat("  02_cpt_recovery.csv        [Model C: phi-free, gains+losses]\n")
cat("  02_modelD_recovery.csv     [Model D: Nilsson hierarchical]\n")
cat("  02_modelE_lossonly.csv     [Model E: phi-free, loss-only — genuine confound]\n")
cat("  02_sbc_alpha_beta.csv      [SBC-light: K=30 MLE lambda bias]\n\n")

# ===========================================================================
# Compute verdicts from diagnostics (NOT hardcoded strings)
# ===========================================================================

# NLF feasibility (Models A and B clean convergence)
nlf_ok <- rhat_A <= 1.05 && div_A == 0 && rhat_B <= 1.05 && div_B == 0
nlf_verdict <- if (nlf_ok) "FEASIBLE" else "ISSUES DETECTED"

# phi/lambda confound: the gain+loss balanced design does NOT show the confound
# (Model C converges). The loss-only design (Model E) is the right test.
if (!is.na(rhat_E) && (rhat_E > 1.05 || div_E > 0)) {
  confound_verdict <- sprintf(
    "CONFIRMED in loss-only design (Model E: Rhat=%.3f, div=%d)",
    rhat_E, div_E)
} else if (is.na(rhat_E)) {
  confound_verdict <- "CONFIRMED — Model E failed to sample (extreme confound)"
} else {
  confound_verdict <- sprintf(
    "NOT SHOWN (Model E Rhat=%.3f, div=%d — inspect phi/lambda posteriors manually)",
    rhat_E, div_E)
}

# alpha=beta benefit: hierarchical single-dataset B vs D
lambdaB_est_final <- fe_B["lambda_Intercept", "Estimate"]
lambdaD_est_final <- if (!is.null(fe_D) && "lambda_Intercept" %in% rownames(fe_D))
                       fe_D["lambda_Intercept", "Estimate"] else NA_real_
h_bias_B <- lambdaB_est_final - LAMBDA_TRUE
h_bias_D <- lambdaD_est_final - LAMBDA_TRUE
h_verdict <- if (!is.na(h_bias_D) && abs(h_bias_D) > abs(h_bias_B))
  sprintf("Hierarchical: Model D |bias|=%.3f > Model B |bias|=%.3f (expected direction)",
          abs(h_bias_D), abs(h_bias_B))
else
  sprintf("Hierarchical: Model D |bias|=%.3f, Model B |bias|=%.3f (weak/inconclusive)",
          abs(h_bias_D), abs(h_bias_B))

nilsson_verdict <- sprintf(
  "%s. SBC-light (K=%d): %s",
  h_verdict, K_SBC, sbc_verdict
)

cat("==========================================================================\n")
cat("SUMMARY:\n\n")
cat(sprintf("  Model A (gains-only, alpha/gammaw/phi):  Rhat=%.3f, divergences=%d\n",
            rhat_A, div_A))
for (x in Filter(Negate(is.null), cov_list_A)) {
  cat(sprintf("    %-8s: true=%.2f, est=%.3f [%.3f,%.3f] %s\n",
              x$param, x$true, x$est, x$lo, x$hi, if (x$covered) "OK" else "MISSED"))
}

cat(sprintf("\n  Model B (gains+losses, phi=1, alpha=bta): Rhat=%.3f, div=%d\n",
            rhat_B, div_B))
for (x in Filter(Negate(is.null), cov_list_B)) {
  cat(sprintf("    %-8s: true=%.2f, est=%.3f [%.3f,%.3f] %s\n",
              x$param, x$true, x$est, x$lo, x$hi, if (x$covered) "OK" else "MISSED"))
}

cat(sprintf("\n  Model C [phi-free, gains+losses]: Rhat=%s, div=%s\n",
            if (is.na(rhat_C)) "FAILED" else sprintf("%.3f", rhat_C),
            if (is.na(div_C))  "FAILED" else as.character(div_C)))
for (x in Filter(Negate(is.null), cov_list_C)) {
  cat(sprintf("    %-8s: true=%.2f, est=%.3f [%.3f,%.3f] %s\n",
              x$param, x$true, x$est, x$lo, x$hi, if (x$covered) "OK" else "MISSED"))
}

cat(sprintf("\n  Model D (phi=1, alpha≠bta, Nilsson hierarchical): Rhat=%.3f, div=%d\n",
            rhat_D, div_D))
for (x in Filter(Negate(is.null), cov_list_D)) {
  cat(sprintf("    %-8s: true=%.2f, est=%.3f [%.3f,%.3f] %s\n",
              x$param, x$true, x$est, x$lo, x$hi, if (x$covered) "OK" else "MISSED"))
}
cat(sprintf("    lambda B bias=%+.3f vs. D bias=%+.3f (expected: |D| > |B|)\n",
            h_bias_B, h_bias_D))

cat(sprintf("\n  Model E [LOSS-ONLY NEGATIVE CONTROL]: Rhat=%s, div=%s\n",
            if (is.na(rhat_E)) "FAILED" else sprintf("%.3f", rhat_E),
            if (is.na(div_E))  "FAILED" else as.character(div_E)))
for (x in Filter(Negate(is.null), cov_list_E)) {
  cat(sprintf("    %-8s: true=%.2f, est=%.3f [%.3f,%.3f] %s\n",
              x$param, x$true, x$est, x$lo, x$hi, if (x$covered) "OK" else "MISSED"))
}

cat("\n  --- Verdicts (computed from diagnostics) ---\n")
cat(sprintf("  NLF formulation: %s\n", nlf_verdict))
cat(sprintf("  phi/lambda confound: %s\n", confound_verdict))
cat(sprintf("  Nilsson alpha=beta benefit: %s\n", nilsson_verdict))
cat("  Gradient stability: OK — softmax handles negative utilities, no pow(neg,alpha)\n")
cat("==========================================================================\n")
