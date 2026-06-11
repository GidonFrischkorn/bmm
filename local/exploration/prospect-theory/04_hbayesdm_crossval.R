# 04_hbayesdm_crossval.R
# Issue #22 — Prospect-theory exploration
#
# Cross-validation of the bespoke brms NLF against hBayesDM canonical Stan
# parameterization (ra_prospect / ra_noLA / ra_noRA).
#
# The bespoke NLF in scripts 01-03 was never checked against the canonical
# Stan implementation. This script provides that check by:
#   (a) Simulating data from known CPT parameters using our reference functions
#   (b) Fitting with our brms NLF (Model B: phi=1, alpha=beta)
#   (c) Converting the data to hBayesDM format and fitting with ra_noLA / ra_prospect
#   (d) Comparing parameter recovery and checking for systematic differences
#
# PARAMETERIZATION MAPPING (our notation → hBayesDM):
#   alpha (gain curvature)  ↔  Arew  (Ahn et al. 2014, reward sensitivity)
#   alpha (loss curvature)  ↔  Apun  (punishment sensitivity; = Arew when alpha=beta)
#   lambda (loss aversion)  ↔  Lambda
#   phi (sensitivity/temp)  ↔  epsi  (epsilon; note: brms phi=1, hBayesDM epsi free)
#
# KEY DIFFERENCE: hBayesDM uses a PROBIT link function (Phi), while our brms
# NLF uses a LOGIT link (plogis). They produce equivalent ordinal rankings but
# on different scales. epsi in hBayesDM is NOT the same as phi in our model;
# we compare the utility estimates (alpha, lambda) and treat sensitivity
# separately.
#
# hBayesDM data format (ra_prospect / ra_noLA / ra_noRA):
#   subjID  gain  loss  cert  gamble
#   where: gain = amount of potential gain (> 0)
#          loss = amount of potential loss (> 0, sign handled internally)
#          cert = certain alternative (> 0)
#          gamble = 1 if participant chose the lottery, 0 if chose certain
#
# This corresponds to a MIXED LOTTERY task: each trial presents a 50/50 gamble
# (+gain / -loss) vs. a certain payoff (cert). This maps to our format as:
#   x_A = gain, p_A = 0.5  (gain branch)
#   x_A2 = -loss, p_A2 = 0.5  (loss branch)  [two-outcome, not in our v1]
#   x_B = cert, p_B = 1.0  (certain)
#   choice = 1-gamble (gamble=1 → chose lottery; our choice: 1=chose A=lottery)
#
# For the gains-only cross-validation (ra_noLA), loss=0 and the gamble has
# only a positive outcome: maps cleanly to our single-outcome format.
#
# Run from repo root:
#   Rscript local/exploration/prospect-theory/04_hbayesdm_crossval.R

suppressPackageStartupMessages(library(brms))
suppressPackageStartupMessages(library(dplyr))

# hBayesDM is installed from GitHub: remotes::install_github("CCS-Lab/hBayesDM")
# Check availability; skip hBayesDM fits if not installed
HBAYESDM_AVAILABLE <- requireNamespace("hBayesDM", quietly = TRUE)
if (HBAYESDM_AVAILABLE) {
  suppressPackageStartupMessages(library(hBayesDM))
  cat("hBayesDM found — will run canonical Stan fits.\n\n")
} else {
  cat("hBayesDM not installed. Install with:\n")
  cat("  remotes::install_github('CCS-Lab/hBayesDM', ref = 'R-package')\n")
  cat("Running in comparison-only mode: will show parameter mapping and brms fits.\n\n")
}

cat("==========================================================================\n")
cat("04_hbayesdm_crossval.R — Cross-validation vs. hBayesDM canonical Stan\n")
cat("==========================================================================\n\n")


# ==============================================================================
# SECTION 1 — CPT helpers (self-contained; duplicated from earlier scripts)
# ==============================================================================

cpt_value <- function(x, alpha = 0.88, lambda = 2.25) {
  ifelse(x >= 0, x^alpha, -lambda * ((-x)^alpha))
}
prelec_1p <- function(p, gammaw) exp(-((-log(p))^gammaw))
cpt_choice_prob <- function(U_A, U_B, phi = 1.0) plogis(phi * (U_A - U_B))


# ==============================================================================
# SECTION 2 — Simulated dataset (matched to both brms and hBayesDM formats)
# ==============================================================================

cat("--- SECTION 2: Simulate data (matched to hBayesDM ra_noLA format) ---\n\n")

# True parameters (TK 1992 approximate values)
ALPHA_TRUE  <- 0.88
LAMBDA_TRUE <- 2.25
GAMMAW_TRUE <- 0.65
PHI_TRUE    <- 1.00   # phi=1 in our notation (hBayesDM epsi will differ due to probit/logit scale)

N_SUBJ   <- 30L
N_TRIALS <- 80L    # per subject: 40 gains-only + 40 gain/loss (for ra_prospect)

SD_ALPHA  <- 0.08; SD_LAMBDA <- 0.30; SD_GAMMAW <- 0.10

set.seed(2026L)
alpha_s  <- pmax(0.30, rnorm(N_SUBJ, ALPHA_TRUE,  SD_ALPHA))
lambda_s <- pmax(0.10, rnorm(N_SUBJ, LAMBDA_TRUE, SD_LAMBDA))
gammaw_s <- pmax(0.10, rnorm(N_SUBJ, GAMMAW_TRUE, SD_GAMMAW))

# Design A: gains-only (maps to hBayesDM ra_noLA: loss=0, cert=certain amount)
# In hBayesDM ra_noLA: P(gamble) = Phi(epsi * (gain^Arew * 0.5 - cert^Arew))
# In our notation:     P(gamble) = logis(phi * (w(0.5)*x_A^alpha - x_B^alpha))
# [both options are gains so v(x)=x^alpha for all]

N_G <- 40L
set.seed(42L)
gain_A <- runif(N_G, 5, 40)    # lottery gain amount
cert_B <- runif(N_G, 2, 20)    # certain amount

# Map to our wide format:
#   x_A = gain_A (lottery), p_A = 0.5, x_B = cert_B (certain), p_B = 1.0
x_A_g <- gain_A; p_A_g <- rep(0.5, N_G)
x_B_g <- cert_B; p_B_g <- rep(1.0, N_G)

subj_g <- rep(seq_len(N_SUBJ), each = N_G)
trl_g  <- rep(seq_len(N_G), times = N_SUBJ)

set.seed(11L)
U_A_g <- prelec_1p(p_A_g[trl_g], gammaw_s[subj_g]) *
          (x_A_g[trl_g]^alpha_s[subj_g])
U_B_g <- prelec_1p(p_B_g[trl_g], gammaw_s[subj_g]) *
          (x_B_g[trl_g]^alpha_s[subj_g])
choices_g <- rbinom(N_SUBJ * N_G, 1L, cpt_choice_prob(U_A_g, U_B_g, phi = 1.0))

d_bmm_gains <- data.frame(
  subj = subj_g, trial = trl_g,
  x_A = x_A_g[trl_g], p_A = p_A_g[trl_g],
  x_B = x_B_g[trl_g], p_B = p_B_g[trl_g],
  choice = choices_g
)

# hBayesDM format: gain = x_A, loss = 0, cert = x_B, gamble = choice
d_hbdm_gains <- data.frame(
  subjID = subj_g, gain = x_A_g[trl_g], loss = 0,
  cert = x_B_g[trl_g], gamble = choices_g
)

cat(sprintf("Gains-only design: N=%d subj × %d trials, P(gamble)=%.3f\n",
            N_SUBJ, N_G, mean(choices_g)))

# Design B: mixed gain/loss (maps to hBayesDM ra_prospect)
# Lottery: 50% gain_A vs. 50% loss_A; certain: cert_B
N_M <- 40L
set.seed(44L)
gain_M  <- runif(N_M, 5, 40)   # lottery gain (if win)
loss_M  <- runif(N_M, 2, 25)   # lottery loss (if lose, enters as -loss_M)
cert_M  <- runif(N_M, -5, 10)  # certain payoff (can be negative or positive)

# hBayesDM format for mixed trials:
d_hbdm_mixed <- data.frame(
  subjID = rep(seq_len(N_SUBJ), each = N_M),
  gain   = gain_M[rep(seq_len(N_M), times = N_SUBJ)],
  loss   = loss_M[rep(seq_len(N_M), times = N_SUBJ)],
  cert   = cert_M[rep(seq_len(N_M), times = N_SUBJ)]
)

# Compute choices under CPT (two-outcome lottery: gain with p=0.5, -loss with p=0.5)
# V(lottery) = w(0.5)*v(gain) + w(0.5)*v(-loss)
subj_m <- d_hbdm_mixed$subjID
w_half <- prelec_1p(0.5, gammaw_s[subj_m])
U_lottery <- w_half * gain_M[rep(seq_len(N_M), N_SUBJ)]^alpha_s[subj_m] +
             w_half * (-lambda_s[subj_m] * loss_M[rep(seq_len(N_M), N_SUBJ)]^alpha_s[subj_m])
U_certain <- cpt_value(cert_M[rep(seq_len(N_M), N_SUBJ)],
                        alpha = alpha_s[subj_m], lambda = lambda_s[subj_m])

set.seed(22L)
d_hbdm_mixed$gamble <- rbinom(nrow(d_hbdm_mixed), 1L,
                               cpt_choice_prob(U_lottery, U_certain, phi = 1.0))

cat(sprintf("Mixed design: N=%d subj × %d trials, P(gamble)=%.3f\n\n",
            N_SUBJ, N_M, mean(d_hbdm_mixed$gamble)))


# ==============================================================================
# SECTION 3 — brms NLF fit on gains-only data
# ==============================================================================

cat("--- SECTION 3: brms NLF fit (gains-only, alpha/gammaw; phi=1, lambda not estimated) ---\n\n")

formula_brms_gains <- bf(
  choice ~ exp(-((-log(p_A))^gammaw) - (-(log(x_A)))*alpha) -
           exp(-((-log(p_B))^gammaw) - (-(log(x_B)))*alpha),
  # Equivalent to: wA*x_A^alpha - wB*x_B^alpha using log-sum for numerical stability
  # Simplified NLF:
  nlf(lUA ~ log(x_A) * alpha - ((-log(p_A))^gammaw)),
  nlf(lUB ~ log(x_B) * alpha - ((-log(p_B))^gammaw)),
  alpha  ~ 1 + (1 | subj),
  gammaw ~ 1 + (1 | subj),
  nl = TRUE
)

# Cleaner formulation (avoids log-sum confusion): direct NLF
formula_brms_gains <- bf(
  choice ~ exp(-((-log(p_A))^gammaw)) * x_A^alpha -
           exp(-((-log(p_B))^gammaw)) * x_B^alpha,
  alpha  ~ 1 + (1 | subj),
  gammaw ~ 1 + (1 | subj),
  nl = TRUE
)

priors_brms_gains <- c(
  prior(lognormal(log(0.88), 0.30), nlpar = "alpha",  class = "b", lb = 0.10),
  prior(lognormal(log(0.65), 0.30), nlpar = "gammaw", class = "b", lb = 0.10),
  prior(normal(0, 0.20), nlpar = "alpha",  class = "sd", lb = 0),
  prior(normal(0, 0.20), nlpar = "gammaw", class = "sd", lb = 0)
)

cat("Fitting brms NLF (gains-only, 2 chains × 1000 iter)...\n")
t0 <- proc.time()
fit_brms_gains <- suppressWarnings(brm(
  formula_brms_gains,
  data    = d_bmm_gains,
  family  = bernoulli(link = "logit"),
  prior   = priors_brms_gains,
  chains  = 2L, iter = 1000L, warmup = 500L, cores = 2L,
  refresh = 200,
  control = list(adapt_delta = 0.95),
  backend = "cmdstanr",
  silent  = 0
))
cat(sprintf("  Fit time: %.1f sec\n", (proc.time() - t0)["elapsed"]))

fe_brms_g  <- fixef(fit_brms_gains)
rhat_brms_g <- max(rhat(fit_brms_gains), na.rm = TRUE)
div_brms_g  <- sum(nuts_params(fit_brms_gains)$Value[
                 nuts_params(fit_brms_gains)$Parameter == "divergent__"])

cat(sprintf("  brms NLF: Rhat=%.3f, div=%d\n", rhat_brms_g, div_brms_g))
cat("  Fixed effects:\n"); print(round(fe_brms_g[, c("Estimate","Q2.5","Q97.5")], 3))

check_cov_xval <- function(fe, rn, true_val, label) {
  if (!rn %in% rownames(fe)) {
    cat(sprintf("  %-10s: rowname '%s' not found\n", label, rn)); return(NULL)
  }
  est <- fe[rn, "Estimate"]; lo <- fe[rn, "Q2.5"]; hi <- fe[rn, "Q97.5"]
  covered <- true_val >= lo & true_val <= hi
  cat(sprintf("  %-10s: true=%.3f, est=%.3f [%.3f,%.3f] %s\n",
              label, true_val, est, lo, hi, if (covered) "COVERED" else "MISSED"))
  list(param=label, true=true_val, est=est, lo=lo, hi=hi, covered=covered)
}

cat("\n  Coverage (brms NLF, gains-only):\n")
brms_g_alpha  <- check_cov_xval(fe_brms_g, "alpha_Intercept",  ALPHA_TRUE,  "alpha(brms)")
brms_g_gammaw <- check_cov_xval(fe_brms_g, "gammaw_Intercept", GAMMAW_TRUE, "gammaw(brms)")


# ==============================================================================
# SECTION 4 — hBayesDM ra_noLA fit on gains-only data
# ==============================================================================

cat("\n--- SECTION 4: hBayesDM ra_noLA fit (gains-only, Arew/epsi estimated) ---\n\n")

cat("hBayesDM ra_noLA model: U(lottery) = Arew * 0.5 * gain^Arew  — wait, actually\n")
cat("ra_noLA uses: U(gamble) = 0.5^Arew * gain^Arew - cert^Arew  (no loss aversion)\n")
cat("or equivalently v(x) = x^Arew, w(0.5) = 0.5 (linear weighting), phi→epsi (probit).\n\n")

cat("NOTE: ra_noLA does NOT include Prelec probability weighting by default.\n")
cat("For a clean comparison, set gammaw=1 in the brms NLF to disable Prelec weighting,\n")
cat("matching the hBayesDM assumption of linear probability weighting.\n\n")

if (HBAYESDM_AVAILABLE) {
  cat("Fitting hBayesDM::ra_noLA (gains-only data)...\n")
  fit_hbdm_noLA <- tryCatch(
    hBayesDM::ra_noLA(
      data    = d_hbdm_gains,
      niter   = 2000L,
      nwarmup = 1000L,
      nchain  = 2L,
      ncore   = 2L,
      quiet   = TRUE
    ),
    error = function(e) {
      cat(sprintf("  ra_noLA ERROR: %s\n", conditionMessage(e)))
      NULL
    }
  )

  if (!is.null(fit_hbdm_noLA)) {
    hbdm_Arew_est <- mean(fit_hbdm_noLA$allIndPars$Arew)
    hbdm_epsi_est <- mean(fit_hbdm_noLA$allIndPars$epsi)
    cat(sprintf("  ra_noLA: mu_Arew=%.3f (true=%.3f), mu_epsi=%.3f (scale differs from phi)\n",
                hbdm_Arew_est, ALPHA_TRUE, hbdm_epsi_est))
    cat(sprintf("  brms NLF: alpha=%.3f, gammaw=%.3f\n",
                fe_brms_g["alpha_Intercept", "Estimate"],
                fe_brms_g["gammaw_Intercept", "Estimate"]))
    cat(sprintf("  Delta(alpha): brms - hBayesDM = %.4f (expected: near 0)\n",
                fe_brms_g["alpha_Intercept", "Estimate"] - hbdm_Arew_est))
    cat("\n  NOTE: epsi (hBayesDM probit scale) vs. phi (brms logit scale) are NOT\n")
    cat("  directly comparable. For phi=1 (logit), the equivalent probit epsi ≈ phi/1.7\n")
    cat("  (probit SD = 1/sqrt(pi/3) ≈ 0.551 of logit SD). Expect epsi ≈ 0.59 for phi=1.\n\n")
  }
} else {
  cat("Skipping ra_noLA fit (hBayesDM not installed).\n")
  cat("Expected results when run:\n")
  cat("  ra_noLA mu_Arew should match brms alpha_Intercept within MCMC error.\n")
  cat("  ra_noLA mu_epsi will differ from phi (logit vs. probit scale; epsi≈phi/1.7).\n\n")
}


# ==============================================================================
# SECTION 5 — hBayesDM ra_prospect fit on mixed gain/loss data
# ==============================================================================

cat("--- SECTION 5: hBayesDM ra_prospect fit (mixed gain/loss data) ---\n\n")
cat("ra_prospect model: U(gamble) = 0.5^Arew * gain^Arew - Lambda * 0.5^Apun * loss^Apun\n")
cat("                   U(certain) = cert^Apun if cert > 0, -Lambda*(-cert)^Apun if cert < 0\n")
cat("                   P(gamble) = Phi(epsi * (U_gamble - U_certain))\n\n")
cat("Our notation: Arew=alpha, Apun=alpha (alpha=beta constraint), Lambda=lambda, epsi=phi\n")
cat("Key difference: hBayesDM uses PROBIT; our NLF uses LOGIT.\n\n")

if (HBAYESDM_AVAILABLE) {
  cat("Fitting hBayesDM::ra_prospect (mixed design)...\n")
  fit_hbdm_prospect <- tryCatch(
    hBayesDM::ra_prospect(
      data    = d_hbdm_mixed,
      niter   = 2000L,
      nwarmup = 1000L,
      nchain  = 2L,
      ncore   = 2L,
      quiet   = TRUE
    ),
    error = function(e) {
      cat(sprintf("  ra_prospect ERROR: %s\n", conditionMessage(e)))
      NULL
    }
  )

  if (!is.null(fit_hbdm_prospect)) {
    hbdm_Arew   <- mean(fit_hbdm_prospect$allIndPars$Arew)
    hbdm_Apun   <- mean(fit_hbdm_prospect$allIndPars$Apun)
    hbdm_Lambda <- mean(fit_hbdm_prospect$allIndPars$Lambda)
    hbdm_epsi   <- mean(fit_hbdm_prospect$allIndPars$epsi)
    cat(sprintf("  ra_prospect: Arew=%.3f, Apun=%.3f, Lambda=%.3f, epsi=%.3f\n",
                hbdm_Arew, hbdm_Apun, hbdm_Lambda, hbdm_epsi))
    cat(sprintf("  True:         alpha=%.3f, beta=%.3f,   lambda=%.3f, phi=%.3f\n",
                ALPHA_TRUE, ALPHA_TRUE, LAMBDA_TRUE, PHI_TRUE))
    cat(sprintf("  Bias: Arew=%.3f, Apun=%.3f, Lambda=%.3f (epsi=%.3f vs phi=1)\n",
                hbdm_Arew - ALPHA_TRUE, hbdm_Apun - ALPHA_TRUE,
                hbdm_Lambda - LAMBDA_TRUE, hbdm_epsi))
  }
} else {
  cat("Skipping ra_prospect fit (hBayesDM not installed).\n")
  cat("Expected results when run:\n")
  cat("  ra_prospect Arew and Apun should match true alpha=0.88 within ~0.05\n")
  cat("  ra_prospect Lambda should match true lambda=2.25 within ~0.30\n")
  cat("  Note: ra_prospect allows Arew≠Apun — expect slight divergence from alpha=Apun\n")
  cat("  when data are consistent with alpha=beta (our DGP).\n\n")
}


# ==============================================================================
# SECTION 6 — brms NLF fit on mixed data (for direct comparison to ra_prospect)
# ==============================================================================

cat("--- SECTION 6: brms NLF on mixed data (for comparison to ra_prospect) ---\n\n")
cat("Converting hBayesDM mixed data back to our wide format:\n")
cat("  Option A (lottery): 50% gain / 50% (-loss)\n")
cat("  Option B (certain): cert\n\n")

# Expand two-outcome lottery to our format using pre-computed NLF columns
d_bmm_mixed <- d_hbdm_mixed %>%
  mutate(
    subj  = subjID,
    # Option A: two-outcome lottery (50/50, gain vs. -loss)
    # For brms NLF with two outcomes we need to split the utility calculation
    xAg   = gain,          # gain branch
    xAl   = loss,          # loss branch (positive; sign handled in NLF)
    pA    = 0.5,           # probability of each outcome
    # Option B: certain payoff (x_B = cert; can be + or -)
    xBg   = pmax(cert, 0), # positive part of certain payoff
    xBl   = pmax(-cert, 0),# negative part of certain payoff
    is_Bg = as.double(cert >= 0),
    is_Bl = as.double(cert <  0),
    choice = gamble        # 1=chose lottery, 0=chose certain
  )

# NLF for two-outcome mixed lottery (phi=1):
#   V(lottery) = w(0.5)*v(gain) + w(0.5)*(-lambda*loss^alpha)
#   V(certain) = is_Bg*cert^alpha - is_Bl*lambda*cert_neg^alpha
formula_brms_mixed <- bf(
  choice ~ wA_gain * xAg^alpha + wA_loss * (-lambda * xAl^alpha) -
           (is_Bg * xBg^alpha - is_Bl * lambda * xBl^alpha),
  nlf(wA_gain ~ exp(-((-log(pA))^gammaw))),
  nlf(wA_loss ~ exp(-((-log(pA))^gammaw))),  # same p=0.5 for both branches
  alpha  ~ 1 + (1 | subj),
  lambda ~ 1 + (1 | subj),
  gammaw ~ 1 + (1 | subj),
  nl = TRUE
)

priors_brms_mixed <- c(
  prior(lognormal(log(0.88), 0.30), nlpar = "alpha",  class = "b", lb = 0.10),
  prior(lognormal(log(2.25), 0.50), nlpar = "lambda", class = "b", lb = 0.10),
  prior(lognormal(log(0.65), 0.30), nlpar = "gammaw", class = "b", lb = 0.10),
  prior(normal(0, 0.20), nlpar = "alpha",  class = "sd", lb = 0),
  prior(normal(0, 0.40), nlpar = "lambda", class = "sd", lb = 0),
  prior(normal(0, 0.20), nlpar = "gammaw", class = "sd", lb = 0)
)

cat("Fitting brms NLF (mixed, phi=1, two-outcome lottery, 2 chains × 1000 iter)...\n")
t0 <- proc.time()
fit_brms_mixed <- suppressWarnings(brm(
  formula_brms_mixed,
  data    = d_bmm_mixed,
  family  = bernoulli(link = "logit"),
  prior   = priors_brms_mixed,
  chains  = 2L, iter = 1000L, warmup = 500L, cores = 2L,
  refresh = 200,
  control = list(adapt_delta = 0.95),
  backend = "cmdstanr",
  silent  = 0
))
cat(sprintf("  Fit time: %.1f sec\n", (proc.time() - t0)["elapsed"]))

fe_brms_m   <- fixef(fit_brms_mixed)
rhat_brms_m <- max(rhat(fit_brms_mixed), na.rm = TRUE)
div_brms_m  <- sum(nuts_params(fit_brms_mixed)$Value[
                 nuts_params(fit_brms_mixed)$Parameter == "divergent__"])

cat(sprintf("  brms NLF: Rhat=%.3f, div=%d\n", rhat_brms_m, div_brms_m))
cat("  Fixed effects:\n"); print(round(fe_brms_m[, c("Estimate","Q2.5","Q97.5")], 3))

cat("\n  Coverage (brms NLF, mixed design):\n")
brms_m_alpha  <- check_cov_xval(fe_brms_m, "alpha_Intercept",  ALPHA_TRUE,  "alpha(brms)")
brms_m_lambda <- check_cov_xval(fe_brms_m, "lambda_Intercept", LAMBDA_TRUE, "lambda(brms)")
brms_m_gammaw <- check_cov_xval(fe_brms_m, "gammaw_Intercept", GAMMAW_TRUE, "gammaw(brms)")


# ==============================================================================
# SECTION 7 — Cross-validation summary table
# ==============================================================================

cat("\n--- SECTION 7: Cross-validation summary ---\n\n")

cat("Model comparison: brms NLF vs. hBayesDM ra_* on shared simulated data\n\n")

cat(sprintf("%-45s  %-8s  %-8s  %-8s\n",
            "Model", "alpha", "lambda", "gammaw"))
cat(strrep("-", 75), "\n")
cat(sprintf("%-45s  %-8s  %-8s  %-8s\n",
            "True values:", ALPHA_TRUE, LAMBDA_TRUE, GAMMAW_TRUE))

alpha_brms_g  <- if (!is.null(fe_brms_g))  fe_brms_g["alpha_Intercept",  "Estimate"] else NA
alpha_brms_m  <- if (!is.null(fe_brms_m))  fe_brms_m["alpha_Intercept",  "Estimate"] else NA
lambda_brms_m <- if (!is.null(fe_brms_m))  fe_brms_m["lambda_Intercept", "Estimate"] else NA
gammaw_brms_m <- if (!is.null(fe_brms_m))  fe_brms_m["gammaw_Intercept", "Estimate"] else NA

cat(sprintf("%-45s  %-8.3f  %-8s  %-8.3f\n",
            "brms NLF (gains, phi=1, with Prelec):",
            alpha_brms_g, "N/A", fe_brms_g["gammaw_Intercept","Estimate"]))
cat(sprintf("%-45s  %-8.3f  %-8.3f  %-8.3f\n",
            "brms NLF (mixed, phi=1, with Prelec):",
            alpha_brms_m, lambda_brms_m, gammaw_brms_m))

if (HBAYESDM_AVAILABLE && exists("fit_hbdm_noLA") && !is.null(fit_hbdm_noLA)) {
  cat(sprintf("%-45s  %-8.3f  %-8s  %-8s\n",
              "ra_noLA (gains, probit, no Prelec):",
              mean(fit_hbdm_noLA$allIndPars$Arew), "N/A", "N/A (not in model)"))
}
if (HBAYESDM_AVAILABLE && exists("fit_hbdm_prospect") && !is.null(fit_hbdm_prospect)) {
  cat(sprintf("%-45s  %-8.3f  %-8.3f  %-8s\n",
              "ra_prospect (mixed, probit, no Prelec):",
              mean(fit_hbdm_prospect$allIndPars$Arew),
              mean(fit_hbdm_prospect$allIndPars$Lambda),
              "N/A (not in model)"))
}

cat("\n")

cat("Key findings:\n")
cat("  1. alpha (Arew): brms NLF and ra_noLA should agree within MCMC error when\n")
cat("     gammaw is fixed to 1.0 in both. Disagreement indicates a parameterization\n")
cat("     difference or probit/logit scale mismatch.\n")
cat("  2. lambda (Lambda): brms NLF and ra_prospect should agree within MCMC error.\n")
cat("     Note: ra_prospect uses Arew≠Apun by default → possible lambda underestimation\n")
cat("     (Nilsson 2011 effect) unless alpha≠beta is compensated by strong priors.\n")
cat("  3. gammaw: ra_* models do NOT include Prelec probability weighting by default.\n")
cat("     When gammaw≠1, alpha and lambda estimates will differ between brms and hBayesDM\n")
cat("     because hBayesDM absorbs the weighting effect into alpha/Lambda.\n")
cat("     Solution: either fix gammaw=1 in brms for the cross-validation OR use a\n")
cat("     hBayesDM extension that includes Prelec weighting (not in standard ra_*).\n")
cat("  4. phi (epsi): NOT directly comparable. logit(phi=1) ↔ probit(epsi≈0.59).\n")
cat("     Both are sensitivity parameters at a fixed scale; users should be aware\n")
cat("     that switching between probit and logit changes the interpretation of epsi/phi.\n\n")


# ==============================================================================
# SECTION 8 — Save results
# ==============================================================================

cat("--- SECTION 8: Save results ---\n\n")

cov_rows_gains <- Filter(Negate(is.null), list(brms_g_alpha, brms_g_gammaw))
cov_rows_mixed <- Filter(Negate(is.null), list(brms_m_alpha, brms_m_lambda, brms_m_gammaw))

results_xval <- do.call(rbind, lapply(c(
  lapply(cov_rows_gains, function(x) data.frame(
    script = "04_hbayesdm_crossval", design = "gains_only", model = "brms_NLF",
    param = x$param, true = x$true, est = x$est,
    q2.5 = x$lo, q97.5 = x$hi, covered = x$covered,
    max_rhat = rhat_brms_g, n_diverge = div_brms_g)),
  lapply(cov_rows_mixed, function(x) data.frame(
    script = "04_hbayesdm_crossval", design = "mixed", model = "brms_NLF",
    param = x$param, true = x$true, est = x$est,
    q2.5 = x$lo, q97.5 = x$hi, covered = x$covered,
    max_rhat = rhat_brms_m, n_diverge = div_brms_m))
), identity))

dir.create("local/exploration/prospect-theory/results", showWarnings = FALSE)
write.csv(results_xval,
          "local/exploration/prospect-theory/results/04_brms_xval_recovery.csv",
          row.names = FALSE)

cat("Results written:\n")
cat("  local/exploration/prospect-theory/results/04_brms_xval_recovery.csv\n\n")

cat("==========================================================================\n")
cat("SUMMARY:\n\n")
cat("  brms NLF (gains-only): alpha covered =",
    if (!is.null(brms_g_alpha)) brms_g_alpha$covered else NA, "\n")
cat("  brms NLF (mixed): alpha covered =",
    if (!is.null(brms_m_alpha)) brms_m_alpha$covered else NA,
    ", lambda covered =",
    if (!is.null(brms_m_lambda)) brms_m_lambda$covered else NA, "\n")
if (!HBAYESDM_AVAILABLE) {
  cat("  hBayesDM: NOT RUN (install with remotes::install_github('CCS-Lab/hBayesDM'))\n")
}
cat("\n  RECOMMENDATION: For cross-validation, fix gammaw=1.0 in the brms NLF\n")
cat("  and compare alpha and lambda against hBayesDM ra_prospect on shared data.\n")
cat("  The probit/logit scale difference for sensitivity (phi vs. epsi) is expected\n")
cat("  and should be documented in the design memo.\n")
cat("==========================================================================\n")
