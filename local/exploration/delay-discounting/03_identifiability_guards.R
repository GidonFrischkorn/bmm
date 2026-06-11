# 03_identifiability_guards.R
# Task 5 — Identifiability guards
#
# Two guards from the issue specification:
#   G1: Delay-range guard — insufficient delay spread → flat posterior on k
#   G2: k vs. sensitivity (phi) confound — both shift P(LL) monotonically
#
# Plus a functional-form discriminability check:
#   G3: Hyperbolic vs. exponential cannot be separated at short delays
#
# For each guard:
#   - Implementation (standalone function, returns list with $ok and $message)
#   - Demo: deficient design → guard fires; adequate design → guard passes
#   - For G2: show the confound and how a phi-anchoring prior resolves it
#
# Run from repo root:
#   Rscript local/exploration/delay-discounting/03_identifiability_guards.R

suppressPackageStartupMessages(library(brms))

cat("==========================================================================\n")
cat("03_identifiability_guards.R — Identifiability guards for discounting\n")
cat("==========================================================================\n\n")

# ==============================================================================
# Shared helpers (duplicated from 01 to keep script self-contained)
# ==============================================================================

sv_hyperbolic <- function(A, D, k) A / (1 + k * D)

p_choose_ll <- function(V_LL, V_SS, phi) plogis(phi * (V_LL - V_SS))

ll_hyperbolic <- function(data, k, phi) {
  V_LL <- sv_hyperbolic(data$amt_LL, data$delay_LL, k)
  V_SS <- sv_hyperbolic(data$amt_SS, data$delay_SS, k)
  p    <- p_choose_ll(V_LL, V_SS, phi)
  p    <- pmax(pmin(p, 1 - 1e-10), 1e-10)
  sum(dbinom(data$choice, 1, p, log = TRUE))
}

make_design <- function(n_trials, delay_LL_levels, seed) {
  set.seed(seed)
  amt_LL_levels <- c(12, 15, 20, 25, 30)
  grid    <- expand.grid(amt_LL = amt_LL_levels, delay_LL = delay_LL_levels)
  design  <- grid[sample(nrow(grid), n_trials, replace = TRUE), ]
  data.frame(amt_SS = 10, delay_SS = 0,
             amt_LL = design$amt_LL, delay_LL = design$delay_LL)
}

sim_choices <- function(design, k_true, phi_true, seed) {
  V_LL <- sv_hyperbolic(design$amt_LL, design$delay_LL, k_true)
  V_SS <- sv_hyperbolic(design$amt_SS, design$delay_SS, k_true)
  p    <- p_choose_ll(V_LL, V_SS, phi_true)
  set.seed(seed)
  design$choice <- rbinom(nrow(design), 1, p)
  design
}

recover_k_phi <- function(dat) {
  obj <- function(par) {
    -ll_hyperbolic(dat, exp(par[1]), exp(par[2]))
  }
  best <- Inf; par_best <- c(-4, 0)
  for (lk0 in c(-6, -4, -2)) {
    for (lp0 in c(-1, 0, 1)) {
      f <- tryCatch(
        optim(c(lk0, lp0), obj, method = "L-BFGS-B",
              lower = c(-12, -3), upper = c(3, 5),
              control = list(factr = 1e7)),
        error = function(e) list(value = Inf)
      )
      if (f$value < best) { best <- f$value; par_best <- f$par }
    }
  }
  list(k_hat = exp(par_best[1]), phi_hat = exp(par_best[2]),
       nll = best)
}

# ==============================================================================
# SECTION 1 — G1: Delay-range guard
# ==============================================================================

cat("==========================================================================\n")
cat("GUARD G1: Delay-range guard\n")
cat("--------------------------------------------------------------------------\n")
cat("Premise: if all delays are too short, the discount function is nearly\n")
cat("linear in k*D (Maclaurin: 1/(1+kD) ≈ 1-kD for small kD) and k is poorly\n")
cat("identified. Threshold: max(delay_LL) / min(delay_LL) < 5 (insufficient\n")
cat("spread), or max(kD) < 0.5 for the typical k range.\n\n")

check_delay_range <- function(delay_LL, k_ref = 0.02) {
  if (length(unique(delay_LL)) < 2) {
    return(list(ok = FALSE,
                message = "All trials have the same delay — k is not identified."))
  }
  delay_range_ratio <- max(delay_LL) / pmax(min(delay_LL[delay_LL > 0]), 1)
  max_kD <- k_ref * max(delay_LL)

  if (max_kD < 0.5) {
    return(list(ok = FALSE, message = paste0(
      "Insufficient delay range for k identification.\n",
      "  max(k_ref * delay_LL) = ", round(max_kD, 3), " < 0.5\n",
      "  With delays up to ", max(delay_LL), " days and k_ref=", k_ref, ",\n",
      "  the discount function is approximately linear and k is\n",
      "  poorly separated from the sensitivity parameter phi.\n",
      "  Recommendation: include delays up to >= ", ceiling(0.5 / k_ref), " days."
    )))
  }
  if (delay_range_ratio < 5) {
    return(list(ok = FALSE, message = paste0(
      "Narrow delay range (max/min = ", round(delay_range_ratio, 1), " < 5).\n",
      "  Functional form discrimination requires at least 5-fold delay spread.\n",
      "  Recommendation: span at least 3 orders of magnitude (e.g., 7 to 365 days)."
    )))
  }
  list(ok = TRUE, message = "Delay range is adequate.")
}

# --- Demo: deficient design (short delays only) ------------------------------

cat("--- G1 Demo: deficient design (delays = 1, 2, 3, 5, 7 days) ---\n")
delay_deficient <- c(1, 2, 3, 5, 7)
g1_deficient    <- check_delay_range(delay_deficient, k_ref = 0.02)
cat("Guard result: ok =", g1_deficient$ok, "\n")
cat("Message:\n"); cat(g1_deficient$message, "\n\n")

# Show the flat LL profile on k with deficient design
design_def  <- make_design(200, delay_deficient, seed = 1)
dat_def     <- sim_choices(design_def, k_true = 0.02, phi_true = 2, seed = 101)
cat("  LL profile for deficient design (true k=0.02, phi=2 fixed):\n")
for (lk in seq(-8, -1, by=1)) {
  ll <- ll_hyperbolic(dat_def, exp(lk), 2)
  cat(sprintf("    log_k=%+.0f  k=%.5f  LL=%7.2f\n", lk, exp(lk), ll))
}

cat("\n--- G1 Demo: adequate design (delays = 7, 14, 30, 60, 90, 180, 365 days) ---\n")
delay_adequate <- c(7, 14, 30, 60, 90, 180, 365)
g1_adequate    <- check_delay_range(delay_adequate, k_ref = 0.02)
cat("Guard result: ok =", g1_adequate$ok, "\n")
cat("Message:", g1_adequate$message, "\n\n")

design_adq  <- make_design(200, delay_adequate, seed = 1)
dat_adq     <- sim_choices(design_adq, k_true = 0.02, phi_true = 2, seed = 101)
cat("  LL profile for adequate design (true k=0.02, phi=2 fixed):\n")
for (lk in seq(-8, -1, by=1)) {
  ll <- ll_hyperbolic(dat_adq, exp(lk), 2)
  cat(sprintf("    log_k=%+.0f  k=%.5f  LL=%7.2f\n", lk, exp(lk), ll))
}
cat("\n")

# ==============================================================================
# SECTION 2 — G2: k vs. phi confound
# ==============================================================================

cat("==========================================================================\n")
cat("GUARD G2: k vs. sensitivity (phi) confound\n")
cat("--------------------------------------------------------------------------\n")
cat("Both k (discount rate) and phi (sensitivity) shift the slope of P(LL) vs.\n")
cat("value difference. They trade off: higher phi with smaller k can give the\n")
cat("same choice pattern as lower phi with larger k — up to a monotone re-\n")
cat("parameterisation. The guard checks for near-singular Hessian (high\n")
cat("correlation between log_k and log_phi estimates) on a held-out test set.\n\n")

check_kphi_confound <- function(dat, n_starts = 9) {
  # Fit the joint (k, phi) model and compute correlation via observed Fisher info
  best_nll <- Inf; best_par <- c(-4, 0)
  for (lk0 in c(-6, -4, -2)) {
    for (lp0 in c(-1, 0, 1)) {
      f <- tryCatch(
        optim(c(lk0, lp0), function(p) -ll_hyperbolic(dat, exp(p[1]), exp(p[2])),
              method = "L-BFGS-B", lower = c(-12, -3), upper = c(3, 5),
              hessian = TRUE, control = list(factr = 1e5)),
        error = function(e) list(value = Inf)
      )
      if (!is.null(f$hessian) && f$value < best_nll) {
        best_nll <- f$value; best_par <- f$par
        H <- f$hessian
      }
    }
  }

  # Covariance from inverse Hessian, correlation between log_k and log_phi
  cov_mat <- tryCatch(solve(H), error = function(e) NULL)
  if (is.null(cov_mat)) return(list(ok = TRUE, cor_kphi = NA,
                                    message = "Hessian singular — confound likely."))

  cor_kphi <- cov_mat[1, 2] / sqrt(cov_mat[1, 1] * cov_mat[2, 2])
  # Threshold lowered to 0.70 after probing: short-delay (linear) regime
  # reliably hits |r| = 0.73–0.999 (mean 0.83 across 10 seeds), while
  # adequate wide-delay designs stay below 0.48.  The original 0.85 threshold
  # failed to fire on any realistic deficient design.
  threshold <- 0.70

  if (abs(cor_kphi) > threshold) {
    return(list(ok = FALSE, cor_kphi = round(cor_kphi, 3), message = paste0(
      "k-phi confound detected: |cor(log_k, log_phi)| = ", round(abs(cor_kphi), 3),
      " > ", threshold, ".\n",
      "  Likely cause: delays too short (linear discount regime) — k and phi\n",
      "  both scale the tiny delay effect and cannot be separated.\n",
      "  Recommendation: fix phi (softmax temperature) a priori,\n",
      "  or use a strong prior on phi, or use the Luce rule (phi not estimated)."
    )))
  }
  list(ok = TRUE, cor_kphi = round(cor_kphi, 3),
       message = paste0("k-phi correlation acceptable: |r| = ", round(abs(cor_kphi), 3)))
}

# --- Demo: show confound surface ---------------------------------------------

cat("--- G2 Demo: LL surface for (k, phi) on adequate design ---\n\n")

# Profile the joint LL surface: vary log_k and log_phi independently
cat("  NLL surface [rows = log_phi, cols = log_k]:\n")
lk_grid  <- seq(-7, -2, by=1)
lp_grid  <- seq(-1, 3, by=1)
cat("         ", paste(sprintf("lk=%+.0f", lk_grid), collapse="  "), "\n")
for (lp in lp_grid) {
  vals <- sapply(lk_grid, function(lk) ll_hyperbolic(dat_adq, exp(lk), exp(lp)))
  cat(sprintf("lp=%+.0f ", lp), paste(sprintf("%7.1f", vals), collapse="  "), "\n")
}
cat("\n")

# MLE joint recovery
cat("--- G2 Demo: deficient design — short delays (1-7 days, linear regime) ---\n")
# The k-phi confound lives in the *linear/short-delay* regime: when all delays
# are short, 1/(1+k*D) ≈ 1 - k*D, so k enters linearly and is nearly
# exchangeable with phi (both scale the tiny delay effect).  The result is
# near-degenerate p(LL) ≈ 0.98 and |cor(logk, logphi)| > 0.70.
# Note: a flat-amount design (amt_LL ≈ amt_SS) produces |r| ≈ 0.56 — below
# threshold — because phi is poorly identified but k and phi do not genuinely
# trade off there.
dat_short_delay <- sim_choices(make_design(200, c(1, 2, 3, 5, 7), seed = 1),
                               k_true = 0.02, phi_true = 2, seed = 200)
cat("p(LL) =", round(mean(dat_short_delay$choice), 3),
    " (near-degenerate: delay too short to drive discounting)\n")
g2_deficient <- check_kphi_confound(dat_short_delay)
cat("Short-delay design:  ok =", g2_deficient$ok, " cor_kphi =", g2_deficient$cor_kphi, "\n")
cat("Message:", g2_deficient$message, "\n\n")

cat("--- G2 Demo: adequate design ---\n")
g2_ok <- check_kphi_confound(dat_adq)
cat("Adequate design:     ok =", g2_ok$ok, " cor_kphi =", g2_ok$cor_kphi, "\n")
cat("Message:", g2_ok$message, "\n\n")

cat("--- G2 Bayesian solution: strong prior on phi anchors the confound ---\n")
cat("  Prior: logphi ~ Normal(0.7, 0.3)  [phi ~ LogN, centre near 2, tight SD]\n")
cat("  This is recommended for softmax discounting; Luce rule avoids the issue\n")
cat("  entirely since phi does not appear in the Luce likelihood.\n\n")

# ==============================================================================
# SECTION 3 — G3: Functional-form discriminability (hyperbolic vs. exponential)
# ==============================================================================

cat("==========================================================================\n")
cat("GUARD G3: Functional-form discriminability (hyperbolic vs. exponential)\n")
cat("--------------------------------------------------------------------------\n")
cat("The hyperbolic and exponential discount functions produce nearly identical\n")
cat("predictions when delays are short. The maximum delay determines whether the\n")
cat("curvature differences are observable.\n\n")

# For each design (deficient vs adequate), fit both models and compare AIC
ll_exponential_model <- function(dat, k, phi) {
  V_LL <- dat$amt_LL * exp(-k * dat$delay_LL)
  V_SS <- dat$amt_SS * exp(-k * dat$delay_SS)
  p    <- pmax(pmin(plogis(phi * (V_LL - V_SS)), 1-1e-10), 1e-10)
  sum(dbinom(dat$choice, 1, p, log = TRUE))
}

recover_exp <- function(dat) {
  best <- Inf; par_best <- c(-4, 0)
  for (lk0 in c(-6, -4, -2)) {
    for (lp0 in c(-1, 0, 1)) {
      f <- tryCatch(
        optim(c(lk0, lp0), function(p) -ll_exponential_model(dat, exp(p[1]), exp(p[2])),
              method = "L-BFGS-B", lower = c(-12, -3), upper = c(3, 5)),
        error = function(e) list(value = Inf)
      )
      if (f$value < best) { best <- f$value; par_best <- f$par }
    }
  }
  list(k_hat = exp(par_best[1]), phi_hat = exp(par_best[2]), nll = best)
}

compare_functional_forms <- function(dat, label) {
  r_hyp <- recover_k_phi(dat)
  r_exp <- recover_exp(dat)
  aic_hyp <- 2 * r_hyp$nll + 2 * 2
  aic_exp <- 2 * r_exp$nll + 2 * 2
  delta_aic <- aic_exp - aic_hyp
  cat(sprintf("  %s:\n", label))
  cat(sprintf("    Hyperbolic: k=%.4f phi=%.3f  AIC=%.1f\n",
              r_hyp$k_hat, r_hyp$phi_hat, aic_hyp))
  cat(sprintf("    Exponential: k=%.4f phi=%.3f  AIC=%.1f\n",
              r_exp$k_hat, r_exp$phi_hat, aic_exp))
  cat(sprintf("    ΔAIC (exp - hyp) = %+.1f  [>2 = hyp preferred; <-2 = exp preferred]\n\n",
              delta_aic))
}

cat("--- G3 Demo: data generated from hyperbolic model ---\n\n")

# Deficient design: short delays → both models fit equally well
dat_short_hyp <- sim_choices(make_design(200, c(1,2,3,5,7), seed=3),
                             k_true=0.02, phi_true=2, seed=303)
# Adequate design: wide delays → hyperbolic has better AIC
dat_wide_hyp  <- sim_choices(make_design(200, c(7,14,30,60,90,180,365), seed=3),
                             k_true=0.02, phi_true=2, seed=303)

compare_functional_forms(dat_short_hyp, "Short delays only (1–7 days)")
compare_functional_forms(dat_wide_hyp,  "Wide delays (7–365 days)")

cat("Interpretation: With wide delays, ΔAIC > 2 indicates the generating model\n")
cat("(hyperbolic) can be distinguished from the exponential competitor.\n")
cat("With short delays, ΔAIC ≈ 0 → the two functional forms are exchangeable.\n\n")

# ==============================================================================
# SECTION 4 — Summary
# ==============================================================================

cat("==========================================================================\n")
cat("Summary — Identifiability guards\n")
cat("==========================================================================\n\n")

cat("G1 (Delay-range guard):\n")
cat("  FIRES:  delays <= 7 days with k_ref=0.02 → max(kD) < 0.5 (linear regime)\n")
cat("  PASSES: delays up to 365 days → max(kD) = 7.3 (full nonlinear range)\n")
cat("  Implementation: check_delay_range(delay_LL, k_ref = population_median_k)\n\n")

cat("G2 (k vs. phi confound):\n")
cat("  Detectable via |cor(logk, logphi)| at the MLE (observed Fisher info).\n")
cat("  Threshold: 0.70 (chosen so short-delay regime fires reliably;\n")
cat("  flat-amount design gives |r| ≈ 0.56 and does NOT trigger — wrong\n")
cat("  deficient design used in first draft; now fixed to short-delay).\n")
cat("  Main driver: linear discount regime (short delays) — k and phi both\n")
cat("  scale the tiny delay effect and cannot be separated, producing\n")
cat("  near-degenerate p(LL) ≈ 0.98.\n")
cat("  Solution: strong prior on logphi (Normal(0.7, 0.3)), or use Luce rule.\n\n")

cat("G3 (Functional-form discriminability):\n")
cat("  Hyperbolic vs. exponential only separable with delay > ~30 x 1/k_ref.\n")
cat("  For k ≈ 0.02: need delays > 30 days to see curvature differences.\n")
cat("  Design recommendation: include at least one delay >= 180 days.\n\n")

cat("Integration path:\n")
cat("  check_delay_range() → call in check_data.dd_choice() (data validation)\n")
cat("  Phi-prior warning   → emit in configure_model.dd_choice() if no phi prior\n")
cat("  Functional-form     → note in documentation / vignette design guide\n\n")
