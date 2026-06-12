# 03_identifiability_guards.R
# Issue #22 — Prospect-theory exploration
#
# Identifiability guards for CPT constructor (G-style, ported from
# 16_identifiability_guards.R in the m3-utility exploration).
#
# Guards:
#   G1: Probability-weighting identifiability (gammaw) — needs sufficient
#       probability range in the design (both low and high probabilities)
#   G2: Value-function identifiability (alpha) — needs sufficient outcome
#       magnitude range; min_x / max_x < 0.5 required
#   G3: phi / lambda confound — cannot estimate both simultaneously with
#       only gain-vs-gain and loss-vs-loss trials; require mixed-option
#       trials OR fix phi=1. (Nilsson et al. 2011 main finding)
#   G4: Loss-domain identifiability (lambda) — gains-only data → lambda
#       unidentified; requires BOTH gain trials AND loss trials
#   G5: Outcome sign check — outcomes must include both positive and negative
#       values to identify lambda; gains-only data fails this check
#
# For each guard:
#   - Standalone function (same pattern as 16_identifiability_guards.R)
#   - Demo: deficient design → guard fires; adequate design → guard passes
#
# Run from repo root:
#   Rscript local/exploration/prospect-theory/03_identifiability_guards.R

library(dplyr)

cat("==========================================================================\n")
cat("03_identifiability_guards.R — CPT identifiability guards\n")
cat("==========================================================================\n\n")


# ==============================================================================
# SECTION 1 — Guard function implementations
# ==============================================================================

cat("--- SECTION 1: Guard function implementations ---\n\n")

# ---- G1: Probability weighting (gammaw) needs sufficient probability range ----
# Prelec w(p) = exp(-(-ln p)^gammaw) is only distinguishable from no weighting
# (gammaw=1) when the design covers a range of probabilities. If all p are
# clustered near 0.5, gammaw is poorly identified (the inverse-S shape is flat
# near p=0.5). Both low probabilities (p < 0.2) and high probabilities (p > 0.8)
# should be present.

check_prob_range <- function(data, p_A_col = "prob_A", p_B_col = "prob_B",
                              low_threshold = 0.20, high_threshold = 0.80) {
  p_vals <- c(data[[p_A_col]], data[[p_B_col]])
  p_vals <- p_vals[!is.na(p_vals)]
  has_low  <- any(p_vals < low_threshold)
  has_high <- any(p_vals > high_threshold)
  if (!has_low || !has_high) {
    msg <- sprintf(paste0(
      "G1 (probability weighting, gammaw): insufficient probability range.\n",
      "  Prelec w(p) = exp(-(-ln p)^gammaw) requires both small and large\n",
      "  probabilities to identify the inverse-S shape of gammaw.\n",
      "  Current range: [%.3f, %.3f]; needed: at least some p < %.2f AND p > %.2f.\n",
      "  Fix: ensure the design covers probabilities below %.2f and above %.2f."
    ), min(p_vals), max(p_vals), low_threshold, high_threshold,
    low_threshold, high_threshold)
    if (!has_low && !has_high) stop(msg)
    warning(msg)
  }
  invisible(TRUE)
}

# ---- G2: Value-function curvature (alpha) needs outcome magnitude range ------
# v(x) = x^alpha for gains; the curvature parameter alpha is only identifiable
# when the design spans a sufficient range of outcome magnitudes. If all |x| are
# clustered in a narrow band (e.g. all between $4 and $5), alpha is aliased with
# a simple scaling of utility.

check_outcome_range <- function(data, x_A_col = "amt_A", x_B_col = "amt_B",
                                 min_ratio_threshold = 3.0) {
  x_A <- data[[x_A_col]]; x_B <- data[[x_B_col]]
  x_gains <- abs(c(x_A[x_A > 0], x_B[x_B > 0]))
  if (length(x_gains) == 0) {
    stop("G2 (value curvature, alpha): no gains trials found — check x_A, x_B column signs.")
  }
  range_ratio <- max(x_gains) / max(min(x_gains), 1e-6)
  if (range_ratio < min_ratio_threshold) {
    stop(sprintf(paste0(
      "G2 (value curvature, alpha): insufficient outcome magnitude range.\n",
      "  v(x) = x^alpha requires range ratio max(|x|)/min(|x|) >= %.1f to\n",
      "  identify alpha separately from a scale parameter.\n",
      "  Current ratio: %.2f (max=%.2f, min=%.2f).\n",
      "  Fix: include outcomes ranging from small to large (e.g. $1 to $20;\n",
      "  ratio = 20). Outcome range {4, 5} (ratio 1.25) is insufficient."
    ), min_ratio_threshold, range_ratio, max(x_gains), min(x_gains)))
  }
  invisible(TRUE)
}

# ---- G3: phi/lambda confound — cannot estimate both jointly -----------------
# When the design has ONLY gain-vs-gain and loss-vs-loss trials (no gain-vs-loss
# or multi-outcome mixed gambles), phi and lambda are jointly unidentified:
#   - gain-vs-gain: P(A) depends on phi (given alpha, gammaw)
#   - loss-vs-loss: P(A) depends on phi*lambda (given alpha, gammaw)
# With informative priors on phi, lambda CAN be recovered but requires a very
# tight phi prior, which is rarely available in practice.
# Recommendation: either fix phi=1 OR use multi-outcome mixed gambles.

check_phi_lambda_confound <- function(data, x_A_col = "amt_A", x_B_col = "amt_B",
                                       phi_fixed = FALSE) {
  if (phi_fixed) return(invisible(TRUE))
  x_A <- data[[x_A_col]]; x_B <- data[[x_B_col]]
  has_gain_trials <- any(x_A > 0 | x_B > 0)
  has_loss_trials <- any(x_A < 0 | x_B < 0)
  has_gainA_lossB <- any(x_A > 0 & x_B < 0)
  has_lossA_gainB <- any(x_A < 0 & x_B > 0)
  has_mixed_option <- has_gainA_lossB | has_lossA_gainB

  if (!phi_fixed && has_gain_trials && has_loss_trials && !has_mixed_option) {
    warning(sprintf(paste0(
      "G3 (phi/lambda confound): design has gain trials and loss trials, but no\n",
      "  mixed-option trials (gain vs. loss in the same trial).\n",
      "  In loss-vs-loss trials: U_A - U_B = -phi*lambda*(w_A*|x_A|^alpha - w_B*|x_B|^alpha)\n",
      "  Only phi*lambda is identified (not phi and lambda separately).\n",
      "  With gain-vs-gain trials phi IS identified, but jointly fitting phi AND\n",
      "  lambda leads to strongly correlated posteriors (Nilsson et al. 2011,\n",
      "  J. Math. Psych. 55:84-93, Table 1: lambda underestimated when alpha!=beta\n",
      "  and phi is free).\n\n",
      "  Options to resolve:\n",
      "    1. Fix phi = 1 (standard in CPT literature): phi is absorbed into utility\n",
      "       scale. Estimable with formula: choice ~ wA * vA - wB * vB\n",
      "    2. Use informative prior on phi from pilot data.\n",
      "    3. Add mixed-option trials (gain x vs. loss y) to pin relative scale.\n",
      "  Recommended: phi = 1 (simplest; see 02_brms_prototype.R Model B).\n",
      "  If phi varies systematically across conditions, treat as a fixed predictor."
    )))
  }
  invisible(TRUE)
}

# ---- G4: Loss-domain identifiability — lambda requires both gain and loss ----
# Lambda (loss aversion) is NOT identified in gains-only data because the loss
# branch of v(x) is never evaluated. If the data has no negative outcomes,
# lambda is structurally unidentified: any value of lambda gives identical fits.

check_lambda_identifiability <- function(data, x_A_col = "amt_A", x_B_col = "amt_B") {
  x_all <- c(data[[x_A_col]], data[[x_B_col]])
  has_losses <- any(x_all < 0, na.rm = TRUE)
  if (!has_losses) {
    stop(sprintf(paste0(
      "G4 (loss aversion, lambda): no loss outcomes detected.\n",
      "  All outcomes in columns '%s' and '%s' are non-negative.\n",
      "  Lambda (loss aversion) is defined only for the loss branch:\n",
      "    v(x) = -lambda * (-x)^alpha  for x < 0\n",
      "  In a gains-only design (x >= 0 always), v(x) = x^alpha regardless of\n",
      "  lambda. Lambda is therefore structurally unidentified.\n\n",
      "  Fix: include trials with negative outcomes (losses) in the design.\n",
      "  Minimum requirement: at least 30%% of trials should be loss trials\n",
      "  (both options have negative outcomes) to estimate lambda reliably."
    ), x_A_col, x_B_col))
  }
  invisible(TRUE)
}

# ---- G5: alpha=beta design requirement (Nilsson et al. 2011) ----------------
# When alpha and beta are estimated freely (separate gain and loss curvature),
# they are strongly correlated with lambda (Nilsson et al. 2011 main finding):
# lambda is systematically underestimated. The recommended fix is to constrain
# alpha = beta. This guard checks that the user is aware of this issue when
# they specify a model with free alpha and beta.

check_alpha_beta_constraint <- function(alpha_free = TRUE, beta_free = TRUE) {
  if (alpha_free && beta_free) {
    warning(sprintf(paste0(
      "G5 (alpha/beta/lambda entanglement, Nilsson et al. 2011):\n",
      "  Both alpha (gain curvature) and beta (loss curvature) are free.\n",
      "  When alpha != beta, lambda is systematically underestimated because:\n",
      "    v_gains(x) = x^alpha → scale depends on alpha\n",
      "    v_losses(x) = -lambda * (-x)^beta → scale depends on both beta and lambda\n",
      "    Lambda compensates for alpha/beta difference by shrinking toward 1.\n\n",
      "  Nilsson et al. (2011, J. Math. Psych. 55:84-93) show that constraining\n",
      "  alpha = beta substantially reduces this bias (see 01_cpt_reference_impl.R,\n",
      "  Section 7: unconstrained lambda bias=-1.39; constrained bias=-0.29).\n\n",
      "  Recommendation: constrain alpha = beta (standard CPT parameterization).\n",
      "  If gain/loss asymmetry in curvature is theoretically important, use very\n",
      "  large N (> 200 trials per domain per subject) and strong priors."
    )))
  }
  invisible(TRUE)
}

# ---- Combined pre-flight validator -------------------------------------------
check_cpt_design <- function(data,
                              x_A_col = "amt_A", x_B_col = "amt_B",
                              p_A_col = "prob_A", p_B_col = "prob_B",
                              estimate_lambda = TRUE,
                              alpha_free = TRUE,
                              beta_free = FALSE,  # default: alpha=beta constraint
                              phi_fixed = TRUE) {
  errors <- character(0)
  warnings_list <- character(0)

  # G1: probability range
  tryCatch(
    withCallingHandlers(
      check_prob_range(data, p_A_col, p_B_col),
      warning = function(w) {
        warnings_list <<- c(warnings_list, paste("G1:", conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) errors <<- c(errors, paste("G1:", conditionMessage(e)))
  )

  # G2: outcome range
  tryCatch(
    check_outcome_range(data, x_A_col, x_B_col),
    error = function(e) errors <<- c(errors, paste("G2:", conditionMessage(e)))
  )

  # G3: phi/lambda confound
  tryCatch(
    withCallingHandlers(
      check_phi_lambda_confound(data, x_A_col, x_B_col, phi_fixed),
      warning = function(w) {
        warnings_list <<- c(warnings_list, paste("G3:", conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) errors <<- c(errors, paste("G3:", conditionMessage(e)))
  )

  # G4: lambda identifiability
  if (estimate_lambda) {
    tryCatch(
      check_lambda_identifiability(data, x_A_col, x_B_col),
      error = function(e) errors <<- c(errors, paste("G4:", conditionMessage(e)))
    )
  }

  # G5: alpha=beta constraint
  tryCatch(
    withCallingHandlers(
      check_alpha_beta_constraint(alpha_free, beta_free),
      warning = function(w) {
        warnings_list <<- c(warnings_list, paste("G5:", conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) errors <<- c(errors, paste("G5:", conditionMessage(e)))
  )

  if (length(errors) > 0) {
    stop(paste0("CPT design check failed:\n",
                paste0("  ", errors, collapse = "\n"), "\n",
                "Fix the design or model specification before fitting."))
  }

  if (length(warnings_list) > 0) {
    for (w in warnings_list) warning(w, call. = FALSE)
  }

  invisible(TRUE)
}


# ==============================================================================
# SECTION 2 — Demonstrations: guards firing and passing
# ==============================================================================

cat("--- SECTION 2: Guard demonstrations ---\n\n")


# ---- G1: Probability range ---------------------------------------------------

cat("G1: Probability weighting identifiability\n\n")

# Deficient: all p near 0.5
d_bad_p <- data.frame(
  subj  = rep(1:10, each = 50),
  amt_A = runif(500, 1, 10), prob_A = runif(500, 0.40, 0.60),
  amt_B = runif(500, 1, 10), prob_B = runif(500, 0.40, 0.60),
  choice = rbinom(500, 1, 0.5)
)
cat("Deficient (all p in [0.40, 0.60]):\n")
r_g1_fail <- tryCatch(
  withCallingHandlers(
    check_prob_range(d_bad_p),
    warning = function(w) {
      cat("  GUARD FIRES (warning, correct):", conditionMessage(w), "\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) {
    cat("  GUARD FIRES (error, correct):", conditionMessage(e), "\n")
    e
  }
)

# Adequate: p covers low and high range
d_good_p <- data.frame(
  subj  = rep(1:10, each = 60),
  amt_A = runif(600, 1, 10), prob_A = runif(600, 0.05, 0.95),
  amt_B = runif(600, 1, 10), prob_B = runif(600, 0.05, 0.95),
  choice = rbinom(600, 1, 0.5)
)
cat("\nAdequate (p in [0.05, 0.95]):\n")
r_g1_pass <- tryCatch(
  withCallingHandlers(
    check_prob_range(d_good_p),
    warning = function(w) {
      cat("  GUARD FIRES (unexpected warning)\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) cat("  GUARD FIRES (unexpected error):", conditionMessage(e), "\n")
)
if (!inherits(r_g1_pass, "error")) cat("  GUARD PASSES (correct) ✓\n")
cat("\n")


# ---- G2: Outcome range -------------------------------------------------------

cat("G2: Value-function curvature identifiability\n\n")

# Deficient: narrow outcome range
d_narrow <- data.frame(
  amt_A  = runif(500, 4.5, 5.5),
  amt_B  = runif(500, 4.5, 5.5),
  prob_A = runif(500, 0.1, 0.9),
  prob_B = runif(500, 0.1, 0.9),
  choice = rbinom(500, 1, 0.5)
)
cat("Deficient (x in [4.5, 5.5], range ratio = 1.22):\n")
r_g2_fail <- tryCatch(
  check_outcome_range(d_narrow),
  error = function(e) { cat("  GUARD FIRES (correct):", conditionMessage(e), "\n"); e }
)

# Adequate: wide outcome range
d_wide <- data.frame(
  amt_A  = runif(500, 1, 20),
  amt_B  = runif(500, 1, 20),
  prob_A = runif(500, 0.1, 0.9),
  prob_B = runif(500, 0.1, 0.9),
  choice = rbinom(500, 1, 0.5)
)
cat("\nAdequate (x in [1, 20], range ratio = 20):\n")
r_g2_pass <- tryCatch(
  check_outcome_range(d_wide),
  error = function(e) cat("  GUARD FIRES (unexpected):", conditionMessage(e), "\n")
)
if (!inherits(r_g2_pass, "error")) cat("  GUARD PASSES (correct) ✓\n")
cat("\n")


# ---- G3: phi/lambda confound -------------------------------------------------

cat("G3: phi/lambda confound\n\n")

# Deficient: gain-vs-gain and loss-vs-loss only (no cross-domain trials)
d_nocross <- data.frame(
  amt_A  = c(runif(250, 1, 10), -runif(250, 1, 10)),
  amt_B  = c(runif(250, 1, 10), -runif(250, 1, 10)),
  prob_A = runif(500, 0.1, 0.9),
  prob_B = runif(500, 0.1, 0.9),
  choice = rbinom(500, 1, 0.5)
)
cat("Deficient (gain-vs-gain + loss-vs-loss only; phi NOT fixed):\n")
r_g3_fail <- tryCatch(
  withCallingHandlers(
    check_phi_lambda_confound(d_nocross, phi_fixed = FALSE),
    warning = function(w) {
      cat("  GUARD FIRES (warning, correct):", conditionMessage(w), "\n\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) { cat("  GUARD FIRES (error):", conditionMessage(e), "\n\n"); e }
)

# Adequate: phi fixed to 1
cat("Adequate (phi=1 fixed — standard CPT parameterization):\n")
r_g3_pass <- tryCatch(
  withCallingHandlers(
    check_phi_lambda_confound(d_nocross, phi_fixed = TRUE),
    warning = function(w) {
      cat("  GUARD FIRES (unexpected):", conditionMessage(w), "\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) cat("  GUARD FIRES (unexpected error):", conditionMessage(e), "\n")
)
if (!inherits(r_g3_pass, "error")) cat("  GUARD PASSES (correct) ✓\n")
cat("\n")


# ---- G4: Lambda identifiability (gains-only fails) ---------------------------

cat("G4: Lambda identifiability (gains-only design)\n\n")

d_gains_only <- data.frame(
  amt_A  = runif(500, 1, 10),
  amt_B  = runif(500, 1, 10),
  prob_A = runif(500, 0.1, 0.9),
  prob_B = runif(500, 0.1, 0.9),
  choice = rbinom(500, 1, 0.5)
)
cat("Deficient (gains-only, no negative outcomes):\n")
r_g4_fail <- tryCatch(
  check_lambda_identifiability(d_gains_only),
  error = function(e) { cat("  GUARD FIRES (correct):", conditionMessage(e), "\n\n"); e }
)

d_mixed <- data.frame(
  amt_A  = c(runif(250, 1, 10), -runif(250, 1, 10)),
  amt_B  = c(runif(250, 1, 10), -runif(250, 1, 10)),
  prob_A = runif(500, 0.1, 0.9),
  prob_B = runif(500, 0.1, 0.9),
  choice = rbinom(500, 1, 0.5)
)
cat("Adequate (gains + losses design):\n")
r_g4_pass <- tryCatch(
  check_lambda_identifiability(d_mixed),
  error = function(e) cat("  GUARD FIRES (unexpected):", conditionMessage(e), "\n")
)
if (!inherits(r_g4_pass, "error")) cat("  GUARD PASSES (correct) ✓\n\n")


# ---- G5: alpha=beta entanglement warning ------------------------------------

cat("G5: alpha=beta/lambda entanglement (Nilsson et al. 2011)\n\n")

cat("Deficient (both alpha and beta free):\n")
r_g5_fail <- tryCatch(
  withCallingHandlers(
    check_alpha_beta_constraint(alpha_free = TRUE, beta_free = TRUE),
    warning = function(w) {
      cat("  GUARD FIRES (warning, correct):", conditionMessage(w), "\n\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) { cat("  GUARD FIRES (error):", conditionMessage(e), "\n"); e }
)

cat("Adequate (alpha=beta constraint applied):\n")
r_g5_pass <- tryCatch(
  withCallingHandlers(
    check_alpha_beta_constraint(alpha_free = TRUE, beta_free = FALSE),
    warning = function(w) {
      cat("  GUARD FIRES (unexpected):", conditionMessage(w), "\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) cat("  GUARD FIRES (unexpected error)\n")
)
if (!inherits(r_g5_pass, "error")) cat("  GUARD PASSES (correct) ✓\n\n")


# ==============================================================================
# SECTION 3 — Combined pre-flight demos
# ==============================================================================

cat("--- SECTION 3: Combined check_cpt_design() demos ---\n\n")

cat("Demo 1: Gains-only with free phi + lambda estimation (should fail G4)\n")
r_demo1 <- tryCatch(
  check_cpt_design(d_gains_only, estimate_lambda = TRUE, phi_fixed = FALSE),
  error = function(e) { cat("  FAIL (expected):", conditionMessage(e), "\n"); e }
)

cat("\nDemo 2: Narrow probability range (p in [0.40, 0.60]) (should warn G1)\n")
r_demo2 <- tryCatch(
  withCallingHandlers(
    check_cpt_design(d_bad_p, estimate_lambda = FALSE, phi_fixed = TRUE),
    warning = function(w) {
      cat("  WARN G1 (expected):", conditionMessage(w), "\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) { cat("  FAIL:", conditionMessage(e), "\n"); e }
)

cat("\nDemo 3: Well-designed CPT study (should pass all)\n")
d_good <- data.frame(
  amt_A  = c(runif(250, 1, 15), -runif(250, 1, 15)),
  amt_B  = c(runif(250, 1, 15), -runif(250, 1, 15)),
  prob_A = runif(500, 0.05, 0.95),
  prob_B = runif(500, 0.05, 0.95),
  choice = rbinom(500, 1, 0.5)
)
r_demo3 <- tryCatch(
  withCallingHandlers(
    check_cpt_design(d_good, estimate_lambda = TRUE, phi_fixed = TRUE,
                     alpha_free = TRUE, beta_free = FALSE),
    warning = function(w) {
      cat("  WARN (unexpected):", conditionMessage(w), "\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) { cat("  FAIL (unexpected):", conditionMessage(e), "\n"); e }
)
if (!inherits(r_demo3, "error")) cat("  ALL GUARDS PASS ✓\n\n")


# ==============================================================================
# SECTION 4 — Guard placement in production architecture
# ==============================================================================

cat("--- SECTION 4: Guard placement in future binary_pt() constructor ---\n\n")

cat(
"In a production binary_pt() constructor, guards would be placed as S3 methods:\n\n",
"  check_data.binary_pt(model, data, formula):\n",
"    - G1: check_prob_range() — error if no low/high probabilities\n",
"    - G2: check_outcome_range() — error if range ratio < 3\n",
"    - G4: check_lambda_identifiability() — error if gains-only and lambda free\n\n",
"  check_model.binary_pt(model, data, formula):\n",
"    - G3: check_phi_lambda_confound() — warning if phi free + mixed design\n",
"    - G5: check_alpha_beta_constraint() — warning if alpha AND beta free\n\n",
"  These guards fire automatically via S3 dispatch in bmm().\n",
"  Guards G1, G2, G4 require data → live in check_data.\n",
"  Guards G3, G5 require only model spec → live in check_model.\n\n",
sep = ""
)

cat(
"Comparison with m3-utility guards (16_identifiability_guards.R):\n",
"  m3-utility guard pattern: G1 (power utility range) = CPT G2 (outcome range)\n",
"  m3-utility guard G2 (Prelec, set size) = CPT G1 (probability range)\n",
"  m3-utility guard G3 (EU slope gamma) = CPT G4 (lambda identifiability)\n",
"  NEW in CPT: G3 (phi/lambda confound), G5 (alpha=beta entanglement)\n\n",
sep = ""
)

cat("==========================================================================\n")
cat("SECTION 4 summary: guard functions complete.\n")
cat("  G1: probability range → DEMONSTRATED\n")
cat("  G2: outcome range → DEMONSTRATED\n")
cat("  G3: phi/lambda confound → DEMONSTRATED\n")
cat("  G4: gains-only/lambda → DEMONSTRATED\n")
cat("  G5: alpha=beta/lambda entanglement → DEMONSTRATED\n")
cat("  check_cpt_design(): combined pre-flight validator → DEMONSTRATED\n")
cat("==========================================================================\n")
