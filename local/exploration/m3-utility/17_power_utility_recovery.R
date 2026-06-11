# 17_power_utility_recovery.R — WP1 (Round 6)
#
# Measured justification for guard G1: ">=3 value levels required for joint
# (gamma, rho) identification in power utility models."
# Analytic argument is in 07_power_utility.R; this script provides empirical
# evidence via actual Stan fits.
#
# Design:
#   3-level: V_corr in {1, 3, 5}, 30 trials per level per subject, N=30x90
#   2-level: V_corr in {1, 2}, 40 trials per level per subject, N=30x80
#   True: gamma=0.5, rho=0.7, a=1.5, c=2.0, b=0
#   Fits: 2 chains x 1000 iter (500 warmup), backend=cmdstanr
#
# Results saved to:
#   results/17_power_utility_3lev.csv
#   results/17_power_utility_2lev.csv
#
# Run from repo root:
#   Rscript local/exploration/m3-utility/17_power_utility_recovery.R

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
TRUE_B   <- 0.0; TRUE_A <- 1.5; TRUE_C <- 2.0
TRUE_GAM <- 0.5; TRUE_RHO <- 0.7

log_Z_fn <- function(act, n) {
  lv <- act + log(n); lv[1L] + log(sum(exp(lv - lv[1L])))
}

# ---- Simulate power utility data ----
sim_pu <- function(V_levels, n_per_lev, seed) {
  set.seed(seed)
  Vc_s  <- rep(V_levels, each = n_per_lev)    # per-subject trial sequence
  Vo_s  <- pmax(1, 6 - Vc_s)                  # anti-correlated V_other
  N_per <- length(Vc_s)
  N_tot <- N_SUBJ * N_per
  Vc    <- rep(Vc_s, N_SUBJ)
  Vo    <- rep(Vo_s, N_SUBJ)
  subj  <- rep(seq_len(N_SUBJ), each = N_per)

  resp <- integer(N_tot)
  for (i in seq_len(N_tot)) {
    act <- c(
      TRUE_B + TRUE_A + TRUE_C + TRUE_GAM * Vc[i]^TRUE_RHO,
      TRUE_B + TRUE_A             + TRUE_GAM * Vo[i]^TRUE_RHO,
      TRUE_B
    )
    n_i <- c(1L, 4L, 5L)
    lZ  <- log_Z_fn(act, n_i)
    resp[i] <- sample.int(3L, 1L, prob = exp(act + log(n_i) - lZ))
  }
  Y <- matrix(0L, N_tot, 3L)
  for (i in seq_len(N_tot)) Y[i, resp[i]] <- 1L
  colnames(Y) <- c("corr", "other", "npl")
  data.frame(
    subj      = subj, V_corr = Vc, V_other = Vo,
    n_corr = 1L, n_other = 4L, n_npl = 5L,
    Idx_corr = 1L, Idx_other = 1L, Idx_npl = 1L,
    nTrials = 1L, Y = I(Y)
  )
}

# ---- brms formula: power utility (gamma * V^rho; non-linear in rho) ----
pu_form <- bf(
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

true_vals <- c(a = TRUE_A, c = TRUE_C, gamma = TRUE_GAM, rho = TRUE_RHO)

# ---- Run fit and save CSV ----
run_and_save <- function(d, label, outfile) {
  cat(sprintf("\nFitting %s (N=%d rows, %d subj)...\n",
              label, nrow(d), N_SUBJ))
  fit <- brm(
    pu_form, data = d,
    family  = multinomial(refcat = NA),
    prior   = pu_priors,
    chains  = 2L, iter = 1000L, warmup = 500L, cores = 2L,
    refresh = 0, backend = "cmdstanr", silent = 2
  )

  fe  <- fixef(fit)
  rh  <- max(rhat(fit), na.rm = TRUE)
  np  <- nuts_params(fit)
  div <- sum(np$Value[np$Parameter == "divergent__"])
  post <- as.data.frame(fit)

  rows <- lapply(names(true_vals), function(p) {
    rn <- paste0(p, "_Intercept")
    if (!rn %in% rownames(fe)) return(NULL)
    lo <- fe[rn, "Q2.5"]; hi <- fe[rn, "Q97.5"]; est <- fe[rn, "Estimate"]
    tv <- true_vals[p]
    data.frame(fit = label, param = p, true_value = tv,
               estimate = est, ci_lower = lo, ci_upper = hi,
               covered = (tv >= lo & tv <= hi),
               ci_width = hi - lo,
               max_rhat = rh, divergences = div,
               stringsAsFactors = FALSE)
  })
  res <- do.call(rbind, Filter(Negate(is.null), rows))
  write.csv(res, outfile, row.names = FALSE)
  cat(sprintf("  Saved -> %s\n", outfile))
  cat(sprintf("  Max R-hat: %.3f  Divergences: %d\n", rh, div))
  for (p in c("gamma", "rho")) {
    rn <- paste0(p, "_Intercept")
    if (rn %in% rownames(fe)) {
      cov <- ifelse(true_vals[p] >= fe[rn,"Q2.5"] & true_vals[p] <= fe[rn,"Q97.5"],
                    "COVERED", "MISSED")
      cat(sprintf("  %s: %.3f [%.3f, %.3f] width=%.3f  %s\n",
          p, fe[rn,"Estimate"], fe[rn,"Q2.5"], fe[rn,"Q97.5"],
          fe[rn,"Q97.5"] - fe[rn,"Q2.5"], cov))
    }
  }
  list(fit = fit, res = res)
}

# ---- 3-level design: V in {1, 3, 5} ----
cat("=== WP1: Power utility recovery — 3-level design ===\n")
cat(sprintf("True: gamma=%.2f, rho=%.2f\n\n", TRUE_GAM, TRUE_RHO))
d3   <- sim_pu(c(1, 3, 5), 30L, 2025L)
out3 <- run_and_save(d3, "power_utility_3level",
                     "local/exploration/m3-utility/results/17_power_utility_3lev.csv")

# ---- 2-level design: V in {1, 2} ----
cat("\n=== WP1: Power utility recovery — 2-level design ===\n")
d2   <- sim_pu(c(1, 2), 40L, 2026L)
out2 <- run_and_save(d2, "power_utility_2level",
                     "local/exploration/m3-utility/results/17_power_utility_2lev.csv")

# ---- CI width comparison ----
fe3 <- fixef(out3$fit); fe2 <- fixef(out2$fit)
get_width <- function(fe, p) {
  rn <- paste0(p, "_Intercept")
  fe[rn, "Q97.5"] - fe[rn, "Q2.5"]
}

rho_w3 <- get_width(fe3, "rho");   rho_w2 <- get_width(fe2, "rho")
gam_w3 <- get_width(fe3, "gamma"); gam_w2 <- get_width(fe2, "gamma")

post3 <- as.data.frame(out3$fit); post2 <- as.data.frame(out2$fit)
gc3 <- grep("^b_gamma", names(post3), value = TRUE)[1]
rc3 <- grep("^b_rho",   names(post3), value = TRUE)[1]
gc2 <- grep("^b_gamma", names(post2), value = TRUE)[1]
rc2 <- grep("^b_rho",   names(post2), value = TRUE)[1]

cat("\n=== CI width comparison (guard G1 empirical justification) ===\n")
cat(sprintf("  gamma: 3-lev CI=%.3f  2-lev CI=%.3f  ratio=%.1fx\n",
            gam_w3, gam_w2, gam_w2 / gam_w3))
cat(sprintf("  rho:   3-lev CI=%.3f  2-lev CI=%.3f  ratio=%.1fx\n",
            rho_w3, rho_w2, rho_w2 / rho_w3))
cat(sprintf("  Post. cor(gamma,rho): 3-lev=%.3f  2-lev=%.3f\n",
            cor(post3[[gc3]], post3[[rc3]]),
            cor(post2[[gc2]], post2[[rc2]])))

if (rho_w2 / rho_w3 > 1.5) {
  cat(sprintf("\n  Guard G1 justified: rho CI %.1fx wider with 2 vs 3 value levels.\n",
              rho_w2 / rho_w3))
} else {
  cat("\n  WARNING: rho CI similar across designs — guard G1 may need revision.\n")
}
cat("\nDone.\n")
