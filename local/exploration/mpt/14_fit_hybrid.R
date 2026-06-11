# 14_fit_hybrid.R
#
# Fit the Hybrid model from Al Hadhrami, Bartsch & Oberauer (2025) to
# Experiment 1 response frequencies and reproduce Table 5.
#
# Source: Al Hadhrami, A., Bartsch, L. M., & Oberauer, K. (2025). A
# multinomial model-based analysis of bindings in working memory.
# Psychological Review. OSF: https://osf.io/nu4sb/
#
# Run from the repository root:
#   Rscript local/exploration/mpt/14_fit_hybrid.R

suppressPackageStartupMessages({
  library(brms)
  library(cmdstanr)
})

source("local/exploration/mpt/12_eqn_converter.R")

set.seed(2025)
options(mc.cores = 4)

# ---------------------------------------------------------------------------
# Build model spec and data
# ---------------------------------------------------------------------------
eqn_dir <- "local/exploration/mpt/binding_models"
spec     <- build_binding_spec(file.path(eqn_dir, "hybrid.eqn"), constants)
long     <- build_long_data(file.path(eqn_dir, "exp1_response_frequencies.csv"))

cat("Hybrid model spec:\n")
print(spec)

# ---------------------------------------------------------------------------
# Predictor formulas: all 18 free params with full correlated RE (|p| id)
# ---------------------------------------------------------------------------
pred_forms <- setNames(
  lapply(spec$params, function(p) ~ 1 + (1 | p | id)),
  spec$params
)

out <- mpt_to_brms(
  spec,
  predictor_formulas = pred_forms,
  response_col       = spec$resp_cats[1],
  trials_col         = "n"
)

data_fit <- out$data_prep(long)
cat("\nData dimensions:", nrow(data_fit), "rows,", ncol(data_fit), "cols\n")

cat("\nbrms formula:\n")
print(out$brms_formula)

# ---------------------------------------------------------------------------
# Priors (same as Unitization)
# ---------------------------------------------------------------------------
l_params <- as.character(out$link_params)  # values = l-param names
priors <- do.call(c, lapply(l_params, function(lp) {
  brms::prior_string("normal(0, 1)", nlpar = lp, class = "b", coef = "Intercept")
}))
priors <- priors +
  brms::prior_string("student_t(3, 0, 1)", class = "sd") +
  brms::prior_string("lkj(1)", class = "cor")

cat("\nPriors:\n")
print(priors)

# ---------------------------------------------------------------------------
# Time guard: check if we should reduce iterations
# ---------------------------------------------------------------------------
# Use 4 chains × 2000 iter as primary; fall back to 2 × 1500 if projected slow
CHAINS  <- 4L
ITER    <- 2000L
WARMUP  <- 1000L
SMOKE   <- FALSE

cat(sprintf("\nFitting Hybrid model (%d chains × %d iter, %d warmup) ...\n",
            CHAINS, ITER, WARMUP))
if (SMOKE) cat("NOTE: smoke test settings (reduced chains/iter)\n")

t0 <- proc.time()
fit_hyb <- brm(
  formula     = out$brms_formula,
  data        = data_fit,
  family      = out$family_obj,
  prior       = priors,
  chains      = CHAINS,
  iter        = ITER,
  warmup      = WARMUP,
  adapt_delta = 0.95,
  cores       = 4,
  backend     = "cmdstanr",
  seed        = 2025
)
elapsed <- (proc.time() - t0)[["elapsed"]]
cat(sprintf("Sampling time: %.1f s\n", elapsed))

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
rhats    <- brms::rhat(fit_hyb)
max_rhat <- max(rhats, na.rm = TRUE)
div      <- sum(nuts_params(fit_hyb, pars = "divergent__")$Value)
cat(sprintf("\nMax Rhat: %.3f | Divergent transitions: %d\n", max_rhat, div))

# WAIC
waic_hyb <- brms::waic(fit_hyb)
cat("\nWAIC (Hybrid):\n")
print(waic_hyb)

# WAIC from Unitization for comparison (load if available)
results_dir <- "local/exploration/mpt/results"
waic_unit_path <- file.path(results_dir, "unitization_waic.rds")
if (file.exists(waic_unit_path)) {
  waic_unit <- readRDS(waic_unit_path)
  cat("\nWAIC comparison:\n")
  print(loo::loo_compare(waic_unit, waic_hyb))
  delta_waic <- waic_unit$estimates["elpd_waic", "Estimate"] -
                waic_hyb$estimates["elpd_waic", "Estimate"]
  cat(sprintf("\nDelta WAIC (Unitization - Hybrid) in WAIC units: %.2f\n",
              -2 * delta_waic))
  cat(sprintf("(paper reports 2546.73; >1000 is passing criterion)\n"))
}

# ---------------------------------------------------------------------------
# Posterior summary and Table 5 reproduction
# ---------------------------------------------------------------------------
smry <- as.data.frame(fixef(fit_hyb))
cat("\nGroup-level intercepts (probit scale):\n")
print(round(smry, 3))

# Table 5 reference values from the paper: P = Phi(mu), 95% BCI
# Tree column mapping: CC = Color Cue, WC = Word Cue, LC = Location Cue
paper_table5 <- list(
  # Parameter name (after underscore stripping) : [cue, paper_mean, lo, hi]
  CCPU       = list(cue = "CC", mean = .55, lo = .47, hi = .63),
  WCPU       = list(cue = "WC", mean = .28, lo = .21, hi = .34),
  LCPU       = list(cue = "LC", mean = .12, lo = .07, hi = .18),
  CCPUTgivenU = list(cue = "CC", mean = .61, lo = .55, hi = .66),
  WCPUTgivenU = list(cue = "WC", mean = .85, lo = .78, hi = .92),
  LCPUTgivenU = list(cue = "LC", mean = .96, lo = .89, hi = .99),
  LCPrWord   = list(cue = "LC", mean = .59, lo = .49, hi = .67),
  CCPrWord   = list(cue = "CC", mean = .06, lo = .00, hi = .13),
  WCPrLocation = list(cue = "WC", mean = .59, lo = .50, hi = .67),
  CCPrLocation = list(cue = "CC", mean = .39, lo = .31, hi = .48),
  WCPrColor   = list(cue = "WC", mean = .04, lo = .00, hi = .10),
  LCPrColor   = list(cue = "LC", mean = .12, lo = .05, hi = .20),
  LCPItemWord  = list(cue = "LC", mean = .30, lo = .12, hi = .48),
  CCPItemWord  = list(cue = "CC", mean = .29, lo = .08, hi = .51),
  WCPItemLocation = list(cue = "WC", mean = .66, lo = .39, hi = .87),
  CCPItemLocation = list(cue = "CC", mean = .52, lo = .30, hi = .69),
  LCPItemColor = list(cue = "LC", mean = .28, lo = .18, hi = .48),
  WCPItemColor = list(cue = "WC", mean = .31, lo = .20, hi = .41)
)

# Convert intercepts to probability scale
int_rows <- rownames(smry)[grep("Intercept", rownames(smry))]
prob_df <- smry[int_rows, , drop = FALSE]
prob_df[, c("Estimate", "Q2.5", "Q97.5")] <-
  pnorm(prob_df[, c("Estimate", "Q2.5", "Q97.5")])
# Extract param name from row name: "lWCPU_Intercept" -> "WCPU"
prob_df$param <- sub("^l(.+)_Intercept$", "\\1", int_rows)

cat("\n--- Table 5 Reproduction ---\n")
cat(sprintf("%-22s %5s %14s %14s %6s\n",
            "Parameter", "Cue", "brms [95% CI]", "Paper [95% BCI]", "Pass?"))
cat(strrep("-", 75), "\n")

passes <- 0L
total  <- 0L
for (nm in names(paper_table5)) {
  ref  <- paper_table5[[nm]]
  ridx <- which(prob_df$param == nm)
  if (length(ridx) == 0L) {
    cat(sprintf("%-22s %5s  NOT FOUND in posterior\n", nm, ref$cue))
    next
  }
  est <- prob_df$Estimate[ridx]
  lo  <- prob_df$Q2.5[ridx]
  hi  <- prob_df$Q97.5[ridx]
  inside <- (est >= ref$lo) & (est <= ref$hi)
  total  <- total + 1L
  if (inside) passes <- passes + 1L
  cat(sprintf("%-22s %5s  %.2f [%.2f, %.2f]  %.2f [%.2f, %.2f]  %s\n",
              nm, ref$cue, est, lo, hi,
              ref$mean, ref$lo, ref$hi,
              if (inside) "PASS" else "FAIL"))
}
cat(strrep("-", 75), "\n")
cat(sprintf("Passes: %d / %d (criterion: >= 15/18)\n", passes, total))

# Qualitative ordering checks
cat("\nQualitative ordering checks:\n")
# P(U): CC > WC > LC
pu_cc <- prob_df$Estimate[prob_df$param == "CCPU"]
pu_wc <- prob_df$Estimate[prob_df$param == "WCPU"]
pu_lc <- prob_df$Estimate[prob_df$param == "LCPU"]
cat(sprintf("  P(U): CC(%.2f) > WC(%.2f) > LC(%.2f): %s\n",
            pu_cc, pu_wc, pu_lc,
            if (pu_cc > pu_wc && pu_wc > pu_lc) "PASS" else "FAIL"))
# P(UT|U): LC > WC > CC
putgu_cc <- prob_df$Estimate[prob_df$param == "CCPUTgivenU"]
putgu_wc <- prob_df$Estimate[prob_df$param == "WCPUTgivenU"]
putgu_lc <- prob_df$Estimate[prob_df$param == "LCPUTgivenU"]
cat(sprintf("  P(UT|U): LC(%.2f) > WC(%.2f) > CC(%.2f): %s\n",
            putgu_lc, putgu_wc, putgu_cc,
            if (putgu_lc > putgu_wc && putgu_wc > putgu_cc) "PASS" else "FAIL"))

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)
write.csv(smry,
  file.path(results_dir, "hybrid_posterior_summary.csv"))
saveRDS(waic_hyb,
  file.path(results_dir, "hybrid_waic.rds"))
write.csv(prob_df,
  file.path(results_dir, "hybrid_prob_scale.csv"),
  row.names = FALSE)

cat("\nDone. Results saved to", results_dir, "\n")
