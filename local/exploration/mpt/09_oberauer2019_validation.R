# WP2 (Round 2): Real-data validation — Oberauer (2019) Exp. 1 simple span
#
# Source:
#   Oberauer, K. (2019). Is rehearsal an effective maintenance strategy for
#   working memory? Journal of Cognition, 2(1), 1–14.
#   https://doi.org/10.5334/joc.58
#
# Data: per-participant x condition aggregation of the openly published
#   recognition data (Exp. 1), filtered to rsizeList > 1 and rsizeNPL > 0;
#   response categories: correct / other-list / not-presented lure;
#   23 trials per cell. (21 participants; IDs 1-21 excluding 4.)
#
# Model — Oberauer (2019) MPT with design-fixed guessing rates (single tree,
#   three response categories; fitted with custom JAGS by the original author):
#   - correct = Pb + (1-Pb)*Pi*GcorrPi + (1-Pb)*(1-Pi)*GcorrNoPi
#   - other   = (1-Pb)*Pi*(1-GcorrPi) + (1-Pb)*(1-Pi)*(1-GcorrNoPi)*GotherNoPi
#   - npl     = (1-Pb)*(1-Pi)*(1-GcorrNoPi)*(1-GotherNoPi)
#   Pb = probability of binding to probe; Pi = probability of feature inference
#   Guessing rates fixed by response-set composition (not estimated):
#     GcorrPi    = 1/rsizeList
#     GcorrNoPi  = 1/(rsizeList + rsizeNPL)
#     GotherNoPi = (rsizeList - 1)/(rsizeList - 1 + rsizeNPL)
#
# Reference (JAGS, 4 chains x 10000/5000):
#   Pi intercept (meanPi): 2.420 [1.617, 3.436]
#   Pb intercept (meanPb): 2.459 [2.140, 2.799]
#   Pi set-size slope (dPi): -0.236 [-0.569, 0.079]
#   Pb set-size slope (dPb): -0.596 [-0.684, -0.518]
#
# Acceptance: brms posterior means of all four quantities fall inside JAGS
#   95% CIs; dPb is credibly negative; max Rhat <= 1.02.
#
# Run from repo root:  Rscript local/exploration/mpt/09_oberauer2019_validation.R

suppressMessages(library(brms))
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))

source("local/exploration/mpt/03_mpt_parser.R", local = TRUE)

# =============================================================================
# 1. Load and prepare data
# =============================================================================
dat <- read.csv("local/exploration/mpt/data/oberauer2019_ss_exp1_agg.csv")
cat("=== Oberauer (2019) — simple span recognition MPT ===\n\n")
cat(sprintf("Participants: %d, Conditions: %d, Total rows: %d\n",
            length(unique(dat$id)), length(unique(dat$setsize)), nrow(dat)))

# Design-fixed guessing covariates
# Rename f* columns to match tree response-category names used by data_prep
dat <- dat |>
  rename(correct = fcorrect, other = fother, npl = fnpl) |>
  mutate(
    GcorrPi    = 1 / rsizeList,
    GcorrNoPi  = 1 / (rsizeList + rsizeNPL),
    GotherNoPi = (rsizeList - 1) / (rsizeList - 1 + rsizeNPL),
    # Centred set size (matching the reference JAGS centering)
    Csetsize   = setsize - mean(sort(unique(setsize))),
    id         = factor(id)
  )

cat("\nCovariate ranges:\n")
cat(sprintf("  GcorrPi   : [%.3f, %.3f]\n", min(dat$GcorrPi), max(dat$GcorrPi)))
cat(sprintf("  GcorrNoPi : [%.3f, %.3f]\n", min(dat$GcorrNoPi), max(dat$GcorrNoPi)))
cat(sprintf("  GotherNoPi: [%.3f, %.3f]\n", min(dat$GotherNoPi), max(dat$GotherNoPi)))
cat(sprintf("  Csetsize  : [%.1f, %.1f] (mean of unique setsizes = %.1f)\n",
            min(dat$Csetsize), max(dat$Csetsize),
            mean(sort(unique(dat$setsize)))))

cat("\nAggregate response proportions:\n")
dat |>
  summarise(
    correct = sum(correct) / sum(n),
    other   = sum(other)   / sum(n),
    npl     = sum(npl)     / sum(n)
  ) |>
  print()

# =============================================================================
# 2. Build MPT spec with covariates
# =============================================================================
tree_ss <- mpt_tree("ss", list(
  correct = paste0("Pb + (1 - Pb) * Pi * GcorrPi + ",
                   "(1 - Pb) * (1 - Pi) * GcorrNoPi"),
  other   = paste0("(1 - Pb) * Pi * (1 - GcorrPi) + ",
                   "(1 - Pb) * (1 - Pi) * (1 - GcorrNoPi) * GotherNoPi"),
  npl     = "(1 - Pb) * (1 - Pi) * (1 - GcorrNoPi) * (1 - GotherNoPi)"
))

spec_ss <- mpt(
  trees      = list(tree_ss),
  covariates = c("GcorrPi", "GcorrNoPi", "GotherNoPi")
  # condition = NULL: single tree, no condition column needed
)
print(spec_ss)
cat(sprintf("Parameters identified: %s\n", paste(spec_ss$params, collapse = ", ")))
stopifnot(setequal(spec_ss$params, c("Pb", "Pi")))

# =============================================================================
# 3. Build brms formula
# =============================================================================
out_ss <- mpt_to_brms(
  spec_ss,
  predictor_formulas = list(
    Pi = ~ 1 + Csetsize + (1 + Csetsize || id),
    Pb = ~ 1 + Csetsize + (1 + Csetsize || id)
  )
)

cat("\nBrms formula:\n")
print(out_ss$brms_formula)
cat("\nFamily:", out_ss$family, "\n")
cat("Response categories:", paste(out_ss$family_obj$dpars, collapse = ", "), "\n")

# Prepare data: add indicator column + Y response matrix
dat_prep <- out_ss$data_prep(dat)
cat(sprintf("\nPrepared data: %d rows, .ind_ss range [%d, %d]\n",
            nrow(dat_prep), min(dat_prep$.ind_ss), max(dat_prep$.ind_ss)))
stopifnot(all(dat_prep$.ind_ss == 1L))

# =============================================================================
# 4. Priors — logistic(0,1) intercepts for Pb/Pi; normal(0,1) for slopes;
#    half-normal(0,1) for SDs (no correlation prior — uncorrelated REs via ||)
# =============================================================================
priors_ss <- c(
  out_ss$suggested_priors,
  prior("normal(0, 1)", class = "b", nlpar = "lPi"),
  prior("normal(0, 1)", class = "b", nlpar = "lPb"),
  prior("normal(0, 1)", class = "sd", nlpar = "lPi"),
  prior("normal(0, 1)", class = "sd", nlpar = "lPb")
)

cat("\nPriors:\n")
print(priors_ss)

# =============================================================================
# 5. Fit — 4 chains x 2000 iter (1000 warmup), backend = cmdstanr
# =============================================================================
cat("\nFitting Oberauer SS model (4 chains x 2000 iter, 1000 warmup)...\n")
fit_ss <- brm(
  formula  = out_ss$brms_formula,
  data     = dat_prep,
  prior    = priors_ss,
  family   = out_ss$family_obj,
  backend  = "cmdstanr",
  chains   = 4,
  iter     = 2000,
  warmup   = 1000,
  cores    = 4,
  seed     = 42,
  silent   = 2,
  refresh  = 0
)

cat("\nModel summary:\n")
print(summary(fit_ss))

# =============================================================================
# 6. Extract estimates and compare to JAGS reference
# =============================================================================
draws <- as_draws_df(fit_ss)

# Group-level means (fixed effects, logit scale)
Pi_int   <- draws$b_lPi_Intercept
Pb_int   <- draws$b_lPb_Intercept
Pi_slope <- draws$b_lPi_Csetsize
Pb_slope <- draws$b_lPb_Csetsize

# Random-effect SDs (uncorrelated via ||)
sd_Pi_int   <- draws$`sd_id__lPi_Intercept`
sd_Pb_int   <- draws$`sd_id__lPb_Intercept`
sd_Pi_slope <- draws$`sd_id__lPi_Csetsize`
sd_Pb_slope <- draws$`sd_id__lPb_Csetsize`

ci95 <- function(x) quantile(x, c(0.025, 0.975))

max_rhat <- max(brms::rhat(fit_ss), na.rm = TRUE)
ndiv     <- sum(nuts_params(fit_ss)$Value[nuts_params(fit_ss)$Parameter == "divergent__"])

cat("\n=== Estimation Results vs JAGS Reference ===\n")
cat(sprintf("%-30s  %8s  %-16s  %8s  %-16s  %s\n",
            "Quantity", "brms", "brms 95% CI", "JAGS", "JAGS 95% CI", "Pass?"))
cat(strrep("-", 100), "\n")

jags_ref <- list(
  Pi_int   = list(mean =  2.420, lo =  1.617, hi = 3.436),
  Pb_int   = list(mean =  2.459, lo =  2.140, hi = 2.799),
  Pi_slope = list(mean = -0.236, lo = -0.569, hi = 0.079),
  Pb_slope = list(mean = -0.596, lo = -0.684, hi = -0.518)
)

check_row <- function(label, draws_vec, ref) {
  est  <- mean(draws_vec)
  ci   <- ci95(draws_vec)
  pass <- (est >= ref$lo) & (est <= ref$hi)
  cat(sprintf("%-30s  %8.3f  [%6.3f, %6.3f]  %8.3f  [%6.3f, %6.3f]  %s\n",
              label, est, ci[1], ci[2], ref$mean, ref$lo, ref$hi,
              ifelse(pass, "PASS", "FAIL")))
  pass
}

pass_results <- c(
  check_row("Pi intercept (meanPi)",   Pi_int,   jags_ref$Pi_int),
  check_row("Pb intercept (meanPb)",   Pb_int,   jags_ref$Pb_int),
  check_row("Pi slope (dPi)",          Pi_slope, jags_ref$Pi_slope),
  check_row("Pb slope (dPb)",          Pb_slope, jags_ref$Pb_slope)
)

cat(strrep("-", 100), "\n")

# Credible negativity of Pb slope (dPb)
p_neg_Pb <- mean(Pb_slope < 0)

cat("\nAdditional checks:\n")
cat(sprintf("  Pb slope credibly negative (P(dPb<0)): %.3f (threshold: > 0.95)\n", p_neg_Pb))
cat(sprintf("  Max Rhat: %.4f (threshold: <= 1.02)\n", max_rhat))
cat(sprintf("  Divergent transitions: %d (threshold: < 0.5%% = %d)\n",
            ndiv, ceiling(0.005 * 4 * 1000)))

cat("\nRandom-effect SDs (no JAGS reference for comparison):\n")
cat(sprintf("  sd_Pi_int   : est = %.3f  95%% CI = [%.3f, %.3f]\n",
            mean(sd_Pi_int),   ci95(sd_Pi_int)[1],   ci95(sd_Pi_int)[2]))
cat(sprintf("  sd_Pb_int   : est = %.3f  95%% CI = [%.3f, %.3f]\n",
            mean(sd_Pb_int),   ci95(sd_Pb_int)[1],   ci95(sd_Pb_int)[2]))
cat(sprintf("  sd_Pi_slope : est = %.3f  95%% CI = [%.3f, %.3f]\n",
            mean(sd_Pi_slope), ci95(sd_Pi_slope)[1], ci95(sd_Pi_slope)[2]))
cat(sprintf("  sd_Pb_slope : est = %.3f  95%% CI = [%.3f, %.3f]\n",
            mean(sd_Pb_slope), ci95(sd_Pb_slope)[1], ci95(sd_Pb_slope)[2]))

cat(sprintf("\nJAGS reference SDs (for qualitative comparison):\n"))
cat("  sgPi=0.938, sgPb=0.577, sgSlopePi=0.188, sgSlopePb=0.045\n")

# Summary verdict
all_pass <- all(pass_results) & (p_neg_Pb > 0.95) & (max_rhat <= 1.02)
cat(sprintf("\n=== Overall verdict: %s ===\n",
            ifelse(all_pass, "PASS — all acceptance criteria met",
                   "PARTIAL — check individual criteria above")))

cat("\nDone.\n")
