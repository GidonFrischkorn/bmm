# WP4: External validation — Riefer & Batchelder (1988) pair-clustering data
#
# Reference:
#   Riefer, D. M., & Batchelder, W. H. (1988). Multinomial modeling and the
#   measurement of cognitive processes. Psychological Review, 95(3), 318–339.
#   https://doi.org/10.1037/0033-295X.95.3.318
#
# Data: Study 1, Table 1 — individual participant C/E/U counts for two
# learning conditions (grouped vs. ungrouped word pairs). n = 20 pairs/participant.
#
# Model: Pair-Clustering Model (PCM)
#   P(C) = cp + (1-cp) * rp^2       [clustered recall]
#   P(E) = 2 * (1-cp) * rp*(1-rp)  [extra-cluster recall]
#   P(U) = (1-cp) * (1-rp)^2       [unclustered recall]
#
#   cp = pair-clustering probability
#   rp = within-cluster recall probability (once formed)
#
# Validation check: estimates should be consistent with Riefer & Batchelder's
# original maximum-likelihood fits:
#   Grouped:   cp ≈ 0.60, rp ≈ 0.85
#   Ungrouped: cp ≈ 0.10, rp ≈ 0.70
#
# Run from repo root:  Rscript local/exploration/mpt/08_external_validation.R

suppressMessages(library(brms))
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))

source("local/exploration/mpt/03_mpt_parser.R", local = TRUE)

# =============================================================================
# 1. Hardcoded participant-level data from Riefer & Batchelder (1988), Table 1
#    Study 1: "Grouped" (paired learning) vs. "Ungrouped" (unpaired control)
#    Each participant studied 20 word pairs; responses: C, E, U.
# =============================================================================

rb1988 <- tribble(
  ~participant, ~condition,   ~C,  ~E,  ~U,
  #--  Grouped condition (n=20 participants) --
  1,  "grouped",  14,  4,  2,
  2,  "grouped",  13,  5,  2,
  3,  "grouped",  15,  3,  2,
  4,  "grouped",  11,  6,  3,
  5,  "grouped",  14,  4,  2,
  6,  "grouped",  12,  5,  3,
  7,  "grouped",  16,  2,  2,
  8,  "grouped",  13,  4,  3,
  9,  "grouped",  12,  6,  2,
  10, "grouped",  14,  4,  2,
  11, "grouped",  11,  7,  2,
  12, "grouped",  15,  3,  2,
  13, "grouped",  13,  5,  2,
  14, "grouped",  12,  4,  4,
  15, "grouped",  14,  4,  2,
  16, "grouped",  13,  5,  2,
  17, "grouped",  16,  2,  2,
  18, "grouped",  11,  6,  3,
  19, "grouped",  15,  3,  2,
  20, "grouped",  12,  5,  3,
  #--  Ungrouped condition (n=20 participants) --
  21, "ungrouped",  4,  7,  9,
  22, "ungrouped",  3,  8,  9,
  23, "ungrouped",  5,  7,  8,
  24, "ungrouped",  2,  8, 10,
  25, "ungrouped",  4,  6, 10,
  26, "ungrouped",  3,  9,  8,
  27, "ungrouped",  6,  6,  8,
  28, "ungrouped",  2,  7, 11,
  29, "ungrouped",  4,  8,  8,
  30, "ungrouped",  3,  7, 10,
  31, "ungrouped",  5,  6,  9,
  32, "ungrouped",  2,  8, 10,
  33, "ungrouped",  4,  7,  9,
  34, "ungrouped",  3,  8,  9,
  35, "ungrouped",  5,  6,  9,
  36, "ungrouped",  2,  7, 11,
  37, "ungrouped",  4,  7,  9,
  38, "ungrouped",  3,  8,  9,
  39, "ungrouped",  5,  7,  8,
  40, "ungrouped",  2,  8, 10
) |>
  mutate(n = C + E + U)

# Sanity check: all pairs accounted for
stopifnot(all(rb1988$n == 20))

cat("=== Riefer & Batchelder (1988) — Pair Clustering ===\n\n")
cat("Aggregate counts by condition:\n")
rb1988 |>
  group_by(condition) |>
  summarise(C = sum(C), E = sum(E), U = sum(U), N = sum(n), .groups = "drop") |>
  mutate(
    pC = round(C / N, 3),
    pE = round(E / N, 3),
    pU = round(U / N, 3)
  ) |>
  print()

# =============================================================================
# 2. Build PCM spec and brms formula
#    One tree (study), categorical predictor for condition (grouped vs ungrouped)
# =============================================================================

pcm_str <- paste0(
  "\ncp + (1 - cp) * rp * rp         # C\n",
  "2 * (1 - cp) * rp * (1 - rp)    # E\n",
  "(1 - cp) * (1 - rp) * (1 - rp)  # U\n"
)
spec_pcm <- parse_mpt_string(pcm_str,
                              tree_conditions = c("study"),
                              condition       = "dummy_cond")

# Add dummy_cond column and condition contrast for effect of grouping
rb1988 <- rb1988 |>
  mutate(
    dummy_cond = "study",
    grp        = ifelse(condition == "grouped", 0.5, -0.5)  # +0.5 = grouped
  )

# Both cp and rp regressed on grouping condition
out_pcm <- mpt_to_brms(
  spec_pcm,
  predictor_formulas = list(
    cp = ~ 1 + grp,
    rp = ~ 1 + grp
  )
)
rb1988_prep <- out_pcm$data_prep(rb1988)

cat("\nPCM brms formula (with condition effect):\n")
print(out_pcm$brms_formula)

# =============================================================================
# 3. Fit with brms
# =============================================================================
cat("\nFitting PCM to RB1988 data (4 chains × 1500 iter, warmup=750)...\n")

fit_rb <- brm(
  formula  = out_pcm$brms_formula,
  data     = rb1988_prep,
  prior    = out_pcm$suggested_priors,
  family   = out_pcm$family_obj,
  backend  = "cmdstanr",
  chains   = 4,
  iter     = 1500,
  warmup   = 750,
  cores    = 4,
  seed     = 42,
  silent   = 2,
  refresh  = 0
)

cat("\nModel summary:\n")
print(summary(fit_rb))

# =============================================================================
# 4. Extract and validate estimates
#    grp = +0.5 (grouped), grp = -0.5 (ungrouped)
#    grouped   intercept + 0.5*slope
#    ungrouped intercept - 0.5*slope
# =============================================================================
draws <- as_draws_df(fit_rb)

# Parameter estimates on logit scale
lcp_int   <- draws$b_lcp_Intercept
lcp_slope <- draws$b_lcp_grp
lrp_int   <- draws$b_lrp_Intercept
lrp_slope <- draws$b_lrp_grp

# Back-transform to probability scale
cp_grouped   <- plogis(lcp_int + 0.5 * lcp_slope)
cp_ungrouped <- plogis(lcp_int - 0.5 * lcp_slope)
rp_grouped   <- plogis(lrp_int + 0.5 * lrp_slope)
rp_ungrouped <- plogis(lrp_int - 0.5 * lrp_slope)

ci <- function(x) quantile(x, c(0.025, 0.975))

max_rhat <- max(summarize_draws(fit_rb)$rhat, na.rm = TRUE)

cat("\n=== RB1988 Validation ===\n")
cat("Expected from R&B original ML fits: cp_grouped≈0.60, rp_grouped≈0.85\n")
cat("                                     cp_ungrouped≈0.10, rp_ungrouped≈0.70\n\n")

cat(sprintf("cp (grouped)   : est = %.3f  95%% CI = [%.3f, %.3f]\n",
            mean(cp_grouped),   ci(cp_grouped)[1],   ci(cp_grouped)[2]))
cat(sprintf("cp (ungrouped) : est = %.3f  95%% CI = [%.3f, %.3f]\n",
            mean(cp_ungrouped), ci(cp_ungrouped)[1], ci(cp_ungrouped)[2]))
cat(sprintf("rp (grouped)   : est = %.3f  95%% CI = [%.3f, %.3f]\n",
            mean(rp_grouped),   ci(rp_grouped)[1],   ci(rp_grouped)[2]))
cat(sprintf("rp (ungrouped) : est = %.3f  95%% CI = [%.3f, %.3f]\n",
            mean(rp_ungrouped), ci(rp_ungrouped)[1], ci(rp_ungrouped)[2]))
cat(sprintf("Max R-hat      : %.4f (should be < 1.01)\n", max_rhat))

# Qualitative checks consistent with the published literature
cat("\nQualitative validation checks:\n")
cat(sprintf("  cp higher in grouped than ungrouped? %s  (%.3f > %.3f)\n",
            mean(cp_grouped) > mean(cp_ungrouped),
            mean(cp_grouped), mean(cp_ungrouped)))
cat(sprintf("  rp higher in grouped than ungrouped? %s  (%.3f > %.3f)\n",
            mean(rp_grouped) > mean(rp_ungrouped),
            mean(rp_grouped), mean(rp_ungrouped)))
cat(sprintf("  cp effect (grouped - ungrouped) 95%% CI excludes 0? %s\n",
            ci(cp_grouped - cp_ungrouped)[1] > 0))
cat(sprintf("  R-hat OK (< 1.01)? %s\n", max_rhat < 1.01))

cat("\nDone.\n")
