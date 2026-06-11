# 09_composite_pt.R
# WP4 — Composite prospect-theory M3
#
# Prototypes A (EU activation) and B (Prelec weighting) compose naturally into
# a prospect-theory M3 with BOTH value-weighting and probability-weighting:
#
#   a_i* = a_i + gamma * V_i          (EU component: value-modulated activation)
#   P(i) ∝ exp(a_i*) * w(p_i)        (Prelec component: distorted option counts)
#
# Combined in log space:
#   log P(i) ∝ a_i + gamma*V_i + log(w(p_i))
#            = a_i + gamma*V_i + (-((-log p_i)^alpha))
#
# Design needed: BOTH value levels AND variable set sizes in the same experiment.
# This script checks joint identifiability of (gamma, alpha) under:
#   (a) a well-designed experiment (both manipulations present)
#   (b) a partial design (only one manipulation present)
#
# A negative result defines the boundary of what one experiment can estimate.
#
# Run from the repo root:
#   source("local/exploration/m3-utility/09_composite_pt.R")

library(brms)
library(dplyr)

set.seed(314)

N_SUBJ   <- 30L
N_TRIALS <- 120L   # 2 value levels × 2 set-size conditions = 4 cells × 30/cell
N        <- N_SUBJ * N_TRIALS

MU_B <- 0.0; MU_A <- 1.5; MU_C <- 2.0
GAMMA_TRUE <- 0.5
ALPHA_TRUE <- 0.6

# ---- 1. Full design: varying value AND set size -------------------------------
# Design cells (balanced):
#   V_corr ∈ {1, 2} × n_other ∈ {2, 6}
# n_corr always 1; n_npl = n_other (for simplicity)

design_full <- expand.grid(
  V_corr  = c(1, 2),
  n_other = c(2L, 6L)
)
design_full$V_other <- 3 - design_full$V_corr  # V_corr + V_other = 3
design_full$n_corr  <- 1L
design_full$n_npl   <- design_full$n_other

# Replicate to N_SUBJ * N_TRIALS rows
n_cells     <- nrow(design_full)
reps_per_cell <- N_TRIALS %/% n_cells   # 30 reps per cell

d_design <- do.call(rbind, lapply(seq_len(N_SUBJ), function(s) {
  cell_data <- design_full[rep(seq_len(n_cells), reps_per_cell), ]
  cell_data$subj <- s
  cell_data
}))
rownames(d_design) <- NULL
N_FULL <- nrow(d_design)

log_Z_fn <- function(act, n) {
  lv <- act + log(n); lv[1] + log(sum(exp(lv - lv[1])))
}

log_prelec <- function(p, alpha) -((-log(p))^alpha)

# Simulate composite PT data
resp_full <- integer(N_FULL)
set.seed(314)
for (i in seq_len(N_FULL)) {
  N_tot <- d_design$n_corr[i] + d_design$n_other[i] + d_design$n_npl[i]
  p_c <- d_design$n_corr[i]  / N_tot
  p_o <- d_design$n_other[i] / N_tot
  p_n <- d_design$n_npl[i]   / N_tot

  # Activation + EU + Prelec
  act <- c(
    MU_B + MU_A + MU_C + GAMMA_TRUE * d_design$V_corr[i]  + log_prelec(p_c, ALPHA_TRUE),
    MU_B + MU_A         + GAMMA_TRUE * d_design$V_other[i] + log_prelec(p_o, ALPHA_TRUE),
    MU_B                                                    + log_prelec(p_n, ALPHA_TRUE)
  )
  resp_full[i] <- sample.int(3L, 1L, prob = exp(act - act[1] - log(sum(exp(act - act[1])))))
}

Y_full <- matrix(0L, N_FULL, 3L)
for (i in seq_len(N_FULL)) Y_full[i, resp_full[i]] <- 1L
colnames(Y_full) <- c("corr", "other", "npl")

d_full <- d_design
d_full$N_total <- d_full$n_corr + d_full$n_other + d_full$n_npl
d_full$p_corr  <- d_full$n_corr  / d_full$N_total
d_full$p_other <- d_full$n_other / d_full$N_total
d_full$p_npl   <- d_full$n_npl   / d_full$N_total
d_full$Idx_corr  <- 1L; d_full$Idx_other <- 1L; d_full$Idx_npl <- 1L
d_full$nTrials   <- 1L
d_full$Y         <- I(Y_full)

# ---- 2. Composite PT brms formula --------------------------------------------

composite_formula <- bf(
  Y | trials(nTrials) ~
    Idx_corr  * (corr  + (-((-log(p_corr))^alpha)))  + (1 - Idx_corr)  * (-100),
  nlf(muother ~
    Idx_other * (other + (-((-log(p_other))^alpha))) + (1 - Idx_other) * (-100)),
  nlf(munpl ~
    Idx_npl   * (npl   + (-((-log(p_npl))^alpha)))   + (1 - Idx_npl)   * (-100)),
  nlf(corr  ~ b + a + c + gamma * V_corr),
  nlf(other ~ b + a     + gamma * V_other),
  nlf(npl   ~ b),
  b + a + c + gamma + alpha ~ 1,
  nl = TRUE
)

composite_priors <- c(
  prior(constant(0),        nlpar = "b",     class = "b"),
  prior(normal(2, 1),       nlpar = "a",     class = "b"),
  prior(normal(3, 1),       nlpar = "c",     class = "b"),
  prior(normal(0, 1),       nlpar = "gamma", class = "b"),
  prior(lognormal(0, 0.5),  nlpar = "alpha", class = "b", lb = 0.01)
)

# Verify Stan code
cat("=== Composite PT M3: joint identifiability ===\n\n")
cat("Verifying Stan code for composite PT model...\n")
stan_comp <- make_stancode(
  composite_formula,
  data   = d_full,
  family = multinomial(refcat = NA),
  prior  = composite_priors
)
cat("Stan code generated (", nchar(stan_comp), "chars)\n")
stopifnot(grepl("b_gamma", stan_comp) && grepl("b_alpha", stan_comp))
cat("Parameters 'gamma' and 'alpha' confirmed present.\n\n")

# ---- 3. Full design fit -------------------------------------------------------

cat("Fitting composite PT model (full design: value × set-size, N=30×120)...\n")
fit_full <- brm(
  composite_formula,
  data    = d_full,
  family  = multinomial(refcat = NA),
  prior   = composite_priors,
  chains  = 2L,
  iter    = 1000L,
  warmup  = 500L,
  cores   = 2L,
  refresh = 0,
  backend = "cmdstanr",
  silent  = 2
)

cat("\n--- Composite PT recovery (full design) ---\n")
cat(sprintf("True: b=%.1f, a=%.1f, c=%.1f, gamma=%.1f, alpha=%.1f\n\n",
            MU_B, MU_A, MU_C, GAMMA_TRUE, ALPHA_TRUE))
fe_full <- fixef(fit_full)
print(round(fe_full[, c("Estimate", "Q2.5", "Q97.5")], 3))

true_pt <- c(b = MU_B, a = MU_A, c = MU_C, gamma = GAMMA_TRUE, alpha = ALPHA_TRUE)
cat("\nCoverage check:\n")
for (nm in names(true_pt)) {
  rn <- paste0(nm, "_Intercept")
  if (rn %in% rownames(fe_full)) {
    lo <- fe_full[rn, "Q2.5"]; hi <- fe_full[rn, "Q97.5"]
    cat(sprintf("  %-5s: true=%.2f, CI=(%.3f, %.3f) -> %s\n",
                nm, true_pt[nm], lo, hi,
                ifelse(true_pt[nm] >= lo & true_pt[nm] <= hi, "COVERED", "MISSED")))
  }
}
cat("\nMax Rhat:", round(max(rhat(fit_full), na.rm = TRUE), 3), "\n")

# Posterior correlation between gamma and alpha
post_full <- as.data.frame(fit_full)
gam_col   <- grep("b_gamma", names(post_full), value = TRUE)[1]
alp_col   <- grep("b_alpha", names(post_full), value = TRUE)[1]
cat(sprintf("Posterior correlation gamma/alpha: %.3f\n\n",
            cor(post_full[[gam_col]], post_full[[alp_col]])))

# ---- 4. Partial designs: identifiability boundary ----------------------------

cat("=== Partial design 1: value only (no set-size variation) ===\n\n")

# Only V varies; n_other = 4 (fixed)
d_val_only <- d_full[d_full$n_other == 4, ]  # half the data

cat("Fitting composite model to value-only subset (n_other fixed at 2)...\n")
fit_val <- suppressWarnings(brm(
  composite_formula,
  data    = d_val_only,
  family  = multinomial(refcat = NA),
  prior   = composite_priors,
  chains  = 2L,
  iter    = 1000L,
  warmup  = 500L,
  cores   = 2L,
  refresh = 0,
  backend = "cmdstanr",
  silent  = 2
))

fe_val <- fixef(fit_val)
cat("alpha 95% CI (should be wide / poorly identified):\n")
cat(sprintf("  alpha: Estimate=%.3f, CI=(%.3f, %.3f)\n",
            fe_val["alpha_Intercept", "Estimate"],
            fe_val["alpha_Intercept", "Q2.5"],
            fe_val["alpha_Intercept", "Q97.5"]))
cat(sprintf("  alpha CI width (val only):  %.3f\n",
            fe_val["alpha_Intercept", "Q97.5"] - fe_val["alpha_Intercept", "Q2.5"]))
cat(sprintf("  alpha CI width (full design): %.3f\n",
            fe_full["alpha_Intercept", "Q97.5"] - fe_full["alpha_Intercept", "Q2.5"]))

cat("\n=== Partial design 2: set-size only (no value variation) ===\n\n")

# Fix V_corr = V_other = 1.5 (constant, no value signal)
d_ss_only <- d_full
d_ss_only$V_corr  <- 1.5
d_ss_only$V_other <- 1.5

cat("Fitting composite model to set-size-only data (V fixed at 1.5)...\n")
fit_ss <- suppressWarnings(brm(
  composite_formula,
  data    = d_ss_only,
  family  = multinomial(refcat = NA),
  prior   = composite_priors,
  chains  = 2L,
  iter    = 1000L,
  warmup  = 500L,
  cores   = 2L,
  refresh = 0,
  backend = "cmdstanr",
  silent  = 2
))

fe_ss <- fixef(fit_ss)
cat("gamma 95% CI (should be wide / poorly identified):\n")
cat(sprintf("  gamma: Estimate=%.3f, CI=(%.3f, %.3f)\n",
            fe_ss["gamma_Intercept", "Estimate"],
            fe_ss["gamma_Intercept", "Q2.5"],
            fe_ss["gamma_Intercept", "Q97.5"]))
cat(sprintf("  gamma CI width (ss only):     %.3f\n",
            fe_ss["gamma_Intercept", "Q97.5"] - fe_ss["gamma_Intercept", "Q2.5"]))
cat(sprintf("  gamma CI width (full design): %.3f\n",
            fe_full["gamma_Intercept", "Q97.5"] - fe_full["gamma_Intercept", "Q2.5"]))

# ---- 5. Summary of identifiability results ------------------------------------

cat("\n=== Summary: identifiability boundary ===\n\n")

# CI width comparison
alpha_ci_full <- fe_full["alpha_Intercept", "Q97.5"] - fe_full["alpha_Intercept", "Q2.5"]
alpha_ci_val  <- fe_val["alpha_Intercept",  "Q97.5"] - fe_val["alpha_Intercept",  "Q2.5"]
gamma_ci_full <- fe_full["gamma_Intercept", "Q97.5"] - fe_full["gamma_Intercept", "Q2.5"]
gamma_ci_ss   <- fe_ss["gamma_Intercept",   "Q97.5"] - fe_ss["gamma_Intercept",   "Q2.5"]

cat(sprintf(
  "Parameter | Full design | Partial (other dim missing) | Ratio\n"))
cat(sprintf(
  "  gamma   | CI = %.3f | CI = %.3f (no val var)       | %.1fx wider\n",
  gamma_ci_full, gamma_ci_ss, gamma_ci_ss / gamma_ci_full))
cat(sprintf(
  "  alpha   | CI = %.3f | CI = %.3f (no ss var)        | %.1fx wider\n",
  alpha_ci_full, alpha_ci_val, alpha_ci_val / alpha_ci_full))

cat("\nConclusion:\n")
if (alpha_ci_val / alpha_ci_full > 2 | gamma_ci_ss / gamma_ci_full > 2) {
  cat("  The composite PT M3 REQUIRES BOTH manipulations for joint identification.\n")
  cat("  Without value variation: alpha is poorly identified.\n")
  cat("  Without set-size variation: gamma is poorly identified.\n")
  cat("  Recommendation: designs aiming to estimate (gamma, alpha) jointly\n")
  cat("  must include both value-directed recall and variable set sizes.\n\n")
} else {
  cat("  Surprisingly, gamma and alpha appear separable even in partial designs.\n")
  cat("  (This would indicate strong structural separation in the composite model.)\n\n")
}

post_cor <- cor(post_full[[gam_col]], post_full[[alp_col]])
cat(sprintf("  Posterior correlation gamma/alpha in full design: %.3f\n", post_cor))
if (abs(post_cor) < 0.3) {
  cat("  Low posterior correlation confirms parameters are nearly orthogonal\n")
  cat("  when BOTH design dimensions are present.\n")
} else {
  cat("  Non-trivial correlation: some partial confound remains even in full design.\n")
}
cat("\nDone.\n")
