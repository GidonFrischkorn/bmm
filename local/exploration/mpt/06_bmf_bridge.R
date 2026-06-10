# WP3: bmf() bridge for mpt_to_brms()
#
# Demonstrates:
#   1. Accepting a bmmformula (bmf()) object and mapping it to mpt_to_brms()
#   2. Correlated random effects via (1 |p| id) syntax
#   3. Recovery of a known latent correlation (rho = 0.5) between lD and lg
#
# Run from the repo root:  Rscript local/exploration/mpt/06_bmf_bridge.R

suppressMessages(library(brms))
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))

source("local/exploration/mpt/03_mpt_parser.R", local = TRUE)

# =============================================================================
# 1. bmf() bridge helper
# =============================================================================
# Convert a bmmformula object (from bmf()) to the predictor_formulas list
# expected by mpt_to_brms().
#
# bmmformula structure: named list where each element is either
#   - a two-sided formula  param ~ predictors
#   - a numeric value (fixed parameter, currently unsupported for MPT)
#
# The bridge strips the LHS and produces: list(param = ~ predictors, ...)
# Parameters in the bmmformula that are not in spec$params are ignored.
# Parameters in spec$params that are not in the bmmformula default to ~ 1.

mpt_bridge_bmf <- function(spec, bmf_obj) {
  stopifnot(inherits(spec, "mpt_spec"))
  stopifnot(inherits(bmf_obj, "bmmformula"))

  predictor_formulas <- lapply(bmf_obj, function(fm) {
    if (!inherits(fm, "formula")) {
      warning("mpt_bridge_bmf: fixed numeric parameters are not yet supported; using ~ 1.")
      return(as.formula("~ 1"))
    }
    as.formula(sprintf("~ %s", deparse(fm[[3]])))
  })
  # Only keep entries whose names are in spec$params
  predictor_formulas[intersect(names(predictor_formulas), spec$params)]
}

# =============================================================================
# 2. Simulate hierarchical 2HTM data with known correlation between lD and lg
# =============================================================================
# Population parameters (logit scale):
#   mu_lD  = logit(0.70) ≈ 0.847
#   mu_lg  = logit(0.50) = 0
#   sd_lD  = 0.30, sd_lg = 0.25
#   rho(lD, lg) = 0.50 (known target)

n_participants <- 100
n_items        <- 50

mu_lD  <- qlogis(0.70)
mu_lg  <- 0.00
sd_lD  <- 0.30
sd_lg  <- 0.25
rho    <- 0.50

Sigma  <- matrix(c(sd_lD^2, rho * sd_lD * sd_lg,
                   rho * sd_lD * sd_lg, sd_lg^2),
                 nrow = 2)

set.seed(2025)
u      <- MASS::mvrnorm(n_participants, mu = c(0, 0), Sigma = Sigma)
lD_i   <- mu_lD + u[, 1]
lg_i   <- mu_lg + u[, 2]
D_i    <- plogis(lD_i)
g_i    <- plogis(lg_i)

sim_data <- bind_rows(
  tibble(
    id        = seq_len(n_participants),
    condition = "old",
    old       = rbinom(n_participants, n_items, D_i + (1 - D_i) * g_i),
    n         = n_items
  ),
  tibble(
    id        = seq_len(n_participants),
    condition = "new",
    old       = rbinom(n_participants, n_items, (1 - D_i) * g_i),
    n         = n_items
  )
) |>
  arrange(id, condition) |>
  mutate(id = factor(id))

cat("=== Simulated data ===\n")
cat(sprintf("True rho(lD, lg) = %.2f\n", rho))
cat(sprintf("True sample correlation in simulated individual params: %.3f\n", cor(lD_i, lg_i)))
print(head(sim_data, 6))

# =============================================================================
# 3. Build formula with correlated random effects using bmf()
# =============================================================================
devtools::load_all(quiet = TRUE)

formula_bmf <- bmf(
  D ~ 1 + (1 |p| id),   # correlated random intercept for D
  g ~ 1 + (1 |p| id)    # correlated random intercept for g
)
cat("\nbmmformula object:\n")
print(formula_bmf)

# Build mpt_spec
tree_old <- mpt_tree("old", list(
  old = "D + (1 - D) * g",
  new = "(1 - D) * (1 - g)"
))
tree_new <- mpt_tree("new", list(
  old = "(1 - D) * g",
  new = "D + (1 - D) * (1 - g)"
))
spec <- mpt(list(tree_old, tree_new), condition = "condition")

# Convert bmf -> predictor_formulas and build brms formula
pred_fms  <- mpt_bridge_bmf(spec, formula_bmf)
out       <- mpt_to_brms(spec,
                          predictor_formulas = pred_fms,
                          response_col       = "old",
                          trials_col         = "n")
sim_prep  <- out$data_prep(sim_data)

cat("\nBrms formula with correlated REs:\n")
print(out$brms_formula)

# =============================================================================
# 4. Fit with brms  (smoke: 4 chains × 1000 iter for speed)
# =============================================================================
cat("\nFitting 2HTM with correlated random effects ...\n")

prior_corr <- c(
  out$suggested_priors,
  prior("lkj(2)", class = "cor", group = "id")
)

fit_corr <- brm(
  formula = out$brms_formula,
  data    = sim_prep,
  prior   = prior_corr,
  family  = binomial(),
  chains  = 4,
  iter    = 1000,
  warmup  = 500,
  cores   = 4,
  seed    = 42,
  silent  = 2,
  refresh = 0
)

cat("\nModel summary:\n")
print(summary(fit_corr))

# =============================================================================
# 5. Recovery: population means, SDs, and latent correlation
# =============================================================================
draws <- as_draws_df(fit_corr)

mu_lD_hat <- mean(draws$b_lD_Intercept)
mu_lg_hat <- mean(draws$b_lg_Intercept)
sd_lD_hat <- mean(draws$`sd_id__lD_Intercept`)
sd_lg_hat <- mean(draws$`sd_id__lg_Intercept`)
rho_hat   <- mean(draws$`cor_id__lD_Intercept__lg_Intercept`)

ci_rho <- quantile(draws$`cor_id__lD_Intercept__lg_Intercept`, c(0.025, 0.975))

cat("\n=== Recovery Summary ===\n")
cat(sprintf("Population means (logit scale):\n"))
cat(sprintf("  mu_lD : true = %.3f  est = %.3f  (D: true = %.3f, est = %.3f)\n",
            mu_lD, mu_lD_hat, plogis(mu_lD), plogis(mu_lD_hat)))
cat(sprintf("  mu_lg : true = %.3f  est = %.3f  (g: true = %.3f, est = %.3f)\n",
            mu_lg, mu_lg_hat, plogis(mu_lg), plogis(mu_lg_hat)))

cat(sprintf("\nRandom-effect SDs:\n"))
cat(sprintf("  sd_lD : true = %.3f  est = %.3f\n", sd_lD, sd_lD_hat))
cat(sprintf("  sd_lg : true = %.3f  est = %.3f\n", sd_lg, sd_lg_hat))

cat(sprintf("\nLatent correlation rho(lD, lg):\n"))
cat(sprintf("  true = %.3f  est = %.3f  95%% CI = [%.3f, %.3f]\n",
            rho, rho_hat, ci_rho[1], ci_rho[2]))
cat(sprintf("  Coverage: true rho IN 95%% CI? %s\n",
            rho >= ci_rho[1] & rho <= ci_rho[2]))

# Max R-hat
draws_summary <- summarize_draws(fit_corr)
max_rhat <- max(draws_summary$rhat, na.rm = TRUE)
cat(sprintf("\nMax R-hat: %.4f (should be < 1.01)\n", max_rhat))

cat("\nDone.\n")
