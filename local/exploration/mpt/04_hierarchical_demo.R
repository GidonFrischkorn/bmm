# Hierarchical 2HTM: participant random effects on probit-transformed parameters
# Demonstrates parameter recovery using the mpt_to_brms() infrastructure
# from 03_mpt_parser.R.

source("local/exploration/mpt/03_mpt_parser.R", local = TRUE)

library(brms)
library(dplyr)
library(tidyr)

set.seed(2024)

# ---------------------------------------------------------------------------
# 1.  Simulate hierarchical data
# ---------------------------------------------------------------------------
# Group-level (population) parameters on logit scale:
#   lD ~ N(logit(0.70), 0.30^2)  (detection)
#   lg ~ N(logit(0.50), 0.25^2)  (guessing)

n_participants <- 80
n_items        <- 50          # per tree (old / new)

mu_lD  <- qlogis(0.70)       # ≈ 0.847
sd_lD  <- 0.30
mu_lg  <- qlogis(0.50)       # = 0
sd_lg  <- 0.25

lD_i   <- rnorm(n_participants, mu_lD, sd_lD)
lg_i   <- rnorm(n_participants, mu_lg, sd_lg)
D_i    <- plogis(lD_i)
g_i    <- plogis(lg_i)

# Simulate responses per participant × condition
sim_hier <- bind_rows(
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
  mutate(id = as.factor(id))

cat("Simulated data (first 8 rows):\n")
print(head(sim_hier, 8))
cat(sprintf("True population: D ~ N(%.3f, %.3f^2),  g ~ N(%.3f, %.3f^2)\n",
            plogis(mu_lD), sd_lD, plogis(mu_lg), sd_lg))
cat(sprintf("  mu_lD = %.3f, sd_lD = %.3f\n", mu_lD, sd_lD))
cat(sprintf("  mu_lg = %.3f, sd_lg = %.3f\n", mu_lg, sd_lg))

# ---------------------------------------------------------------------------
# 2.  Build model spec and brms formula via parser
# ---------------------------------------------------------------------------
tree_old <- mpt_tree("old", list(
  old = "D + (1 - D) * g",
  new = "(1 - D) * (1 - g)"
))
tree_new <- mpt_tree("new", list(
  old = "(1 - D) * g",
  new = "D + (1 - D) * (1 - g)"
))

spec <- mpt(list(tree_old, tree_new), condition = "condition", link = "logit")

out <- mpt_to_brms(
  spec,
  predictor_formulas = list(
    D = ~ 1 + (1 | id),
    g = ~ 1 + (1 | id)
  ),
  response_col = "old",
  trials_col   = "n"
)

# Add indicator columns to data
sim_hier_prep <- out$data_prep(sim_hier)

cat("\nGenerated brms formula:\n")
print(out$brms_formula)

# ---------------------------------------------------------------------------
# 3.  Fit with brms
# ---------------------------------------------------------------------------
cat("\nFitting hierarchical 2HTM via mpt_to_brms() output...\n")

fit_hier <- brm(
  formula  = out$brms_formula,
  data     = sim_hier_prep,
  prior    = out$suggested_priors,
  family   = binomial(),
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
print(summary(fit_hier))

# ---------------------------------------------------------------------------
# 4.  Parameter recovery checks
# ---------------------------------------------------------------------------
# Extract population-level intercepts (= mu_lD, mu_lg on logit scale)
draws <- as_draws_df(fit_hier)

# Population means
mu_lD_hat <- mean(draws$b_lD_Intercept)
mu_lg_hat <- mean(draws$b_lg_Intercept)

cat(sprintf("\n--- Population-level means (logit scale) ---\n"))
cat(sprintf("  mu_lD: true = %.3f, estimated = %.3f (D: true = %.3f, est = %.3f)\n",
            mu_lD, mu_lD_hat, plogis(mu_lD), plogis(mu_lD_hat)))
cat(sprintf("  mu_lg: true = %.3f, estimated = %.3f (g: true = %.3f, est = %.3f)\n",
            mu_lg, mu_lg_hat, plogis(mu_lg), plogis(mu_lg_hat)))

# Random-effects SDs
sd_lD_hat <- mean(draws$`sd_id__lD_Intercept`)
sd_lg_hat <- mean(draws$`sd_id__lg_Intercept`)

cat(sprintf("\n--- Random-effects SDs ---\n"))
cat(sprintf("  sd_lD: true = %.3f, estimated = %.3f\n", sd_lD, sd_lD_hat))
cat(sprintf("  sd_lg: true = %.3f, estimated = %.3f\n", sd_lg, sd_lg_hat))

# Participant-level recovery: compare true lD_i with posterior medians
r_lD_col <- grep("^r_id__lD", names(draws), value = TRUE)
r_lg_col <- grep("^r_id__lg", names(draws), value = TRUE)

if (length(r_lD_col) == n_participants) {
  lD_post <- apply(draws[r_lD_col], 2, median) + mu_lD_hat
  lg_post <- apply(draws[r_lg_col], 2, median) + mu_lg_hat

  cor_D <- cor(lD_i, lD_post)
  cor_g <- cor(lg_i, lg_post)
  cat(sprintf("\n--- Participant-level recovery (logit scale) ---\n"))
  cat(sprintf("  Correlation lD (true vs estimated): r = %.3f\n", cor_D))
  cat(sprintf("  Correlation lg (true vs estimated): r = %.3f\n", cor_g))
} else {
  cat("\n[Note: r_id columns not matching expected count — skipping participant-level check]\n")
}

# ---------------------------------------------------------------------------
# 5.  Posterior predictive check (visual, if bayesplot available)
# ---------------------------------------------------------------------------
if (requireNamespace("bayesplot", quietly = TRUE)) {
  cat("\nGenerating posterior predictive check plot...\n")
  pp <- pp_check(fit_hier, type = "rootogram", ndraws = 100)
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    ggplot2::ggsave(
      "local/exploration/mpt/04_ppc_rootogram.png",
      pp, width = 8, height = 5
    )
    cat("  Saved to 04_ppc_rootogram.png\n")
  }
} else {
  cat("\nbayesplot not installed — skipping pp_check plot.\n")
}

cat("\nDone.\n")
