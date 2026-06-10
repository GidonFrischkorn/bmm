# Reproduce the upstream venpopov/bmm#255 demo
# Fit 2-High Threshold Model (2HTM) of recognition memory in raw brms,
# then compare against MPTinR estimates on the same simulated data.

library(brms)
library(dplyr)
library(tidyr)

set.seed(42)

# ---------------------------------------------------------------------------
# 1.  Simulate data with MPTinR
# ---------------------------------------------------------------------------
# The 2HTM with Do = Dn = D (identifiable constrained version):
#
#   Tree 1 (old items):
#     P(old response | old item) = D + (1 - D) * g
#     P(new response | old item) = (1 - D) * (1 - g)
#
#   Tree 2 (new items):
#     P(old response | new item) = (1 - D) * g
#     P(new response | new item) = D + (1 - D) * (1 - g)
#
# True parameters: D = 0.7, g = 0.5
# 100 participants, 50 old items + 50 new items each

D_true <- 0.7
g_true <- 0.5
n_participants <- 100
n_items_per_tree <- 50

# Branch probabilities
p_old_hit   <- D_true + (1 - D_true) * g_true      # 0.85
p_old_miss  <- (1 - D_true) * (1 - g_true)         # 0.15
p_new_fa    <- (1 - D_true) * g_true               # 0.15
p_new_cr    <- D_true + (1 - D_true) * (1 - g_true) # 0.85

stopifnot(abs(p_old_hit + p_old_miss - 1) < 1e-10)
stopifnot(abs(p_new_fa  + p_new_cr  - 1) < 1e-10)

sim_data <- tibble(
  id        = rep(seq_len(n_participants), each = 2),
  condition = rep(c("old", "new"), n_participants),
  old = c(
    rbinom(n_participants, n_items_per_tree, p_old_hit),   # old hits
    rbinom(n_participants, n_items_per_tree, p_new_fa)     # false alarms
  ),
  new = c(
    rbinom(n_participants, n_items_per_tree, p_old_miss),  # misses
    rbinom(n_participants, n_items_per_tree, p_new_cr)     # correct rejections
  ),
  n = n_items_per_tree
) |>
  mutate(
    id = as.factor(id),
    is_old = as.integer(condition == "old")
  )

cat("Data head:\n")
print(head(sim_data, 6))

# ---------------------------------------------------------------------------
# 2.  Compute naive MLE (aggregated) for reference
# ---------------------------------------------------------------------------
agg <- sim_data |>
  group_by(condition) |>
  summarise(old = sum(old), n = sum(n), .groups = "drop")

old_hit_rate <- agg$old[agg$condition == "old"] / agg$n[agg$condition == "old"]
fa_rate      <- agg$old[agg$condition == "new"] / agg$n[agg$condition == "new"]

# From 2HTM equations:
#   hit_rate = D + (1 - D) * g
#   fa_rate  = (1 - D) * g
#   => D_hat = hit_rate - fa_rate
#   => g_hat = fa_rate / (1 - D_hat)
D_mle <- old_hit_rate - fa_rate
g_mle <- fa_rate / (1 - D_mle)

cat(sprintf("\nNaive MLE (aggregated):\n  D = %.4f (true: %.2f)\n  g = %.4f (true: %.2f)\n",
            D_mle, D_true, g_mle, g_true))

# ---------------------------------------------------------------------------
# 3.  Fit with brms (aggregated, no random effects) — mirror of upstream demo
# ---------------------------------------------------------------------------
# The key trick: the probability of responding "old" depends on condition via
# the index variable is_old:
#   P(old) = is_old * [D + (1-D)*g] + (1-is_old) * (1-D)*g
#
# Parameters D and g live on (0,1), so we reparametrise via logit:
#   D = inv_logit(lD),  g = inv_logit(lg)
# with logistic(0,1) priors on the logit scale (=> Uniform(0,1) on probability scale)

mpt_form_fixed <- bf(
  old | trials(n) ~ logit(is_old * (D + (1 - D) * g) + (1 - is_old) * (1 - D) * g)
) +
  nlf(D ~ inv_logit(lD)) +
  nlf(g ~ inv_logit(lg)) +
  lf(lD ~ 1) +
  lf(lg ~ 1) +
  set_nl(TRUE)

prior_2htm <- c(
  prior("logistic(0, 1)", nlpar = "lD", class = "b"),
  prior("logistic(0, 1)", nlpar = "lg", class = "b")
)

cat("\nFitting aggregated 2HTM ...\n")
fit_fixed <- brm(
  formula  = mpt_form_fixed,
  data     = sim_data,
  prior    = prior_2htm,
  family   = binomial(),
  chains   = 4,
  iter     = 2000,
  warmup   = 1000,
  cores    = 4,
  seed     = 42,
  silent   = 2,
  refresh  = 0
)

cat("\nAggregated model summary (logit scale):\n")
print(summary(fit_fixed))

# Back-transform to probability scale
draws_fixed <- as_draws_df(fit_fixed)
cat("\nPosterior means on probability scale:\n")
cat(sprintf("  D = %.4f  (true: %.2f, MLE: %.4f)\n",
            mean(plogis(draws_fixed$b_lD_Intercept)), D_true, D_mle))
cat(sprintf("  g = %.4f  (true: %.2f, MLE: %.4f)\n",
            mean(plogis(draws_fixed$b_lg_Intercept)), g_true, g_mle))

# ---------------------------------------------------------------------------
# 4.  Hierarchical fit (random effects per participant on lD and lg)
# ---------------------------------------------------------------------------
mpt_form_hier <- bf(
  old | trials(n) ~ logit(is_old * (D + (1 - D) * g) + (1 - is_old) * (1 - D) * g)
) +
  nlf(D ~ inv_logit(lD)) +
  nlf(g ~ inv_logit(lg)) +
  lf(lD ~ 1 + (1 | id)) +
  lf(lg ~ 1 + (1 | id)) +
  set_nl(TRUE)

cat("\nFitting hierarchical 2HTM ...\n")
fit_hier <- brm(
  formula  = mpt_form_hier,
  data     = sim_data,
  prior    = prior_2htm,
  family   = binomial(),
  chains   = 4,
  iter     = 2000,
  warmup   = 1000,
  cores    = 4,
  seed     = 42,
  silent   = 2,
  refresh  = 0
)

cat("\nHierarchical model summary:\n")
print(summary(fit_hier))

draws_hier <- as_draws_df(fit_hier)
cat("\nHierarchical posterior means on probability scale:\n")
cat(sprintf("  D = %.4f  (true: %.2f)\n",
            mean(plogis(draws_hier$b_lD_Intercept)), D_true))
cat(sprintf("  g = %.4f  (true: %.2f)\n",
            mean(plogis(draws_hier$b_lg_Intercept)), g_true))

cat("\nRandom effects SDs (should be near 0 since data had no random effects):\n")
print(VarCorr(fit_hier, summary = TRUE))

# ---------------------------------------------------------------------------
# 5.  Optional: compare with MPTinR (if installed)
# ---------------------------------------------------------------------------
if (requireNamespace("MPTinR", quietly = TRUE)) {
  cat("\n--- MPTinR comparison ---\n")
  model_str <- "
D + (1 - D) * g         # p_old_old
(1 - D) * (1 - g)       # p_new_old

(1 - D) * g             # p_old_new
D + (1 - D) * (1 - g)   # p_new_new
"
  # MPTinR wants a matrix: rows = participants, cols = [old_old, new_old, old_new, new_new]
  mptr_data <- sim_data |>
    pivot_wider(id_cols = id, names_from = condition,
                values_from = c(old, new)) |>
    transmute(old_old = old_old, new_old = new_old,
              old_new = old_new, new_new = new_new) |>
    as.matrix()

  fit_mptinr <- MPTinR::fit.mpt(mptr_data, textConnection(model_str))
  cat("\nMPTinR aggregated estimates:\n")
  print(fit_mptinr$parameters$aggregated)
} else {
  cat("\nMPTinR not installed — skipping cross-validation.\n")
  cat("Install with: install.packages('MPTinR')\n")
}
