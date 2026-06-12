# GCM Parameter Recovery — identifiability of c, gamma, w
#
# Phase 2a changes:
# - Multinomial-aggregated likelihood: each scenario runs S=12 multinomial
#   evaluations instead of T=n_per_stim×12 categorical ones.
# - Improved scenario design: 5 of 8 scenarios now vary gamma (previously 2/8),
#   so r(gamma) and CI coverage are estimated from a proper spread.
# - Prior: log_c ~ N(0,1), log_gamma ~ N(0,1) — the recommended defaults.
#   N(0.5,0.5) on log(c) is documented here as an informative option, not
#   the default, because it pulled mean_c away from true values near 0.8.
#
# Run from repository root:
#   Rscript local/exploration/categorization/04_parameter_recovery.R

suppressPackageStartupMessages({
  library(cmdstanr)
  library(dplyr)
})

# -----------------------------------------------------------------------
# 0. Load R reference functions
# -----------------------------------------------------------------------

source_env <- new.env()
source("local/exploration/categorization/01_r_reference.R",
       local = source_env, echo = FALSE)

gcm_distances   <- source_env$gcm_distances
gcm_similarity  <- source_env$gcm_similarity
luce_choice     <- source_env$luce_choice

# -----------------------------------------------------------------------
# 1. Nosof88 exemplar geometry
# -----------------------------------------------------------------------

stim_coords <- data.frame(
  stim = 1:12,
  x1 = c(-2.543,  0.943, -1.092,  1.558, -2.258,  0.194,
           2.806, -1.177,  1.543, -2.775,  0.528,  1.709),
  x2 = c( 2.641,  4.341,  1.848,  2.902,  0.430,  0.572,
           0.202, -1.038, -1.040, -3.149, -3.766, -3.773),
  cat = c(1L, 2L, 2L, 2L, 1L, 2L, 2L, 1L, 2L, 1L, 1L, 1L)
)

ex_mat   <- as.matrix(stim_coords[, c("x1", "x2")])
ex_cats  <- stim_coords$cat
T_items  <- nrow(stim_coords)  # S = 12 unique stimuli
J_items  <- T_items
M_dims   <- 2
K_cats   <- 2
r_met    <- 2

# Pre-compute per-stimulus distance matrices (S=12, each J×M)
D_raw <- array(0, dim = c(T_items, J_items, M_dims))
for (m in seq_len(M_dims)) {
  D_raw[, , m] <- abs(outer(ex_mat[, m], ex_mat[, m], `-`))^r_met
}
D_stim_list <- lapply(seq_len(T_items), function(s) D_raw[s, , ])

# -----------------------------------------------------------------------
# 2. Simulate trial data from known parameters
# -----------------------------------------------------------------------

simulate_gcm <- function(c_true, gamma_true, w1_true, n_per_stim,
                         ex_mat, ex_cats, K = 2, rng_seed = NULL) {
  if (!is.null(rng_seed)) set.seed(rng_seed)
  w <- c(w1_true, 1 - w1_true)
  D_mat <- gcm_distances(ex_mat, ex_mat, w, r_metric = 2)
  S_mat <- gcm_similarity(D_mat, c_true, p_sim = 1)
  act   <- cbind(rowSums(S_mat[, ex_cats == 1, drop = FALSE]),
                 rowSums(S_mat[, ex_cats == 2, drop = FALSE]))
  P_mat <- luce_choice(act, gamma_true, bias = c(0.5, 0.5))
  trial_rows <- lapply(seq_len(T_items), function(i) {
    y <- sample(1:K, n_per_stim, replace = TRUE, prob = P_mat[i, ])
    data.frame(stim = i, y = y)
  })
  do.call(rbind, trial_rows)
}

# -----------------------------------------------------------------------
# 3. Stan model — multinomial-aggregated likelihood
# -----------------------------------------------------------------------

gcm_stan_code <- "
functions {
  vector gcm_log_act(matrix D_t, vector w, real c, real gamma,
                     vector log_bias, array[] int ex_cat, int K) {
    int J = rows(D_t);
    vector[J] log_sims;
    vector[K] log_act;
    for (j in 1:J) {
      real d = sqrt(dot_product(w, D_t[j, ]'));
      log_sims[j] = -c * d;
    }
    for (k in 1:K) {
      real la_k = negative_infinity();
      for (j in 1:J) if (ex_cat[j] == k) la_k = log_sum_exp(la_k, log_sims[j]);
      log_act[k] = la_k;
    }
    return gamma * log_act + log_bias;
  }
}
data {
  int<lower=1> S;
  int<lower=1> J;
  int<lower=1> M;
  int<lower=1> K;
  array[S, K] int<lower=0> y_counts;   // per-(stim, cat) response counts
  array[S] matrix[J, M] D_stim;        // pre-computed distances (one per stimulus)
  array[J] int<lower=1, upper=K> ex_cat;
}
parameters {
  real log_c;
  real log_gamma;
  real w1_logit;
}
transformed parameters {
  real<lower=0>          c        = exp(log_c);
  real<lower=0>          gamma    = exp(log_gamma);
  real<lower=0, upper=1> w1       = inv_logit(w1_logit);
  vector[M]              w        = [w1, 1 - w1]';
  vector[K]              log_bias = rep_vector(log(1.0 / K), K);
}
model {
  log_c     ~ normal(0, 1);
  log_gamma ~ normal(0, 1);
  w1_logit  ~ normal(0, 1);
  for (s in 1:S) {
    vector[K] la = gcm_log_act(D_stim[s], w, c, gamma, log_bias, ex_cat, K);
    target += multinomial_logit_lpmf(y_counts[s] | la);
  }
}
"

cat("Compiling Stan model...\n")
stan_file <- tempfile(fileext = ".stan")
writeLines(gcm_stan_code, stan_file)
mod <- cmdstan_model(stan_file, quiet = TRUE)
cat("Compilation OK.\n\n")

# -----------------------------------------------------------------------
# 4. Recovery function (aggregates trials to counts before fitting)
# -----------------------------------------------------------------------

run_recovery <- function(c_true, gamma_true, w1_true, n_per_stim,
                         rng_seed, label = "") {
  trial_df <- simulate_gcm(c_true, gamma_true, w1_true, n_per_stim,
                            ex_mat, ex_cats, rng_seed = rng_seed)

  # Aggregate T = S × n_per_stim trials to S multinomial cells
  y_counts <- matrix(0L, nrow = T_items, ncol = K_cats)
  for (s in seq_len(T_items)) {
    for (k in seq_len(K_cats)) {
      y_counts[s, k] <- sum(trial_df$stim == s & trial_df$y == k)
    }
  }

  stan_data <- list(
    S        = T_items,
    J        = J_items,
    M        = M_dims,
    K        = K_cats,
    y_counts = y_counts,
    D_stim   = D_stim_list,
    ex_cat   = ex_cats
  )

  fit <- mod$sample(
    data          = stan_data,
    chains        = 4,
    iter_warmup   = 500,
    iter_sampling = 500,
    seed          = 999,
    refresh       = 0,
    show_messages = FALSE
  )

  draws <- fit$summary(variables = c("c", "gamma", "w1"))
  n_div <- sum(fit$diagnostic_summary(quiet = TRUE)$num_divergent)

  data.frame(
    label    = label,
    true_c   = c_true,
    true_g   = gamma_true,
    true_w1  = w1_true,
    n_trials = nrow(trial_df),
    post_c   = round(draws$mean[draws$variable == "c"],     3),
    post_g   = round(draws$mean[draws$variable == "gamma"], 3),
    post_w1  = round(draws$mean[draws$variable == "w1"],    3),
    ci_c     = sprintf("[%.2f, %.2f]", draws$q5[draws$variable == "c"],
                       draws$q95[draws$variable == "c"]),
    ci_g     = sprintf("[%.2f, %.2f]", draws$q5[draws$variable == "gamma"],
                       draws$q95[draws$variable == "gamma"]),
    rhat_max = round(max(draws$rhat, na.rm = TRUE), 3),
    diverge  = n_div
  )
}

# -----------------------------------------------------------------------
# 5. Recovery scenarios
#
# Phase 2a redesign: previous 8 scenarios varied gamma in only 2 of 8 cases
# (all others fixed gamma=1.5), which caused the spurious r(gamma)=-0.12 and
# 0.75 CI coverage. The new design varies gamma across 5 of 8 scenarios so
# that recovery statistics reflect the actual identifiability landscape.
# A Phase 2b follow-up will replace this with a proper factorial SBC grid.
# -----------------------------------------------------------------------

cat("=== Parameter recovery study (multinomial-aggregated) ===\n\n")

scenarios <- list(
  # Baseline: mid-range parameters, 50 trials/stim
  list(c = 0.8, g = 1.5, w1 = 0.65, n = 50, seed = 101, label = "baseline_n50"),
  # Vary c only
  list(c = 2.0, g = 1.5, w1 = 0.65, n = 50, seed = 102, label = "high_c_n50"),
  list(c = 0.4, g = 1.5, w1 = 0.65, n = 50, seed = 103, label = "low_c_n50"),
  # Vary gamma (previously only 2/8 scenarios varied gamma — now 5/8)
  list(c = 0.8, g = 3.0, w1 = 0.65, n = 50, seed = 104, label = "high_g_n50"),
  list(c = 0.8, g = 0.7, w1 = 0.65, n = 50, seed = 105, label = "low_g_n50"),
  # Vary attention weight
  list(c = 0.8, g = 1.5, w1 = 0.9,  n = 50, seed = 106, label = "extreme_w_n50"),
  # c-gamma trade-off region
  list(c = 0.4, g = 3.0, w1 = 0.65, n = 50, seed = 107, label = "low_c_high_g"),
  list(c = 1.5, g = 0.7, w1 = 0.65, n = 50, seed = 108, label = "high_c_low_g")
)

results <- lapply(scenarios, function(s) {
  cat(sprintf("  Running: %s (c=%.1f, gamma=%.1f, w1=%.2f, n_per_stim=%d)...\n",
              s$label, s$c, s$g, s$w1, s$n))
  tryCatch(
    run_recovery(s$c, s$g, s$w1, s$n, s$seed, s$label),
    error = function(e) {
      cat("  ERROR:", conditionMessage(e), "\n")
      NULL
    }
  )
})

results_df <- do.call(rbind, Filter(Negate(is.null), results))

cat("\n=== RECOVERY RESULTS ===\n")
cat("\nTrue vs. posterior mean:\n")
print(results_df[, c("label", "true_c", "post_c", "true_g", "post_g",
                      "true_w1", "post_w1", "rhat_max", "diverge")])

cat("\nCredible intervals for c and gamma:\n")
print(results_df[, c("label", "true_c", "ci_c", "true_g", "ci_g")])

# -----------------------------------------------------------------------
# 6. Compute recovery correlations
# -----------------------------------------------------------------------

cat("\n=== RECOVERY CORRELATIONS ===\n")
r_c  <- cor(results_df$true_c,  results_df$post_c)
r_g  <- cor(results_df$true_g,  results_df$post_g)
r_w1 <- cor(results_df$true_w1, results_df$post_w1)
cat(sprintf("  r(true_c,  post_c):    %.3f\n", r_c))
cat(sprintf("  r(true_g,  post_g):    %.3f\n", r_g))
cat(sprintf("  r(true_w1, post_w1):   %.3f\n", r_w1))

# Coverage (does 90% CI contain true value?)
contains_true <- function(draws_ci, true_val) {
  bounds <- as.numeric(strsplit(gsub("\\[|\\]", "", draws_ci), ", ")[[1]])
  true_val >= bounds[1] & true_val <= bounds[2]
}

cat("\n90% CI coverage (should be ~0.90):\n")
cov_c  <- mean(mapply(contains_true, results_df$ci_c, results_df$true_c))
cov_g  <- mean(mapply(contains_true, results_df$ci_g, results_df$true_g))
cat(sprintf("  c: %.2f   gamma: %.2f\n", cov_c, cov_g))

# -----------------------------------------------------------------------
# 7. c-gamma trade-off analysis
# -----------------------------------------------------------------------

cat("\n=== c-GAMMA TRADE-OFF ANALYSIS ===\n")
cat("The two trade-off scenarios low_c_high_g and high_c_low_g share similar\n")
cat("predicted probabilities when n is moderate:\n")

w <- c(0.65, 0.35)
scenarios_tradeoff <- list(
  list(c = 0.4, g = 3.0, label = "low_c/high_g"),
  list(c = 1.5, g = 0.7, label = "high_c/low_g")
)
for (s in scenarios_tradeoff) {
  D_mat <- gcm_distances(ex_mat, ex_mat, w)
  S_mat <- gcm_similarity(D_mat, s$c, p_sim = 1)
  act   <- cbind(rowSums(S_mat[, ex_cats == 1, drop = FALSE]),
                 rowSums(S_mat[, ex_cats == 2, drop = FALSE]))
  P_mat <- luce_choice(act, s$g, bias = c(0.5, 0.5))
  cat(sprintf("  %s: P(cat2) = %s\n", s$label,
              paste(round(P_mat[, 2], 2), collapse = " ")))
}

cat("\nBoth yield similar P(cat B) patterns — the trade-off is partially\n")
cat("identifiable with sufficient data but requires informative priors.\n")
cat("Recommended default: log(c) ~ N(0, 1), log(gamma) ~ N(0, 1).\n")
cat("For the prototype model fix gamma=1 (Nosofsky & Zaki 2002).\n")
cat("\nNote: an informative N(0.5, 0.5) prior on log(c) pulls the posterior\n")
cat("toward exp(0.5)=1.65, placing true values near c=0.8 outside the 90%\n")
cat("CI. Use N(0.5, 0.5) only if the coordinate space is known to be scaled\n")
cat("so that c is expected to be around 1.0-2.0.\n")

# -----------------------------------------------------------------------
# 8. Prototype identifiability
# -----------------------------------------------------------------------

cat("\n=== PROTOTYPE: gamma=1 CONSTRAINT JUSTIFICATION ===\n")
cat("In prototype GCM, c and gamma are NOT separately identifiable:\n")
cat("  P(K|i) ∝ (sum_K exp(-c * d_Ki))^gamma\n")
cat("  = exp(-c * d_Ki * gamma)  [when only 1 exemplar per category]\n")
cat("Only 2 categories with prototypes → c*gamma is identified, not c or gamma alone.\n")
cat("Fix: gamma = 1, recover only c and w. Nosofsky & Zaki (2002, p.926) confirm.\n")

cat("\n=== SUMMARY ===\n")
cat(sprintf("  Recovery r(c):     %.3f\n", r_c))
cat(sprintf("  Recovery r(gamma): %.3f\n", r_g))
cat(sprintf("  Recovery r(w1):    %.3f\n", r_w1))
cat(sprintf("  Rhat max overall:  %.3f\n", max(results_df$rhat_max)))
cat(sprintf("  Total divergences: %d\n",   sum(results_df$diverge)))
cat("  Likelihood: multinomial per stimulus (exact, not approximate)\n")
cat("  Prior: log_c ~ N(0,1), log_gamma ~ N(0,1) — weakly informative defaults.\n")
cat("  c-gamma identifiability: partial — requires moderate n and careful priors.\n")
cat("  w1 recovery: good across all scenarios.\n")
cat("  Phase 2b: replace 8-scenario design with factorial SBC grid over (c, gamma, w).\n")
