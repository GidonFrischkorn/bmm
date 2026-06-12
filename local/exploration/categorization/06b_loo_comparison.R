# GCM vs Prototype vs PRM — LOO model comparison (Phase 2b)
#
# Addresses two Phase 2b gaps in 06_realdata_fit.R:
#   1. PRM rote-memory component was unexercised (is_old = all-FALSE → PRM ≡ prototype).
#      This script sets is_old = TRUE for all 12 nosof88 training stimuli, giving
#      PRM a genuine old-item signal to work with.
#   2. No model comparison was performed. Here GCM, prototype, and PRM are
#      each fitted to the same data and compared via LOO-IC (loo package).
#
# Note on leave-one-cell-out LOO: after multinomial aggregation, each "observation"
# is one (stimulus × category) count cell. LOO-IC is therefore leave-one-cell-out,
# not per-trial. This is a coarser unit but is theoretically valid for model
# comparison — the scale difference from per-trial LOO is noted in the output.
#
# All three Stan models use:
#   - Log-space activation (log_sum_exp; eliminates softmax-sum-NaN at large c)
#   - multinomial_logit_lpmf (GCM, prototype) / multinomial_lpmf (PRM — needs
#     explicit simplex because the PRM mixing breaks the logit form)
#   - log_lik[s] in generated quantities for loo()
#
# Run from repository root:
#   Rscript local/exploration/categorization/06b_loo_comparison.R

suppressPackageStartupMessages({
  library(cmdstanr)
  library(catlearn)
  library(dplyr)
  library(loo)
})

source_env <- new.env()
source("local/exploration/categorization/01_r_reference.R",
       local = source_env, echo = FALSE)
gcm_distances  <- source_env$gcm_distances
gcm_similarity <- source_env$gcm_similarity
luce_choice    <- source_env$luce_choice

# -----------------------------------------------------------------------
# 1. Stimulus geometry, observed data, and count matrix
# -----------------------------------------------------------------------

stim_coords <- data.frame(
  stim = 1:12,
  x1 = c(-2.543,  0.943, -1.092,  1.558, -2.258,  0.194,
           2.806, -1.177,  1.543, -2.775,  0.528,  1.709),
  x2 = c( 2.641,  4.341,  1.848,  2.902,  0.430,  0.572,
           0.202, -1.038, -1.040, -3.149, -3.766, -3.773),
  cat = c(1L, 2L, 2L, 2L, 1L, 2L, 2L, 1L, 2L, 1L, 1L, 1L)
)

ex_mat  <- as.matrix(stim_coords[, c("x1", "x2")])
ex_cats <- stim_coords$cat
S <- nrow(stim_coords)
J <- S
M <- 2
K <- 2

# Pre-computed distance array [S, J, M]
D_raw <- array(0, dim = c(S, J, M))
for (m in seq_len(M)) {
  D_raw[, , m] <- abs(outer(ex_mat[, m], ex_mat[, m], `-`))^2
}
D_stim_list <- lapply(seq_len(S), function(s) D_raw[s, , ])

# nosof88 condition 1 (balanced), aggregate P(cat B)
obs <- catlearn::nosof88 |> filter(cond == 1) |> arrange(stim)

# Synthesise count data from aggregate proportions (n=50 per stimulus)
n_per_stim <- 50L
y_counts <- matrix(0L, nrow = S, ncol = K)
y_counts[, 2] <- round(obs$c2acc * n_per_stim)
y_counts[, 1] <- n_per_stim - y_counts[, 2]

# is_old = TRUE for all 12 stimuli (all are training items in nosof88).
# This exercises PRM's rote-memory component on every stimulus — previously
# is_old was all-FALSE, collapsing PRM to prototype.
is_old   <- rep(1L, S)
true_cat <- stim_coords$cat

cat(sprintf("Data: S=%d stim, K=%d cats, n=%d/stim, is_old=TRUE for all stim\n\n",
            S, K, n_per_stim))

# Category prototypes for prototype / PRM models
proto_mat <- do.call(rbind, lapply(seq_len(K), function(k) {
  colMeans(ex_mat[ex_cats == k, , drop = FALSE])
}))
D_proto <- array(0, dim = c(S, K, M))
for (m in seq_len(M)) {
  D_proto[, , m] <- abs(outer(ex_mat[, m], proto_mat[, m], `-`))^2
}
D_proto_list <- lapply(seq_len(S), function(s) D_proto[s, , ])

# -----------------------------------------------------------------------
# 2. Stan model A — GCM (exemplar-based, aggregated)
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
  array[S, K] int<lower=0> y_counts;
  array[S] matrix[J, M] D_stim;
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
generated quantities {
  vector[S] log_lik;
  for (s in 1:S) {
    vector[K] la = gcm_log_act(D_stim[s], w, c, gamma, log_bias, ex_cat, K);
    log_lik[s] = multinomial_logit_lpmf(y_counts[s] | la);
  }
}
"

# -----------------------------------------------------------------------
# 3. Stan model B — Prototype (gamma=1, exemplars = K prototypes)
# -----------------------------------------------------------------------

proto_stan_code <- "
functions {
  // Prototype model: each exemplar IS a category prototype (J = K here).
  // gamma fixed to 1.
  vector proto_log_act(matrix D_t, vector w, real c,
                       vector log_bias, int K) {
    vector[K] log_act;
    for (k in 1:K) {
      real d = sqrt(dot_product(w, D_t[k, ]'));
      log_act[k] = -c * d;
    }
    return log_act + log_bias;
  }
}
data {
  int<lower=1> S;
  int<lower=1> M;
  int<lower=1> K;
  array[S, K] int<lower=0> y_counts;
  array[S] matrix[K, M] D_proto;
}
parameters {
  real log_c;
  real w1_logit;
}
transformed parameters {
  real<lower=0>          c        = exp(log_c);
  real<lower=0, upper=1> w1       = inv_logit(w1_logit);
  vector[M]              w        = [w1, 1 - w1]';
  vector[K]              log_bias = rep_vector(log(1.0 / K), K);
}
model {
  log_c    ~ normal(0, 1);
  w1_logit ~ normal(0, 1);
  for (s in 1:S) {
    vector[K] la = proto_log_act(D_proto[s], w, c, log_bias, K);
    target += multinomial_logit_lpmf(y_counts[s] | la);
  }
}
generated quantities {
  vector[S] log_lik;
  for (s in 1:S) {
    vector[K] la = proto_log_act(D_proto[s], w, c, log_bias, K);
    log_lik[s] = multinomial_logit_lpmf(y_counts[s] | la);
  }
}
"

# -----------------------------------------------------------------------
# 4. Stan model C — PRM (Prototype Recognition Memory)
#
# For old stimuli:
#   P(k|s) = p_mem * I(k = true_cat[s]) + (1 - p_mem) * P_prototype(k|s)
# For new stimuli (is_old = 0):
#   P(k|s) = P_prototype(k|s)
#
# The mixing happens in probability space (not log-space), which means
# multinomial_lpmf must be used (not multinomial_logit_lpmf) — the mixed
# probabilities are a valid simplex but not in standard logit form.
# -----------------------------------------------------------------------

prm_stan_code <- "
functions {
  vector proto_log_act(matrix D_t, vector w, real c,
                       vector log_bias, int K) {
    vector[K] log_act;
    for (k in 1:K) {
      real d = sqrt(dot_product(w, D_t[k, ]'));
      log_act[k] = -c * d;
    }
    return log_act + log_bias;
  }
}
data {
  int<lower=1> S;
  int<lower=1> M;
  int<lower=1> K;
  array[S, K]  int<lower=0>           y_counts;
  array[S]     matrix[K, M]           D_proto;
  array[S]     int<lower=0, upper=1>  is_old;
  array[S]     int<lower=1, upper=K>  true_cat;
}
parameters {
  real                   log_c;
  real                   w1_logit;
  real<lower=0, upper=1> p_mem;
}
transformed parameters {
  real<lower=0>          c        = exp(log_c);
  real<lower=0, upper=1> w1       = inv_logit(w1_logit);
  vector[M]              w        = [w1, 1 - w1]';
  vector[K]              log_bias = rep_vector(log(1.0 / K), K);
}
model {
  log_c    ~ normal(0, 1);
  w1_logit ~ normal(0, 1);
  p_mem    ~ beta(1, 1);  // uniform prior on rote-memory proportion
  for (s in 1:S) {
    vector[K] la = proto_log_act(D_proto[s], w, c, log_bias, K);
    vector[K] p = softmax(la);
    if (is_old[s]) {
      // Mix rote-memory (delta on correct category) with prototype
      p = (1 - p_mem) * p;
      p[true_cat[s]] += p_mem;
    }
    target += multinomial_lpmf(y_counts[s] | p);
  }
}
generated quantities {
  vector[S] log_lik;
  real      p_mem_out = p_mem;  // rename for clear output
  for (s in 1:S) {
    vector[K] la = proto_log_act(D_proto[s], w, c, log_bias, K);
    vector[K] p = softmax(la);
    if (is_old[s]) {
      p = (1 - p_mem) * p;
      p[true_cat[s]] += p_mem;
    }
    log_lik[s] = multinomial_lpmf(y_counts[s] | p);
  }
}
"

# -----------------------------------------------------------------------
# 5. Compile all three models
# -----------------------------------------------------------------------

compile_model <- function(code, label) {
  cat(sprintf("Compiling %s...\n", label))
  f <- tempfile(fileext = ".stan")
  writeLines(code, f)
  m <- tryCatch(cmdstan_model(f, quiet = TRUE),
                error = function(e) { cat("Error:", conditionMessage(e), "\n"); NULL })
  if (!is.null(m)) cat(sprintf("  %s OK\n", label))
  m
}

mod_gcm   <- compile_model(gcm_stan_code,   "GCM")
mod_proto <- compile_model(proto_stan_code, "Prototype")
mod_prm   <- compile_model(prm_stan_code,   "PRM")

if (any(sapply(list(mod_gcm, mod_proto, mod_prm), is.null))) {
  cat("Compilation failed. Exiting.\n")
  quit(status = 1)
}

# -----------------------------------------------------------------------
# 6. Stan data objects
# -----------------------------------------------------------------------

stan_data_gcm <- list(
  S        = S,
  J        = J,
  M        = M,
  K        = K,
  y_counts = y_counts,
  D_stim   = D_stim_list,
  ex_cat   = ex_cats
)

stan_data_proto <- list(
  S        = S,
  M        = M,
  K        = K,
  y_counts = y_counts,
  D_proto  = D_proto_list
)

stan_data_prm <- c(
  stan_data_proto,
  list(
    is_old   = is_old,
    true_cat = true_cat
  )
)

# -----------------------------------------------------------------------
# 7. Fit all three models
# -----------------------------------------------------------------------

fit_model <- function(mod, data, label, seed) {
  cat(sprintf("\nFitting %s (4 chains × 1000 draws)...\n", label))
  t0 <- proc.time()
  fit <- tryCatch(
    mod$sample(
      data          = data,
      chains        = 4,
      iter_warmup   = 500,
      iter_sampling = 1000,
      seed          = seed,
      refresh       = 0,
      show_messages = FALSE
    ),
    error = function(e) { cat("Error:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(fit)) return(NULL)
  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("  Done in %.1f s | diverg: %d | Rhat max: %.3f\n",
              elapsed,
              sum(fit$diagnostic_summary(quiet = TRUE)$num_divergent),
              max(fit$summary()$rhat, na.rm = TRUE)))
  fit
}

fit_gcm   <- fit_model(mod_gcm,   stan_data_gcm,   "GCM",       seed = 101)
fit_proto <- fit_model(mod_proto, stan_data_proto, "Prototype", seed = 102)
fit_prm   <- fit_model(mod_prm,   stan_data_prm,   "PRM",       seed = 103)

# -----------------------------------------------------------------------
# 8. LOO-IC comparison
# -----------------------------------------------------------------------

cat("\n=== LOO MODEL COMPARISON (leave-one-cell-out) ===\n")
cat("Note: each observation is one (stimulus, category) count cell,\n")
cat("  not one individual trial. LOO-IC values are on the cell scale.\n\n")

extract_loo <- function(fit, label) {
  if (is.null(fit)) return(NULL)
  ll_draws <- fit$draws("log_lik")
  loo_res  <- loo(ll_draws, r_eff = relative_eff(exp(ll_draws)))
  cat(sprintf("  %s: ELPD_loo=%.1f (SE=%.1f)  p_loo=%.1f  LOO-IC=%.1f\n",
              label,
              loo_res$estimates["elpd_loo",  "Estimate"],
              loo_res$estimates["elpd_loo",  "SE"],
              loo_res$estimates["p_loo",     "Estimate"],
              loo_res$estimates["looic",     "Estimate"]))
  list(label = label, loo = loo_res)
}

loo_gcm   <- extract_loo(fit_gcm,   "GCM")
loo_proto <- extract_loo(fit_proto, "Prototype")
loo_prm   <- extract_loo(fit_prm,   "PRM")

# Pairwise comparison
loo_list <- Filter(Negate(is.null),
                   list(GCM = loo_gcm$loo, Prototype = loo_proto$loo, PRM = loo_prm$loo))
if (length(loo_list) >= 2) {
  cat("\nPairwise LOO difference (GCM as reference):\n")
  for (nm in setdiff(names(loo_list), "GCM")) {
    cmp <- loo_compare(loo_list[["GCM"]], loo_list[[nm]])
    cat(sprintf("  GCM vs %s: ΔELPD=%.1f (SE=%.1f)\n",
                nm,
                cmp[2, "elpd_diff"],
                cmp[2, "se_diff"]))
  }
}

# -----------------------------------------------------------------------
# 9. PRM p_mem estimate — confirms rote-memory component is exercised
# -----------------------------------------------------------------------

if (!is.null(fit_prm)) {
  cat("\n=== PRM ROTE-MEMORY COMPONENT ===\n")
  p_mem_sum <- fit_prm$summary("p_mem_out")
  cat(sprintf("  p_mem: mean=%.3f  90%%CI=[%.3f, %.3f]\n",
              p_mem_sum$mean, p_mem_sum$q5, p_mem_sum$q95))
  cat("  Interpretation: proportion of responses attributed to rote recall.\n")
  cat("  CI clearly above 0 → rote component is exercised (PRM ≠ prototype).\n")
}

# -----------------------------------------------------------------------
# 10. Parameter summary across models
# -----------------------------------------------------------------------

cat("\n=== PARAMETER ESTIMATES BY MODEL ===\n")
summarise_params <- function(fit, vars, label) {
  if (is.null(fit)) return(invisible(NULL))
  s <- fit$summary(vars)
  cat(sprintf("\n  %s:\n", label))
  for (i in seq_len(nrow(s))) {
    cat(sprintf("    %-12s  mean=%.3f  90%%CI=[%.3f, %.3f]  Rhat=%.3f\n",
                s$variable[i], s$mean[i], s$q5[i], s$q95[i], s$rhat[i]))
  }
}

summarise_params(fit_gcm,   c("c", "gamma", "w1"), "GCM")
summarise_params(fit_proto, c("c", "w1"),           "Prototype (gamma=1)")
summarise_params(fit_prm,   c("c", "w1", "p_mem"),  "PRM")

# -----------------------------------------------------------------------
# 11. Summary
# -----------------------------------------------------------------------

cat("\n=== LOO COMPARISON SUMMARY ===\n")
cat("Expected ordering (Nosofsky et al. 2022): GCM < PRM < Prototype (BIC).\n")
cat("LOO-IC here is on cell scale (12 cells) vs BIC on trial scale — ranks\n")
cat("should agree in direction. Absolute values are not comparable.\n")
cat("PRM p_mem > 0 confirms the rote-memory component is identified.\n")
cat("Phase 2b: repeat on nosof94 / OSF rocks data for trial-scale comparison.\n")
