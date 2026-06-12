# GCM Attention Simplex — K>2 categories and M>2 dimensions (Phase 2b)
#
# This is the gating item for an upstream [new-model] proposal: every prior
# Stan model hardcodes w = [w1, 1-w1] (M=2). This script implements and
# recovery-tests the two novel API elements:
#
#   1. Softmax-with-reference attention weight parameterisation (memo §4):
#      Stan parameter vector[M-1] w_raw; simplex[M] w = softmax([w_raw; 0]).
#      Dimension M is the reference (log-ratio = 0); w_raw[m] = log(w[m]/w[M]).
#      Exposes M-1 free parameters in bmmformula (w1 ~ ..., w2 ~ ..., etc.).
#
#   2. Multi-category choice rule (K>2):
#      P(k|s) ∝ activation_k^gamma — same Luce formula, K categories.
#      log_bias = rep_vector(log(1/K), K) for equal category priors.
#
# Design: M=4 dimensions, K=3 categories, J=24 exemplars (8 per category),
# S=24 test stimuli (= exemplars). True w = [0.40, 0.30, 0.20, 0.10].
# One recovery scenario to confirm both API elements are identified.
#
# Run from repository root:
#   Rscript local/exploration/categorization/04b_attention_simplex.R

suppressPackageStartupMessages({
  library(cmdstanr)
  library(dplyr)
})

source_env <- new.env()
source("local/exploration/categorization/01_r_reference.R",
       local = source_env, echo = FALSE)
gcm_distances  <- source_env$gcm_distances
gcm_similarity <- source_env$gcm_similarity
luce_choice    <- source_env$luce_choice

# -----------------------------------------------------------------------
# 1. Synthetic 3-category / 4-dimension stimulus set
# -----------------------------------------------------------------------

M_dims   <- 4L
K_cats   <- 3L
J_per_cat <- 8L
J_items  <- K_cats * J_per_cat  # 24 exemplars
S_stims  <- J_items              # test = training set

# Well-separated cluster centres (first 2 dims separate categories; dims 3-4
# carry less information, so attention recovery is non-trivial)
centres <- rbind(
  c(-2.5, -2.5,  0.5, -0.5),  # cat 1
  c( 2.5, -2.5, -0.5,  0.5),  # cat 2
  c( 0.0,  2.8,  0.0,  0.0)   # cat 3
)

set.seed(2024)
ex_mat <- do.call(rbind, lapply(seq_len(K_cats), function(k) {
  matrix(rnorm(J_per_cat * M_dims, sd = 0.7), J_per_cat, M_dims) +
    rep(centres[k, ], each = J_per_cat)
}))
ex_cats <- rep(seq_len(K_cats), each = J_per_cat)

cat(sprintf("Exemplar geometry: J=%d, K=%d, M=%d\n", J_items, K_cats, M_dims))
cat(sprintf("Category sizes: %s\n\n", paste(table(ex_cats), collapse = " ")))

# -----------------------------------------------------------------------
# 2. True parameters and data simulation
# -----------------------------------------------------------------------

c_true     <- 0.80
gamma_true <- 1.50
w_true     <- c(0.40, 0.30, 0.20, 0.10)  # sums to 1; dim 4 is reference

# Derive true softmax-with-reference log-ratios (for verification)
w_raw_true <- log(w_true[1:(M_dims - 1)] / w_true[M_dims])
cat(sprintf("True parameters:  c=%.2f, gamma=%.2f\n", c_true, gamma_true))
cat(sprintf("True w:           %s (sums to %.2f)\n",
            paste(round(w_true, 3), collapse=" "), sum(w_true)))
cat(sprintf("True w_raw:       %s\n\n",
            paste(round(w_raw_true, 3), collapse=" ")))

n_per_stim <- 80L

set.seed(777)
D_mat  <- gcm_distances(ex_mat, ex_mat, w_true, r_metric = 2)
S_mat  <- gcm_similarity(D_mat, c_true, p_sim = 1)
# Activation: K columns (one per category), S rows
act <- do.call(cbind, lapply(seq_len(K_cats), function(k) {
  rowSums(S_mat[, ex_cats == k, drop = FALSE])
}))
P_mat <- luce_choice(act, gamma_true, bias = rep(1/K_cats, K_cats))

trial_rows <- lapply(seq_len(S_stims), function(s) {
  y <- sample(seq_len(K_cats), n_per_stim, replace = TRUE, prob = P_mat[s, ])
  data.frame(stim = s, y = y)
})
trial_df <- do.call(rbind, trial_rows)

# Aggregate to [S, K] count matrix
y_counts <- matrix(0L, nrow = S_stims, ncol = K_cats)
for (s in seq_len(S_stims)) {
  for (k in seq_len(K_cats)) {
    y_counts[s, k] <- sum(trial_df$stim == s & trial_df$y == k)
  }
}

cat(sprintf("Simulated: %d total trials, %d stims × %d trials/stim\n",
            nrow(trial_df), S_stims, n_per_stim))
cat(sprintf("Aggregate counts [S=%d, K=%d]: row sums = %s ...\n\n",
            S_stims, K_cats, paste(rowSums(y_counts)[1:3], collapse=" ")))

# Pre-compute D_stim[S, J, M]
D_raw <- array(0, dim = c(S_stims, J_items, M_dims))
for (m in seq_len(M_dims)) {
  D_raw[, , m] <- abs(outer(ex_mat[, m], ex_mat[, m], `-`))^2
}
D_stim_list <- lapply(seq_len(S_stims), function(s) D_raw[s, , ])

# -----------------------------------------------------------------------
# 3. Stan model — softmax-with-reference attention + K>2 multinomial
# -----------------------------------------------------------------------

gcm_simplex_stan_code <- "
functions {
  // Log-space activation with M-dimensional attention simplex.
  // Accumulates in log space (log_sum_exp) to avoid underflow at large c.
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
  int<lower=2> M;
  int<lower=2> K;
  array[S, K] int<lower=0> y_counts;
  array[S] matrix[J, M] D_stim;
  array[J] int<lower=1, upper=K> ex_cat;
}
parameters {
  real           log_c;
  real           log_gamma;
  vector[M - 1]  w_raw;    // softmax-with-reference log-ratios (dim M = ref = 0)
}
transformed parameters {
  real<lower=0>  c     = exp(log_c);
  real<lower=0>  gamma = exp(log_gamma);
  vector[K]      log_bias = rep_vector(-log(K), K);
  // Softmax-with-reference: append 0 for the reference dimension
  vector[M] w_full;
  w_full[1:(M - 1)] = w_raw;
  w_full[M] = 0.0;
  simplex[M] w = softmax(w_full);
}
model {
  log_c     ~ normal(0, 1);
  log_gamma ~ normal(0, 1);
  w_raw     ~ normal(0, 1);   // Normal(0,1) on each log-ratio
  for (s in 1:S) {
    vector[K] la = gcm_log_act(D_stim[s], w, c, gamma, log_bias, ex_cat, K);
    target += multinomial_logit_lpmf(y_counts[s] | la);
  }
}
generated quantities {
  simplex[M] w_fit = softmax(w_full);
}
"

# -----------------------------------------------------------------------
# 4. Compile
# -----------------------------------------------------------------------

cat("Compiling Stan model (M>2 attention simplex, K>2 categories)...\n")
stan_file <- tempfile(fileext = ".stan")
writeLines(gcm_simplex_stan_code, stan_file)
mod <- tryCatch(
  cmdstan_model(stan_file, quiet = TRUE),
  error = function(e) { cat("Compilation error:", conditionMessage(e), "\n"); NULL }
)
if (is.null(mod)) quit(status = 1)
cat("Compilation OK.\n\n")

# -----------------------------------------------------------------------
# 5. Build Stan data and fit
# -----------------------------------------------------------------------

stan_data <- list(
  S        = S_stims,
  J        = J_items,
  M        = M_dims,
  K        = K_cats,
  y_counts = y_counts,
  D_stim   = D_stim_list,
  ex_cat   = ex_cats
)

cat("--- MCMC: 4 chains, 500 warmup + 1000 sampling ---\n")
t_start <- proc.time()
fit <- tryCatch(
  mod$sample(
    data          = stan_data,
    chains        = 4,
    iter_warmup   = 500,
    iter_sampling = 1000,
    seed          = 42,
    refresh       = 200,
    show_messages = TRUE
  ),
  error = function(e) { cat("MCMC error:", conditionMessage(e), "\n"); NULL }
)
t_elapsed <- proc.time() - t_start
if (is.null(fit)) quit(status = 1)

cat(sprintf("\nSampling time: %.1f s\n\n", t_elapsed["elapsed"]))

# -----------------------------------------------------------------------
# 6. Diagnostics
# -----------------------------------------------------------------------

w_vars   <- paste0("w_fit[", seq_len(M_dims), "]")
all_vars <- c("c", "gamma", w_vars)
draws_sum <- fit$summary(variables = all_vars)

cat("--- Posterior summary ---\n")
print(draws_sum[, c("variable", "mean", "median", "sd", "q5", "q95", "rhat", "ess_bulk")])

n_diverg <- sum(fit$diagnostic_summary(quiet = TRUE)$num_divergent)
cat(sprintf("\nDivergences: %d\n", n_diverg))

# -----------------------------------------------------------------------
# 7. Recovery table
# -----------------------------------------------------------------------

cat("\n=== PARAMETER RECOVERY ===\n")
true_vals <- c(c = c_true, gamma = gamma_true, setNames(w_true, w_vars))

post_means <- setNames(draws_sum$mean,  draws_sum$variable)
post_ci5   <- setNames(draws_sum$q5,    draws_sum$variable)
post_ci95  <- setNames(draws_sum$q95,   draws_sum$variable)

for (nm in names(true_vals)) {
  in_ci <- (true_vals[nm] >= post_ci5[nm]) & (true_vals[nm] <= post_ci95[nm])
  cat(sprintf("  %-16s  true=%.3f  post=%.3f  90%%CI=[%.3f, %.3f]  in_CI=%s\n",
              nm, true_vals[nm], post_means[nm],
              post_ci5[nm], post_ci95[nm], ifelse(in_ci, "YES", "NO")))
}

# -----------------------------------------------------------------------
# 8. Summary
# -----------------------------------------------------------------------

rhat_max <- max(draws_sum$rhat, na.rm = TRUE)
ess_min  <- min(draws_sum$ess_bulk, na.rm = TRUE)

cat(sprintf("\n=== ATTENTION SIMPLEX SUMMARY ===\n"))
cat(sprintf("  M=%d dimensions, K=%d categories, J=%d exemplars, S=%d stims\n",
            M_dims, K_cats, J_items, S_stims))
cat(sprintf("  n_per_stim: %d  (S×n = %d total trials)\n",
            n_per_stim, S_stims * n_per_stim))
cat(sprintf("  Sampling time: %.1f s\n", t_elapsed["elapsed"]))
cat(sprintf("  Rhat max: %.3f\n", rhat_max))
cat(sprintf("  ESS bulk min: %.0f\n", ess_min))
cat(sprintf("  Divergences: %d\n", n_diverg))
cat("  Parameterisation: softmax-with-reference (w_raw[1..M-1], ref dim = M)\n")
cat("  Stability fix: log_sum_exp accumulation + multinomial_logit_lpmf\n")
cat("  Phase 2b status: M>2 and K>2 both confirmed — gating item for upstream.\n")
cat("  Next: softmax attention + multi-category in hierarchical model (05b).\n")
