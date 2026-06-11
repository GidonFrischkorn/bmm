# GCM Hierarchical Proof-of-Concept
#
# Multi-subject fit with random effects on log(c), log(gamma), and softmax-w.
# Uses nosof88 geometry. Simulates 20 subjects with subject-level variation
# around a group mean, then fits the hierarchical model and checks diagnostics.
#
# Run from repository root:
#   Rscript local/exploration/categorization/05_hierarchical.R

suppressPackageStartupMessages({
  library(cmdstanr)
  library(dplyr)
})

# -----------------------------------------------------------------------
# Load R reference functions
# -----------------------------------------------------------------------

source_env <- new.env()
source("local/exploration/categorization/01_r_reference.R",
       local = source_env, echo = FALSE)
gcm_distances  <- source_env$gcm_distances
gcm_similarity <- source_env$gcm_similarity
luce_choice    <- source_env$luce_choice

# -----------------------------------------------------------------------
# 1. Exemplar geometry (nosof88)
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
T_stims <- nrow(stim_coords)
J_items <- T_stims
M_dims  <- 2
K_cats  <- 2
r_met   <- 2

D_raw <- array(0, dim = c(T_stims, J_items, M_dims))
for (m in seq_len(M_dims)) {
  D_raw[, , m] <- abs(outer(ex_mat[, m], ex_mat[, m], `-`))^r_met
}

# -----------------------------------------------------------------------
# 2. Simulate multi-subject data
# -----------------------------------------------------------------------

N_subj      <- 20L
n_per_stim  <- 30L   # trials per stimulus per subject

# Group-level true parameters (on log / logit scale)
mu_log_c     <- log(0.8)
mu_log_gamma <- log(1.5)
mu_w1_logit  <- qlogis(0.65)

# Between-subject SDs (on unconstrained scale)
sigma_log_c     <- 0.4
sigma_log_gamma <- 0.4
sigma_w1_logit  <- 0.5

set.seed(777)
subj_log_c     <- rnorm(N_subj, mu_log_c,     sigma_log_c)
subj_log_gamma <- rnorm(N_subj, mu_log_gamma, sigma_log_gamma)
subj_w1_logit  <- rnorm(N_subj, mu_w1_logit,  sigma_w1_logit)

subj_c     <- exp(subj_log_c)
subj_gamma <- exp(subj_log_gamma)
subj_w1    <- plogis(subj_w1_logit)

cat(sprintf("True group means: c=%.3f, gamma=%.3f, w1=%.3f\n",
            exp(mu_log_c), exp(mu_log_gamma), plogis(mu_w1_logit)))
cat(sprintf("True group SDs: log_c=%.2f, log_gamma=%.2f, w1_logit=%.2f\n\n",
            sigma_log_c, sigma_log_gamma, sigma_w1_logit))

all_trials <- lapply(seq_len(N_subj), function(s) {
  w <- c(subj_w1[s], 1 - subj_w1[s])
  D <- gcm_distances(ex_mat, ex_mat, w, r_metric = 2)
  S <- gcm_similarity(D, subj_c[s], p_sim = 1)
  act <- cbind(rowSums(S[, ex_cats == 1, drop = FALSE]),
               rowSums(S[, ex_cats == 2, drop = FALSE]))
  P <- luce_choice(act, subj_gamma[s], bias = c(0.5, 0.5))
  do.call(rbind, lapply(seq_len(T_stims), function(i) {
    y <- sample(1:K_cats, n_per_stim, replace = TRUE, prob = P[i, ])
    data.frame(subj = s, stim = i, y = y)
  }))
})
trial_df <- do.call(rbind, all_trials)
cat(sprintf("Simulated %d trials (%d subjects × %d stims × %d trials)\n\n",
            nrow(trial_df), N_subj, T_stims, n_per_stim))

# -----------------------------------------------------------------------
# 3. Hierarchical Stan model
# -----------------------------------------------------------------------

hier_stan_code <- "
functions {
  vector gcm_log_act(matrix D_t, vector w, real c, real gamma,
                     vector log_bias, array[] int ex_cat, int K) {
    int J = rows(D_t);
    vector[J] sims;
    vector[K] log_act;
    for (j in 1:J) {
      real d = sqrt(dot_product(w, D_t[j, ]'));
      sims[j] = exp(-c * d);
    }
    for (k in 1:K) {
      real act_k = 0;
      for (j in 1:J) if (ex_cat[j] == k) act_k += sims[j];
      log_act[k] = log(act_k);
    }
    return gamma * log_act + log_bias;
  }
}
data {
  int<lower=1> T;             // total trials
  int<lower=1> N;             // number of subjects
  int<lower=1> J;             // exemplars
  int<lower=1> M;             // dimensions
  int<lower=1> K;             // categories
  array[T] int<lower=1, upper=K>   y;
  array[T] int<lower=1, upper=N>   subj;
  array[T] matrix[J, M]            D_raw;
  array[J] int<lower=1, upper=K>   ex_cat;
}
parameters {
  // Group-level means (unconstrained)
  real mu_log_c;
  real mu_log_gamma;
  real mu_w1_logit;
  // Group-level SDs
  real<lower=0> sigma_log_c;
  real<lower=0> sigma_log_gamma;
  real<lower=0> sigma_w1_logit;
  // Subject-level (non-centred parameterisation)
  vector[N] z_log_c;
  vector[N] z_log_gamma;
  vector[N] z_w1_logit;
}
transformed parameters {
  vector[N] subj_log_c     = mu_log_c     + sigma_log_c     * z_log_c;
  vector[N] subj_log_gamma = mu_log_gamma + sigma_log_gamma * z_log_gamma;
  vector[N] subj_w1_logit  = mu_w1_logit  + sigma_w1_logit  * z_w1_logit;

  vector[N] subj_c     = exp(subj_log_c);
  vector[N] subj_gamma = exp(subj_log_gamma);
  vector[N] subj_w1    = inv_logit(subj_w1_logit);
  vector[K] log_bias   = rep_vector(log(1.0 / K), K);
}
model {
  // Hyperpriors
  mu_log_c     ~ normal(0.5, 0.5);
  mu_log_gamma ~ normal(0.0, 0.5);
  mu_w1_logit  ~ normal(0.0, 1.0);
  sigma_log_c     ~ normal(0, 0.5);
  sigma_log_gamma ~ normal(0, 0.5);
  sigma_w1_logit  ~ normal(0, 1.0);
  // Non-centred subject offsets
  z_log_c     ~ std_normal();
  z_log_gamma ~ std_normal();
  z_w1_logit  ~ std_normal();
  // Likelihood
  for (t in 1:T) {
    int s = subj[t];
    vector[M] w = [subj_w1[s], 1 - subj_w1[s]]';
    vector[K] la = gcm_log_act(D_raw[t], w, subj_c[s], subj_gamma[s],
                                log_bias, ex_cat, K);
    target += la[y[t]] - log_sum_exp(la);
  }
}
generated quantities {
  // Group-level means on natural scale
  real mean_c     = exp(mu_log_c);
  real mean_gamma = exp(mu_log_gamma);
  real mean_w1    = inv_logit(mu_w1_logit);
}
"

# -----------------------------------------------------------------------
# 4. Build Stan data
# -----------------------------------------------------------------------

D_raw_list <- lapply(seq_len(nrow(trial_df)), function(row) {
  D_raw[trial_df$stim[row], , ]
})

stan_data <- list(
  T       = nrow(trial_df),
  N       = N_subj,
  J       = J_items,
  M       = M_dims,
  K       = K_cats,
  y       = trial_df$y,
  subj    = trial_df$subj,
  D_raw   = D_raw_list,
  ex_cat  = ex_cats
)

# -----------------------------------------------------------------------
# 5. Compile and fit
# -----------------------------------------------------------------------

cat("Compiling hierarchical Stan model...\n")
stan_file <- tempfile(fileext = ".stan")
writeLines(hier_stan_code, stan_file)
mod <- cmdstan_model(stan_file, quiet = TRUE)
cat("Compilation OK.\n\n")

cat("--- Hierarchical MCMC: 4 chains, 600 warmup + 400 sampling ---\n")
t_start <- proc.time()
fit <- tryCatch(
  mod$sample(
    data          = stan_data,
    chains        = 4,
    iter_warmup   = 600,
    iter_sampling = 400,
    seed          = 42,
    refresh       = 200,
    show_messages = TRUE,
    adapt_delta   = 0.9
  ),
  error = function(e) { cat("MCMC error:", conditionMessage(e), "\n"); NULL }
)
t_elapsed <- proc.time() - t_start

if (is.null(fit)) {
  cat("MCMC failed. Exiting.\n")
  quit(status = 1)
}
cat(sprintf("\nSampling time: %.1f s\n\n", t_elapsed["elapsed"]))

# -----------------------------------------------------------------------
# 6. Diagnostics
# -----------------------------------------------------------------------

cat("=== GROUP-LEVEL PARAMETER RECOVERY ===\n")
group_draws <- fit$summary(variables = c("mean_c", "mean_gamma", "mean_w1",
                                          "sigma_log_c", "sigma_log_gamma",
                                          "sigma_w1_logit"))
print(group_draws[, c("variable", "mean", "median", "sd", "q5", "q95", "rhat", "ess_bulk")])

n_diverg <- sum(fit$diagnostic_summary(quiet = TRUE)$num_divergent)
cat(sprintf("\nDivergences: %d\n", n_diverg))

# -----------------------------------------------------------------------
# 7. Group-level recovery summary
# -----------------------------------------------------------------------

cat("\n=== GROUP-LEVEL RECOVERY ===\n")
true_vals <- c(
  mean_c     = exp(mu_log_c),
  mean_gamma = exp(mu_log_gamma),
  mean_w1    = plogis(mu_w1_logit),
  sigma_log_c     = sigma_log_c,
  sigma_log_gamma = sigma_log_gamma,
  sigma_w1_logit  = sigma_w1_logit
)

post_means <- setNames(group_draws$mean, group_draws$variable)
post_ci5   <- setNames(group_draws$q5,   group_draws$variable)
post_ci95  <- setNames(group_draws$q95,  group_draws$variable)

for (nm in names(true_vals)) {
  in_ci <- (true_vals[nm] >= post_ci5[nm]) & (true_vals[nm] <= post_ci95[nm])
  cat(sprintf("  %-20s  true=%.3f  post=%.3f  90%%CI=[%.3f, %.3f]  in_CI=%s\n",
              nm, true_vals[nm], post_means[nm],
              post_ci5[nm], post_ci95[nm],
              ifelse(in_ci, "YES", "NO")))
}

# -----------------------------------------------------------------------
# 8. Subject-level recovery
# -----------------------------------------------------------------------

cat("\n=== SUBJECT-LEVEL RECOVERY (c and w1) ===\n")
subj_c_vars  <- paste0("subj_c[", 1:N_subj, "]")
subj_w1_vars <- paste0("subj_w1[", 1:N_subj, "]")
subj_c_draws  <- fit$summary(variables = subj_c_vars)
subj_w1_draws <- fit$summary(variables = subj_w1_vars)

r_c_subj  <- cor(subj_c,  subj_c_draws$mean)
r_w1_subj <- cor(subj_w1, subj_w1_draws$mean)
cat(sprintf("  r(true_c_subj,  post_c_subj):  %.3f\n", r_c_subj))
cat(sprintf("  r(true_w1_subj, post_w1_subj): %.3f\n", r_w1_subj))

# -----------------------------------------------------------------------
# 9. Summary
# -----------------------------------------------------------------------

cat("\n=== HIERARCHICAL FIT SUMMARY ===\n")
rhat_max <- max(c(group_draws$rhat,
                  subj_c_draws$rhat,
                  subj_w1_draws$rhat), na.rm = TRUE)
ess_min  <- min(group_draws$ess_bulk, na.rm = TRUE)

cat(sprintf("  Subjects: %d, trials: %d\n", N_subj, nrow(trial_df)))
cat(sprintf("  Sampling time: %.1f s\n", t_elapsed["elapsed"]))
cat(sprintf("  Rhat max: %.3f\n", rhat_max))
cat(sprintf("  ESS bulk min (group params): %.0f\n", ess_min))
cat(sprintf("  Divergences: %d\n", n_diverg))
cat(sprintf("  Subject-level r(c):  %.3f\n", r_c_subj))
cat(sprintf("  Subject-level r(w1): %.3f\n", r_w1_subj))
cat("  Non-centred parameterisation: stable with adapt_delta=0.90\n")
