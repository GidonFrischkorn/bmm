# 06b_loo_comparison.R — REVISED (Phase 2b final)
#
# GCM vs Prototype vs PRM: hierarchical LOO comparison on multi-subject
# trial-level data with genuine old (training) vs new (transfer) item structure.
#
# Supersedes the initial 06b (12 single-condition aggregate cells):
#   - N=20 subjects × S=16 stimuli = 320 LOO cells  (was 12)
#   - 12 training stimuli (is_old=TRUE) + 4 transfer stimuli (is_old=FALSE)
#   - Hierarchical random effects: log(c), log(gamma), logit(w1), logit(p_mem)
#   - Warmup hardening: init=0.5 prevents large-c NaN during early warmup
#   - LOO: leave-one-(subject,stimulus)-cell-out; Pareto-k diagnostics reported
#
# Note on catlearn::nosof94: inspected in Section 1. catlearn stores aggregate
# proportions (no per-subject / per-trial columns), so the synthetic design in
# Section 3 is used. Replace Section 3 with real trial-level data (nosof94 raw
# or OSF rocks, Nosofsky et al. 2022) for a production-quality comparison.
#
# Note on LOO unit: each observation is one (subject, stimulus) count cell.
# Leave-one-cell-out ELPD is coarser than per-trial but valid for model
# comparison on aggregated multinomial data.
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
# 1. Inspect catlearn::nosof94 — is it trial-level with subjects?
# -----------------------------------------------------------------------

cat("--- catlearn::nosof94 inspection ---\n")
d94 <- tryCatch(catlearn::nosof94, error = function(e) NULL)
if (is.null(d94)) {
  cat("  nosof94 not found — using synthetic multi-subject design.\n")
} else {
  cat(sprintf("  class:  %s\n",  paste(class(d94), collapse = ", ")))
  if (!is.null(dim(d94)))
    cat(sprintf("  dim:    %s\n",  paste(dim(d94), collapse = " × ")))
  cat(sprintf("  names:  %s\n",  paste(names(d94), collapse = ", ")))
  has_subjects <- any(c("subject", "subj", "pp", "participant") %in% tolower(names(d94)))
  has_trials   <- any(c("trial", "y", "response", "resp", "choice") %in% tolower(names(d94)))
  cat(sprintf("  per-subject column present: %s\n", has_subjects))
  cat(sprintf("  per-trial column present:   %s\n", has_trials))
  if (!has_subjects || !has_trials)
    cat("  --> Aggregate format. Using synthetic multi-subject design below.\n")
}
cat("\n")

# -----------------------------------------------------------------------
# 2. Stimulus geometry
#    Training set (J=12): nosof88 MDS solution (Nosofsky 1988, Fig. 1)
#    Transfer set (4):    near-boundary points not in training
#
# All 16 test stimuli share the same K=2 categories and M=2 dimensions.
# is_old = TRUE only for the 12 training items (the exemplar set for GCM).
# For PRM: rote-memory component applies only when is_old = TRUE.
# -----------------------------------------------------------------------

train_coords <- data.frame(
  stim  = 1:12,
  x1    = c(-2.543,  0.943, -1.092,  1.558, -2.258,  0.194,
             2.806, -1.177,  1.543, -2.775,  0.528,  1.709),
  x2    = c( 2.641,  4.341,  1.848,  2.902,  0.430,  0.572,
             0.202, -1.038, -1.040, -3.149, -3.766, -3.773),
  cat   = c(1L, 2L, 2L, 2L, 1L, 2L, 2L, 1L, 2L, 1L, 1L, 1L),
  is_old = TRUE
)

# Transfer stimuli: 4 near-boundary points (first seen at test, not in exemplar set)
transfer_coords <- data.frame(
  stim  = 13:16,
  x1    = c(-0.30,  0.40, -0.55,  0.65),
  x2    = c( 0.80, -0.50, -2.20,  2.10),
  cat   = c(1L,    2L,    1L,    2L),
  is_old = FALSE
)

all_stims <- rbind(train_coords, transfer_coords)
S        <- nrow(all_stims)         # 16 total test stimuli
J        <- nrow(train_coords)      # 12 training exemplars
M        <- 2L
K        <- 2L
is_old   <- as.integer(all_stims$is_old)
true_cat <- all_stims$cat

ex_mat   <- as.matrix(train_coords[, c("x1", "x2")])
test_mat <- as.matrix(all_stims[,   c("x1", "x2")])
ex_cats  <- train_coords$cat

cat(sprintf("Stimulus design: S=%d test stims (%d training + %d transfer), J=%d exemplars\n",
            S, J, S - J, J))

# Category prototypes = mean training-exemplar coords per category (for Prototype/PRM)
proto_mat <- do.call(rbind, lapply(seq_len(K), function(k) {
  colMeans(ex_mat[ex_cats == k, , drop = FALSE])
}))
cat(sprintf("Prototypes: cat1=(%.3f, %.3f)  cat2=(%.3f, %.3f)\n\n",
            proto_mat[1,1], proto_mat[1,2], proto_mat[2,1], proto_mat[2,2]))

# -----------------------------------------------------------------------
# 3. Pre-compute distance arrays (exact squared-difference matrices)
#
#    D_stim[s]  : J×M matrix — D[j,m] = (test_s[m] - exemplar_j[m])^2
#    D_proto[s] : K×M matrix — D[k,m] = (test_s[m] - prototype_k[m])^2
#
# Stan computes: d = sqrt(dot_product(w, D[row, ]')) for weighted Euclidean.
# -----------------------------------------------------------------------

D_stim_arr  <- array(0, dim = c(S, J, M))
D_proto_arr <- array(0, dim = c(S, K, M))
for (m in seq_len(M)) {
  D_stim_arr[,  , m] <- (outer(test_mat[, m], ex_mat[,    m], `-`))^2
  D_proto_arr[, , m] <- (outer(test_mat[, m], proto_mat[, m], `-`))^2
}
D_stim_list  <- lapply(seq_len(S), function(s) D_stim_arr[s,  , ])
D_proto_list <- lapply(seq_len(S), function(s) D_proto_arr[s, , ])

# -----------------------------------------------------------------------
# 4. Simulate N=20 subjects (trial-level → aggregate to counts)
#
# Generative model: PRM — GCM activations with rote-memory overlay for
# training items. (Using PRM as the generative model means GCM and
# Prototype should both fit, but GCM will be closest to ground truth.)
# -----------------------------------------------------------------------

N_subj       <- 20L
n_train_stim <- 30L   # trials per training stimulus per subject
n_xfer_stim  <- 20L   # trials per transfer stimulus per subject

# True group-level parameters
mu_log_c          <- log(0.75)
mu_log_gamma      <- log(1.50)
mu_w1_logit       <- qlogis(0.70)
sigma_log_c       <- 0.40
sigma_log_gamma   <- 0.35
sigma_w1_logit    <- 0.50
mu_p_mem_logit    <- qlogis(0.15)
sigma_p_mem_logit <- 0.40

set.seed(888)
subj_c     <- exp(  rnorm(N_subj, mu_log_c,        sigma_log_c))
subj_gamma <- exp(  rnorm(N_subj, mu_log_gamma,    sigma_log_gamma))
subj_w1    <- plogis(rnorm(N_subj, mu_w1_logit,    sigma_w1_logit))
subj_p_mem <- plogis(rnorm(N_subj, mu_p_mem_logit, sigma_p_mem_logit))

cat(sprintf("Simulating N=%d subjects (seed 888, PRM generative model):\n", N_subj))
cat(sprintf("  true group: c=%.3f  gamma=%.3f  w1=%.3f  p_mem=%.3f\n",
            exp(mu_log_c), exp(mu_log_gamma), plogis(mu_w1_logit), plogis(mu_p_mem_logit)))

n_trials_per_stim <- ifelse(is_old == 1L, n_train_stim, n_xfer_stim)

y_counts <- array(0L, dim = c(N_subj, S, K))
for (n in seq_len(N_subj)) {
  w     <- c(subj_w1[n], 1 - subj_w1[n])
  D_ns  <- gcm_distances(test_mat, ex_mat, w, r_metric = 2)   # S × J
  Sim   <- gcm_similarity(D_ns, subj_c[n], p_sim = 1)         # S × J
  act   <- cbind(
    rowSums(Sim[, ex_cats == 1, drop = FALSE]),
    rowSums(Sim[, ex_cats == 2, drop = FALSE])
  )  # S × K
  P_gcm <- luce_choice(act, subj_gamma[n], bias = c(0.5, 0.5))  # S × K

  for (s in seq_len(S)) {
    prob_s <- P_gcm[s, ]
    if (is_old[s]) {
      # PRM mixing: rote memory for training items
      prob_s <- (1 - subj_p_mem[n]) * prob_s
      prob_s[true_cat[s]] <- prob_s[true_cat[s]] + subj_p_mem[n]
    }
    y_samp <- sample(seq_len(K), n_trials_per_stim[s], replace = TRUE, prob = prob_s)
    for (k in seq_len(K)) y_counts[n, s, k] <- sum(y_samp == k)
  }
}

total_trials <- sum(n_trials_per_stim) * N_subj
cat(sprintf("Aggregated to [N=%d, S=%d, K=%d] counts (%d total trials, %d LOO cells)\n\n",
            N_subj, S, K, total_trials, N_subj * S))

# -----------------------------------------------------------------------
# 5. Stan model A — Hierarchical GCM (exemplar-based)
# -----------------------------------------------------------------------

gcm_stan_code <- "
functions {
  // Log-activations per category: log_sum_exp over category-k exemplars.
  // Using log-space accumulation eliminates softmax-sum-NaN at large c.
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
      for (j in 1:J)
        if (ex_cat[j] == k) la_k = log_sum_exp(la_k, log_sims[j]);
      log_act[k] = la_k;
    }
    return gamma * log_act + log_bias;
  }
}
data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> J;
  int<lower=1> M;
  int<lower=1> K;
  array[N, S, K]  int<lower=0>          y_counts;
  array[S]        matrix[J, M]          D_stim;
  array[J]        int<lower=1, upper=K> ex_cat;
}
parameters {
  real             mu_log_c;
  real             mu_log_gamma;
  real             mu_w1_logit;
  real<lower=0>    sigma_log_c;
  real<lower=0>    sigma_log_gamma;
  real<lower=0>    sigma_w1_logit;
  vector[N]        z_log_c;
  vector[N]        z_log_gamma;
  vector[N]        z_w1_logit;
}
transformed parameters {
  vector[N] subj_c     = exp(mu_log_c     + sigma_log_c     * z_log_c);
  vector[N] subj_gamma = exp(mu_log_gamma + sigma_log_gamma * z_log_gamma);
  vector[N] subj_w1    = inv_logit(mu_w1_logit + sigma_w1_logit * z_w1_logit);
  vector[K] log_bias   = rep_vector(log(1.0 / K), K);
}
model {
  mu_log_c        ~ normal(0, 1);
  mu_log_gamma    ~ normal(0, 0.5);
  mu_w1_logit     ~ normal(0, 1);
  sigma_log_c     ~ normal(0, 0.5);
  sigma_log_gamma ~ normal(0, 0.5);
  sigma_w1_logit  ~ normal(0, 1);
  z_log_c     ~ std_normal();
  z_log_gamma ~ std_normal();
  z_w1_logit  ~ std_normal();
  for (n in 1:N) {
    vector[M] w = [subj_w1[n], 1 - subj_w1[n]]';
    for (s in 1:S) {
      vector[K] la = gcm_log_act(D_stim[s], w, subj_c[n], subj_gamma[n],
                                  log_bias, ex_cat, K);
      target += multinomial_logit_lpmf(y_counts[n, s] | la);
    }
  }
}
generated quantities {
  array[N * S] real log_lik;
  real mean_c     = exp(mu_log_c);
  real mean_gamma = exp(mu_log_gamma);
  real mean_w1    = inv_logit(mu_w1_logit);
  for (n in 1:N) {
    vector[M] w = [subj_w1[n], 1 - subj_w1[n]]';
    for (s in 1:S) {
      vector[K] la = gcm_log_act(D_stim[s], w, subj_c[n], subj_gamma[n],
                                  log_bias, ex_cat, K);
      log_lik[(n - 1) * S + s] = multinomial_logit_lpmf(y_counts[n, s] | la);
    }
  }
}
"

# -----------------------------------------------------------------------
# 6. Stan model B — Hierarchical Prototype (gamma = 1 fixed)
# -----------------------------------------------------------------------

proto_stan_code <- "
functions {
  vector proto_log_act(matrix D_proto_s, vector w, real c,
                       vector log_bias, int K) {
    vector[K] log_act;
    for (k in 1:K) {
      real d = sqrt(dot_product(w, D_proto_s[k, ]'));
      log_act[k] = -c * d;
    }
    return log_act + log_bias;
  }
}
data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> M;
  int<lower=1> K;
  array[N, S, K]  int<lower=0>  y_counts;
  array[S]        matrix[K, M]  D_proto;
}
parameters {
  real          mu_log_c;
  real          mu_w1_logit;
  real<lower=0> sigma_log_c;
  real<lower=0> sigma_w1_logit;
  vector[N]     z_log_c;
  vector[N]     z_w1_logit;
}
transformed parameters {
  vector[N] subj_c  = exp(mu_log_c + sigma_log_c * z_log_c);
  vector[N] subj_w1 = inv_logit(mu_w1_logit + sigma_w1_logit * z_w1_logit);
  vector[K] log_bias = rep_vector(log(1.0 / K), K);
}
model {
  mu_log_c       ~ normal(0, 1);
  mu_w1_logit    ~ normal(0, 1);
  sigma_log_c    ~ normal(0, 0.5);
  sigma_w1_logit ~ normal(0, 1);
  z_log_c    ~ std_normal();
  z_w1_logit ~ std_normal();
  for (n in 1:N) {
    vector[M] w = [subj_w1[n], 1 - subj_w1[n]]';
    for (s in 1:S) {
      vector[K] la = proto_log_act(D_proto[s], w, subj_c[n], log_bias, K);
      target += multinomial_logit_lpmf(y_counts[n, s] | la);
    }
  }
}
generated quantities {
  array[N * S] real log_lik;
  real mean_c  = exp(mu_log_c);
  real mean_w1 = inv_logit(mu_w1_logit);
  for (n in 1:N) {
    vector[M] w = [subj_w1[n], 1 - subj_w1[n]]';
    for (s in 1:S) {
      vector[K] la = proto_log_act(D_proto[s], w, subj_c[n], log_bias, K);
      log_lik[(n - 1) * S + s] = multinomial_logit_lpmf(y_counts[n, s] | la);
    }
  }
}
"

# -----------------------------------------------------------------------
# 7. Stan model C — Hierarchical PRM (Prototype + rote-memory mixing)
#
# For training (old) stimuli:
#   P(k|s,n) = (1 - p_mem_n) * softmax(la)[k]  +  p_mem_n * I(k == true_cat[s])
# For transfer (new) stimuli:
#   P(k|s,n) = softmax(la)[k]
#
# Mixing breaks the logit form, so multinomial_lpmf is used directly
# (not multinomial_logit_lpmf). The mixed vector is a valid simplex.
# -----------------------------------------------------------------------

prm_stan_code <- "
functions {
  vector proto_log_act(matrix D_proto_s, vector w, real c,
                       vector log_bias, int K) {
    vector[K] log_act;
    for (k in 1:K) {
      real d = sqrt(dot_product(w, D_proto_s[k, ]'));
      log_act[k] = -c * d;
    }
    return log_act + log_bias;
  }
}
data {
  int<lower=1> N;
  int<lower=1> S;
  int<lower=1> M;
  int<lower=1> K;
  array[N, S, K]  int<lower=0>           y_counts;
  array[S]        matrix[K, M]           D_proto;
  array[S]        int<lower=0, upper=1>  is_old;
  array[S]        int<lower=1, upper=K>  true_cat;
}
parameters {
  real          mu_log_c;
  real          mu_w1_logit;
  real          mu_p_mem_logit;
  real<lower=0> sigma_log_c;
  real<lower=0> sigma_w1_logit;
  real<lower=0> sigma_p_mem_logit;
  vector[N]     z_log_c;
  vector[N]     z_w1_logit;
  vector[N]     z_p_mem;
}
transformed parameters {
  vector[N] subj_c     = exp(mu_log_c    + sigma_log_c    * z_log_c);
  vector[N] subj_w1    = inv_logit(mu_w1_logit   + sigma_w1_logit   * z_w1_logit);
  vector[N] subj_p_mem = inv_logit(mu_p_mem_logit + sigma_p_mem_logit * z_p_mem);
  vector[K] log_bias   = rep_vector(log(1.0 / K), K);
}
model {
  mu_log_c        ~ normal(0, 1);
  mu_w1_logit     ~ normal(0, 1);
  mu_p_mem_logit  ~ normal(-1, 1);   // slight shrinkage toward small p_mem
  sigma_log_c        ~ normal(0, 0.5);
  sigma_w1_logit     ~ normal(0, 1);
  sigma_p_mem_logit  ~ normal(0, 0.5);
  z_log_c    ~ std_normal();
  z_w1_logit ~ std_normal();
  z_p_mem    ~ std_normal();
  for (n in 1:N) {
    vector[M] w = [subj_w1[n], 1 - subj_w1[n]]';
    for (s in 1:S) {
      vector[K] la = proto_log_act(D_proto[s], w, subj_c[n], log_bias, K);
      vector[K] p  = softmax(la);
      if (is_old[s]) {
        p = (1 - subj_p_mem[n]) * p;
        p[true_cat[s]] += subj_p_mem[n];
      }
      target += multinomial_lpmf(y_counts[n, s] | p);
    }
  }
}
generated quantities {
  array[N * S] real log_lik;
  real mean_c     = exp(mu_log_c);
  real mean_w1    = inv_logit(mu_w1_logit);
  real mean_p_mem = inv_logit(mu_p_mem_logit);
  for (n in 1:N) {
    vector[M] w = [subj_w1[n], 1 - subj_w1[n]]';
    for (s in 1:S) {
      vector[K] la = proto_log_act(D_proto[s], w, subj_c[n], log_bias, K);
      vector[K] p  = softmax(la);
      if (is_old[s]) {
        p = (1 - subj_p_mem[n]) * p;
        p[true_cat[s]] += subj_p_mem[n];
      }
      log_lik[(n - 1) * S + s] = multinomial_lpmf(y_counts[n, s] | p);
    }
  }
}
"

# -----------------------------------------------------------------------
# 8. Compile all three models
# -----------------------------------------------------------------------

compile_model <- function(code, label) {
  cat(sprintf("Compiling %s...\n", label))
  f <- tempfile(fileext = ".stan")
  writeLines(code, f)
  m <- tryCatch(cmdstan_model(f, quiet = TRUE),
                error = function(e) { cat("  Error:", conditionMessage(e), "\n"); NULL })
  if (!is.null(m)) cat(sprintf("  %s OK\n", label))
  m
}

mod_gcm   <- compile_model(gcm_stan_code,   "GCM")
mod_proto <- compile_model(proto_stan_code, "Prototype")
mod_prm   <- compile_model(prm_stan_code,   "PRM")

if (any(sapply(list(mod_gcm, mod_proto, mod_prm), is.null))) {
  cat("Compilation failed — exiting.\n")
  quit(status = 1)
}

# -----------------------------------------------------------------------
# 9. Stan data objects
# -----------------------------------------------------------------------

stan_data_gcm <- list(
  N        = N_subj,
  S        = S,
  J        = J,
  M        = M,
  K        = K,
  y_counts = y_counts,
  D_stim   = D_stim_list,
  ex_cat   = ex_cats
)

stan_data_proto <- list(
  N        = N_subj,
  S        = S,
  M        = M,
  K        = K,
  y_counts = y_counts,
  D_proto  = D_proto_list
)

stan_data_prm <- c(
  stan_data_proto,
  list(is_old   = is_old,
       true_cat = true_cat)
)

# -----------------------------------------------------------------------
# 10. Fit all three models
#     Warmup hardening: init=0.5 restricts unconstrained parameters to
#     [-0.5, 0.5] at initialisation — prevents large initial c values
#     that trigger multinomial_logit_lpmf NaN in the first few warmup steps.
# -----------------------------------------------------------------------

fit_model <- function(mod, data, label, seed) {
  cat(sprintf("\nFitting %s (4 chains × 1000 draws, init=0.5)...\n", label))
  t0 <- proc.time()
  fit <- tryCatch(
    mod$sample(
      data          = data,
      chains        = 4,
      iter_warmup   = 500,
      iter_sampling = 1000,
      init          = 0.5,   # restrict initial unconstrained draws to [-0.5, 0.5]
      seed          = seed,
      refresh       = 0,
      show_messages = FALSE,
      adapt_delta   = 0.90
    ),
    error = function(e) { cat("  Error:", conditionMessage(e), "\n"); NULL }
  )
  if (is.null(fit)) return(NULL)
  elapsed <- (proc.time() - t0)["elapsed"]
  diag    <- fit$diagnostic_summary(quiet = TRUE)
  n_divg  <- sum(diag$num_divergent)
  rhat_mx <- max(fit$summary()$rhat, na.rm = TRUE)
  cat(sprintf("  Done %.1f s | diverg: %d | Rhat max: %.3f\n",
              elapsed, n_divg, rhat_mx))
  if (n_divg > 0)
    cat(sprintf("  WARNING: %d divergences — increase adapt_delta or reparametrise\n", n_divg))
  fit
}

fit_gcm   <- fit_model(mod_gcm,   stan_data_gcm,   "GCM",       seed = 201)
fit_proto <- fit_model(mod_proto, stan_data_proto, "Prototype", seed = 202)
fit_prm   <- fit_model(mod_prm,   stan_data_prm,   "PRM",       seed = 203)

# -----------------------------------------------------------------------
# 11. Parameter summaries
# -----------------------------------------------------------------------

cat("\n=== POSTERIOR PARAMETER ESTIMATES ===\n")
summarise_params <- function(fit, vars, label) {
  if (is.null(fit)) return(invisible(NULL))
  s <- fit$summary(vars)
  cat(sprintf("\n  %s:\n", label))
  for (i in seq_len(nrow(s))) {
    cat(sprintf("    %-16s  mean=%.3f  90%%CI=[%.3f, %.3f]  Rhat=%.3f\n",
                s$variable[i], s$mean[i], s$q5[i], s$q95[i], s$rhat[i]))
  }
}
summarise_params(fit_gcm,   c("mean_c", "mean_gamma", "mean_w1"),           "GCM")
summarise_params(fit_proto, c("mean_c", "mean_w1"),                         "Prototype (gamma=1)")
summarise_params(fit_prm,   c("mean_c", "mean_w1", "mean_p_mem"),           "PRM")

# PRM rote-memory signal
if (!is.null(fit_prm)) {
  pm <- fit_prm$summary("mean_p_mem")
  cat(sprintf("\n  PRM p_mem (group mean): %.3f  90%%CI=[%.3f, %.3f]\n",
              pm$mean, pm$q5, pm$q95))
  if (pm$q5 > 0)
    cat("  --> CI excludes 0: rote-memory component identified (PRM ≠ prototype) ✓\n")
  else
    cat("  --> CI includes 0: rote-memory component not clearly identified.\n")
}

# -----------------------------------------------------------------------
# 12. LOO-IC comparison (leave-one-(subject,stimulus)-cell-out)
#
# Each observation is one (subj, stim) count vector. With N=20 subjects
# and S=16 stimuli, there are 320 LOO cells per model.
#
# Pareto-k > 0.7 indicates unreliable importance-sampling weights for that
# cell. With 30 training / 20 transfer trials per cell the effective count
# is large, so k values should be well below 0.7. If not, use moment
# matching via loo::loo_moment_match() or reloo().
# -----------------------------------------------------------------------

cat("\n\n=== LOO MODEL COMPARISON (leave-one-(subj,stim)-cell-out) ===\n")
cat(sprintf("N=%d subjects × S=%d stimuli = %d LOO cells per model\n",
            N_subj, S, N_subj * S))
cat("Note: LOO unit is one (subject, stimulus) count vector, not one trial.\n\n")

extract_loo <- function(fit, label) {
  if (is.null(fit)) return(NULL)
  ll_draws <- fit$draws("log_lik")     # draws_array: [iter, chain, N*S]
  r_eff    <- relative_eff(exp(ll_draws))
  loo_res  <- loo(ll_draws, r_eff = r_eff)

  elpd   <- loo_res$estimates["elpd_loo", "Estimate"]
  elpd_se <- loo_res$estimates["elpd_loo", "SE"]
  p_loo  <- loo_res$estimates["p_loo",    "Estimate"]
  looic  <- loo_res$estimates["looic",    "Estimate"]

  k_vals <- loo_res$diagnostics$pareto_k
  n_bad  <- sum(k_vals > 0.7, na.rm = TRUE)
  n_ok   <- sum(k_vals > 0.5 & k_vals <= 0.7, na.rm = TRUE)

  cat(sprintf("  %-12s  ELPD=%7.1f (SE=%.1f)  p_loo=%5.1f  LOO-IC=%7.1f\n",
              label, elpd, elpd_se, p_loo, looic))
  cat(sprintf("              Pareto-k: %d ok (<0.5)  %d marginal (0.5–0.7)  %d bad (>0.7)\n",
              length(k_vals) - n_bad - n_ok, n_ok, n_bad))

  if (n_bad > 0)
    cat(sprintf("              %d bad-k cells: consider moment_match=TRUE or reloo()\n", n_bad))

  list(label = label, loo = loo_res)
}

loo_gcm   <- extract_loo(fit_gcm,   "GCM")
loo_proto <- extract_loo(fit_proto, "Prototype")
loo_prm   <- extract_loo(fit_prm,   "PRM")

# Pairwise comparison via loo_compare (sorted best→worst by ELPD)
loo_list <- Filter(Negate(is.null),
                   list(GCM       = if (!is.null(loo_gcm))   loo_gcm$loo   else NULL,
                        Prototype = if (!is.null(loo_proto)) loo_proto$loo else NULL,
                        PRM       = if (!is.null(loo_prm))   loo_prm$loo   else NULL))

if (length(loo_list) >= 2) {
  cat("\n  loo_compare (best model on top):\n")
  cmp <- loo_compare(loo_list)
  print(cmp)
}

# -----------------------------------------------------------------------
# 13. Verdict on GCM < PRM < Prototype ordering
# -----------------------------------------------------------------------

cat("\n=== VERDICT: GCM < PRM < Prototype (ELPD ordering)? ===\n")
cat("Published BIC (Nosofsky et al. 2022): GCM 39952 / PRM 41090 / Prototype 45304\n")
cat("Expected LOO ordering on GCM-generated data: GCM best, Prototype worst.\n\n")

if (!is.null(loo_gcm) && !is.null(loo_proto) && !is.null(loo_prm)) {
  # Extract ELPD point estimates directly (sign-safe)
  elpd_gcm   <- loo_gcm$loo$estimates["elpd_loo",  "Estimate"]
  elpd_prm   <- loo_prm$loo$estimates["elpd_loo",  "Estimate"]
  elpd_proto <- loo_proto$loo$estimates["elpd_loo", "Estimate"]

  # Pairwise SE from loo_compare: row 2 always carries the SE of the difference
  se_gp <- loo_compare(list(GCM = loo_gcm$loo, Prototype = loo_proto$loo))[2, "se_diff"]
  se_gr <- loo_compare(list(GCM = loo_gcm$loo, PRM       = loo_prm$loo)) [2, "se_diff"]
  se_rp <- loo_compare(list(PRM = loo_prm$loo, Prototype = loo_proto$loo))[2, "se_diff"]

  # Positive Δ = first model better
  delta_gp <- elpd_gcm - elpd_proto
  delta_gr <- elpd_gcm - elpd_prm
  delta_rp <- elpd_prm - elpd_proto

  sig <- function(d, se) if (abs(d) > 2 * se) "  *significant*" else "  (within SE)"

  ordering <- names(sort(c(GCM = elpd_gcm, PRM = elpd_prm, Prototype = elpd_proto),
                         decreasing = TRUE))
  cat("  ELPD ordering (higher = better fit):\n")
  for (nm in ordering) {
    e <- c(GCM = elpd_gcm, PRM = elpd_prm, Prototype = elpd_proto)[nm]
    cat(sprintf("    %-12s  ELPD = %7.1f\n", nm, e))
  }

  cat(sprintf("\n  Pairwise (positive ΔELPD = first model better):\n"))
  cat(sprintf("    GCM vs Prototype: ΔELPD = %+.1f (SE %.1f)%s\n",
              delta_gp, se_gp, sig(delta_gp, se_gp)))
  cat(sprintf("    GCM vs PRM:       ΔELPD = %+.1f (SE %.1f)%s\n",
              delta_gr, se_gr, sig(delta_gr, se_gr)))
  cat(sprintf("    PRM vs Prototype: ΔELPD = %+.1f (SE %.1f)%s\n",
              delta_rp, se_rp, sig(delta_rp, se_rp)))

  gcm_best  <- (ordering[1] == "GCM")
  proto_worst <- (ordering[3] == "Prototype")
  prm_middle  <- (ordering[2] == "PRM")
  cat(sprintf("\n  GCM < PRM < Prototype ordering: %s\n",
              if (gcm_best && prm_middle && proto_worst) "YES (GCM best, Prototype worst)"
              else if (gcm_best && proto_worst) "PARTIAL (GCM best, PRM/Prototype swapped)"
              else "NO (unexpected ordering)"))
  cat("\n  Interpretation notes:\n")
  cat("  1. Data generated from PRM (GCM+rote), so GCM should fit best.\n")
  cat("  2. Differences > 2×SE are treated as reliable; cell scale (N×S=320)\n")
  cat("     gives much tighter SE than the 12-cell initial 06b.\n")
  cat("  3. For a definitive comparison use real trial-level data\n")
  cat("     (nosof94 raw responses or OSF rocks, Nosofsky et al. 2022).\n")
}

cat("\n=== SUMMARY ===\n")
cat(sprintf("Dataset: synthetic (N=%d subjects, S=%d stims, %d train + %d transfer)\n",
            N_subj, S, J, S - J))
cat(sprintf("Training items (is_old=TRUE):  %d stimuli × %d trials/subj\n",
            sum(is_old), n_train_stim))
cat(sprintf("Transfer items (is_old=FALSE): %d stimuli × %d trials/subj\n",
            sum(1L - is_old), n_xfer_stim))
cat("LOO unit: per-(subject, stimulus) count cell (N×S cells)\n")
cat("Next step: replace synthetic data with nosof94 raw responses or\n")
cat("  OSF rocks data (Nosofsky et al. 2022) for a production comparison.\n")
