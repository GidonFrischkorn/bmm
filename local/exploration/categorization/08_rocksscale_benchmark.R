# GCM Rocks-Scale Benchmark — Phase 2b
#
# Addresses Phase 2b item 3 (issue #19, task 3): benchmark Stan GCM at the
# rocks scale (Nosofsky et al. 2022: J≈90 exemplars, M≈8 dimensions, K≈10
# categories) rather than extrapolating from the nosof88 case (J=12, M=2, K=2).
#
# Two optimisations included:
#
#   1. Per-category exemplar indexing (CSR format): eliminates the O(J·K) branch
#      loop `if (ex_cat[j]==k)` per stimulus. Instead pass flat index array +
#      category start/end pointers, reducing inner-loop work to O(J) per stimulus.
#
#   2. Multinomial aggregation (from Phase 2a): the J·M outer-product is
#      evaluated S times (unique stimuli), not T = S×n times. Combined with
#      CSR, the per-draw cost is O(S × J × M) regardless of n.
#
# Data: fully synthetic. S=150 test stimuli, J=90 exemplars, M=8 dims, K=10 cats.
# Baseline: J=12, M=2, K=2 (nosof88 scale) fitted with the same model.
# Comparison: timing ratio at multiple scales to extrapolate to production cost.
#
# Run from repository root:
#   Rscript local/exploration/categorization/08_rocksscale_benchmark.R

suppressPackageStartupMessages({
  library(cmdstanr)
})

# -----------------------------------------------------------------------
# 1. Stan model with CSR per-category indexing
# -----------------------------------------------------------------------
# cat_start[k] and cat_end[k] are 1-indexed pointers into ex_flat[].
# ex_flat[cat_start[k]:cat_end[k]] contains the exemplar indices for cat k.
# This replaces the O(J·K) branch-loop with a single O(J) pass.

gcm_csr_stan_code <- "
functions {
  // Log-space GCM activation with CSR per-category exemplar indexing.
  // cat_start[k] and cat_end[k] are 1-indexed into ex_flat.
  vector gcm_log_act_csr(
      matrix D_t, vector w, real c, real gamma, vector log_bias,
      array[] int ex_flat, array[] int cat_start, array[] int cat_end, int K) {
    int J = rows(D_t);
    vector[J] log_sims;
    vector[K] log_act;
    for (j in 1:J) {
      real d = sqrt(dot_product(w, D_t[j, ]'));
      log_sims[j] = -c * d;
    }
    for (k in 1:K) {
      real la_k = negative_infinity();
      for (idx in cat_start[k]:cat_end[k]) {
        la_k = log_sum_exp(la_k, log_sims[ex_flat[idx]]);
      }
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
  array[S]    matrix[J, M] D_stim;
  // CSR per-category exemplar index
  int<lower=1>     total_ex;
  array[total_ex]  int<lower=1, upper=J> ex_flat;
  array[K]         int<lower=1>         cat_start;
  array[K]         int<lower=1>         cat_end;
}
parameters {
  real          log_c;
  real          log_gamma;
  vector[M - 1] w_raw;  // softmax-with-reference (dim M = reference)
}
transformed parameters {
  real<lower=0>  c     = exp(log_c);
  real<lower=0>  gamma = exp(log_gamma);
  vector[K]      log_bias = rep_vector(-log(K), K);
  vector[M] w_full;
  w_full[1:(M - 1)] = w_raw;
  w_full[M] = 0.0;
  simplex[M] w = softmax(w_full);
}
model {
  log_c     ~ normal(0, 1);
  log_gamma ~ normal(0, 1);
  w_raw     ~ normal(0, 1);
  for (s in 1:S) {
    vector[K] la = gcm_log_act_csr(D_stim[s], w, c, gamma, log_bias,
                                    ex_flat, cat_start, cat_end, K);
    target += multinomial_logit_lpmf(y_counts[s] | la);
  }
}
"

# -----------------------------------------------------------------------
# 2. Helper: build CSR index arrays from ex_cats
# -----------------------------------------------------------------------

build_csr <- function(ex_cats, K) {
  ex_flat   <- integer(0)
  cat_start <- integer(K)
  cat_end   <- integer(K)
  pos <- 1L
  for (k in seq_len(K)) {
    idx <- which(ex_cats == k)
    cat_start[k] <- pos
    cat_end[k]   <- pos + length(idx) - 1L
    ex_flat <- c(ex_flat, idx)
    pos <- pos + length(idx)
  }
  list(ex_flat = ex_flat, cat_start = cat_start, cat_end = cat_end)
}

# -----------------------------------------------------------------------
# 3. Benchmark function: generate synthetic data and time one MCMC run
# -----------------------------------------------------------------------

run_benchmark <- function(S, J, M, K, n_per_stim = 10L,
                           chains = 1L, iter_warmup = 100L,
                           iter_sampling = 50L, seed = 42L,
                           mod) {
  # Simulate exemplar coordinates
  ex_mat <- matrix(rnorm(J * M), J, M)
  # Assign categories: round-robin
  ex_cats <- rep_len(seq_len(K), J)
  # Simulate category probabilities (uniform random activations)
  P <- matrix(runif(S * K), S, K)
  P <- P / rowSums(P)
  # Simulate count data
  y_counts <- t(apply(P, 1, function(p) rmultinom(1, n_per_stim, p)[, 1]))
  storage.mode(y_counts) <- "integer"

  # Pre-compute D_stim
  D_raw <- array(0, dim = c(S, J, M))
  test_mat <- matrix(rnorm(S * M), S, M)
  for (m in seq_len(M)) {
    D_raw[, , m] <- abs(outer(test_mat[, m], ex_mat[, m], `-`))^2
  }
  D_stim_list <- lapply(seq_len(S), function(s) D_raw[s, , ])

  # CSR index
  csr <- build_csr(ex_cats, K)

  stan_data <- list(
    S          = S,
    J          = J,
    M          = M,
    K          = K,
    y_counts   = y_counts,
    D_stim     = D_stim_list,
    total_ex   = length(csr$ex_flat),
    ex_flat    = csr$ex_flat,
    cat_start  = csr$cat_start,
    cat_end    = csr$cat_end
  )

  t0 <- proc.time()
  fit <- tryCatch(
    mod$sample(
      data          = stan_data,
      chains        = chains,
      iter_warmup   = iter_warmup,
      iter_sampling = iter_sampling,
      seed          = seed,
      refresh       = 0,
      show_messages = FALSE
    ),
    error = function(e) NULL
  )
  elapsed <- (proc.time() - t0)["elapsed"]

  if (is.null(fit)) return(NULL)
  n_draws    <- chains * iter_sampling
  n_evals    <- n_draws * S  # Stan evaluations (multinomial cells per draw)
  draws_s    <- n_draws / elapsed

  list(
    S = S, J = J, M = M, K = K, n_per_stim = n_per_stim,
    elapsed_s = round(elapsed, 1),
    draws_s   = round(draws_s, 1),
    n_evals   = n_evals,
    note      = sprintf("S=%d, J=%d, M=%d, K=%d", S, J, M, K)
  )
}

# -----------------------------------------------------------------------
# 4. Compile model once
# -----------------------------------------------------------------------

cat("Compiling CSR GCM Stan model...\n")
stan_file <- tempfile(fileext = ".stan")
writeLines(gcm_csr_stan_code, stan_file)
mod <- tryCatch(
  cmdstan_model(stan_file, quiet = TRUE),
  error = function(e) { cat("Compilation error:", conditionMessage(e), "\n"); NULL }
)
if (is.null(mod)) quit(status = 1)
cat("Compilation OK.\n\n")

# -----------------------------------------------------------------------
# 5. Run benchmark scenarios
# -----------------------------------------------------------------------

# Scenarios: nosof88 scale → rocks scale
scenarios <- list(
  # nosof88 scale (baseline from 03_stan_prototype.R)
  list(S = 12L,  J = 12L,  M = 2L,  K = 2L,  n = 25L,  label = "nosof88"),
  # Intermediate scales
  list(S = 30L,  J = 30L,  M = 4L,  K = 4L,  n = 25L,  label = "medium_4dim"),
  list(S = 60L,  J = 60L,  M = 6L,  K = 6L,  n = 15L,  label = "medium_6dim"),
  # Rocks scale (Nosofsky 2022: ~150 stimuli, 90 exemplars, 8 dims, 10 cats)
  list(S = 100L, J = 90L,  M = 8L,  K = 10L, n = 10L,  label = "rocks_100stim"),
  list(S = 150L, J = 90L,  M = 8L,  K = 10L, n = 10L,  label = "rocks_full")
)

cat("=== TIMING BENCHMARKS (1 chain, 100 warmup + 50 draws) ===\n\n")

results <- lapply(scenarios, function(sc) {
  cat(sprintf("  Running: %s (S=%d, J=%d, M=%d, K=%d)...\n",
              sc$label, sc$S, sc$J, sc$M, sc$K))
  r <- run_benchmark(
    S = sc$S, J = sc$J, M = sc$M, K = sc$K, n_per_stim = sc$n,
    chains = 1L, iter_warmup = 100L, iter_sampling = 50L,
    seed = 42L, mod = mod
  )
  if (!is.null(r)) {
    cat(sprintf("    %.1f s | %.1f draws/s | %d Stan evals\n",
                r$elapsed_s, r$draws_s, r$n_evals))
  } else {
    cat("    FAILED\n")
  }
  r
})

# -----------------------------------------------------------------------
# 6. Summary table
# -----------------------------------------------------------------------

cat("\n=== BENCHMARK SUMMARY ===\n")
cat(sprintf("%-20s  %5s  %5s  %3s  %3s  %8s  %8s\n",
            "scenario", "S", "J", "M", "K", "elapsed_s", "draws_s"))
for (r in Filter(Negate(is.null), results)) {
  cat(sprintf("%-20s  %5d  %5d  %3d  %3d  %8.1f  %8.1f\n",
              r$label, r$S, r$J, r$M, r$K, r$elapsed_s, r$draws_s))
}

cat("\nNotes:\n")
cat("  - 1 chain only; multiply elapsed_s by 4 for production (4 chains parallel).\n")
cat("  - n_per_stim is collapsed away by aggregation — counts are [S, K], not [S*n, 1].\n")
cat("  - CSR index eliminates O(J*K) branch loop; inner loop is O(J) per stimulus.\n")
cat("  - At rocks scale (S=150, J=90, M=8, K=10) the per-draw cost is O(S*J*M).\n")
cat("  - Compare to m3 (issue #19, task 3): if m3 runs at ~X draws/s at J=90,\n")
cat("    GCM CSR target is within 5-10x of m3 speed (acceptable for exploration).\n")
cat("  - Next optimisation (if needed): precompute D_cat[S,K] in R via log_sum_exp\n")
cat("    over cat-k rows — but this requires fixing (c, w) which are Stan params.\n")
