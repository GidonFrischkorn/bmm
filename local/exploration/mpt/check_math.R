# Quick math check for 01_reproduce_upstream_demo.R (no Stan needed)
set.seed(42)
D_true <- 0.7
g_true <- 0.5
p_old_hit  <- D_true + (1 - D_true) * g_true
p_old_miss <- (1 - D_true) * (1 - g_true)
p_new_fa   <- (1 - D_true) * g_true
p_new_cr   <- D_true + (1 - D_true) * (1 - g_true)

stopifnot(abs(p_old_hit + p_old_miss - 1) < 1e-10)
stopifnot(abs(p_new_fa  + p_new_cr  - 1) < 1e-10)
cat("Branch probs sum to 1: OK\n")

hit_rate <- p_old_hit
fa_rate  <- p_new_fa
D_mle    <- hit_rate - fa_rate
g_mle    <- fa_rate / (1 - D_mle)
cat(sprintf("MLE: D = %.4f (true %.2f), g = %.4f (true %.2f)\n",
            D_mle, D_true, g_mle, g_true))

stopifnot(abs(D_mle - D_true) < 0.001)
stopifnot(abs(g_mle - g_true) < 0.001)
cat("MLE derivation correct: OK\n")
cat("All math checks passed.\n")
