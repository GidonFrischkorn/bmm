# 01_formal_mapping.R
# Formal verification that M3 choice rules map to Luce / Random-Utility / EU theory
#
# Run from the repo root:
#   source("local/exploration/m3-utility/01_formal_mapping.R")

# ---- helpers ----------------------------------------------------------------

softmax_m3 <- function(a, n) {
  log_num <- a + log(n)
  exp(log_num - log(sum(exp(log_num))))
}

luce_m3 <- function(a, n) {
  strengths <- a * n
  strengths / sum(strengths)
}

# ---- 1. Softmax M3 is the Multinomial Logit (Luce choice with exp-utilities) ----

cat("=== 1. Softmax M3 = Multinomial Logit / Random-Utility Model ===\n\n")

a <- c(3.0, 2.0, 0.0)   # activations:  correct, other, npl
n <- c(1L,  4L,  5L)    # option counts

probs_analytic <- softmax_m3(a, n)
cat("Analytic softmax M3 probabilities:", round(probs_analytic, 5), "\n")

# Random-utility derivation:
#   If U_i = a_i + log(n_i) + eps_i, eps_i ~ Gumbel(0,1) iid,
#   then P(argmax_i U_i = i) = softmax_m3(a, n).
n_sim <- 200000L
u0 <- matrix(a + log(n), nrow = n_sim, ncol = length(a), byrow = TRUE)
eps <- -log(-log(matrix(runif(n_sim * length(a)), n_sim)))  # Gumbel(0,1) samples
choices <- max.col(u0 + eps)
probs_sim <- tabulate(choices, nbins = length(a)) / n_sim

cat("Simulated RU probabilities:       ", round(probs_sim, 5), "\n")
cat("Max abs difference:               ", max(abs(probs_analytic - probs_sim)), "\n\n")

# ---- 2. Temperature is the inverse activation scale ----

cat("=== 2. Temperature = inverse activation scale ===\n\n")
cat("Rescaling all activations by 1/tau gives softmax with 'temperature' tau:\n")
for (tau in c(0.5, 1.0, 2.0, 5.0)) {
  p <- softmax_m3(a / tau, n)
  cat(sprintf("  tau=%.1f: P = (%.3f, %.3f, %.3f)  [entropy=%.3f]\n",
              tau, p[1], p[2], p[3], -sum(p * log(p))))
}
cat("\n")

# ---- 3. Simple rule = Luce ratio rule ----

cat("=== 3. Simple rule = Luce (1959) ratio rule ===\n\n")

# After the log link, simple-rule activations are non-negative:
a_s <- c(exp(1.5), exp(1.0), exp(0.1))  # activations after link (a > 0)
b <- 0.1
probs_luce <- luce_m3(a_s + b, n)
cat("Simple-rule (Luce) P = (%.5f, %.5f, %.5f) for strengths:\n")
cat("  strengths = n_i * (a_i + b) = ",
    sprintf("(%.2f, %.2f, %.2f)", (a_s + b)[1] * n[1], (a_s + b)[2] * n[2], (a_s + b)[3] * n[3]), "\n")
cat("  P =", round(probs_luce, 5), "\n\n")

# ---- 4. Affine-invariance analysis: where M3 diverges from vNM ----

cat("=== 4. Affine-invariance (vNM vs M3 scale uniqueness) ===\n\n")
cat("vNM utility: unique up to U' = alpha*U + beta  (interval scale)\n")
cat("  -> invariant to BOTH additive shift AND positive scaling\n\n")

beta_shift  <- 5
alpha_scale <- 2

# Softmax: shift-invariant (adding a constant beta to all activations does nothing)
p_shifted <- softmax_m3(a + beta_shift, n)
cat("Softmax + shift(", beta_shift, "):",
    "max|delta| =", round(max(abs(probs_analytic - p_shifted)), 8),
    " <- INVARIANT (good)\n")

# Softmax: NOT scale-invariant
p_scaled <- softmax_m3(alpha_scale * a, n)
cat("Softmax * scale(", alpha_scale, "):",
    "max|delta| =", round(max(abs(probs_analytic - p_scaled)), 5),
    " <- NOT invariant\n\n")

# Simple rule: NOT shift-invariant
p_luce_shifted <- luce_m3(a_s + b + beta_shift, n)
cat("Simple + shift(", beta_shift, "):",
    "max|delta| =", round(max(abs(probs_luce - p_luce_shifted)), 5),
    " <- NOT invariant\n")

# Simple rule: scale-invariant
p_luce_scaled <- luce_m3(alpha_scale * (a_s + b), n)
cat("Simple * scale(", alpha_scale, "):",
    "max|delta| =", round(max(abs(probs_luce - p_luce_scaled)), 8),
    " <- INVARIANT (good)\n\n")

cat("Summary of scale uniqueness:\n")
cat("  Softmax:     interval scale (shift-invariant)         -- differs from vNM\n")
cat("  Simple rule: ratio scale  (positive-scaling invariant)-- differs from vNM\n")
cat("  vNM utility: affine class (both shift AND scaling)    -- stricter than both\n\n")
cat("Implication: neither M3 rule straightforwardly implements vNM utility.\n")
cat("The softmax parameterisation is closer: both are shift-invariant.\n")
cat("The key gap is that vNM utilities also scale freely, while\n")
cat("the M3 softmax scale is pinned by fixing b=0 (the reference category).\n\n")

# ---- 5. IIA holds for both choice rules ----

cat("=== 5. Independence of Irrelevant Alternatives (IIA) ===\n\n")

check_iia <- function(prob_fn, a, n, label) {
  p_full <- prob_fn(a, n)
  # Remove category 3 (npl) and recompute
  p_reduced <- prob_fn(a[1:2], n[1:2])
  ratio_full    <- p_full[1]    / p_full[2]
  ratio_reduced <- p_reduced[1] / p_reduced[2]
  cat(sprintf("%s: P(corr)/P(other) full=%.4f, reduced=%.4f, |delta|=%.8f -> %s\n",
              label, ratio_full, ratio_reduced,
              abs(ratio_full - ratio_reduced),
              ifelse(abs(ratio_full - ratio_reduced) < 1e-6, "IIA HOLDS", "IIA VIOLATED")))
}

check_iia(softmax_m3, a, n, "Softmax")
check_iia(luce_m3,    a_s + b, n, "Simple ")

cat("\nBoth rules satisfy IIA because the ratio P(i)/P(j) depends only on a_i, a_j, n_i, n_j.\n")
cat("(The background noise b is per-category; removing a category removes its contribution\n")
cat("from the denominator but leaves the numerator ratio unchanged.)\n\n")
cat("Where the rules differ is not in IIA but in how they handle varying n_i:\n")

n_alt <- c(1L, 8L, 5L)  # more other items
p_soft_alt  <- softmax_m3(a, n_alt)
p_luce_alt  <- luce_m3(a_s + b, n_alt)
cat(sprintf("  Doubling n_other from 4->8:\n"))
cat(sprintf("    Softmax: P = (%.3f, %.3f, %.3f)\n", p_soft_alt[1], p_soft_alt[2], p_soft_alt[3]))
cat(sprintf("    Simple:  P = (%.3f, %.3f, %.3f)\n", p_luce_alt[1], p_luce_alt[2], p_luce_alt[3]))
cat("  The ratio P(corr)/P(npl):\n")
cat(sprintf("    Softmax: %.4f -> %.4f (unchanged by doubling n_other)\n",
            softmax_m3(a, n)[1] / softmax_m3(a, n)[3],
            p_soft_alt[1] / p_soft_alt[3]))
cat(sprintf("    Simple:  %.4f -> %.4f (unchanged by doubling n_other)\n",
            luce_m3(a_s + b, n)[1] / luce_m3(a_s + b, n)[3],
            p_luce_alt[1] / p_luce_alt[3]))
cat("  IIA holds in both cases (doubling options in 'other' leaves corr/npl ratio fixed).\n")
