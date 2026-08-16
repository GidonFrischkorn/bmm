// log(exp(a) - exp(b)) for survivor differences. Floating-point rounding can
// push b >= a when the true difference underflows; the continuation there is
// probability zero (-inf), not the NaN that log_diff_exp would emit, which
// poisons the gradient of the entire proposal
real swald_log_diff_exp(real a, real b) {
  if (b >= a) return negative_infinity();
  return log_diff_exp(a, b);
}

// log-PDF of the shifted Wald distribution
// Optimized to compute entirely in log-space for numerical stability
real swald_lpdf(real rt, real drift, real bound, real ndt, real sigma) {
  // compute shifted response time
  real t_shifted = rt - ndt;
  if (t_shifted <= 0) return negative_infinity();

  // pre-compute common terms
  real sigma_sq = square(sigma);
  real log_t = log(t_shifted);

  // log-normalization: log(bound / sqrt(2*pi*sigma^2*t^3))
  // = log(bound) - 0.5*(log(2*pi) + 2*log(sigma) + 3*log(t))
  real log_norm = log(bound) - 0.5 * (log(2 * pi()) + 2 * log(sigma) + 3 * log_t);

  // log-kernel: -((bound - drift*t)^2) / (2*sigma^2*t)
  real residual = bound - drift * t_shifted;
  real log_kernel = -0.5 * square(residual) / (sigma_sq * t_shifted);

  return log_norm + log_kernel;
}

// log shifted Wald survivor function (complementary CDF)
// Optimized for numerical stability using Stan's log-space CDF functions
// and log_diff_exp to avoid overflow/underflow issues
real swald_lccdf(real rt, real drift, real bound, real ndt, real sigma) {
  // compute shifted response time
  real t_shifted = rt - ndt;
  if (t_shifted <= 0) return 0;  // process hasn't started, survival = 1

  // pre-compute common terms
  real sqrt_t = sqrt(t_shifted);
  real sigma_sqrt_t = sigma * sqrt_t;
  real sigma_sq = square(sigma);

  // standardized arguments for the normal CDF
  real z1 = (drift * t_shifted - bound) / sigma_sqrt_t;
  real z2 = -(drift * t_shifted + bound) / sigma_sqrt_t;

  // log of the scaling constant (kept in log-space to avoid overflow)
  real log_c = 2 * bound * drift / sigma_sq;

  // Survival function: S(t) = 1 - Phi(z1) - exp(c)*Phi(z2)
  //                         = Phi(-z1) - exp(c)*Phi(z2)
  //
  // Compute in log-space for numerical stability:
  // log(S(t)) = log(exp(log_Phi(-z1)) - exp(log_c + log_Phi(z2)))
  //           = log_diff_exp(std_normal_lcdf(-z1), log_c + std_normal_lcdf(z2))
  // std_normal_lcdf(-z1) is used instead of the equivalent std_normal_lccdf(z1),
  // which underflows to -Inf for z1 > ~8.3
  real log_term1 = std_normal_lcdf(-z1 | );  // log(1 - Phi(z1)) = log(Phi(-z1))
  real log_term2 = log_c + std_normal_lcdf(z2 | );  // log(exp(c) * Phi(z2))

  return swald_log_diff_exp(log_term1, log_term2);
}

// Vectorized counterparts of swald_lpdf / swald_lccdf, used by the loop = FALSE
// family overloads. They take the ALREADY-SHIFTED time t = rt - ndt so that the
// two model chunks share one implementation of the algebra rather than
// transcribing it per version. Keep them algebraically in step with the scalar
// forms above; the R mirrors are .dwald() and .pwald() in R/distributions.R.

// t <= 0 anywhere makes the summed target -inf, exactly as the scalar sum would
vector swald_log_dens_vec(vector t, vector drift, vector bound, vector sigma) {
  int n = rows(t);
  if (min(t) <= 0) return rep_vector(negative_infinity(), n);
  vector[n] residual = bound - drift .* t;
  return log(bound)
         - 0.5 * (log(2 * pi()) + 2 * log(sigma) + 3 * log(t))
         - 0.5 * square(residual) ./ (square(sigma) .* t);
}

// requires t > 0 elementwise; callers drop or early-return on non-positive t.
// The z-score preparation vectorizes, but Stan has no elementwise log-CDF, so
// the std_normal_lcdf pair stays in a scalar loop
vector swald_log_surv_vec(vector t, vector drift, vector bound, vector sigma) {
  int n = rows(t);
  vector[n] denom = sigma .* sqrt(t);
  vector[n] dxt = drift .* t;
  vector[n] z1 = (dxt - bound) ./ denom;
  vector[n] z2 = -(dxt + bound) ./ denom;
  vector[n] log_c = 2 * (bound .* drift) ./ square(sigma);
  vector[n] out;
  for (k in 1:n) {
    out[k] = swald_log_diff_exp(std_normal_lcdf(-z1[k] | ),
                                log_c[k] + std_normal_lcdf(z2[k] | ));
  }
  return out;
}
