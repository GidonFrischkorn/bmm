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

// log-CDF of the shifted Wald: F_W(t) = Phi(z1) + exp(c) * Phi(z2), a sum of
// two log-space terms and therefore stable at any gap between them
real swald_lcdf(real rt, real drift, real bound, real ndt, real sigma) {
  real t_shifted = rt - ndt;
  if (t_shifted <= 0) return negative_infinity();

  real sigma_sqrt_t = sigma * sqrt(t_shifted);
  real z1 = (drift * t_shifted - bound) / sigma_sqrt_t;
  real z2 = -(drift * t_shifted + bound) / sigma_sqrt_t;
  real log_c = 2 * bound * drift / square(sigma);

  return log_sum_exp(std_normal_lcdf(z1 | ), log_c + std_normal_lcdf(z2 | ));
}

// G(x) = int_0^x S_W(u) du = x * S_W(x) + M1(x), extended by G(x) = x for
// x <= 0 where S_W = 1, which lets the convolution below cover the
// partial-support strip without a special case. M1(x) = int_0^x u f_W(u) du is
// the partial expectation of the (possibly defective) Wald. Mirrors .gwald()
// in R/distributions.R; the survivor is taken from swald_lccdf so that both
// implementations reach it by the same log-space route
real swald_gint(real x, real drift, real bound, real sigma) {
  if (x <= 0) return x;

  real sigma_sq = square(sigma);
  real sqrt_x = sqrt(x);
  real dx = drift * x;
  real z1 = (dx - bound) / (sigma * sqrt_x);
  real z2 = -(dx + bound) / (sigma * sqrt_x);
  real q = exp(2 * bound * drift / sigma_sq + std_normal_lcdf(z2 | ));
  real m1;

  if (abs(drift) < 1e-6 * sigma_sq / bound) {
    // mu = bound / drift diverges as drift -> 0 while the Phi-bracket vanishes;
    // use the exact drift = 0 limit of M1 instead of the 0 * inf cancellation
    real w = bound / (sigma * sqrt_x);
    m1 = 2 * bound * (sqrt_x * exp(std_normal_lpdf(w | )) / sigma
                      - (bound / sigma_sq) * Phi(-w));
  } else {
    m1 = (bound / drift) * (Phi(z1) - q);
  }

  return x * exp(swald_lccdf(x | drift, bound, 0, sigma)) + m1;
}

// log of (1/sndt) * int_{x-sndt}^{x} S_W(u) du by composite Simpson on four
// intervals, evaluated in log space. Every Simpson weight is positive, so this
// cannot cancel however deep the tail gets; it is the recovery path for the
// G-difference below. Mirrors .simpson_log_mean() in R/distributions.R
real swald_log_surv_mean(real x, real drift, real bound, real sigma, real sndt) {
  vector[5] weights = log(to_vector({1, 4, 2, 4, 1}) / 12);
  vector[5] terms;
  for (k in 1:5) {
    terms[k] = swald_lccdf(x - sndt + (k - 1) * sndt / 4 | drift, bound, 0, sigma)
               + weights[k];
  }
  return log_sum_exp(terms);
}

// log-PDF of the shifted Wald with uniform trial-to-trial variability in the
// non-decision time, onset convention NDT ~ uniform(ndt, ndt + sndt):
// f(t) = [S_W(t - ndt - sndt) - S_W(t - ndt)] / sndt (Miller et al. 2018, Eq. 6)
real swald_sndt_lpdf(real rt, real drift, real bound, real ndt, real sndt, real sigma) {
  if (sndt < 0) return negative_infinity();
  // the convolution is continuous at sndt = 0, so tiny sndt takes the plain
  // density (this is also the path for the default fixed sndt = 0)
  if (sndt < 1e-8) return swald_lpdf(rt | drift, bound, ndt, sigma);

  real t1 = rt - ndt;
  if (t1 <= 0) return negative_infinity();

  // strip ndt < rt <= ndt + sndt: the earlier survivor is exactly 1, so the
  // density reduces to F_W(t1) / sndt. The log-CDF is stable at any gap; the
  // survivor route log_diff_exp(0, log S(t1)) returns -inf once log S(t1)
  // ~ -F(t1) underflows below the smallest subnormal, although the true log
  // density ~ -bound^2 / (2 sigma^2 t1) is still perfectly representable
  if (t1 <= sndt) return swald_lcdf(rt | drift, bound, ndt, sigma) - log(sndt);

  real surv_early = swald_lccdf(rt | drift, bound, ndt + sndt, sigma);
  real surv_late = swald_lccdf(rt | drift, bound, ndt, sigma);

  // for defective (negative-drift) accumulators both survivors converge to the
  // same positive constant in the deep tail, so their log-difference drops
  // below fp precision; the midpoint rule is second order in sndt and stable
  if (surv_early - surv_late < 1e-8) {
    return swald_lpdf(rt | drift, bound, ndt + sndt / 2, sigma);
  }
  return swald_log_diff_exp(surv_early, surv_late) - log(sndt);
}

// log survivor of the shifted Wald + uniform NDT, for censored observations:
// S_conv(t) = [G(x1) - G(x1 - sndt)] / sndt with x1 = t - ndt. Once that
// difference falls below 1e-8 of the terms themselves it has lost half its
// mantissa and log() of it is noise -- non-monotone in t, with a gradient that
// is zero or garbage -- so the Simpson route recovers the same integral
// without cancelling. An absolute floor would admit hundreds of nats of that
// noise before firing
real swald_sndt_lccdf(real rt, real drift, real bound, real ndt, real sndt, real sigma) {
  if (sndt < 0) return negative_infinity();
  if (sndt < 1e-8) return swald_lccdf(rt | drift, bound, ndt, sigma);

  real x1 = rt - ndt;
  if (x1 <= 0) return 0;

  real g_hi = swald_gint(x1, drift, bound, sigma);
  real g_lo = swald_gint(x1 - sndt, drift, bound, sigma);
  real delta = g_hi - g_lo;

  if (delta <= 1e-8 * fmax(abs(g_hi), abs(g_lo))) {
    return swald_log_surv_mean(x1, drift, bound, sigma, sndt);
  }
  return log(delta / sndt);
}
