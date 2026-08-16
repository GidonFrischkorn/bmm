// log-PDF of competing risks shifted Wald model. With sndt > 0 the two
// accumulators receive independent non-decision-time draws (a race between
// total finishing times) rather than one shared draw per trial. The two agree
// exactly on choice probabilities at zr = 0.5, where the second-order term
// cancels by symmetry; away from it the densities differ by up to ~10% and the
// choice probabilities by up to ~1 percentage point at sndt = 0.3. See the
// "Trial-to-trial variability in the non-decision time" section of ?cswald
real cswald_crisk_lpdf(real rt, real mu, real drift, real bound, real ndt,
                       real zr, real s, real sndt, int response) {
  // compute bounds for upper and lower response
  real bound_upper = bound - zr*bound;
  real bound_lower = zr*bound;

  // compute lpdf dependent on response type
  if (response == 1) {
    return swald_sndt_lpdf(rt | drift, bound_upper, ndt, sndt, s)
           + swald_sndt_lccdf(rt | -drift, bound_lower, ndt, sndt, s);
  } else {
    return swald_sndt_lpdf(rt | -drift, bound_lower, ndt, sndt, s)
           + swald_sndt_lccdf(rt | drift, bound_upper, ndt, sndt, s);
  }
}

// vectorized overload used as the loop = FALSE family, which configure_model
// selects only while sndt is fixed at 0: every observation contributes the
// winning accumulator's density and the losing accumulator's survivor, both
// from the shared helpers. A non-zero sndt delegates to the scalar form, where
// the lccdf-bound convolution makes the per-observation loop faster
real cswald_crisk_lpdf(vector rt, vector mu, vector drift, vector bound,
                       vector ndt, vector zr, vector s, vector sndt,
                       array[] int dec) {
  int N = rows(rt);

  if (min(sndt) < 0) return negative_infinity();
  if (max(sndt) >= 1e-8) {
    real lp = 0;
    for (n in 1:N) {
      real bu = bound[n] - zr[n] * bound[n];
      real bl = zr[n] * bound[n];
      if (dec[n] == 1) {
        lp += swald_sndt_lpdf(rt[n] | drift[n], bu, ndt[n], sndt[n], s[n])
              + swald_sndt_lccdf(rt[n] | -drift[n], bl, ndt[n], sndt[n], s[n]);
      } else {
        lp += swald_sndt_lpdf(rt[n] | -drift[n], bl, ndt[n], sndt[n], s[n])
              + swald_sndt_lccdf(rt[n] | drift[n], bu, ndt[n], sndt[n], s[n]);
      }
    }
    return lp;
  }

  // both accumulators share rt - ndt, so a single rt <= ndt makes the winner's
  // density (and thus the summed target) -inf
  vector[N] t = rt - ndt;
  if (min(t) <= 0) return negative_infinity();

  // winner = the accumulator matching the decision (drift toward its bound),
  // loser = the opposite accumulator with mirrored drift; selecting via the
  // 0/1 data vector w keeps everything vectorized
  vector[N] w = to_vector(dec);
  vector[N] bound_upper = bound - zr .* bound;
  vector[N] bound_lower = zr .* bound;
  vector[N] drift_win = (2 * w - 1) .* drift;
  vector[N] bound_win = w .* bound_upper + (1 - w) .* bound_lower;
  vector[N] bound_lose = bound - bound_win;

  return sum(swald_log_dens_vec(t, drift_win, bound_win, s))
         + sum(swald_log_surv_vec(t, -drift_win, bound_lose, s));
}
