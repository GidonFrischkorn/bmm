// log-PDF of the censored shifted Wald model
real cswald_lpdf(real rt, real mu, real drift, real bound, real ndt, real s,
                 real sndt, int response) {
  if (response == 1) {
    return swald_sndt_lpdf(rt | drift, bound, ndt, sndt, s);
  } else {
    return swald_sndt_lccdf(rt | drift, bound, ndt, sndt, s);
  }
}

// vectorized overload used as the loop = FALSE family, which configure_model
// selects only while sndt is fixed at 0: observed responses take the vectorized
// closed-form density, censored responses the vectorized survivor, both from
// the shared helpers. A non-zero sndt would make the convolution lccdf-bound,
// where the per-observation loop is faster, so it delegates to the scalar form
real cswald_lpdf(vector rt, vector mu, vector drift, vector bound,
                 vector ndt, vector s, vector sndt, array[] int dec) {
  int N = rows(rt);
  real lp = 0;

  if (min(sndt) < 0) return negative_infinity();
  if (max(sndt) >= 1e-8) {
    for (n in 1:N) {
      if (dec[n] == 1) {
        lp += swald_sndt_lpdf(rt[n] | drift[n], bound[n], ndt[n], sndt[n], s[n]);
      } else {
        lp += swald_sndt_lccdf(rt[n] | drift[n], bound[n], ndt[n], sndt[n], s[n]);
      }
    }
    return lp;
  }

  array[N] int idx1;
  array[N] int idx0;
  int n1 = 0;
  int n0 = 0;

  for (n in 1:N) {
    if (dec[n] == 1) {
      n1 += 1;
      idx1[n1] = n;
    } else {
      n0 += 1;
      idx0[n0] = n;
    }
  }

  if (n1 > 0) {
    array[n1] int ii = idx1[1:n1];
    lp += sum(swald_log_dens_vec(rt[ii] - ndt[ii], drift[ii], bound[ii], s[ii]));
  }

  if (n0 > 0) {
    // censored observations with rt <= ndt have survival 1 and contribute 0,
    // so they are dropped rather than passed to the survivor helper
    array[n0] int jj = idx0[1:n0];
    vector[n0] t0 = rt[jj] - ndt[jj];
    array[n0] int keep;
    int nk = 0;
    for (k in 1:n0) {
      if (t0[k] > 0) {
        nk += 1;
        keep[nk] = jj[k];
      }
    }
    if (nk > 0) {
      array[nk] int oo = keep[1:nk];
      lp += sum(swald_log_surv_vec(rt[oo] - ndt[oo], drift[oo], bound[oo], s[oo]));
    }
  }

  return lp;
}
