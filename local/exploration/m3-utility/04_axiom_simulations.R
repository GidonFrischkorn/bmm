# 04_axiom_simulations.R
# Rational-choice axiom simulations for M3 choice rules
#
# Investigates which axioms each M3 variant satisfies and how choice patterns
# differ across rules, focusing on implications for utility-theory extensions.
#
# Run from the repo root:
#   source("local/exploration/m3-utility/04_axiom_simulations.R")

# ---- helpers ----------------------------------------------------------------

softmax_m3 <- function(a, n) {
  log_num <- a + log(n)
  exp(log_num - log(sum(exp(log_num))))
}

luce_m3 <- function(a, n, b = 0.1) {
  strengths <- (a + b) * n
  strengths / sum(strengths)
}

# ---- 1. IIA: both rules satisfy independence of irrelevant alternatives -------

cat("=== 1. IIA (Independence of Irrelevant Alternatives) ===\n\n")
cat("IIA: removing alternative k does not change ratio P(i)/P(j) for i,j != k.\n\n")

a <- c(3.0, 2.0, 1.0, 0.0)   # 4 alternatives
n <- c(1L, 4L, 3L, 5L)

for (label in c("Softmax", "Simple")) {
  prob_fn <- if (label == "Softmax") softmax_m3 else function(a, n) luce_m3(a, n)
  p_full <- prob_fn(a, n)
  p_red  <- prob_fn(a[c(1, 2, 4)], n[c(1, 2, 4)])  # remove alternative 3

  r_full <- p_full[1] / p_full[2]
  r_red  <- p_red[1]  / p_red[2]
  cat(sprintf("%s: P(1)/P(2) full=%.6f, reduced=%.6f, |delta|=%.2e -> IIA %s\n",
              label, r_full, r_red, abs(r_full - r_red),
              ifelse(abs(r_full - r_red) < 1e-9, "HOLDS", "VIOLATED")))
}

cat("\nBoth rules satisfy IIA because:\n")
cat("  Softmax: P(i)/P(j) = exp(a_i+log(n_i)) / exp(a_j+log(n_j)) -- doesn't involve k\n")
cat("  Simple:  P(i)/P(j) = (a_i+b)*n_i / (a_j+b)*n_j             -- doesn't involve k\n\n")

# ---- 2. Regularity: adding alternatives cannot increase P(existing) ----------

cat("=== 2. Regularity (adding alternatives cannot increase P) ===\n\n")
cat("Regularity: P(i | S union {k}) <= P(i | S) for all i in S.\n\n")

a3 <- c(3.0, 2.0, 0.0)
n3 <- c(1L, 4L, 5L)
# Add a new alternative at position 3; original alternatives shift to indices 1, 2, 4
a4 <- c(3.0, 2.0, 1.5, 0.0)
n4 <- c(1L, 4L, 3L, 5L)

for (label in c("Softmax", "Simple")) {
  prob_fn <- if (label == "Softmax") softmax_m3 else function(a, n) luce_m3(a, n)
  p3 <- prob_fn(a3, n3)
  p4 <- prob_fn(a4, n4)
  # Compare probabilities of the SAME three alternatives (indices 1, 2, 4 in p4)
  violated <- any(p4[c(1, 2, 4)] > p3 + 1e-10)
  cat(sprintf("%s: P(orig 3 alts)=(%.3f,%.3f,%.3f) -> with new alt=(%.3f,%.3f,*%.3f*,%.3f) -- Regularity %s\n",
              label, p3[1], p3[2], p3[3],
              p4[1], p4[2], p4[3], p4[4],
              ifelse(!violated, "HOLDS", "VIOLATED")))
}
cat("\nRegularity always holds for any Luce-type model (softmax or ratio rule).\n\n")

# ---- 3. Stochastic transitivity -----------------------------------------------

cat("=== 3. Weak Stochastic Transitivity ===\n\n")
cat("WST: if P(A>{A,B})>=0.5 and P(B>{B,C})>=0.5, then P(A>{A,C})>=0.5.\n\n")

check_wst <- function(prob_fn, a, n, label) {
  pAB <- prob_fn(a[1:2], n[1:2])[1]
  pBC <- prob_fn(a[2:3], n[2:3])[1]
  pAC <- prob_fn(a[c(1, 3)], n[c(1, 3)])[1]
  wst_holds <- !(pAB >= 0.5 & pBC >= 0.5 & pAC < 0.5)
  cat(sprintf("%s: P(A|{A,B})=%.3f, P(B|{B,C})=%.3f, P(A|{A,C})=%.3f -> WST %s\n",
              label, pAB, pBC, pAC,
              ifelse(wst_holds, "HOLDS", "VIOLATED")))
}

check_wst(softmax_m3, a3, n3, "Softmax")
check_wst(function(a, n) luce_m3(a, n), a3, n3, "Simple ")
cat("\nBoth rules imply a complete strict ordering of alternatives -> WST always holds.\n\n")

# ---- 4. Set-size effect: how rules respond differently to varying n_i --------

cat("=== 4. Set-size effects (where the rules DIFFER) ===\n\n")
cat("The two rules agree on IIA but disagree on how option counts scale probabilities.\n\n")

a_base <- c(3.0, 2.0, 0.0)
n_ref  <- c(1L,  4L,  5L)

cat("Varying n_other while keeping a fixed:\n")
cat(sprintf("%-10s  %-18s  %-18s\n", "n_other", "Softmax P(corr)", "Simple P(corr)"))
for (n_other in c(1L, 2L, 4L, 8L, 16L)) {
  n_var <- c(1L, n_other, 5L)
  p_s <- softmax_m3(a_base, n_var)[1]
  p_l <- luce_m3(a_base, n_var)[1]
  cat(sprintf("%-10d  %-18.4f  %-18.4f\n", n_other, p_s, p_l))
}

cat("\nKey difference: how P(correct) scales with n_other.\n")
cat("  Softmax: P(corr) decreases faster (exponential in activation)\n")
cat("  Simple:  P(corr) decreases slower (linear in activation)\n\n")

# ---- 5. Similarity / Debreu effect (IIA limitation) -------------------------

cat("=== 5. Similarity effect: IIA limitation ===\n\n")
cat("IIA assumes proportional stealing when adding alternatives.\n")
cat("In practice (Debreu 1960), similar alternatives steal disproportionately.\n")
cat("Both M3 rules cannot represent this — they always steal proportionally.\n\n")

# Illustration: adding a new 'other' item that is similar to existing 'other'
a_corr  <- 3.0
a_other <- 2.0
a_npl   <- 0.0

# Scenario 1: 4 other items, 5 npl
p1 <- softmax_m3(c(a_corr, a_other, a_npl), c(1L, 4L, 5L))

# Scenario 2: add 1 more item to "other" category
p2 <- softmax_m3(c(a_corr, a_other, a_npl), c(1L, 5L, 5L))

# IIA prediction: ratio P(corr)/P(npl) should be unchanged
ratio1 <- p1[1] / p1[3]
ratio2 <- p2[1] / p2[3]
cat(sprintf("P(corr)/P(npl) with 4 other items: %.4f\n", ratio1))
cat(sprintf("P(corr)/P(npl) with 5 other items: %.4f\n", ratio2))
cat(sprintf("|delta| = %.2e  (should be ~0 under IIA)\n\n", abs(ratio1 - ratio2)))

cat("What IIA CANNOT predict: if the new 'other' item is a near-duplicate of an\n")
cat("existing 'other' item, it should steal mainly from 'other', not from 'correct'.\n")
cat("For such similarity effects, a model outside the Luce/softmax family is needed\n")
cat("(e.g., Nested Logit, Cross-Nested Logit, or elimination-by-aspects models).\n\n")

# ---- 6. Scale calibration: softmax activation ≠ vNM utility -----------------

cat("=== 6. Scale calibration: M3 softmax activation vs vNM utility ===\n\n")
cat("vNM utility is unique up to U' = alpha*U + beta (positive linear transform).\n")
cat("Softmax is invariant only to shifts (beta), not to scales (alpha).\n\n")

a_base <- c(3.0, 2.0, 0.0)
n      <- c(1L, 4L, 5L)

cat("Probability rankings under different utility scalings:\n")
cat(sprintf("%-25s  P(corr)  P(other)  P(npl)\n", "Utility function"))
for (alpha in c(0.5, 1.0, 2.0, 5.0)) {
  beta <- -1.0
  u    <- alpha * a_base + beta
  p    <- softmax_m3(u, n)
  cat(sprintf("U' = %.1f*U + (%.1f)      %.4f   %.4f    %.4f\n",
              alpha, beta, p[1], p[2], p[3]))
}
cat("\nFor true vNM utilities, all rows should give the same probabilities.\n")
cat("For softmax M3, only rows with the SAME alpha give the same probabilities.\n")
cat("=> The scale of activations is substantively meaningful in the softmax M3,\n")
cat("   unlike vNM utilities where scale is arbitrary.\n\n")
cat("This is actually USEFUL for cognitive modelling: the activation scale reflects\n")
cat("something like 'decisiveness' (inverse temperature), which vNM doesn't capture.\n")

# ---- 7. Comparison of EU-extended model vs base M3 --------------------------

cat("=== 7. Distinguishability of EU extension from base M3 ===\n\n")
cat("Can we distinguish the EU-extended M3 (with gamma) from the base M3\n")
cat("(which already allows activation predictors) just by looking at probabilities?\n\n")

a_base_no_eu <- c(3.5, 2.5, 0.0)  # base M3 with higher activations
gamma <- 0.8
V_high <- 2; V_low <- 1

p_eu_high <- softmax_m3(c(3.0 + gamma*V_high, 2.0 + gamma*V_low, 0.0), c(1L, 4L, 5L))
p_eu_low  <- softmax_m3(c(3.0 + gamma*V_low,  2.0 + gamma*V_high, 0.0), c(1L, 4L, 5L))
p_base    <- softmax_m3(a_base_no_eu, c(1L, 4L, 5L))

cat(sprintf("EU model, high-value trial:  P = (%.3f, %.3f, %.3f)\n", p_eu_high[1], p_eu_high[2], p_eu_high[3]))
cat(sprintf("EU model, low-value trial:   P = (%.3f, %.3f, %.3f)\n", p_eu_low[1],  p_eu_low[2],  p_eu_low[3]))
cat(sprintf("Base M3 (no EU):             P = (%.3f, %.3f, %.3f)\n", p_base[1],    p_base[2],    p_base[3]))
cat("\nThe EU model predicts DIFFERENT probabilities depending on item value,\n")
cat("which is NOT possible in the base M3 without a value predictor.\n")
cat("Empirical test: does P(correct) correlate with V(correct) in VDR data?\n")
cat("If yes and value is NOT already a predictor, the EU term adds explanatory power.\n")
