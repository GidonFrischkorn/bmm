suppressMessages(devtools::load_all("."))
suppressMessages(library(brms))
invisible(capture.output(source("local/exploration/m3-utility/11_utility_wrapper.R")))
cat("Functions loaded.\n\n")

dir.create("local/exploration/m3-utility/results", showWarnings = FALSE)

# ============================================================================
# Smoke Fit 1: EU VDR (Arch A, UC1)
# ============================================================================
cat("=== Smoke Fit 1: EU VDR (Arch A, UC1) ===\n\n")

set.seed(42)
N_subj <- 20; N_trials <- 60
a_subj <- rnorm(N_subj, 2.0, 0.3)
c_subj <- rnorm(N_subj, 3.0, 0.3)
g_subj <- rnorm(N_subj, 0.8, 0.2)

rows <- lapply(seq_len(N_subj), function(i) {
  V <- sample(c(1, 3, 5), N_trials, replace = TRUE)
  act_c  <- a_subj[i] + c_subj[i] + g_subj[i] * V
  act_o  <- a_subj[i] + g_subj[i] * V
  act_n  <- rep(0, N_trials)
  denom  <- 1L * exp(act_c) + 4L * exp(act_o) + 5L * exp(act_n)
  p <- cbind(1L * exp(act_c) / denom,
             4L * exp(act_o) / denom,
             5L * exp(act_n) / denom)
  resp <- t(apply(p, 1, function(pr) rmultinom(1, 1, pr)))
  data.frame(subj = i, V_corr = V, V_other = V,
             n_corr = 1L, n_other = 4L, n_npl = 5L,
             corr = resp[, 1L], other = resp[, 2L], npl = resp[, 3L])
})
d1 <- do.call(rbind, rows)
cat("Data: N =", nrow(d1), "rows,", N_subj, "subjects\n")

model1 <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "none"
)

formula1 <- do.call(bmf, c(
  m3_utility_formula(model1),
  list(
    a     ~ 1 + (1 | subj),
    c     ~ 1 + (1 | subj),
    gamma ~ 1 + (1 | subj)
  )
))

cat("Starting fit (1 chain, warmup=500, iter=1000)...\n")
t0 <- proc.time()["elapsed"]
fit1 <- bmm(
  formula = formula1,
  data    = d1,
  model   = model1,
  chains  = 1,
  warmup  = 500,
  iter    = 1000,
  backend = "cmdstanr",
  silent  = 2
)
t1 <- proc.time()["elapsed"]
cat(sprintf("Done: %.1f sec\n\n", t1 - t0))

s1 <- summary(fit1)$fixed
cat("Fixed effects:\n")
print(round(s1[, c("Estimate", "l-95% CI", "u-95% CI", "Rhat", "Bulk_ESS")], 3))

g_row <- s1[grepl("gamma_Intercept", rownames(s1)), ]
a_row <- s1[grepl("^b_a_Intercept",     rownames(s1)), ]
c_row <- s1[grepl("^b_c_Intercept",     rownames(s1)), ]

if (nrow(a_row) == 0) a_row <- s1[grepl("a_Intercept", rownames(s1)), ][1, ]
if (nrow(c_row) == 0) c_row <- s1[grepl("c_Intercept", rownames(s1)), ][1, ]

divs     <- sum(nuts_params(fit1)$Value[nuts_params(fit1)$Parameter == "divergent__"])
max_rhat <- max(s1[, "Rhat"], na.rm = TRUE)

cat(sprintf("\nMax Rhat: %.3f | Divergences: %d\n", max_rhat, divs))
cat(sprintf("gamma: true=0.80 est=%.3f [%.3f, %.3f] covered=%s\n",
            g_row$Estimate, g_row$`l-95% CI`, g_row$`u-95% CI`,
            g_row$`l-95% CI` <= 0.80 & 0.80 <= g_row$`u-95% CI`))
cat(sprintf("a:     true=2.00 est=%.3f [%.3f, %.3f]\n",
            a_row$Estimate, a_row$`l-95% CI`, a_row$`u-95% CI`))
cat(sprintf("c:     true=3.00 est=%.3f [%.3f, %.3f]\n\n",
            c_row$Estimate, c_row$`l-95% CI`, c_row$`u-95% CI`))

res1 <- data.frame(
  fit        = "smoke_A_UC1",
  param      = c("a", "c", "gamma"),
  true_value = c(2.0, 3.0, 0.8),
  estimate   = c(a_row$Estimate,     c_row$Estimate,     g_row$Estimate),
  ci_lower   = c(a_row$`l-95% CI`,   c_row$`l-95% CI`,   g_row$`l-95% CI`),
  ci_upper   = c(a_row$`u-95% CI`,   c_row$`u-95% CI`,   g_row$`u-95% CI`),
  covered    = c(a_row$`l-95% CI` <= 2.0 & 2.0 <= a_row$`u-95% CI`,
                 c_row$`l-95% CI` <= 3.0 & 3.0 <= c_row$`u-95% CI`,
                 g_row$`l-95% CI` <= 0.8 & 0.8 <= g_row$`u-95% CI`),
  max_rhat   = max_rhat,
  divergences = divs,
  elapsed_sec = round(t1 - t0, 1)
)
write.csv(res1, "local/exploration/m3-utility/results/15_smoke_A_UC1.csv",
          row.names = FALSE)
cat("Saved: results/15_smoke_A_UC1.csv\n\n")


# ============================================================================
# Smoke Fit 2: Composite PT (Arch A, UC2)
# ============================================================================
cat("=== Smoke Fit 2: Composite PT (Arch A, UC2) ===\n\n")

set.seed(43)
N_subj2   <- 20
N_trials2 <- 80
a_subj2     <- rnorm(N_subj2, 2.0, 0.3)
c_subj2     <- rnorm(N_subj2, 3.0, 0.3)
gamma_subj2 <- rnorm(N_subj2, 0.6, 0.2)
alpha_subj2 <- rnorm(N_subj2, 0.7, 0.15)

n_corr_val <- 1L
n_npl_val  <- 5L
n_other_levels <- c(2L, 4L, 6L)

rows2 <- lapply(seq_len(N_subj2), function(i) {
  V       <- sample(c(1, 3, 5), N_trials2, replace = TRUE)
  n_other <- sample(n_other_levels, N_trials2, replace = TRUE)
  n_total <- n_corr_val + n_other + n_npl_val

  p_corr_raw  <- n_corr_val  / n_total
  p_other_raw <- n_other     / n_total
  p_npl_raw   <- n_npl_val   / n_total

  ai  <- alpha_subj2[i]
  w_c <- exp(-((-log(p_corr_raw))^ai))
  w_o <- exp(-((-log(p_other_raw))^ai))
  w_n <- exp(-((-log(p_npl_raw))^ai))

  act_c <- a_subj2[i] + c_subj2[i] + gamma_subj2[i] * V + log(w_c)
  act_o <- a_subj2[i]               + gamma_subj2[i] * V + log(w_o)
  act_n <- log(w_n)

  denom <- exp(act_c) + exp(act_o) + exp(act_n)
  p <- cbind(exp(act_c) / denom, exp(act_o) / denom, exp(act_n) / denom)
  resp <- t(apply(p, 1, function(pr) rmultinom(1, 1, pr)))

  data.frame(
    subj    = i,
    V_corr  = V, V_other = V,
    n_corr  = n_corr_val,
    n_other = n_other,
    n_npl   = n_npl_val,
    p_corr  = p_corr_raw,
    p_other = p_other_raw,
    p_npl   = p_npl_raw,
    corr    = resp[, 1L],
    other   = resp[, 2L],
    npl     = resp[, 3L]
  )
})
d2 <- do.call(rbind, rows2)
cat("Data: N =", nrow(d2), "rows, n_other in {",
    paste(sort(unique(d2$n_other)), collapse=","), "} (varied for Prelec identifiability)\n")

model2 <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "prelec"
)

formula2 <- do.call(bmf, c(
  m3_utility_formula(model2),
  list(
    a     ~ 1 + (1 | subj),
    c     ~ 1 + (1 | subj),
    gamma ~ 1 + (1 | subj),
    alpha ~ 1 + (1 | subj)
  )
))

cat("Activation formulas generated:\n")
for (nm in names(m3_utility_formula(model2))) {
  cat(sprintf("  %s\n", deparse(m3_utility_formula(model2)[[nm]])))
}
cat("\nStarting fit (1 chain, warmup=500, iter=1000)...\n")

t2 <- proc.time()["elapsed"]
fit2 <- bmm(
  formula = formula2,
  data    = d2,
  model   = model2,
  chains  = 1,
  warmup  = 500,
  iter    = 1000,
  backend = "cmdstanr",
  silent  = 2
)
t3 <- proc.time()["elapsed"]
cat(sprintf("Done: %.1f sec\n\n", t3 - t2))

s2 <- summary(fit2)$fixed
cat("Fixed effects:\n")
print(round(s2[, c("Estimate", "l-95% CI", "u-95% CI", "Rhat", "Bulk_ESS")], 3))

g_row2 <- s2[grepl("gamma_Intercept", rownames(s2)), ]
al_row2 <- s2[grepl("alpha_Intercept", rownames(s2)), ]
a_row2  <- s2[grepl("a_Intercept", rownames(s2)), ][1, ]
c_row2  <- s2[grepl("c_Intercept", rownames(s2)), ][1, ]

divs2     <- sum(nuts_params(fit2)$Value[nuts_params(fit2)$Parameter == "divergent__"])
max_rhat2 <- max(s2[, "Rhat"], na.rm = TRUE)

cat(sprintf("\nMax Rhat: %.3f | Divergences: %d\n", max_rhat2, divs2))
cat(sprintf("gamma: true=0.60 est=%.3f [%.3f, %.3f] covered=%s\n",
            g_row2$Estimate, g_row2$`l-95% CI`, g_row2$`u-95% CI`,
            g_row2$`l-95% CI` <= 0.6 & 0.6 <= g_row2$`u-95% CI`))
cat(sprintf("alpha: true=0.70 est=%.3f [%.3f, %.3f] covered=%s\n",
            al_row2$Estimate, al_row2$`l-95% CI`, al_row2$`u-95% CI`,
            al_row2$`l-95% CI` <= 0.7 & 0.7 <= al_row2$`u-95% CI`))

res2 <- data.frame(
  fit        = "smoke_A_UC2",
  param      = c("a", "c", "gamma", "alpha"),
  true_value = c(2.0, 3.0, 0.6, 0.7),
  estimate   = c(a_row2$Estimate,   c_row2$Estimate,   g_row2$Estimate,   al_row2$Estimate),
  ci_lower   = c(a_row2$`l-95% CI`, c_row2$`l-95% CI`, g_row2$`l-95% CI`, al_row2$`l-95% CI`),
  ci_upper   = c(a_row2$`u-95% CI`, c_row2$`u-95% CI`, g_row2$`u-95% CI`, al_row2$`u-95% CI`),
  covered    = c(a_row2$`l-95% CI` <= 2.0 & 2.0 <= a_row2$`u-95% CI`,
                 c_row2$`l-95% CI` <= 3.0 & 3.0 <= c_row2$`u-95% CI`,
                 g_row2$`l-95% CI` <= 0.6 & 0.6 <= g_row2$`u-95% CI`,
                 al_row2$`l-95% CI` <= 0.7 & 0.7 <= al_row2$`u-95% CI`),
  max_rhat   = max_rhat2,
  divergences = divs2,
  elapsed_sec = round(t3 - t2, 1)
)
write.csv(res2, "local/exploration/m3-utility/results/15_smoke_A_UC2.csv",
          row.names = FALSE)
cat("Saved: results/15_smoke_A_UC2.csv\n\n")

cat("=== All smoke fits complete ===\n")
