# WP2: Multinomial MPT validation — PCM and SMM end-to-end
#
# Validates the multinomial formula emitter in mpt_to_brms() on:
#   (a) Pair-Clustering Model (PCM): 3 response cats (C, E, U), 1 tree
#   (b) 3-source Source Monitoring Model (SMM): 3 cats (A, B, New), 3 trees,
#       with stick-breaking reparameterization for the guessing simplex
#
# Both models: simulate data with known parameters -> fit -> check recovery.
# Run from repo root:  Rscript local/exploration/mpt/07_multinomial_demo.R
#
# Note: parameter names use no underscores (brms nlpar constraint):
#   dA, dB (not d_A, d_B), gA, gB, gNew (not g_A, g_B, g_New).

suppressMessages(library(brms))
suppressMessages(library(dplyr))
suppressMessages(library(tidyr))

source("local/exploration/mpt/03_mpt_parser.R", local = TRUE)

# =============================================================================
# PART A: Pair-Clustering Model (PCM)
# Batchelder & Riefer (1980)
#   P(C) = cp + (1-cp)*rp^2
#   P(E) = 2*(1-cp)*rp*(1-rp)
#   P(U) = (1-cp)*(1-rp)^2
# True parameters: cp = 0.60, rp = 0.75
# =============================================================================

cat("====================================================\n")
cat("PART A: Pair-Clustering Model (PCM)  — multinomial\n")
cat("====================================================\n\n")

cp_true          <- 0.60
rp_true          <- 0.75
n_participants_a <- 60
n_pairs          <- 40

pC_true <- cp_true + (1 - cp_true) * rp_true^2
pE_true <- 2 * (1 - cp_true) * rp_true * (1 - rp_true)
pU_true <- (1 - cp_true) * (1 - rp_true)^2
stopifnot(abs(pC_true + pE_true + pU_true - 1) < 1e-10)

cat(sprintf("True: cp = %.2f, rp = %.2f\n", cp_true, rp_true))
cat(sprintf("Branch probs: P(C)=%.3f, P(E)=%.3f, P(U)=%.3f\n",
            pC_true, pE_true, pU_true))

set.seed(2024)
sim_pcm <- tibble(
  participant = seq_len(n_participants_a),
  dummy_cond  = "study"
) |>
  mutate(
    counts_raw = lapply(seq_len(n()), function(i) {
      rmultinom(1, n_pairs, c(pC_true, pE_true, pU_true))[, 1]
    }),
    C = sapply(counts_raw, `[`, 1),
    E = sapply(counts_raw, `[`, 2),
    U = sapply(counts_raw, `[`, 3),
    n = n_pairs
  ) |>
  select(-counts_raw)

cat("\nSimulated PCM data (first 5 rows):\n")
print(head(sim_pcm, 5))
cat(sprintf("Aggregate proportions: P(C)=%.3f, P(E)=%.3f, P(U)=%.3f\n",
            sum(sim_pcm$C) / (n_participants_a * n_pairs),
            sum(sim_pcm$E) / (n_participants_a * n_pairs),
            sum(sim_pcm$U) / (n_participants_a * n_pairs)))

pcm_str <- paste0(
  "\ncp + (1 - cp) * rp * rp         # C\n",
  "2 * (1 - cp) * rp * (1 - rp)    # E\n",
  "(1 - cp) * (1 - rp) * (1 - rp)  # U\n"
)
spec_pcm <- parse_mpt_string(pcm_str, tree_conditions = c("study"),
                              condition = "dummy_cond")

out_pcm      <- mpt_to_brms(spec_pcm, predictor_formulas = list(cp = ~ 1, rp = ~ 1))
sim_pcm_prep <- out_pcm$data_prep(sim_pcm)

cat("\nPCM brms formula:\n")
print(out_pcm$brms_formula)

cat("\nFitting PCM (4 chains x 1000 iter, warmup=500)...\n")
fit_pcm <- brm(
  formula  = out_pcm$brms_formula,
  data     = sim_pcm_prep,
  prior    = out_pcm$suggested_priors,
  family   = out_pcm$family_obj,
  backend  = "cmdstanr",
  chains   = 4,
  iter     = 1000,
  warmup   = 500,
  cores    = 4,
  seed     = 42,
  silent   = 2,
  refresh  = 0
)

cat("\nPCM model summary:\n")
print(summary(fit_pcm))

draws_pcm <- as_draws_df(fit_pcm)
cp_hat    <- plogis(mean(draws_pcm$b_lcp_Intercept))
rp_hat    <- plogis(mean(draws_pcm$b_lrp_Intercept))
cp_ci     <- plogis(quantile(draws_pcm$b_lcp_Intercept, c(0.025, 0.975)))
rp_ci     <- plogis(quantile(draws_pcm$b_lrp_Intercept, c(0.025, 0.975)))
max_rhat_pcm <- max(brms::rhat(fit_pcm), na.rm = TRUE)

cat("\n=== PCM Recovery ===\n")
cat(sprintf("cp : true = %.3f  est = %.3f  95%% CI = [%.3f, %.3f]\n",
            cp_true, cp_hat, cp_ci[1], cp_ci[2]))
cat(sprintf("rp : true = %.3f  est = %.3f  95%% CI = [%.3f, %.3f]\n",
            rp_true, rp_hat, rp_ci[1], rp_ci[2]))
cat(sprintf("Max R-hat: %.4f\n", max_rhat_pcm))

# =============================================================================
# PART B: 3-category Source Monitoring Model (SMM) with stick-breaking
# Parameters: dA=0.70, dB=0.60, gA=0.40, gB=0.30, gNew=0.30
# Trees: sourceA, sourceB, new
# Branch expressions:
#   Tree A: P(A)=dA+(1-dA)*gA, P(B)=(1-dA)*gB, P(New)=(1-dA)*gNew
#   Tree B: P(A)=(1-dB)*gA,    P(B)=dB+(1-dB)*gB, P(New)=(1-dB)*gNew
#   Tree N: P(A)=gA,           P(B)=gB,         P(New)=gNew
# Constraint: gA + gB + gNew = 1  (guessing simplex)
# =============================================================================

cat("\n====================================================\n")
cat("PART B: Source Monitoring Model (SMM)  — multinomial + simplex\n")
cat("====================================================\n\n")

dA_true  <- 0.70;  dB_true  <- 0.60
gA_true  <- 0.40;  gB_true  <- 0.30;  gNew_true <- 1 - gA_true - gB_true
stopifnot(abs(gA_true + gB_true + gNew_true - 1) < 1e-10)

n_participants_b <- 60
n_items_b        <- 40

pA_tA <- dA_true + (1 - dA_true) * gA_true
pB_tA <- (1 - dA_true) * gB_true
pN_tA <- (1 - dA_true) * gNew_true
stopifnot(abs(pA_tA + pB_tA + pN_tA - 1) < 1e-10)

pA_tB <- (1 - dB_true) * gA_true
pB_tB <- dB_true + (1 - dB_true) * gB_true
pN_tB <- (1 - dB_true) * gNew_true
stopifnot(abs(pA_tB + pB_tB + pN_tB - 1) < 1e-10)

pA_tN <- gA_true;  pB_tN <- gB_true;  pN_tN <- gNew_true

cat(sprintf("True: dA=%.2f, dB=%.2f, gA=%.2f, gB=%.2f, gNew=%.2f\n",
            dA_true, dB_true, gA_true, gB_true, gNew_true))

lgA_true <- qlogis(gA_true)
lgB_true <- qlogis(gB_true / (1 - gA_true))
cat(sprintf("Stick-breaking: lgA=%.3f, lgB=%.3f\n", lgA_true, lgB_true))

set.seed(2024)
sim_smm <- bind_rows(
  tibble(participant = seq_len(n_participants_b), source = "sourceA") |>
    mutate(
      counts_raw = lapply(seq_len(n()), function(i)
        rmultinom(1, n_items_b, c(pA_tA, pB_tA, pN_tA))[, 1]),
      A   = sapply(counts_raw, `[`, 1),
      B   = sapply(counts_raw, `[`, 2),
      New = sapply(counts_raw, `[`, 3),
      n   = n_items_b
    ),
  tibble(participant = seq_len(n_participants_b), source = "sourceB") |>
    mutate(
      counts_raw = lapply(seq_len(n()), function(i)
        rmultinom(1, n_items_b, c(pA_tB, pB_tB, pN_tB))[, 1]),
      A   = sapply(counts_raw, `[`, 1),
      B   = sapply(counts_raw, `[`, 2),
      New = sapply(counts_raw, `[`, 3),
      n   = n_items_b
    ),
  tibble(participant = seq_len(n_participants_b), source = "new") |>
    mutate(
      counts_raw = lapply(seq_len(n()), function(i)
        rmultinom(1, n_items_b, c(pA_tN, pB_tN, pN_tN))[, 1]),
      A   = sapply(counts_raw, `[`, 1),
      B   = sapply(counts_raw, `[`, 2),
      New = sapply(counts_raw, `[`, 3),
      n   = n_items_b
    )
) |>
  select(-counts_raw) |>
  arrange(participant, source)

cat("\nSimulated SMM data (first 6 rows):\n")
print(head(sim_smm, 6))

smm3_str <- paste0(
  "\ndA + (1 - dA) * gA       # A\n",
  "(1 - dA) * gB             # B\n",
  "(1 - dA) * gNew           # New\n",
  "\n(1 - dB) * gA             # A\n",
  "dB + (1 - dB) * gB       # B\n",
  "(1 - dB) * gNew           # New\n",
  "\ngA                        # A\n",
  "gB                        # B\n",
  "gNew                      # New\n"
)
spec_smm3 <- parse_mpt_string(smm3_str,
                               tree_conditions = c("sourceA", "sourceB", "new"),
                               condition       = "source")

out_smm3 <- mpt_to_brms(
  spec_smm3,
  predictor_formulas = list(dA = ~ 1, dB = ~ 1, gA = ~ 1, gB = ~ 1),
  simplex_params     = c("gA", "gB", "gNew")
)
sim_smm_prep <- out_smm3$data_prep(sim_smm)

cat("\nSMM3 brms formula:\n")
print(out_smm3$brms_formula)

cat("\nFitting SMM3 (4 chains x 1000 iter, warmup=500)...\n")
fit_smm <- brm(
  formula  = out_smm3$brms_formula,
  data     = sim_smm_prep,
  prior    = out_smm3$suggested_priors,
  family   = out_smm3$family_obj,
  backend  = "cmdstanr",
  chains   = 4,
  iter     = 1000,
  warmup   = 500,
  cores    = 4,
  seed     = 42,
  silent   = 2,
  refresh  = 0
)

cat("\nSMM model summary:\n")
print(summary(fit_smm))

draws_smm <- as_draws_df(fit_smm)
dA_hat   <- plogis(mean(draws_smm$b_ldA_Intercept))
dB_hat   <- plogis(mean(draws_smm$b_ldB_Intercept))
gA_hat   <- plogis(mean(draws_smm$b_lgA_Intercept))
gB_prop  <- plogis(mean(draws_smm$b_lgB_Intercept))  # gB / (1 - gA)
gB_hat   <- (1 - gA_hat) * gB_prop
gNew_hat <- 1 - gA_hat - gB_hat

dA_ci <- plogis(quantile(draws_smm$b_ldA_Intercept, c(0.025, 0.975)))
dB_ci <- plogis(quantile(draws_smm$b_ldB_Intercept, c(0.025, 0.975)))
gA_ci <- plogis(quantile(draws_smm$b_lgA_Intercept, c(0.025, 0.975)))

max_rhat_smm <- max(brms::rhat(fit_smm), na.rm = TRUE)

cat("\n=== SMM Recovery ===\n")
cat(sprintf("dA   : true = %.3f  est = %.3f  95%% CI = [%.3f, %.3f]\n",
            dA_true, dA_hat, dA_ci[1], dA_ci[2]))
cat(sprintf("dB   : true = %.3f  est = %.3f  95%% CI = [%.3f, %.3f]\n",
            dB_true, dB_hat, dB_ci[1], dB_ci[2]))
cat(sprintf("gA   : true = %.3f  est = %.3f  95%% CI = [%.3f, %.3f]\n",
            gA_true, gA_hat, gA_ci[1], gA_ci[2]))
cat(sprintf("gB   : true = %.3f  est = %.3f  (via stick-breaking)\n", gB_true, gB_hat))
cat(sprintf("gNew : true = %.3f  est = %.3f  (derived: 1-gA-gB)\n", gNew_true, gNew_hat))
cat(sprintf("Max R-hat: %.4f\n", max_rhat_smm))

cat("\n=== Summary Table ===\n")
cat(sprintf("%-8s  %-6s  %-6s  %-12s  %s\n", "Model", "Param", "True", "Estimate", "Max R-hat"))
cat(sprintf("%-8s  %-6s  %-6.3f  %-12.3f  %.4f\n", "PCM",  "cp",   cp_true,   cp_hat,   max_rhat_pcm))
cat(sprintf("%-8s  %-6s  %-6.3f  %-12.3f\n",        "PCM",  "rp",   rp_true,   rp_hat))
cat(sprintf("%-8s  %-6s  %-6.3f  %-12.3f  %.4f\n", "SMM",  "dA",   dA_true,   dA_hat,   max_rhat_smm))
cat(sprintf("%-8s  %-6s  %-6.3f  %-12.3f\n",        "SMM",  "dB",   dB_true,   dB_hat))
cat(sprintf("%-8s  %-6s  %-6.3f  %-12.3f\n",        "SMM",  "gA",   gA_true,   gA_hat))
cat(sprintf("%-8s  %-6s  %-6.3f  %-12.3f\n",        "SMM",  "gB",   gB_true,   gB_hat))
cat(sprintf("%-8s  %-6s  %-6.3f  %-12.3f\n",        "SMM",  "gNew", gNew_true, gNew_hat))

cat("\nDone.\n")
