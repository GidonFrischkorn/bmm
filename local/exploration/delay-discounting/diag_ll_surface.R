sv_exponential <- function(A, D, k) A * exp(-k * D)
sv_hyperboloid <- function(A, D, k, s) A / (1 + k * D)^s

ll_exp <- function(data, k, phi) {
  V_LL <- sv_exponential(data$amt_LL, data$delay_LL, k)
  V_SS <- 10
  p <- plogis(phi * (V_LL - V_SS))
  p <- pmax(pmin(p, 1-1e-10), 1e-10)
  sum(dbinom(data$choice, 1, p, log=TRUE))
}

ll_hyp <- function(data, k, s, phi) {
  V_LL <- sv_hyperboloid(data$amt_LL, data$delay_LL, k, s)
  V_SS <- 10
  p <- plogis(phi * (V_LL - V_SS))
  p <- pmax(pmin(p, 1-1e-10), 1e-10)
  sum(dbinom(data$choice, 1, p, log=TRUE))
}

# Simulate exponential data
set.seed(102)
amt_LL_levels <- c(12,15,20,25,30); delay_LL_levels <- c(7,14,30,60,90,180,365)
grid <- expand.grid(amt_LL=amt_LL_levels, delay_LL=delay_LL_levels)
design <- grid[sample(nrow(grid), 200, replace=TRUE),]
design$amt_SS <- 10; design$delay_SS <- 0
V_LL <- sv_exponential(design$amt_LL, design$delay_LL, 0.01)
V_SS <- 10
p <- plogis(2*(V_LL - V_SS))
set.seed(1102)
design$choice <- rbinom(200, 1, p)
cat("p(LL) =", round(mean(design$choice),3), "\n")

cat("\n=== Exponential LL surface (varying log_k, phi fixed at 2) ===\n")
for (lk in seq(-10, 0, by=1)) {
  ll <- ll_exp(design, exp(lk), 2)
  cat(sprintf("log_k=%+.1f  k=%.6f  LL=%8.2f\n", lk, exp(lk), ll))
}

cat("\n=== Exponential LL surface (varying log_phi, k fixed at 0.01) ===\n")
for (lp in seq(-2, 3, by=0.5)) {
  ll <- ll_exp(design, 0.01, exp(lp))
  cat(sprintf("log_phi=%+.1f  phi=%.4f  LL=%8.2f\n", lp, exp(lp), ll))
}

cat("\n=== Hyperboloid LL surface (k varying, s=0.8, phi=2) ===\n")
set.seed(103)
design2 <- grid[sample(nrow(grid), 200, replace=TRUE),]
design2$amt_SS <- 10; design2$delay_SS <- 0
V_LL2 <- sv_hyperboloid(design2$amt_LL, design2$delay_LL, 0.02, 0.8)
p2 <- plogis(2*(V_LL2 - 10))
set.seed(1103)
design2$choice <- rbinom(200, 1, p2)
cat("p(LL) =", round(mean(design2$choice),3), "\n")
for (lk in seq(-10, 0, by=1)) {
  ll <- ll_hyp(design2, exp(lk), 0.8, 2)
  cat(sprintf("log_k=%+.1f  k=%.6f  LL=%8.2f\n", lk, exp(lk), ll))
}
