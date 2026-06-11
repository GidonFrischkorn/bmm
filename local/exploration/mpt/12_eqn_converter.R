# 12_eqn_converter.R
#
# EQN converter for the Al Hadhrami, Bartsch & Oberauer (2025) binding models.
# Source: Al Hadhrami, A., Bartsch, L. M., & Oberauer, K. (2025). A
# multinomial model-based analysis of bindings in working memory.
# Psychological Review. OSF: https://osf.io/nu4sb/
#
# This script is also the working prototype for a future mpt_from_eqn()
# package function. Run from the repository root:
#   Rscript local/exploration/mpt/12_eqn_converter.R

source("local/exploration/mpt/03_mpt_parser.R")

# ---------------------------------------------------------------------------
# Design-fixed guessing constants (TreeBUGS `restrictions` in the original)
# ---------------------------------------------------------------------------
constants <- c(
  P_G1       = .2,
  P_g1       = .125,
  P_G2       = .2,
  P_g2       = .125,
  P_g1_Lure  = .571,
  P_g2_Lure  = .571,
  P_G2_Lure_s = .25,
  P_g2_Lure_s = .143,
  P_g2_Lure_d = .5
)

# ---------------------------------------------------------------------------
# build_binding_spec(): EQN file -> mpt_spec
# ---------------------------------------------------------------------------
build_binding_spec <- function(eqn_file, constants, link = "probit") {
  lines <- trimws(readLines(eqn_file))
  lines <- lines[nzchar(lines)]
  parts <- strsplit(lines, "\\s+")
  stopifnot(all(vapply(parts, length, 1L) == 3L))
  eqn <- data.frame(
    tree = vapply(parts, `[`, "", 1),
    cat  = vapply(parts, `[`, "", 2),
    expr = vapply(parts, `[`, "", 3),
    stringsAsFactors = FALSE
  )

  # Substitute constants as decimal literals (avoids Stan integer-division)
  for (nm in names(constants)) {
    eqn$expr <- gsub(paste0("\\b", nm, "\\b"),
                     format(constants[[nm]], digits = 6), eqn$expr)
  }

  # Strip underscores from the remaining (free) parameter names.
  # Note: real published EQN models systematically use underscores in param
  # names, which violates brms nlpar naming rules. The mechanical rename works
  # but leaks sanitised names into priors and summaries; mpt_from_eqn() should
  # automate this with a reported mapping (Phase 2 requirement).
  ids <- unique(unlist(regmatches(eqn$expr,
    gregexpr("[A-Za-z_][A-Za-z0-9_.]*", eqn$expr))))
  rename <- setNames(gsub("[._]", "", ids), ids)
  stopifnot(!any(duplicated(rename)))
  for (nm in names(rename)) {
    eqn$expr <- gsub(paste0("\\b", nm, "\\b"), rename[[nm]], eqn$expr)
  }

  # Categories: drop tree prefix, strip underscores
  # e.g. WC_Lure_s_Lure_s -> LuresLures
  eqn$cat_clean <- gsub("_", "", sub("^(WC|LC|CC)_", "", eqn$cat))

  # EQN semantics: multiple branch lines per category -> sum
  trees <- lapply(unique(eqn$tree), function(tr) {
    sub      <- eqn[eqn$tree == tr, ]
    branches <- tapply(sub$expr, sub$cat_clean,
                       function(e) paste0("(", e, ")", collapse = " + "))
    mpt_tree(tr, as.list(branches))
  })

  mpt(trees = trees, condition = "cue", link = link)
}

# ---------------------------------------------------------------------------
# Long-format data prep (matching the category renaming in the converter)
# ---------------------------------------------------------------------------
build_long_data <- function(freq_file) {
  freq     <- read.csv(freq_file)
  freq$id  <- seq_len(nrow(freq))
  long <- do.call(rbind, lapply(c("WC", "LC", "CC"), function(tr) {
    cols <- grep(paste0("^", tr, "_"), names(freq), value = TRUE)
    d    <- freq[, cols, drop = FALSE]
    names(d) <- gsub("_", "", sub(paste0("^", tr, "_"), "", cols))
    d$cue <- tr
    d$id  <- freq$id
    d
  }))
  long$n <- rowSums(long[, setdiff(names(long), c("cue", "id"))])
  long
}

# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------
cat("=== EQN converter self-checks ===\n\n")

eqn_dir  <- "local/exploration/mpt/binding_models"
freq_csv <- file.path(eqn_dir, "exp1_response_frequencies.csv")

# ---- Unitization ----
cat("--- Unitization model ---\n")
spec_unit <- build_binding_spec(
  file.path(eqn_dir, "unitization.eqn"), constants
)
print(spec_unit)

n_params_unit <- length(spec_unit$params)
cat(sprintf("Free parameters: %d (expected 12)\n", n_params_unit))
if (n_params_unit != 12L)
  stop(sprintf("FAIL: expected 12 free parameters, got %d", n_params_unit))

n_trees_unit  <- length(spec_unit$trees)
n_cats_unit   <- length(spec_unit$resp_cats)
cat(sprintf("Trees: %d (expected 3), Categories: %d (expected 10)\n",
            n_trees_unit, n_cats_unit))
if (n_trees_unit != 3L || n_cats_unit != 10L)
  stop("FAIL: wrong number of trees or categories for Unitization model")

cat("Unitization self-checks PASSED.\n\n")

# ---- Hybrid ----
cat("--- Hybrid model ---\n")
spec_hyb <- build_binding_spec(
  file.path(eqn_dir, "hybrid.eqn"), constants
)
print(spec_hyb)

n_params_hyb <- length(spec_hyb$params)
cat(sprintf("Free parameters: %d (expected 18)\n", n_params_hyb))
if (n_params_hyb != 18L)
  stop(sprintf("FAIL: expected 18 free parameters, got %d", n_params_hyb))

n_trees_hyb  <- length(spec_hyb$trees)
n_cats_hyb   <- length(spec_hyb$resp_cats)
cat(sprintf("Trees: %d (expected 3), Categories: %d (expected 10)\n",
            n_trees_hyb, n_cats_hyb))
if (n_trees_hyb != 3L || n_cats_hyb != 10L)
  stop("FAIL: wrong number of trees or categories for Hybrid model")

cat("Hybrid self-checks PASSED.\n\n")

# ---- Branch-sum validation at multiple test points ----
cat("--- Branch-sum validation (both models) ---\n")
test_pts <- c(0.137, 0.421, 0.683, 0.852)

for (mdl_name in c("Unitization", "Hybrid")) {
  spec <- if (mdl_name == "Unitization") spec_unit else spec_hyb
  for (tr in spec$trees) {
    exprs   <- unlist(tr$branches)
    n_par   <- length(spec$params)
    for (tp in test_pts) {
      env <- new.env(parent = baseenv())
      for (p in spec$params) assign(p, tp, envir = env)
      probs <- sapply(exprs, function(e) eval(parse(text = e), envir = env))
      s     <- sum(probs)
      if (abs(s - 1) > 1e-8)
        stop(sprintf("%s tree '%s' sums to %.10f at tp=%.3f (expected 1)",
                     mdl_name, tr$name, s, tp))
    }
    cat(sprintf("  %s tree '%s': branch sums = 1 at all test points PASS\n",
                mdl_name, tr$name))
  }
}

cat("\n=== Long-format data ===\n")
long <- build_long_data(freq_csv)
cat(sprintf("Rows: %d (expected %d)\n", nrow(long), 32 * 3))
cat(sprintf("Columns: %s\n", paste(names(long), collapse = ", ")))
cat(sprintf("n range: [%d, %d] (all should be 50)\n",
            min(long$n), max(long$n)))
if (min(long$n) != 50L || max(long$n) != 50L)
  warning("Some n values differ from 50 — check row sums.")

cat("\n=== All EQN converter checks passed ===\n")
cat("\nSpec objects for downstream scripts:\n")
cat("  spec_unit  — Unitization (12 free params)\n")
cat("  spec_hyb   — Hybrid (18 free params)\n")
cat("  long data  — use build_long_data() in fitting scripts\n")
