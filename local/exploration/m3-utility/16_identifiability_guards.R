# 16_identifiability_guards.R
# WP4 — Identifiability guards as API behaviour
#
# Converts the identifiability findings from rounds 1-4 into enforceable checks.
# For each architecture, shows where in the S3 pipeline the guard can live
# and demonstrates it firing on a deficient dataset.
#
# Three guards:
#   G1: Power utility (rho) requires >= 3 distinct value levels
#   G2: Prelec weighting (alpha) requires variable set size across trials
#   G3: EU slope (gamma) requires value variation (>= 2 distinct values)
#
# For each guard:
#   - Implementation (standalone function)
#   - Integration path for each architecture
#   - Demo: deficient dataset -> guard fires; adequate dataset -> guard passes
#
# Run from repo root:
#   source("local/exploration/m3-utility/16_identifiability_guards.R")

library(brms)
if (requireNamespace("bmm", quietly = TRUE)) {
  library(bmm)
} else {
  suppressMessages(devtools::load_all("."))
}

cat("==========================================================================\n")
cat("16_identifiability_guards.R — Prototype guards\n")
cat("==========================================================================\n\n")


# ==============================================================================
# SECTION 1 — Guard function implementations
# ==============================================================================

# ---- G1: Power utility needs >= 3 distinct value levels ----------------------

check_power_utility_ident <- function(data, value_cols, param_name = "rho") {
  for (col_name in value_cols) {
    if (!col_name %in% names(data)) {
      stop(sprintf(
        "Value column '%s' not found in data. Columns available: %s",
        col_name, paste(names(data), collapse = ", ")
      ))
    }
    vals <- unique(data[[col_name]])
    n_unique <- length(vals)
    if (n_unique < 3) {
      stop(sprintf(
        paste0(
          "Power utility (%s) is not identifiable with fewer than 3 distinct value levels.\n",
          "Column '%s' has only %d distinct value(s): %s.\n",
          "Use >= 3 distinct value levels (e.g., V in {1, 3, 5}) ",
          "or switch to utility_fn = 'linear' (which requires only variation).\n",
          "Identifiability condition: With 2 levels, gamma and rho are ",
          "negatively correlated with posterior correlation typically > |0.95|; ",
          "neither is individually recoverable."
        ),
        param_name, col_name, n_unique,
        paste(sort(vals), collapse = ", ")
      ))
    }
  }
  invisible(TRUE)
}

# ---- G2: Prelec weighting needs variable set size ----------------------------

check_prelec_ident <- function(data, n_opt_cols, param_name = "alpha") {
  for (col_name in n_opt_cols) {
    if (!col_name %in% names(data)) {
      stop(sprintf(
        "Set-size column '%s' not found in data.", col_name
      ))
    }
    n_unique <- length(unique(data[[col_name]]))
    if (n_unique < 2) {
      stop(sprintf(
        paste0(
          "Prelec probability weighting (%s) is not identifiable with constant set size.\n",
          "Column '%s' has only 1 unique value: %s.\n",
          "Variable set sizes across trials are required to identify %s.\n",
          "In a fixed-set-size design, w(p_i) = w(n_i/N) is a constant scalar ",
          "absorbed into the activation intercept and cannot be separated from it.\n",
          "Identifiability condition: >= 2 distinct set sizes are required."
        ),
        param_name, col_name, unique(data[[col_name]]), param_name
      ))
    }
  }
  invisible(TRUE)
}

# ---- G3: EU slope needs value variation (>= 2 distinct values) ---------------

check_gamma_ident <- function(data, value_cols, param_name = "gamma") {
  for (col_name in value_cols) {
    if (!col_name %in% names(data)) {
      stop(sprintf(
        "Value column '%s' not found in data.", col_name
      ))
    }
    vals <- unique(data[[col_name]])
    n_unique <- length(vals)
    if (n_unique < 2) {
      stop(sprintf(
        paste0(
          "EU activation slope (%s) is not identifiable when value is constant.\n",
          "Column '%s' has only 1 unique value: %s.\n",
          "Value must vary across trials to identify %s.\n",
          "Identifiability condition: >= 2 distinct value levels are required."
        ),
        param_name, col_name, unique(data[[col_name]]), param_name
      ))
    }
  }
  invisible(TRUE)
}


# ==============================================================================
# SECTION 2 — Demonstrations: guards firing and passing
# ==============================================================================

cat("--- G1: Power utility identifiability guard ---\n\n")

# Deficient dataset: only 2 value levels (V in {1, 2})
data_2level <- data.frame(
  subj   = rep(1:10, each = 50),
  V_corr = rep(c(1, 2), 250),  # only 2 distinct values
  V_other = rep(c(1, 2), 250)
)

cat("Deficient data: V_corr has", length(unique(data_2level$V_corr)), "distinct values\n")
result_fail <- tryCatch(
  check_power_utility_ident(data_2level, c("V_corr"), "rho"),
  error = function(e) e
)
if (inherits(result_fail, "error")) {
  cat("GUARD FIRES (correct):\n")
  cat(conditionMessage(result_fail), "\n\n")
} else {
  cat("GUARD DID NOT FIRE (incorrect)\n\n")
}

# Adequate dataset: 3 value levels (V in {1, 3, 5})
data_3level <- data.frame(
  subj   = rep(1:10, each = 60),
  V_corr = rep(c(1, 3, 5), 200),
  V_other = rep(c(1, 3, 5), 200)
)

cat("Adequate data: V_corr has", length(unique(data_3level$V_corr)), "distinct values\n")
result_pass <- tryCatch(
  check_power_utility_ident(data_3level, c("V_corr"), "rho"),
  error = function(e) e
)
if (inherits(result_pass, "error")) {
  cat("GUARD FIRES (incorrect — should pass)\n\n")
} else {
  cat("GUARD PASSES (correct) ✓\n\n")
}


cat("--- G2: Prelec weighting identifiability guard ---\n\n")

# Deficient dataset: fixed set size (n_other = 4 always)
data_fixed_set <- data.frame(
  subj    = rep(1:10, each = 50),
  n_other = rep(4L, 500),   # constant — alpha unidentified
  n_corr  = rep(1L, 500),
  n_npl   = rep(5L, 500)
)

cat("Deficient data: n_other has", length(unique(data_fixed_set$n_other)), "distinct value\n")
result2_fail <- tryCatch(
  check_prelec_ident(data_fixed_set, c("n_other"), "alpha"),
  error = function(e) e
)
if (inherits(result2_fail, "error")) {
  cat("GUARD FIRES (correct):\n")
  cat(conditionMessage(result2_fail), "\n\n")
} else {
  cat("GUARD DID NOT FIRE (incorrect)\n\n")
}

# Adequate dataset: variable set sizes
data_var_set <- data.frame(
  subj    = rep(1:10, each = 60),
  n_other = rep(c(2L, 4L, 6L), 200),  # 3 set sizes
  n_corr  = rep(1L, 600),
  n_npl   = rep(5L, 600)
)

cat("Adequate data: n_other has", length(unique(data_var_set$n_other)), "distinct values\n")
result2_pass <- tryCatch(
  check_prelec_ident(data_var_set, c("n_other"), "alpha"),
  error = function(e) e
)
if (inherits(result2_pass, "error")) {
  cat("GUARD FIRES (incorrect — should pass)\n\n")
} else {
  cat("GUARD PASSES (correct) ✓\n\n")
}


cat("--- G3: EU slope (gamma) identifiability guard ---\n\n")

# Deficient dataset: constant value column
data_const_V <- data.frame(
  subj   = rep(1:10, each = 50),
  V_corr = rep(3, 500)   # constant — gamma unidentified
)

cat("Deficient data: V_corr has", length(unique(data_const_V$V_corr)), "distinct value\n")
result3_fail <- tryCatch(
  check_gamma_ident(data_const_V, c("V_corr"), "gamma"),
  error = function(e) e
)
if (inherits(result3_fail, "error")) {
  cat("GUARD FIRES (correct):\n")
  cat(conditionMessage(result3_fail), "\n\n")
} else {
  cat("GUARD DID NOT FIRE (incorrect)\n\n")
}

# Adequate dataset: two value levels
data_var_V <- data.frame(
  subj   = rep(1:10, each = 50),
  V_corr = rep(c(1, 5), 250)
)

cat("Adequate data: V_corr has", length(unique(data_var_V$V_corr)), "distinct values\n")
result3_pass <- tryCatch(
  check_gamma_ident(data_var_V, c("V_corr"), "gamma"),
  error = function(e) e
)
if (inherits(result3_pass, "error")) {
  cat("GUARD FIRES (incorrect — should pass)\n\n")
} else {
  cat("GUARD PASSES (correct) ✓\n\n")
}


# ==============================================================================
# SECTION 3 — Integration paths per architecture
# ==============================================================================

cat("==========================================================================\n")
cat("SECTION 3 — Where each guard lives in each architecture\n")
cat("==========================================================================\n\n")

cat(
"Notation: + = guard can live here automatically (via S3 dispatch)
           ~ = guard CAN be added but requires extra subclassing work
           x = guard cannot live here structurally\n\n"
)

guard_placement <- data.frame(
  Guard = c(
    "G1: power utility (rho)\n  needs >=3 value levels",
    "G2: Prelec (alpha)\n  needs variable set size",
    "G3: EU slope (gamma)\n  needs value variation"
  ),
  Stage_required = c("check_data", "check_data", "check_data"),
  Arch_A_exploration = c(
    "x (no S3 dispatch;\n   no subclass 'm3_utility';\n   only manual call possible)",
    "x (same)",
    "x (same)"
  ),
  Arch_A_production = c(
    "~ (add 'm3_utility' subclass\n   + check_data.m3_utility in R/)",
    "~ (same)",
    "~ (same)"
  ),
  Arch_B_version = c(
    "~ (same as A-production;\n   needs class change AND method)",
    "~ (same)",
    "~ (same)"
  ),
  Arch_C_sibling = c(
    "+ (check_data.utility_memory\n   auto-dispatched by bmm())",
    "+ (same)",
    "+ (same)"
  ),
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(guard_placement))) {
  cat(sprintf("Guard %d (%s)\n", i, c("G1","G2","G3")[i]))
  cat(sprintf("  Stage required:     %s\n", guard_placement$Stage_required[i]))
  cat(sprintf("  Arch A exploration: %s\n", guard_placement$Arch_A_exploration[i]))
  cat(sprintf("  Arch A production:  %s\n", guard_placement$Arch_A_production[i]))
  cat(sprintf("  Arch B version:     %s\n", guard_placement$Arch_B_version[i]))
  cat(sprintf("  Arch C sibling:     %s\n\n", guard_placement$Arch_C_sibling[i]))
}


# ==============================================================================
# SECTION 4 — Integration example for Architecture A (production)
# ==============================================================================
#
# Shows what check_data.m3_utility would look like in R/model_m3_utility.R
# and how it would be triggered by adding "m3_utility" to the class vector.

cat("==========================================================================\n")
cat("SECTION 4 — check_data.m3_utility sketch (Architecture A production)\n")
cat("==========================================================================\n\n")

cat(
'# In R/model_m3_utility.R (production code, NOT in local/exploration/):
#
# 1. m3_utility() returns object with additional class "m3_utility":
#    class(model) <- c("bmmodel", "m3", "m3_custom", "m3_utility")
#
# 2. check_data.m3_utility is dispatched BEFORE check_data.m3:

check_data.m3_utility_SKETCH <- function(model, data, formula) {
  # Retrieve metadata stored by m3_utility()
  value_cols   <- attr(model, "value_cols")   # named vector of V column names
  utility_fn   <- attr(model, "utility")      # "linear" or "power"
  weighting    <- attr(model, "weighting")    # "none" or "prelec"
  num_options  <- attr(model, "num_options")  # column names for set sizes

  # G3: gamma identifiability (always required when value_cols present)
  if (!is.null(value_cols)) {
    for (v_col in value_cols) {
      n_unique <- length(unique(data[[v_col]]))
      if (n_unique < 2) {
        stop(sprintf(
          "EU slope (gamma) not identifiable: column \'%s\' has only 1 unique value.",
          v_col
        ))
      }
    }
  }

  # G1: rho identifiability (power utility only)
  if (!is.null(value_cols) && utility_fn == "power") {
    for (v_col in value_cols) {
      n_unique <- length(unique(data[[v_col]]))
      if (n_unique < 3) {
        stop(sprintf(
          paste0(
            "Power utility (rho) not identifiable: column \'%s\' has only %d distinct ",
            "value(s). Need >= 3 (e.g. V in {1, 3, 5})."
          ),
          v_col, n_unique
        ))
      }
    }
  }

  # G2: alpha identifiability (Prelec weighting only)
  if (weighting == "prelec" && is.character(num_options)) {
    for (n_col in num_options) {
      n_unique <- length(unique(data[[n_col]]))
      if (n_unique < 2) {
        stop(sprintf(
          paste0(
            "Prelec weighting (alpha) not identifiable: \'%s\' has only 1 unique ",
            "set-size value. alpha requires set-size variation across trials."
          ),
          n_col
        ))
      }
    }
  }

  # Delegate to standard m3 data check
  NextMethod("check_data")
}
\n'
)

cat(
"Key point: the guard only fires automatically via S3 dispatch if:\n",
"  (a) m3_utility() adds 'm3_utility' to the class vector, AND\n",
"  (b) check_data.m3_utility is exported from the bmm package (in R/)\n\n",
"In the current exploration-only implementation (local/exploration/):\n",
"  - The guard functions (Section 1) work as standalone validators\n",
"  - Users must call them manually: check_power_utility_ident(data, value_cols)\n",
"  - They do NOT fire automatically when bmm() is called\n\n"
)


# ==============================================================================
# SECTION 5 — Combined guard function for pre-flight validation
# ==============================================================================
#
# A standalone pre-flight function that Arch A users can call manually.
# This is the best available substitute for an auto-dispatched check_data method
# in the exploration phase.

check_utility_design <- function(model, data) {
  value_cols  <- attr(model, "value_cols")
  utility_fn  <- attr(model, "utility")
  weighting   <- attr(model, "weighting")
  num_options <- attr(model, "num_options")

  errors <- character(0)

  # G3: gamma requires value variation
  if (!is.null(value_cols)) {
    for (col_name in value_cols) {
      if (col_name %in% names(data)) {
        n_u <- length(unique(data[[col_name]]))
        if (n_u < 2) {
          errors <- c(errors, sprintf(
            "G3: EU slope (gamma) not identifiable — '%s' has only 1 distinct value.",
            col_name
          ))
        }
      }
    }
  }

  # G1: rho requires >= 3 value levels
  if (!is.null(value_cols) && !is.null(utility_fn) && utility_fn == "power") {
    for (col_name in value_cols) {
      if (col_name %in% names(data)) {
        n_u <- length(unique(data[[col_name]]))
        if (n_u < 3) {
          errors <- c(errors, sprintf(
            "G1: Power utility (rho) not identifiable — '%s' has %d distinct value(s), needs >= 3.",
            col_name, n_u
          ))
        }
      }
    }
  }

  # G2: alpha requires variable set size
  if (!is.null(weighting) && weighting == "prelec" && is.character(num_options)) {
    for (col_name in num_options) {
      if (col_name %in% names(data)) {
        n_u <- length(unique(data[[col_name]]))
        if (n_u < 2) {
          errors <- c(errors, sprintf(
            "G2: Prelec (alpha) not identifiable — '%s' has only 1 distinct set size.",
            col_name
          ))
        }
      }
    }
  }

  if (length(errors) > 0) {
    stop(paste0(
      "Identifiability check failed for this design:\n",
      paste0("  ", errors, collapse = "\n"), "\n",
      "Fix the design or change the model specification."
    ))
  }
  invisible(TRUE)
}

cat("==========================================================================\n")
cat("SECTION 5 — check_utility_design() demo (Architecture A pre-flight guard)\n")
cat("==========================================================================\n\n")

# Load m3_utility for this demo
invisible(capture.output(
  source("local/exploration/m3-utility/11_utility_wrapper.R")
))

# --- Demo 1: Power utility + 2 value levels (should fail) ---
cat("Demo 1: Power utility + 2-level V (should fail)\n")
pow_model <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "power",
  weighting   = "none"
)
deficient_data <- data.frame(
  subj = rep(1:10, each=50), V_corr = rep(c(1,2),250), V_other = rep(c(1,2),250),
  n_corr=1L, n_other=4L, n_npl=5L, corr=0L, other=0L, npl=0L
)
r_demo1 <- tryCatch(check_utility_design(pow_model, deficient_data), error = function(e) e)
cat("Result:", if (inherits(r_demo1, "error")) paste("ERROR (expected):", conditionMessage(r_demo1))
    else "PASS (unexpected)", "\n\n")

# --- Demo 2: Prelec + fixed set size (should fail) ---
cat("Demo 2: Prelec weighting + fixed set size (should fail)\n")
prelec_model <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "prelec"
)
fixed_set_data <- data.frame(
  subj = rep(1:10, each=50), V_corr = rep(c(1,3,5),ceiling(500/3))[1:500],
  V_other = rep(c(1,3,5),ceiling(500/3))[1:500],
  n_corr=1L, n_other=4L, n_npl=5L, p_corr=0.1, p_other=0.4, p_npl=0.5,
  corr=0L, other=0L, npl=0L
)
r_demo2 <- tryCatch(check_utility_design(prelec_model, fixed_set_data), error = function(e) e)
cat("Result:", if (inherits(r_demo2, "error")) paste("ERROR (expected):", conditionMessage(r_demo2))
    else "PASS (unexpected)", "\n\n")

# --- Demo 3: EU linear + good design (should pass) ---
cat("Demo 3: EU linear + 3 value levels + variable set sizes (should pass)\n")
eu_model <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "none"
)
good_data <- data.frame(
  subj = rep(1:10, each=60), V_corr = rep(c(1,3,5),200), V_other = rep(c(1,3,5),200),
  n_corr=1L, n_other=4L, n_npl=5L, corr=0L, other=0L, npl=0L
)
r_demo3 <- tryCatch(check_utility_design(eu_model, good_data), error = function(e) e)
cat("Result:", if (inherits(r_demo3, "error")) paste("ERROR (unexpected):", conditionMessage(r_demo3))
    else "PASS (correct) ✓", "\n\n")

cat("==========================================================================\n")
cat("16_identifiability_guards.R complete.\n")
cat("\nSummary:\n")
cat("  G1 power utility rho:  DEMONSTRATED (fires on 2-level V, passes on 3-level)\n")
cat("  G2 Prelec alpha:       DEMONSTRATED (fires on fixed set size, passes on variable)\n")
cat("  G3 EU slope gamma:     DEMONSTRATED (fires on constant V, passes on variable)\n")
cat("\n  check_utility_design(model, data): pre-flight validator for Architecture A\n")
cat("  In Architecture C (sibling): these guards live in check_data.utility_memory\n")
cat("    and fire automatically via S3 dispatch — no user action required.\n")
cat("==========================================================================\n")
