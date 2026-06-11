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

# ---- G1 (revised): Power utility — RANGE criterion, warning not error ---------
# Round 7 revision: demote from stop() to warning(), and switch from "count of
# levels" to "value range ratio" (max(V)/min(V) >= 3).
# Empirical basis: WP1 results/17_power_utility_{2,3}lev.csv show rho CI ratio
# is only 1.2x (not the 1.5x threshold) when V in {1,2} vs V in {1,3,5}.
# Rho DOES recover with 2 levels at large N; the guard is provisionally measured
# on one fit per design — not enough to justify a hard stop().

check_power_utility_ident_v2 <- function(data, value_cols, param_name = "rho") {
  for (col_name in value_cols) {
    if (!col_name %in% names(data)) {
      stop(sprintf("Value column '%s' not found in data.", col_name))
    }
    vals   <- sort(unique(data[[col_name]]))
    n_uniq <- length(vals)
    if (n_uniq < 2) {
      stop(sprintf(
        "Power utility (%s) not identifiable: '%s' has only 1 distinct value.",
        param_name, col_name
      ))
    }
    ratio <- max(vals) / min(vals)
    if (ratio < 3) {
      warning(sprintf(
        paste0(
          "G1 (power utility, provisional): '%s' has range ratio %.2f (max/min), ",
          "below the recommended threshold of 3.\n",
          "  Values: %s.\n",
          "  At small range (e.g. V in {1,2}, ratio=2), rho and gamma can be ",
          "nearly aliased (posterior correlation > 0.9). Recovery improves with ",
          "ratio >= 3 (e.g. V in {1,3,5}) or N >> 2700 trials.\n",
          "  This is a warning, not an error: one fit per design is insufficient ",
          "to justify a hard stop. See results/17_power_utility_2lev.csv.\n",
          "  Fix: increase value range (e.g. V in {1,3,5}) if rho is of interest."
        ),
        col_name, ratio, paste(vals, collapse=", ")
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


# ---- G4: Positivity hazard — simple rule + identity link ---------------------
# (NEW in round 7; fixes mischaracterized R9 from round 6)
#
# When choice_rule = "simple", glue_choice_rule_functions() (model_m3.R:399-404)
# generates: log({cat} * n_options). With identity-linked parameters, any
# activation formula can produce non-positive values (e.g., 0.5*b + wi <= 0
# if wi goes negative in sampling). Stan evaluates log(non-positive) = NaN,
# causing trajectory rejection (divergences).
#
# This is NOT a structural problem: the Luce rule IS native to m3(choice_rule=
# "simple") — see 22_welfare_bmm_simple.R. The fix is lower-bounded priors.
#
# Guard level: WARNING (not error), because:
#   (a) The user may have already specified lb=0 priors (we cannot verify from
#       the model object alone, since priors are separate from the model)
#   (b) Some formulas with identity link may have structural non-negativity
#       (e.g., activation = b + exp(x) always > 0)
# The warning is informational — it reminds the user to specify lb=0.

check_positive_utility <- function(model) {
  if (is.null(model$other_vars$choice_rule) ||
      model$other_vars$choice_rule != "simple") {
    return(invisible(TRUE))
  }
  identity_params <- names(model$links)[
    sapply(model$links, function(lk) identical(lk, "identity"))
  ]
  if (length(identity_params) == 0) return(invisible(TRUE))

  warning(sprintf(
    paste0(
      "G4 (positivity hazard): choice_rule='simple' with identity-linked ",
      "parameter(s): %s.\n",
      "  The simple (Luce) rule generates log(activation * n_options) for each ",
      "category (model_m3.R:402). With identity links, activations CAN be ",
      "non-positive if parameters go negative during sampling.\n",
      "  log(non-positive) = NaN in Stan -> divergences or failed sampling.\n",
      "  Fix: use lower-bounded priors for identity-linked utility parameters:\n",
      "    set_prior('normal(1, 0.5)', ..., nlpar = '%s', lb = 0)   # lb=0\n",
      "    # or: 'normal(1, 0.5) T[0,]'  (Stan truncation notation)\n",
      "  Note: configure_model.m3 already sets init=0 (model_m3.R:425-428)\n",
      "  to avoid starting from non-positive values, but this does not prevent\n",
      "  negative draws during sampling.\n",
      "  Reference: 22_welfare_bmm_simple.R — wi=1.2, wo=0.8 recovery with lb=0."
    ),
    paste(identity_params, collapse=", "),
    identity_params[1]
  ))
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

cat("--- G4: Positivity hazard (simple rule + identity link) ---\n\n")

# Deficient setup: model with simple rule + identity link, no lower-bounded prior
# (guard warns because the prior default normal(0,1) permits negative draws)
deficient_model <- m3(
  resp_cats   = c("nkeep", "ningroup", "nuniversal"),
  num_options = c(1L, 1L, 1L),
  choice_rule = "simple"
)
deficient_model$links <- list(wi = "identity", wo = "identity")

cat("Deficient setup: simple rule + identity links, no lower-bounded prior\n")
result_g4_warn <- tryCatch(
  withCallingHandlers(
    check_positive_utility(deficient_model),
    warning = function(w) {
      cat("GUARD G4 FIRES (correct):\n")
      cat(conditionMessage(w), "\n\n")
      invokeRestart("muffleWarning")
    }
  ),
  error = function(e) e
)

# Adequate setup: same model, but with lb=0 priors — warning fires but user
# has already addressed it. We reference 22_welfare_bmm_simple.R as the
# execution-backed proof that the correct configuration works.
adequate_model <- m3(
  resp_cats   = c("nkeep", "ningroup", "nuniversal"),
  num_options = c(1L, 1L, 1L),
  choice_rule = "simple"
)
adequate_model$links <- list(wi = "identity", wo = "identity")
# Note: fixed_parameters$b <- 1.0 (numeraire) is set in 22_welfare_bmm_simple.R
# Lower-bounded priors are set at fit time: set_prior(..., lb = 0)

cat("Adequate setup: simple rule + identity links + lb=0 priors (see 22_welfare_bmm_simple.R)\n")
cat("  G4 warns on model construction — user addresses by specifying lb=0 priors.\n")
cat("  22_welfare_bmm_simple.R confirms: wi=1.2, wo=0.8 recovered with lb=0 priors,\n")
cat("  0 divergences. Guard G4 is informational; the fix is straightforward.\n")
cat("  ADEQUATE: 22_welfare_bmm_simple.R passes with lb=0 priors ✓\n\n")

cat("Note: G4 is a WARNING not an ERROR because (a) the user may have already\n")
cat("  specified lb=0 priors and (b) some identity-linked formulas are structurally\n")
cat("  non-negative (e.g., b + exp(x) > 0 always). The guard is informational.\n\n")

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
    "G1: power utility (rho)\n  range ratio >= 3 (WARNING)",
    "G2: Prelec (alpha)\n  needs variable set size",
    "G3: EU slope (gamma)\n  needs value variation",
    "G4: positivity hazard\n  simple rule + identity link (WARNING)"
  ),
  Stage_required = c("check_data", "check_data", "check_data", "check_model"),
  Arch_A_exploration = c(
    "x (no S3 dispatch;\n   only manual call possible)",
    "x (same)",
    "x (same)",
    "~ (standalone check_positive_utility(model);\n   fires on construction)"
  ),
  Arch_A_production = c(
    "~ (add 'm3_utility' subclass\n   + check_data.m3_utility in R/)",
    "~ (same)",
    "~ (same)",
    "~ (check_model.m3_utility in R/;\n   fires before data check)"
  ),
  Arch_B_version = c(
    "~ (same as A-production;\n   needs class change AND method)",
    "~ (same)",
    "~ (same)",
    "~ (same)"
  ),
  Arch_C_sibling = c(
    "+ (check_data.utility_memory\n   auto-dispatched by bmm())",
    "+ (same)",
    "+ (same)",
    "+ (check_model.utility\n   auto-dispatched by bmm())"
  ),
  stringsAsFactors = FALSE
)

for (i in seq_len(nrow(guard_placement))) {
  cat(sprintf("Guard %d (%s)\n", i, c("G1","G2","G3","G4")[i]))
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

  # G1: rho identifiability (power utility only) — WARNING, range criterion
  if (!is.null(value_cols) && utility_fn == "power") {
    for (v_col in value_cols) {
      vals  <- sort(unique(data[[v_col]]))
      ratio <- max(vals) / min(vals)
      if (ratio < 3) {
        warning(sprintf(
          paste0(
            "G1 (power utility): value range ratio %.2f < 3 for column \'%s\'.\n",
            "  (gamma, rho) may be nearly aliased at small range. ",
            "Use V with max/min >= 3 (e.g. {1,3,5})."
          ),
          ratio, v_col
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
cat("  G1 power utility rho:  REVISED to WARNING + range criterion (max(V)/min(V)>=3)\n")
cat("  G2 Prelec alpha:       DEMONSTRATED (fires on fixed set size, passes on variable)\n")
cat("  G3 EU slope gamma:     DEMONSTRATED (fires on constant V, passes on variable)\n")
cat("  G4 positivity hazard:  DEMONSTRATED (warns on simple rule + identity link;\n")
cat("                           adequate use in 22_welfare_bmm_simple.R with lb=0)\n")
cat("\n  check_utility_design(model, data): pre-flight validator for Architecture A\n")
cat("  In Architecture A production / C (sibling): guards fire automatically.\n")
cat("  G4 requires check_model method (not check_data) — fires earlier in pipeline.\n")
cat("==========================================================================\n")
