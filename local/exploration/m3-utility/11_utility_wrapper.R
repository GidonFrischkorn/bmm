# 11_utility_wrapper.R
# WP2 — Prototype of m3_utility() wrapper
#
# The WP5 spike (10_bmm_api_example.R) exposed three sharp edges in the raw
# m3() + bmf() workflow:
#
#   1. Requires manual model$links mutation (was: model$links <- list(...))
#      → FIXED: m3() already accepts links = list(...) directly; line 90 of
#        R/model_m3.R: out$links[names(links)] <- links
#
#   2. No default prior for the value slope (gamma)
#      → FIXED in wrapper: attaches prior(normal(0,1), nlpar="gamma", class="b")
#
#   3. Silent predictor-vs-activation trap: adding V_corr in a ~ 1 + V_corr
#      shares the slope across corr/other, but the prototype requires different
#      V values per category
#      → FIXED in wrapper: generates explicit per-category activation formulas
#
# This script:
#   1. Verifies the m3(links=) approach works without model$links mutation
#   2. Implements m3_utility() prototype
#   3. Shows the clean usage pattern
#
# Everything in local/exploration/m3-utility/ — no changes to R/ or inst/.
#
# Run from repo root:
#   source("local/exploration/m3-utility/11_utility_wrapper.R")

library(bmm)
library(brms)

# ---- 0. Verify: m3(links=) works directly ------------------------------------
#
# R/model_m3.R line 53: .model_m3(..., links = NULL, ...)
# R/model_m3.R line 90: out$links[names(links)] <- links
#
# So m3(..., links = list(gamma = "identity")) should add gamma to the links
# WITHOUT needing post-construction mutation.

cat("=== Verifying m3(links=) approach ===\n\n")

test_model <- m3(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c("n_corr", "n_other", "n_npl"),
  choice_rule = "softmax",
  version     = "custom",
  links       = list(a = "identity", c = "identity", gamma = "identity")
)

cat("Links in model after m3(links=...):\n")
print(test_model$links)

has_gamma <- "gamma" %in% names(test_model$links) &&
             test_model$links[["gamma"]] == "identity"
cat("\nHas gamma with identity link:", has_gamma, "\n")
cat("No model$links mutation needed:", has_gamma, "\n\n")
stopifnot(has_gamma)


# ---- 1. m3_utility() prototype -----------------------------------------------
#
# Arguments:
#   resp_cats    : character vector of response category names (e.g. c("corr","other","npl"))
#   num_options  : named character vector, column names for option counts
#   value_cols   : named character vector, column names for value predictors per category
#                  (e.g. c(corr = "V_corr", other = "V_other"))
#                  Categories not named here get no value term.
#   utility      : "linear"  → gamma * V_i  (just gamma, default)
#                  "power"   → gamma * V_i^rho  (adds rho parameter)
#   weighting    : "none"    → standard softmax n_i adjustment (default)
#                  "prelec"  → Prelec w(p_i), adds alpha parameter
#                  When "prelec": data must contain p_<cat> columns (n_i / N_total)
#   choice_rule  : passed to m3() (default "softmax")
#   ...          : other arguments passed to m3()
#
# Returns: an m3 model object (class "bmmodel", "m3", "m3_custom") with:
#   - links     : identity for all estimated parameters incl. utility params
#   - default_priors : sensible defaults for gamma, rho, alpha
#   - ATTR "value_cols": stored for formula generation
#   - ATTR "utility"   : "linear" or "power"
#   - ATTR "weighting" : "none" or "prelec"
#
# Use m3_utility_formula() to generate the matching bmf() formula.

m3_utility <- function(resp_cats,
                       num_options,
                       value_cols   = NULL,
                       utility      = c("linear", "power"),
                       weighting    = c("none", "prelec"),
                       choice_rule  = "softmax",
                       ...) {

  utility   <- match.arg(utility)
  weighting <- match.arg(weighting)

  # Determine extra parameters and links
  extra_links  <- list(a = "identity", c = "identity")
  extra_priors <- list()

  if (!is.null(value_cols)) {
    extra_links[["gamma"]] <- "identity"
    extra_priors[["gamma"]] <- list(
      main    = "normal(0, 1)",
      effects = "normal(0, 0.3)"
    )
    if (utility == "power") {
      extra_links[["rho"]] <- "log"      # rho > 0; log link
      extra_priors[["rho"]] <- list(
        main    = "lognormal(0, 0.5)",
        effects = "normal(0, 0.3)"
      )
    }
  }

  if (weighting == "prelec") {
    extra_links[["alpha"]] <- "log"      # alpha > 0; log link
    extra_priors[["alpha"]] <- list(
      main    = "lognormal(0, 0.5)",
      effects = "normal(0, 0.2)"
    )
  }

  model <- m3(
    resp_cats    = resp_cats,
    num_options  = num_options,
    choice_rule  = choice_rule,
    version      = "custom",
    links        = extra_links,
    default_priors = extra_priors,
    ...
  )

  # Store metadata for formula generation
  attr(model, "value_cols") <- value_cols
  attr(model, "utility")    <- utility
  attr(model, "weighting")  <- weighting
  attr(model, "resp_cats")  <- resp_cats
  attr(model, "num_options") <- num_options

  model
}


# ---- 2. m3_utility_formula() -------------------------------------------------
#
# Generates the activation sub-formulas for bmf() based on the m3_utility model.
#
# The returned list can be passed to bmf() as:
#   do.call(bmf, c(m3_utility_formula(model), list(a ~ 1 + (1|subj), ...)))
#
# Or used as a reference for building the bmf() manually.

m3_utility_formula <- function(model) {
  resp_cats  <- attr(model, "resp_cats")
  value_cols <- attr(model, "value_cols")
  utility    <- attr(model, "utility")
  weighting  <- attr(model, "weighting")
  num_opts   <- attr(model, "num_options")

  if (is.null(resp_cats)) stop("model must be created with m3_utility()")

  # Base activation terms for each category
  # Standard M3: corr gets b+a+c, other gets b+a, npl gets b
  # (Assumes 3-category M3; generalises to n categories by treating first as
  # "fully activated", last as background only)
  n_cats    <- length(resp_cats)
  base_acts <- vector("character", n_cats)

  # Convention: last category is background (b only)
  base_acts[n_cats] <- "b"
  for (k in seq_len(n_cats - 1L)) {
    if (k == 1L) {
      base_acts[k] <- "b + a + c"
    } else {
      base_acts[k] <- "b + a"
    }
  }

  # Add value terms
  if (!is.null(value_cols)) {
    for (k in seq_len(n_cats - 1L)) {
      cat_k <- resp_cats[k]
      if (cat_k %in% names(value_cols)) {
        v_col <- value_cols[[cat_k]]
        u_term <- if (utility == "linear") {
          sprintf("gamma * %s", v_col)
        } else {
          sprintf("gamma * %s^rho", v_col)
        }
        base_acts[k] <- paste(base_acts[k], u_term, sep = " + ")
      }
    }
  }

  # Add Prelec weighting term to n_i offset
  if (weighting == "prelec") {
    num_opts_nm <- names(num_opts)
    if (is.null(num_opts_nm)) num_opts_nm <- paste0("n_", resp_cats)
    n_total_expr <- paste(num_opts_nm, collapse = " + ")
    for (k in seq_along(resp_cats)) {
      p_col <- sprintf("p_%s", resp_cats[k])
      # Prelec: replace log(n_i) with -((-log(p_i))^alpha)
      # This is encoded in the nlf formula, not the activation formula
      base_acts[k] <- paste0(
        base_acts[k], " + (-((-log(", p_col, "))^alpha))"
      )
    }
  }

  # Build named list of activation formulas
  form_list <- setNames(
    lapply(seq_along(resp_cats), function(k) {
      as.formula(paste(resp_cats[k], "~", base_acts[k]))
    }),
    resp_cats
  )

  form_list
}


# ---- 3. Usage example (no fitting) -------------------------------------------

cat("=== m3_utility() usage example ===\n\n")

# Build model for EU activation (linear utility)
eu_model_v2 <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "none",
  choice_rule = "softmax"
)

cat("Model links (no mutation needed):\n")
print(eu_model_v2$links)

cat("\nDefault priors (gamma included):\n")
print(eu_model_v2$default_priors)

cat("\nGenerated activation formulas:\n")
act_forms <- m3_utility_formula(eu_model_v2)
for (nm in names(act_forms)) {
  cat(sprintf("  %s\n", deparse(act_forms[[nm]])))
}

cat("\nComplete bmf() call:\n")
cat("  bmf(\n")
for (nm in names(act_forms)) {
  cat(sprintf("    %s,\n", deparse(act_forms[[nm]])))
}
cat("    a     ~ 1 + (1 | subj),\n")
cat("    c     ~ 1 + (1 | subj),\n")
cat("    gamma ~ 1 + (1 | subj)\n")
cat("  )\n\n")


# ---- 4. Power utility variant ------------------------------------------------

cat("=== Power utility variant (gamma * V^rho) ===\n\n")

pow_model <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "power",
  weighting   = "none"
)

cat("Power utility links:\n")
print(pow_model$links)

pow_forms <- m3_utility_formula(pow_model)
cat("\nActivation formulas for power utility:\n")
for (nm in names(pow_forms)) {
  cat(sprintf("  %s\n", deparse(pow_forms[[nm]])))
}


# ---- 5. Prelec variant -------------------------------------------------------

cat("\n=== Prelec weighting variant ===\n\n")

prelec_model <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = NULL,
  utility     = "linear",
  weighting   = "prelec"
)

cat("Prelec links:\n")
print(prelec_model$links)

prelec_forms <- m3_utility_formula(prelec_model)
cat("\nActivation formulas for Prelec:\n")
for (nm in names(prelec_forms)) {
  cat(sprintf("  %s\n", deparse(prelec_forms[[nm]])))
}


# ---- 6. Composite PT variant -------------------------------------------------

cat("\n=== Composite PT variant (gamma + Prelec) ===\n\n")

cpt_model <- m3_utility(
  resp_cats   = c("corr", "other", "npl"),
  num_options = c(n_corr = "n_corr", n_other = "n_other", n_npl = "n_npl"),
  value_cols  = c(corr = "V_corr", other = "V_other"),
  utility     = "linear",
  weighting   = "prelec"
)

cat("Composite PT links:\n")
print(cpt_model$links)

cpt_forms <- m3_utility_formula(cpt_model)
cat("\nActivation formulas for composite PT:\n")
for (nm in names(cpt_forms)) {
  cat(sprintf("  %s\n", deparse(cpt_forms[[nm]])))
}

cat("\n=== 11_utility_wrapper.R complete ===\n")
cat("Key result: m3(links=list(gamma='identity')) works without model$links mutation.\n")
cat("m3_utility() wraps this into a clean interface with automatic formula generation.\n")
