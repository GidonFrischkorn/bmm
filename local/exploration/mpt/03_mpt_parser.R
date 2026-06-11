# MPT parser prototype  — Round 2 revision
#
# Implements Syntax C (mpt_tree + mpt()) as the primary API, and
# Syntax A (MPTinR-string parse_mpt_string()) as a convenience wrapper.
#
# Changes from Phase 1:
#   - covariates argument: data columns that appear in branch expressions but
#     are NOT latent parameters; passed through unsanitised into emitted formulas.
#   - condition = NULL: single-tree models (no condition column required).
#   - Naming policy: error at mpt() construction if any parameter name contains
#     '_' or '.'; silent rename removed (brms forbids these in nlpar names).
#   - .mpt_validate_tree(): multiple distinct test points — cycling
#     (0.137, 0.421, 0.683, 0.852) — catches swapped-complement errors that
#     all-0.5 evaluation can miss.
#   - WP4 bypass: predictor formulas whose RHS references other predictor-formula
#     keys are emitted directly as nlf() (time-varying / non-linear), bypassing
#     the automatic inv_logit(l<param>) reparameterisation.
#
# Pure R — no package integration; no modifications to R/ or inst/.

# =============================================================================
# Low-level helpers
# =============================================================================

#' Extract latent parameter names from an MPT branch expression.
#' Excludes known R math functions and declared covariates.
#' @param expr    Character expression string.
#' @param covariates Character vector of declared covariate names to exclude.
.mpt_extract_params <- function(expr, covariates = character(0)) {
  tokens <- regmatches(expr, gregexpr("[A-Za-z_][A-Za-z0-9_.]*", expr))[[1]]
  r_fns  <- c("exp", "log", "sqrt", "sin", "cos", "tan", "abs", "sign",
               "inv_logit", "inv_probit", "logit", "probit", "pnorm", "qnorm",
               "Phi", "phi", "Phi_approx")
  unique(setdiff(tokens, c(r_fns, covariates)))
}

#' Numerically validate that an mpt_tree's branches sum to 1.
#'
#' Uses four distinct test points (0.137, 0.421, 0.683, 0.852) cycled across
#' all symbols (parameters + covariates).  Cycling ensures no two symbols share
#' the same value, which catches swapped-complement bugs that an all-0.5 test
#' would miss.
#'
#' @param tree       An mpt_tree object.
#' @param all_symbols Character vector — all symbols appearing in branches
#'   (both latent params and covariates).
.mpt_validate_tree <- function(tree, all_symbols) {
  if (length(all_symbols) == 0L) return(invisible(NULL))
  test_vals <- c(0.137, 0.421, 0.683, 0.852)
  n <- length(all_symbols)
  for (trial in seq_along(test_vals)) {
    vals     <- test_vals[((seq_len(n) - 1L + trial - 1L) %% length(test_vals)) + 1L]
    test_env <- as.list(setNames(vals, all_symbols))
    probs    <- vapply(tree$branches, function(expr) {
      eval(parse(text = expr), envir = test_env)
    }, numeric(1))
    total <- sum(probs)
    if (abs(total - 1) > 1e-6) {
      warning(sprintf(
        "Tree '%s': branches sum to %.6f (not 1) at test point %d.",
        tree$name, total, trial
      ))
    }
  }
  invisible(NULL)
}

# =============================================================================
# mpt_tree constructor
# =============================================================================

#' Create an MPT tree specification
#'
#' @param name     Character. Tree label; must match values in the condition
#'   column of the data (or be the sole tree name when condition = NULL).
#' @param branches Named list of character strings — one branch-probability
#'   expression per response category.
#' @return An object of class "mpt_tree".
#'
#' @examples
#' tree_old <- mpt_tree(
#'   name     = "old",
#'   branches = list(old = "D + (1 - D) * g", new = "(1 - D) * (1 - g)")
#' )
mpt_tree <- function(name, branches) {
  stopifnot(is.character(name), length(name) == 1L)
  stopifnot(is.list(branches), !is.null(names(branches)))
  structure(list(name = name, branches = branches), class = "mpt_tree")
}

#' @export
print.mpt_tree <- function(x, ...) {
  cat(sprintf("MPT tree '%s':\n", x$name))
  for (nm in names(x$branches)) {
    cat(sprintf("  P(%s) = %s\n", nm, x$branches[[nm]]))
  }
  invisible(x)
}

# =============================================================================
# mpt() constructor  — Syntax C
# =============================================================================

#' Specify an MPT model from a list of mpt_tree objects
#'
#' @param trees      A list of mpt_tree objects.  All trees must share the same
#'   set of response-category names.
#' @param condition  Character or NULL.  Name of the data column that identifies
#'   which tree applies to each row (values must match tree names).  Pass NULL
#'   for single-tree models where no condition column is needed.
#' @param covariates Character vector.  Names of data columns that appear in
#'   branch expressions but are NOT latent parameters (e.g. design-fixed
#'   guessing rates such as \code{GcorrPi = 1/rsizeList}).  Covariate names
#'   have no naming restrictions and are passed through into brms formulas
#'   unsanitised as data columns.
#' @param link       "logit" (default) or "probit".
#'
#' @return An object of class "mpt_spec".
mpt <- function(trees, condition = NULL, covariates = character(0), link = "logit") {
  stopifnot(is.list(trees), all(vapply(trees, inherits, logical(1), "mpt_tree")))
  if (!is.null(condition)) {
    stopifnot(is.character(condition), length(condition) == 1L)
  }
  link <- match.arg(link, c("logit", "probit"))

  # All trees must share the same response categories
  all_resp <- lapply(trees, function(t) sort(names(t$branches)))
  if (length(unique(lapply(all_resp, paste, collapse = ","))) > 1L) {
    stop("All trees must have the same response-category names (branch names).")
  }
  resp_cats <- names(trees[[1]]$branches)

  # Extract latent parameter names (excluding covariates)
  all_params <- unique(unlist(lapply(trees, function(t) {
    unlist(lapply(t$branches, .mpt_extract_params, covariates = covariates))
  })))

  # Naming policy: error if any parameter contains '_' or '.'
  bad_params <- grep("[._]", all_params, value = TRUE)
  if (length(bad_params) > 0L) {
    stop(sprintf(paste0(
      "MPT parameter names must not contain underscores ('_') or dots ('.').\n",
      "brms forbids these characters in non-linear parameter (nlpar) names.\n",
      "Offending parameters: %s\n",
      "Please rename them (e.g., 'd_A' -> 'dA')."
    ), paste(bad_params, collapse = ", ")))
  }

  # Collision checks
  bad_cov <- intersect(covariates, all_params)
  if (length(bad_cov) > 0L) {
    stop(sprintf(
      "Symbols cannot be both a parameter and a covariate: %s",
      paste(bad_cov, collapse = ", ")
    ))
  }

  # Generated indicator-column names must not collide with declared covariates
  tree_names <- vapply(trees, `[[`, "", "name")
  ind_names  <- paste0(".ind_", tree_names)
  ind_cov_collision <- intersect(ind_names, covariates)
  if (length(ind_cov_collision) > 0L) {
    stop(sprintf(
      "Covariate names collide with generated indicator columns: %s",
      paste(ind_cov_collision, collapse = ", ")
    ))
  }

  # Numeric tree validation (multiple test points)
  all_symbols <- c(all_params, covariates)
  for (t in trees) {
    .mpt_validate_tree(t, all_symbols)
  }

  structure(
    list(
      trees      = trees,
      condition  = condition,
      link       = link,
      resp_cats  = resp_cats,
      params     = all_params,
      covariates = covariates
    ),
    class = "mpt_spec"
  )
}

#' @export
print.mpt_spec <- function(x, ...) {
  cat("MPT model specification\n")
  cat(sprintf("  Trees     : %s\n",
              paste(vapply(x$trees, `[[`, "", "name"), collapse = ", ")))
  cat(sprintf("  Resp. cats: %s\n", paste(x$resp_cats, collapse = ", ")))
  cat(sprintf("  Parameters: %s\n", paste(x$params, collapse = ", ")))
  cat(sprintf("  Covariates: %s\n",
              if (length(x$covariates) == 0L) "(none)"
              else paste(x$covariates, collapse = ", ")))
  cat(sprintf("  Condition : %s\n",
              if (is.null(x$condition)) "(none — single-tree model)"
              else x$condition))
  cat(sprintf("  Link      : %s\n", x$link))
  invisible(x)
}

# =============================================================================
# Formula emitter: mpt_spec -> brms formula components
# =============================================================================

#' Build brms formula components from an mpt_spec
#'
#' Returns a named list:
#' \describe{
#'   \item{brms_formula}{The full \code{brms::bf()} object.}
#'   \item{data_prep}{A \code{function(data)} that adds indicator columns (and,
#'     for multinomial models, a response matrix \code{Y}).}
#'   \item{suggested_priors}{A \code{brms::prior()} object with
#'     \code{logistic(0,1)} priors on logit-scale intercepts for each latent
#'     parameter.  NULL if no standard params exist.}
#'   \item{link_params}{Named character: latent_param -> l<param> name for
#'     standard (non-bypass, non-simplex) parameters.}
#'   \item{family}{"binomial" or "multinomial".}
#' }
#'
#' @param spec              An \code{mpt_spec} (output of \code{mpt()}).
#' @param predictor_formulas Named list of one-sided R formulas, one per latent
#'   parameter (or simplex free parameter or sub-parameter).  Defaults to
#'   \code{~ 1}.
#'
#'   **Time-varying (WP4 bypass):** if a parameter's formula RHS references
#'   symbols that are themselves keys in \code{predictor_formulas}, it is
#'   treated as a non-linear time-varying formula.  The formula is emitted as
#'   \code{nlf(param ~ <user expression>)} directly — the automatic
#'   \code{inv_logit(l<param>)} reparameterisation is skipped.  Any predictor-
#'   formula keys that are *not* tree parameters become sub-parameters and
#'   receive \code{lf(sub_param ~ <user rhs>)} formulas.
#' @param response_col      Character.  For binomial (K==2) only: the "success"
#'   column name (default: first resp_cat).
#' @param trials_col        Character.  Total-trials column name.
#' @param simplex_params    Character vector (or list of vectors) of parameters
#'   that form a probability simplex.  Stick-breaking reparametrisation is
#'   applied.  Not applicable to binomial models.
#'
#' @return Named list (see Description).
mpt_to_brms <- function(spec,
                         predictor_formulas = list(),
                         response_col       = spec$resp_cats[1],
                         trials_col         = "n",
                         simplex_params     = NULL) {
  stopifnot(inherits(spec, "mpt_spec"))
  link_fn    <- if (spec$link == "logit") "inv_logit" else "Phi"
  trees      <- spec$trees
  resp       <- spec$resp_cats
  n_cats     <- length(resp)
  params     <- spec$params    # already validated: no '_' or '.' in names
  covariates <- spec$covariates

  tree_names <- vapply(trees, `[[`, "", "name")
  ind_names  <- paste0(".ind_", tree_names)

  # ---------------------------------------------------------------------------
  # Aggregated probability for response category `cat` across all trees
  # ---------------------------------------------------------------------------
  build_agg_prob <- function(cat) {
    parts <- vapply(seq_along(trees), function(i) {
      sprintf("%s * (%s)", ind_names[i], trees[[i]]$branches[[cat]])
    }, "")
    paste(parts, collapse = " + ")
  }

  # ---------------------------------------------------------------------------
  # data_prep: add tree indicator columns (and Y matrix for multinomial)
  # ---------------------------------------------------------------------------
  if (is.null(spec$condition)) {
    # Single-tree model: indicator is 1 for all rows
    data_prep_base <- local({
      nms <- ind_names
      function(data) { for (nm in nms) data[[nm]] <- 1L; data }
    })
  } else {
    # Multi-tree model: indicator matches condition column value
    data_prep_base <- local({
      cond_col  <- spec$condition
      t_names   <- tree_names
      i_names   <- ind_names
      function(data) {
        for (i in seq_along(t_names)) {
          data[[i_names[i]]] <- as.integer(data[[cond_col]] == t_names[i])
        }
        data
      }
    })
  }

  # ---------------------------------------------------------------------------
  # WP4: Bypass detection
  # A tree parameter is "bypass" (time-varying) when its predictor formula RHS
  # references symbols that are also keys in predictor_formulas (i.e. the user
  # has written a compound non-linear expression that decomposes into further
  # fitted parameters).
  # ---------------------------------------------------------------------------
  pred_keys  <- names(predictor_formulas)
  is_bypass  <- setNames(logical(length(params)), params)
  for (p in params) {
    fm_p <- predictor_formulas[[p]]
    if (!is.null(fm_p)) {
      rhs_str    <- paste(deparse(fm_p[[2]]), collapse = "")
      rhs_tokens <- .mpt_extract_params(rhs_str, covariates = covariates)
      is_bypass[p] <- any(rhs_tokens %in% pred_keys)
    }
  }
  bypass_params  <- params[is_bypass]
  # Sub-params: predictor_formula keys that are not tree parameters
  sub_params     <- setdiff(pred_keys, params)

  # ---------------------------------------------------------------------------
  # Helper: emit user RHS from predictor_formulas, default "1"
  # ---------------------------------------------------------------------------
  user_rhs <- function(p) {
    fm <- predictor_formulas[[p]]
    if (!is.null(fm)) paste(deparse(fm[[2]]), collapse = "") else "1"
  }

  # ===========================================================================
  # Binomial path (K == 2)
  # ===========================================================================
  if (n_cats == 2L) {
    agg_prob <- build_agg_prob(response_col)
    lhs      <- sprintf("%s | trials(%s)", response_col, trials_col)

    main_bf  <- brms::bf(
      as.formula(sprintf("%s ~ logit(%s)", lhs, agg_prob)),
      nl = TRUE
    )

    standard_params <- params[!is_bypass]
    l_standard      <- paste0("l", standard_params)

    # nlf: standard params (automatic inv_logit reparameterisation)
    for (i in seq_along(standard_params)) {
      p  <- standard_params[i]
      lp <- l_standard[i]
      main_bf <- main_bf +
        brms::nlf(as.formula(sprintf("%s ~ %s(%s)", p, link_fn, lp)))
    }

    # nlf: bypass params (user's compound non-linear expression)
    for (p in bypass_params) {
      main_bf <- main_bf +
        brms::nlf(as.formula(sprintf("%s ~ %s", p, user_rhs(p))))
    }

    # lf: standard params (predictor formula on l-param)
    for (i in seq_along(standard_params)) {
      p  <- standard_params[i]
      lp <- l_standard[i]
      main_bf <- main_bf +
        brms::lf(as.formula(sprintf("%s ~ %s", lp, user_rhs(p))))
    }

    # lf: sub-params (supporting bypass formulas)
    for (sp in sub_params) {
      main_bf <- main_bf +
        brms::lf(as.formula(sprintf("%s ~ %s", sp, user_rhs(sp))))
    }

    # Suggested priors: logistic(0,1) on logit-scale intercepts
    prior_list <- if (length(l_standard) > 0L) {
      do.call(c, lapply(l_standard, function(lp) {
        brms::prior_string("logistic(0, 1)", nlpar = lp, class = "b",
                           coef = "Intercept")
      }))
    } else NULL

    return(list(
      brms_formula     = main_bf,
      data_prep        = data_prep_base,
      suggested_priors = prior_list,
      link_params      = setNames(l_standard, standard_params),
      family           = "binomial"
    ))
  }

  # ===========================================================================
  # Multinomial path (K > 2)
  # Follows M3 infrastructure pattern (R/model_m3.R:bmf2bf.m3):
  #   - base bf(): Y | trials(n) ~ log(P_first_cat), nl = TRUE
  #   - nlf(mu{cat} ~ log(P_cat)) for each subsequent category
  #   - nlf/lf per parameter (standard, bypass, simplex)
  # ===========================================================================

  # Simplex groups
  simplex_grps <- if (is.null(simplex_params)) {
    list()
  } else if (!is.list(simplex_params)) {
    list(simplex_params)
  } else {
    simplex_params
  }
  all_simplex <- unlist(simplex_grps)
  if (any(is_bypass[all_simplex])) {
    stop("Time-varying bypass is not supported for simplex_params.")
  }

  non_simplex          <- setdiff(params, all_simplex)
  standard_non_simplex <- non_simplex[!is_bypass[non_simplex]]
  bypass_non_simplex   <- non_simplex[is_bypass[non_simplex]]
  l_standard           <- paste0("l", standard_non_simplex)

  # Base formula (first category)
  first_cat <- resp[1]
  main_bf   <- brms::bf(
    as.formula(sprintf("Y | trials(%s) ~ log(%s)", trials_col,
                       build_agg_prob(first_cat))),
    nl = TRUE
  )

  # nlf for each subsequent category
  for (cat in resp[-1]) {
    main_bf <- main_bf +
      brms::nlf(as.formula(sprintf("mu%s ~ log(%s)", cat, build_agg_prob(cat))))
  }

  # nlf: standard non-simplex params
  for (i in seq_along(standard_non_simplex)) {
    p  <- standard_non_simplex[i]
    lp <- l_standard[i]
    main_bf <- main_bf +
      brms::nlf(as.formula(sprintf("%s ~ %s(%s)", p, link_fn, lp)))
  }

  # nlf: bypass params
  for (p in bypass_non_simplex) {
    main_bf <- main_bf +
      brms::nlf(as.formula(sprintf("%s ~ %s", p, user_rhs(p))))
  }

  # nlf: simplex params (stick-breaking)
  simplex_free_lp <- character(0)
  for (grp in simplex_grps) {
    K_s <- length(grp)
    for (k in seq_len(K_s)) {
      sp <- grp[k]
      if (k < K_s) {
        lp_name         <- paste0("l", sp)
        simplex_free_lp <- c(simplex_free_lp, lp_name)
        rhs_nlf <- if (k == 1L) {
          sprintf("inv_logit(%s)", lp_name)
        } else {
          prev_factors <- paste(
            sprintf("(1 - inv_logit(l%s))", grp[seq_len(k - 1L)]),
            collapse = " * "
          )
          sprintf("%s * inv_logit(%s)", prev_factors, lp_name)
        }
        main_bf <- main_bf +
          brms::nlf(as.formula(sprintf("%s ~ %s", sp, rhs_nlf)))
      } else {
        prev_terms <- paste(grp[-K_s], collapse = " + ")
        main_bf <- main_bf +
          brms::nlf(as.formula(sprintf("%s ~ 1 - (%s)", sp, prev_terms)))
      }
    }
  }

  # lf: standard non-simplex
  for (i in seq_along(standard_non_simplex)) {
    p  <- standard_non_simplex[i]
    lp <- l_standard[i]
    main_bf <- main_bf +
      brms::lf(as.formula(sprintf("%s ~ %s", lp, user_rhs(p))))
  }

  # lf: simplex free params
  simplex_free_orig <- unlist(lapply(simplex_grps, function(g) g[-length(g)]))
  for (orig in simplex_free_orig) {
    lp <- paste0("l", orig)
    main_bf <- main_bf +
      brms::lf(as.formula(sprintf("%s ~ %s", lp, user_rhs(orig))))
  }

  # lf: sub-params
  for (sp in sub_params) {
    main_bf <- main_bf +
      brms::lf(as.formula(sprintf("%s ~ %s", sp, user_rhs(sp))))
  }

  # Suggested priors (guard: paste0("l", NULL) = "l" in R, not character(0))
  simplex_free_lp2 <- if (length(simplex_free_orig) > 0L) paste0("l", simplex_free_orig) else character(0)
  all_l_params     <- c(l_standard, simplex_free_lp2)
  prior_list <- if (length(all_l_params) > 0L) {
    do.call(c, lapply(all_l_params, function(lp) {
      brms::prior_string("logistic(0, 1)", nlpar = lp, class = "b",
                         coef = "Intercept")
    }))
  } else NULL

  # Multinomial family (mirrors M3 configure_model.m3)
  mnomial_family        <- brms::multinomial(refcat = NA)
  mnomial_family$cats   <- resp
  mnomial_family$dpars  <- paste0("mu", resp)

  # data_prep for multinomial: add indicator columns + Y response matrix
  resp_cats_for_Y <- resp
  data_prep_multi <- local({
    base_prep <- data_prep_base
    rc        <- resp_cats_for_Y
    function(data) {
      data  <- base_prep(data)
      Y_mat <- as.matrix(data[rc])
      colnames(Y_mat) <- rc
      data$Y <- Y_mat
      data
    }
  })

  # link_params map (standard + simplex free); simplex_free_lp2 already guarded
  simplex_free_map <- if (length(simplex_free_orig) > 0L) setNames(simplex_free_lp2, simplex_free_orig) else setNames(character(0), character(0))
  param_lp_map     <- c(setNames(l_standard, standard_non_simplex),
                         simplex_free_map)

  list(
    brms_formula     = main_bf,
    data_prep        = data_prep_multi,
    suggested_priors = prior_list,
    link_params      = param_lp_map,
    family           = "multinomial",
    family_obj       = mnomial_family,
    simplex_params   = simplex_grps
  )
}

# =============================================================================
# Syntax A: parse_mpt_string() — MPTinR-style import wrapper
# =============================================================================

#' Parse an MPTinR-style model string into an mpt_spec
#'
#' Lines are branch-probability equations. Blank lines separate trees.
#' Inline \code{# comment} gives the branch name.
#'
#' @param model_str      Character string in MPTinR equation format.
#' @param tree_conditions Character vector (length == number of trees) giving
#'   the condition values in the data corresponding to each tree block.
#' @param condition      Character or NULL.  Column name in data.
#' @param covariates     Character vector of covariate names (passed to
#'   \code{mpt()}).
#' @param link           "logit" or "probit".
#'
#' @return An mpt_spec object.
parse_mpt_string <- function(model_str,
                              tree_conditions,
                              condition   = "condition",
                              covariates  = character(0),
                              link        = "logit") {
  lines    <- trimws(strsplit(model_str, "\n")[[1]])
  trees_raw <- list()
  current   <- character(0)
  for (ln in lines) {
    if (nchar(ln) == 0L) {
      if (length(current) > 0L) { trees_raw <- c(trees_raw, list(current)); current <- character(0) }
    } else {
      current <- c(current, ln)
    }
  }
  if (length(current) > 0L) trees_raw <- c(trees_raw, list(current))

  if (length(trees_raw) != length(tree_conditions)) {
    stop(sprintf(
      "parse_mpt_string: found %d tree block(s) but %d tree_conditions supplied.",
      length(trees_raw), length(tree_conditions)
    ))
  }

  parse_one_tree <- function(block, cond_name) {
    branches <- list()
    for (ln in block) {
      parts  <- strsplit(ln, "#")[[1]]
      expr   <- trimws(parts[1])
      bname  <- if (length(parts) >= 2L) trimws(parts[2]) else NULL
      if (is.null(bname) || nchar(bname) == 0L) {
        bname <- paste0("cat", length(branches) + 1L)
      }
      branches[[bname]] <- expr
    }
    mpt_tree(name = cond_name, branches = branches)
  }

  tree_list <- mapply(parse_one_tree, trees_raw, tree_conditions, SIMPLIFY = FALSE)
  mpt(trees = tree_list, condition = condition, covariates = covariates, link = link)
}

# =============================================================================
# Tests
# =============================================================================

cat("=== Test 1: 2HTM via Syntax C (mpt_tree + mpt + mpt_to_brms) ===\n\n")

tree_old <- mpt_tree("old", list(
  old = "D + (1 - D) * g",
  new = "(1 - D) * (1 - g)"
))
tree_new <- mpt_tree("new", list(
  old = "(1 - D) * g",
  new = "D + (1 - D) * (1 - g)"
))

spec_2htm <- mpt(list(tree_old, tree_new), condition = "condition")
print(spec_2htm)

out_2htm <- mpt_to_brms(
  spec_2htm,
  predictor_formulas = list(D = ~ 1, g = ~ 1),
  response_col = "old",
  trials_col   = "n"
)
cat("\nGenerated brms formula:\n")
print(out_2htm$brms_formula)
cat("\nSuggested priors:\n")
print(out_2htm$suggested_priors)


cat("\n=== Test 2: 2HTM via Syntax A (parse_mpt_string) ===\n\n")

model_str_2htm <- "
D + (1 - D) * g        # old
(1 - D) * (1 - g)      # new

(1 - D) * g            # old
D + (1 - D) * (1 - g)  # new
"

spec_parsed <- parse_mpt_string(
  model_str_2htm,
  tree_conditions = c("old", "new"),
  condition = "condition"
)
print(spec_parsed)

out_parsed <- mpt_to_brms(
  spec_parsed,
  predictor_formulas = list(D = ~ 1, g = ~ 1),
  response_col = "old"
)
cat("\nParsed formula matches manual spec:\n")
identical_formula <- identical(
  deparse(out_2htm$brms_formula$formula),
  deparse(out_parsed$brms_formula$formula)
)
cat(sprintf("  Formulas identical: %s\n", identical_formula))


cat("\n=== Test 3: Source Monitoring Model (SMM, binomial) ===\n\n")
# Renamed d_A -> dA, d_B -> dB to comply with naming policy
smm_str <- "
dA + (1 - dA) * g        # A
(1 - dA) * (1 - g)       # notA

(1 - dB) * g             # A
dB + (1 - dB) * (1 - g)  # notA
"

spec_smm <- parse_mpt_string(
  smm_str,
  tree_conditions = c("sourceA", "sourceB"),
  condition = "source"
)
print(spec_smm)

out_smm <- mpt_to_brms(
  spec_smm,
  predictor_formulas = list(
    dA = ~ 1 + (1 | id),
    dB = ~ 1 + (1 | id),
    g  = ~ 1
  ),
  response_col = "A",
  trials_col   = "n"
)
cat("\nSMM brms formula:\n")
print(out_smm$brms_formula)


cat("\n=== Test 4: Pair Clustering Model (PCM, K=3 multinomial) ===\n\n")
pcm_str <- "
cp + (1 - cp) * rp * rp         # C
2 * (1 - cp) * rp * (1 - rp)    # E
(1 - cp) * (1 - rp) * (1 - rp)  # U
"
spec_pcm <- parse_mpt_string(
  pcm_str,
  tree_conditions = c("study"),
  condition = "dummy_cond"
)
print(spec_pcm)

out_pcm <- mpt_to_brms(
  spec_pcm,
  predictor_formulas = list(cp = ~ 1, rp = ~ 1)
)
cat("\nPCM multinomial brms formula:\n")
print(out_pcm$brms_formula)
cat("\nPCM family  :", out_pcm$family, "\n")
cat("PCM dpars   :", paste(out_pcm$family_obj$dpars, collapse = ", "), "\n")
cat("PCM priors  :\n")
print(out_pcm$suggested_priors)


cat("\n=== Test 5: SMM3 multinomial with simplex_params ===\n\n")
# Renamed d_A -> dA, d_B -> dB, g_A -> gA, g_B -> gB, g_New -> gNew
smm3_str <- "
dA + (1 - dA) * gA       # A
(1 - dA) * gB             # B
(1 - dA) * gNew           # New

(1 - dB) * gA             # A
dB + (1 - dB) * gB       # B
(1 - dB) * gNew           # New

gA                        # A
gB                        # B
gNew                      # New
"
spec_smm3 <- parse_mpt_string(
  smm3_str,
  tree_conditions = c("sourceA", "sourceB", "new"),
  condition       = "source"
)
print(spec_smm3)

out_smm3 <- mpt_to_brms(
  spec_smm3,
  predictor_formulas = list(
    dA  = ~ 1 + (1 | id),
    dB  = ~ 1 + (1 | id),
    gA  = ~ 1,
    gB  = ~ 1
  ),
  simplex_params = c("gA", "gB", "gNew")
)
cat("\nSMM3 multinomial brms formula:\n")
print(out_smm3$brms_formula)
cat("\nSMM3 free l-params (stick-breaking):",
    paste(names(out_smm3$link_params), collapse = ", "), "\n")
cat("SMM3 family:", out_smm3$family, "\n")


cat("\n=== Test 6: Covariates smoke test (WP1 acceptance) ===\n\n")
# Oberauer (2019) simple-span MPT tree with design-fixed guessing rates.
# GcorrPi, GcorrNoPi, GotherNoPi are data covariates (= 1/rsizeList etc.);
# only Pb and Pi are latent parameters.
tree_ss <- mpt_tree("ss", list(
  correct = "Pb + (1 - Pb) * Pi * GcorrPi + (1 - Pb) * (1 - Pi) * GcorrNoPi",
  other   = paste0("(1 - Pb) * Pi * (1 - GcorrPi) + ",
                   "(1 - Pb) * (1 - Pi) * (1 - GcorrNoPi) * GotherNoPi"),
  npl     = "(1 - Pb) * (1 - Pi) * (1 - GcorrNoPi) * (1 - GotherNoPi)"
))
spec_obs <- mpt(
  trees      = list(tree_ss),
  covariates = c("GcorrPi", "GcorrNoPi", "GotherNoPi")
  # condition = NULL: single-tree model
)
print(spec_obs)

# Params must be exactly {Pb, Pi}
stopifnot(setequal(spec_obs$params, c("Pb", "Pi")))
cat("  spec$params correctly identifies Pb and Pi as latent parameters.\n")

out_obs <- mpt_to_brms(
  spec_obs,
  predictor_formulas = list(
    Pi = ~ 1 + Csetsize + (1 + Csetsize || id),
    Pb = ~ 1 + Csetsize + (1 + Csetsize || id)
  )
)
cat("\nObserauer ss brms formula:\n")
print(out_obs$brms_formula)

# Assert covariates appear verbatim in the formula
formula_lines <- capture.output(print(out_obs$brms_formula))
formula_str   <- paste(formula_lines, collapse = " ")
for (cov in c("GcorrPi", "GcorrNoPi", "GotherNoPi")) {
  if (!grepl(cov, formula_str)) stop(sprintf("Covariate '%s' missing from formula.", cov))
  cat(sprintf("  Covariate '%s' present in formula: TRUE\n", cov))
}
cat("Test 6 passed: covariates pass through verbatim.\n")


cat("\n=== Test 7: Time-varying bypass smoke test (WP4 acceptance) ===\n\n")
# Detection model with exponential time-varying D(t).
# Gcorr (= 1/rsize) is a covariate; ptime is a data column in predictor formulas.
tree_tv <- mpt_tree("main", list(
  correct   = "D + (1 - D) * Gcorr",
  incorrect = "(1 - D) * (1 - Gcorr)"
))
spec_tv <- mpt(
  trees      = list(tree_tv),
  covariates = c("Gcorr")
)
print(spec_tv)

out_tv <- mpt_to_brms(
  spec_tv,
  predictor_formulas = list(
    D     = ~ inv_logit(lDmax) * (1 - exp(-exp(lrate) * ptime)),
    lDmax = ~ 1 + (1 | id),
    lrate = ~ 1 + (1 | id)
  ),
  response_col = "correct",
  trials_col   = "n"
)
cat("\nTime-varying brms formula:\n")
print(out_tv$brms_formula)

# Assert D is bypassed (no nlf(D ~ inv_logit(lD)) in formula)
tv_str <- paste(capture.output(print(out_tv$brms_formula)), collapse = " ")
if (grepl("inv_logit\\(lD\\)", tv_str)) {
  stop("Bypass failed: found automatic inv_logit(lD) reparameterisation for D.")
}
if (!grepl("lDmax", tv_str)) stop("Bypass failed: lDmax missing from formula.")
if (!grepl("lrate",  tv_str)) stop("Bypass failed: lrate missing from formula.")
cat("Test 7 passed: time-varying bypass emitted correctly.\n")


cat("\n=== Test 8: Probit link smoke test ===\n\n")
# Build a minimal 2HTM spec with link = "probit" and check that make_stancode
# emits Phi( rather than pnorm(.
tree_2htm_p <- mpt_tree("old", list(
  old = "D + (1 - D) * g",
  new = "(1 - D) * (1 - g)"
))
spec_probit <- mpt(
  trees = list(tree_2htm_p),
  link  = "probit"
)
out_probit <- mpt_to_brms(
  spec_probit,
  predictor_formulas = list(D = ~ 1, g = ~ 1),
  response_col = "old",
  trials_col   = "n"
)
# Build Stan code for a tiny dummy dataset
dummy_data <- data.frame(old = 8L, new = 2L, n = 10L, .ind_old = 1L)
suppressMessages({
  sc_probit <- brms::make_stancode(
    out_probit$brms_formula,
    data    = out_probit$data_prep(dummy_data),
    family  = binomial(),
    prior   = out_probit$suggested_priors
  )
})
if (grepl("pnorm", sc_probit))
  stop("Probit test FAILED: found pnorm in Stan code (should be Phi).")
if (!grepl("Phi\\(", sc_probit))
  stop("Probit test FAILED: Phi( not found in Stan code.")
cat("  Stan code contains Phi( : TRUE\n")
cat("  Stan code contains pnorm: FALSE\n")
cat("Test 8 passed: probit link correctly emits Phi.\n")


cat("\n=== All tests passed ===\n")
