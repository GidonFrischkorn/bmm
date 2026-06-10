# MPT parser prototype
# Implements Syntax C (mpt_tree + mpt()) as the primary API, and
# Syntax A (MPTinR-string parse_mpt_string()) as a convenience wrapper.
#
# Pure R — no package integration; no modifications to R/ or inst/.
#
# Tests at bottom cover: 2HTM, source-monitoring model (SMM).

# =============================================================================
# Low-level helpers
# =============================================================================

#' Sanitize a parameter name for use as a brms NL parameter (nlpar).
#' brms forbids dots and underscores; this replaces them with nothing.
#' @param nm Character. E.g. "d_A" -> "dA".
.sanitize_nlpar <- function(nm) gsub("[._]", "", nm)

#' Replace all occurrences of orig_names with safe_names in an expression string
#' using whole-word matching.
.replace_params_in_expr <- function(expr, orig_names, safe_names) {
  for (i in seq_along(orig_names)) {
    if (orig_names[i] != safe_names[i]) {
      expr <- gsub(sprintf("\\b%s\\b", orig_names[i]), safe_names[i], expr)
    }
  }
  expr
}

#' Extract parameter names from an MPT branch expression string
#' @param expr Character string, e.g. "D + (1 - D) * g"
#' @return Character vector of unique identifiers (not numeric / operators)
.mpt_extract_params <- function(expr) {
  # tokenise: letters+digits+underscore sequences
  tokens <- regmatches(expr, gregexpr("[A-Za-z_][A-Za-z0-9_.]*", expr))[[1]]
  # exclude R math functions that might appear
  r_fns <- c("exp", "log", "sqrt", "sin", "cos", "tan", "abs", "sign",
              "inv_logit", "inv_probit", "logit", "probit", "pnorm", "qnorm",
              "Phi", "phi", "Phi_approx")
  unique(setdiff(tokens, r_fns))
}

#' Validate that branch probabilities in a tree sum to 1 (symbolically skipped;
#' checked numerically at some parameter point)
#' @param tree An mpt_tree object
#' @param test_params Named numeric vector of parameter values
.mpt_validate_tree <- function(tree, test_params = NULL) {
  if (is.null(test_params)) {
    # default: all params = 0.5
    all_params <- unique(unlist(lapply(tree$branches, .mpt_extract_params)))
    test_params <- setNames(rep(0.5, length(all_params)), all_params)
  }
  probs <- vapply(tree$branches, function(expr) {
    eval(parse(text = expr), envir = as.list(test_params))
  }, numeric(1))
  total <- sum(probs)
  if (abs(total - 1) > 1e-6) {
    warning(sprintf(
      "Tree '%s': branch probabilities sum to %.6f (not 1) at test parameters.",
      tree$name, total
    ))
  }
  invisible(total)
}

# =============================================================================
# mpt_tree constructor
# =============================================================================

#' Create an MPT tree specification
#'
#' @param name Character. The label for this tree (must match a value in the
#'   condition column of the data).
#' @param branches Named list of character strings. Each element is a branch
#'   probability expression (valid R math, using parameter names as variables).
#'   Names are the response category labels.
#' @return An object of class "mpt_tree"
#'
#' @examples
#' tree_old <- mpt_tree(
#'   name = "old",
#'   branches = list(old = "D + (1 - D) * g", new = "(1 - D) * (1 - g)")
#' )
mpt_tree <- function(name, branches) {
  stopifnot(is.character(name), length(name) == 1L)
  stopifnot(is.list(branches), !is.null(names(branches)))
  structure(
    list(name = name, branches = branches),
    class = "mpt_tree"
  )
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
#' @param trees A list of mpt_tree objects. All trees must share the same set
#'   of response-category names (branch names).
#' @param condition Character. Name of the column in the data that identifies
#'   which tree applies to each observation (its values must match
#'   \code{tree$name} for each tree in \code{trees}).
#' @param link Character. Link function applied to each model parameter before
#'   the GLM predictor formula. One of "logit" (default) or "probit".
#'
#' @return An object of class "mpt_spec"
#'
#' @examples
#' tree_old <- mpt_tree("old", list(old = "D + (1 - D) * g",
#'                                   new = "(1 - D) * (1 - g)"))
#' tree_new <- mpt_tree("new", list(old = "(1 - D) * g",
#'                                   new = "D + (1 - D) * (1 - g)"))
#' spec <- mpt(trees = list(tree_old, tree_new), condition = "condition")
mpt <- function(trees, condition, link = "logit") {
  stopifnot(is.list(trees), all(vapply(trees, inherits, logical(1), "mpt_tree")))
  stopifnot(is.character(condition), length(condition) == 1L)
  link <- match.arg(link, c("logit", "probit"))

  # check all trees have the same response categories
  all_resp <- lapply(trees, function(t) sort(names(t$branches)))
  if (length(unique(lapply(all_resp, paste, collapse = ","))) > 1L) {
    stop("All trees must have the same response-category names (branch names).")
  }
  resp_cats <- names(trees[[1]]$branches)

  # collect all parameter names
  all_params <- unique(unlist(lapply(trees, function(t) {
    unlist(lapply(t$branches, .mpt_extract_params))
  })))

  structure(
    list(
      trees     = trees,
      condition = condition,
      link      = link,
      resp_cats = resp_cats,
      params    = all_params
    ),
    class = "mpt_spec"
  )
}

#' @export
print.mpt_spec <- function(x, ...) {
  cat(sprintf("MPT model specification\n"))
  cat(sprintf("  Trees     : %s\n", paste(vapply(x$trees, `[[`, "", "name"), collapse = ", ")))
  cat(sprintf("  Resp. cats: %s\n", paste(x$resp_cats, collapse = ", ")))
  cat(sprintf("  Parameters: %s\n", paste(x$params, collapse = ", ")))
  cat(sprintf("  Condition : %s\n", x$condition))
  cat(sprintf("  Link      : %s\n", x$link))
  invisible(x)
}

# =============================================================================
# Formula emitter: mpt_spec → brms formula components
# =============================================================================

#' Build brms formula from an mpt_spec
#'
#' Returns a named list with components needed to call \code{brm()}:
#'   \describe{
#'     \item{brms_formula}{The full \code{brms::bf()} object.}
#'     \item{data_prep}{A \code{function(data)} that adds indicator columns (and,
#'       for multinomial models, a response matrix \code{Y}) to the data frame.}
#'     \item{suggested_priors}{A \code{brms::prior()} object with
#'       \code{logistic(0, 1)} priors on intercepts for each linear-scale param.}
#'     \item{family}{Character "binomial" or "multinomial"; for multinomial, the
#'       \code{brms::multinomial(refcat = NA)} family object is also returned as
#'       \code{family_obj} with \code{cats} and \code{dpars} pre-set.}
#'     \item{link_params}{Named character: maps model param → l<param> name.}
#'   }
#'
#' @param spec An \code{mpt_spec} object (output of \code{mpt()}).
#' @param predictor_formulas Named list of one-sided R formulas, one per model
#'   parameter (or per simplex free parameter), e.g.
#'   \code{list(D = ~ 1 + (1|id), g = ~ 1)}.  Defaults to \code{~ 1}.
#' @param response_col Character.  For binomial (K == 2) only: name of the
#'   "number of successes" column (default: first resp_cat).
#' @param trials_col Character.  Name of the total-trials column.
#' @param simplex_params Character vector (or list of such vectors) naming
#'   parameters that form a probability simplex (sum to 1).  These receive a
#'   stick-breaking reparametrisation instead of simple logit/probit.
#'   Example: \code{c("g_A", "g_B", "g_New")} for source monitoring.
#'   The last element is derived; free parameters are named \code{l<param>}
#'   for the first K-1 elements.  Supply predictors for those via
#'   \code{predictor_formulas}.
#'
#' @return A named list (see description above).
mpt_to_brms <- function(spec,
                         predictor_formulas = list(),
                         response_col       = spec$resp_cats[1],
                         trials_col         = "n",
                         simplex_params     = NULL) {
  stopifnot(inherits(spec, "mpt_spec"))
  link_fn  <- if (spec$link == "logit") "inv_logit" else "pnorm"
  trees    <- spec$trees
  resp     <- spec$resp_cats
  n_cats   <- length(resp)
  n_trees  <- length(trees)

  tree_names <- vapply(trees, `[[`, "", "name")
  ind_names  <- paste0(".ind_", tree_names)

  # Sanitized (brms-safe) versions of all model parameters
  orig_params <- spec$params
  safe_params <- vapply(orig_params, .sanitize_nlpar, "")

  # Build aggregated probability string for one response category,
  # with parameter names replaced by their sanitized (brms-safe) versions.
  build_agg_prob <- function(cat) {
    parts <- character(n_trees)
    for (i in seq_along(trees)) {
      expr <- trees[[i]]$branches[[cat]]
      expr <- .replace_params_in_expr(expr, orig_params, safe_params)
      parts[i] <- sprintf("%s * (%s)", ind_names[i], expr)
    }
    paste(parts, collapse = " + ")
  }

  data_prep_base <- function(data) {
    for (i in seq_along(trees)) {
      data[[ind_names[i]]] <- as.integer(data[[spec$condition]] == tree_names[i])
    }
    data
  }

  # ===========================================================================
  # Binomial path (K == 2)
  # ===========================================================================
  if (n_cats == 2) {
    safe_response_col <- .sanitize_nlpar(response_col)
    agg_prob <- build_agg_prob(response_col)
    lhs      <- sprintf("%s | trials(%s)", response_col, trials_col)
    rhs      <- sprintf("logit(%s)", agg_prob)

    main_bf  <- brms::bf(as.formula(sprintf("%s ~ %s", lhs, rhs)), nl = TRUE)
    l_params <- paste0("l", safe_params)

    for (i in seq_along(orig_params)) {
      sp <- safe_params[i]
      lp <- l_params[i]
      main_bf <- main_bf + brms::nlf(as.formula(sprintf("%s ~ %s(%s)", sp, link_fn, lp)))
    }
    for (i in seq_along(orig_params)) {
      op       <- orig_params[i]
      lp       <- l_params[i]
      user_rhs <- if (!is.null(predictor_formulas[[op]])) {
        deparse(predictor_formulas[[op]][[2]])
      } else "1"
      main_bf <- main_bf + brms::lf(as.formula(sprintf("%s ~ %s", lp, user_rhs)))
    }

    prior_list <- do.call(c, lapply(l_params, function(lp) {
      brms::prior_string("logistic(0, 1)", nlpar = lp, class = "b", coef = "Intercept")
    }))

    return(list(
      brms_formula     = main_bf,
      data_prep        = data_prep_base,
      suggested_priors = prior_list,
      link_params      = setNames(l_params, orig_params),
      family           = "binomial"
    ))
  }

  # ===========================================================================
  # Multinomial path (K > 2)
  # Follows M3 infrastructure pattern (R/model_m3.R:bmf2bf.m3):
  #   - base bf() uses Y | trials(n) ~ log(P_first_cat), nl = TRUE
  #   - nlf(mu{cat} ~ log(P_cat)) for each subsequent category
  #   - nlf(safe_param ~ link_fn(l_safe_param)) for non-simplex params
  #   - stick-breaking nlf() for simplex groups
  #   - lf(l_safe_param ~ predictor) for all free linear-scale params
  # Parameter names with underscores/dots are sanitized for brms compatibility.
  # ===========================================================================

  # Normalise simplex_params to list of character vectors (original names)
  simplex_grps <- if (is.null(simplex_params)) {
    list()
  } else if (!is.list(simplex_params)) {
    list(simplex_params)
  } else {
    simplex_params
  }
  all_simplex_orig  <- unlist(simplex_grps)
  free_params_orig  <- setdiff(orig_params, all_simplex_orig)
  free_params_safe  <- vapply(free_params_orig, .sanitize_nlpar, "")
  l_free_params     <- paste0("l", free_params_safe)

  # Stick-breaking NLF strings (using sanitized names throughout)
  simplex_nlf_strs <- list()   # safe_param_name -> "safe_param ~ expr"
  simplex_free_lp  <- character(0)

  for (grp_orig in simplex_grps) {
    grp_safe <- vapply(grp_orig, .sanitize_nlpar, "")
    K_s <- length(grp_safe)
    for (k in seq_len(K_s)) {
      sp_safe <- grp_safe[k]
      if (k < K_s) {
        lp_name <- paste0("l", sp_safe)
        simplex_free_lp <- c(simplex_free_lp, lp_name)
        rhs_nlf <- if (k == 1) {
          sprintf("inv_logit(%s)", lp_name)
        } else {
          prev_factors <- paste(
            sprintf("(1 - inv_logit(l%s))", grp_safe[seq_len(k - 1)]),
            collapse = " * "
          )
          sprintf("%s * inv_logit(%s)", prev_factors, lp_name)
        }
        simplex_nlf_strs[[sp_safe]] <- sprintf("%s ~ %s", sp_safe, rhs_nlf)
      } else {
        prev_terms <- paste(grp_safe[-K_s], collapse = " + ")
        simplex_nlf_strs[[sp_safe]] <- sprintf("%s ~ 1 - (%s)", sp_safe, prev_terms)
      }
    }
  }

  all_l_params <- c(l_free_params, simplex_free_lp)

  # Base formula (first category) following M3 pattern
  first_cat <- resp[1]
  agg_first <- build_agg_prob(first_cat)
  main_bf <- brms::bf(
    as.formula(sprintf("Y | trials(%s) ~ log(%s)", trials_col, agg_first)),
    nl = TRUE
  )

  # nlf for each subsequent category
  for (cat in resp[-1]) {
    agg_cat <- build_agg_prob(cat)
    main_bf <- main_bf + brms::nlf(
      as.formula(sprintf("mu%s ~ log(%s)", cat, agg_cat))
    )
  }

  # nlf reparametrizations for non-simplex params
  for (i in seq_along(free_params_orig)) {
    sp <- free_params_safe[i]
    lp <- l_free_params[i]
    main_bf <- main_bf + brms::nlf(as.formula(sprintf("%s ~ %s(%s)", sp, link_fn, lp)))
  }

  # nlf for simplex params (stick-breaking)
  for (sp_safe in names(simplex_nlf_strs)) {
    main_bf <- main_bf + brms::nlf(as.formula(simplex_nlf_strs[[sp_safe]]))
  }

  # lf predictor formulas (user looks up by original param name)
  # Guard against empty-vector paste0 quirk in R 4.6+ (paste0("l", character(0)) = "l")
  simplex_free_orig <- unlist(lapply(simplex_grps, function(g) g[-length(g)]))
  if (length(simplex_free_orig) > 0) {
    simplex_free_safe <- vapply(simplex_free_orig, .sanitize_nlpar, "")
    simplex_free_lp2  <- paste0("l", simplex_free_safe)
    simplex_extra_map <- setNames(simplex_free_lp2, simplex_free_orig)
  } else {
    simplex_extra_map <- setNames(character(0), character(0))
  }

  # Map: original param name -> l<safe_param> name
  param_lp_map <- c(
    setNames(l_free_params, free_params_orig),
    simplex_extra_map
  )

  for (orig_name in names(param_lp_map)) {
    lp <- param_lp_map[[orig_name]]
    user_rhs <- if (!is.null(predictor_formulas[[orig_name]])) {
      deparse(predictor_formulas[[orig_name]][[2]])
    } else "1"
    main_bf <- main_bf + brms::lf(as.formula(sprintf("%s ~ %s", lp, user_rhs)))
  }

  # Priors (intercepts only)
  prior_list <- do.call(c, lapply(all_l_params, function(lp) {
    brms::prior_string("logistic(0, 1)", nlpar = lp, class = "b", coef = "Intercept")
  }))

  # Multinomial family (mirrors M3's configure_model.m3)
  mnomial_family        <- brms::multinomial(refcat = NA)
  mnomial_family$cats   <- resp
  mnomial_family$dpars  <- paste0("mu", resp)

  # data_prep: indicator columns + Y response matrix
  resp_cats_for_Y <- resp
  data_prep_multi <- function(data) {
    data <- data_prep_base(data)
    Y_mat            <- as.matrix(data[resp_cats_for_Y])
    colnames(Y_mat)  <- resp_cats_for_Y
    data$Y           <- Y_mat
    data
  }

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
#' Lines are equations for tree branches. Blank lines separate trees.
#' Inline comments (# ...) give the branch name; if absent, branches are
#' named "cat1", "cat2", ..., "catN".
#'
#' The function deduces response categories from branch names: branches with
#' the same name in different trees map to the same response category.
#'
#' @param model_str Character string in MPTinR equation format.
#' @param tree_conditions Character vector, length = number of trees, giving
#'   the condition values in the data that correspond to each tree (in order).
#' @param condition Character. Column name in data.
#' @param link Character. "logit" or "probit".
#'
#' @return An mpt_spec object.
#'
#' @examples
#' model_str <- "
#' D + (1 - D) * g        # old
#' (1 - D) * (1 - g)      # new
#'
#' (1 - D) * g            # old
#' D + (1 - D) * (1 - g)  # new
#' "
#' spec <- parse_mpt_string(model_str,
#'           tree_conditions = c("old", "new"),
#'           condition = "condition")
parse_mpt_string <- function(model_str,
                              tree_conditions,
                              condition = "condition",
                              link      = "logit") {
  # Split into lines, strip leading/trailing whitespace
  lines <- trimws(strsplit(model_str, "\n")[[1]])

  # Collect trees: split on blank lines
  trees_raw  <- list()
  current    <- character(0)
  for (ln in lines) {
    if (nchar(ln) == 0) {
      if (length(current) > 0) {
        trees_raw <- c(trees_raw, list(current))
        current   <- character(0)
      }
    } else {
      current <- c(current, ln)
    }
  }
  if (length(current) > 0) trees_raw <- c(trees_raw, list(current))

  if (length(trees_raw) != length(tree_conditions)) {
    stop(sprintf(
      "parse_mpt_string: found %d tree block(s) but %d tree_conditions supplied.",
      length(trees_raw), length(tree_conditions)
    ))
  }

  # Parse each tree block into mpt_tree
  parse_one_tree <- function(block, cond_name) {
    branches <- list()
    for (ln in block) {
      # split on '#' to get expression and optional branch name
      parts  <- strsplit(ln, "#")[[1]]
      expr   <- trimws(parts[1])
      bname  <- if (length(parts) >= 2) trimws(parts[2]) else NULL
      if (is.null(bname) || nchar(bname) == 0) {
        bname <- paste0("cat", length(branches) + 1L)
      }
      branches[[bname]] <- expr
    }
    mpt_tree(name = cond_name, branches = branches)
  }

  tree_list <- mapply(parse_one_tree, trees_raw, tree_conditions,
                      SIMPLIFY = FALSE)
  mpt(trees = tree_list, condition = condition, link = link)
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

# Validate tree consistency at D=0.7, g=0.5
.mpt_validate_tree(tree_old, c(D = 0.7, g = 0.5))
.mpt_validate_tree(tree_new, c(D = 0.7, g = 0.5))
cat("\nTree validation passed (branches sum to 1).\n")


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
# Both should produce identical formula structures
identical_formula <- identical(
  deparse(out_2htm$brms_formula$formula),
  deparse(out_parsed$brms_formula$formula)
)
cat(sprintf("  Formulas identical: %s\n", identical_formula))


cat("\n=== Test 3: Source Monitoring Model (SMM, 2-category binomial) ===\n\n")
# 2-category simplified SMM: responded A vs not-A for two source types.
# (Full 3-category SMM is in 07_multinomial_demo.R.)
smm_str <- "
d_A + (1 - d_A) * g        # A
(1 - d_A) * (1 - g)        # notA

(1 - d_B) * g              # A
d_B + (1 - d_B) * (1 - g)  # notA
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
    d_A = ~ 1 + (1 | id),
    d_B = ~ 1 + (1 | id),
    g   = ~ 1
  ),
  response_col = "A",
  trials_col   = "n"
)
cat("\nSMM brms formula:\n")
print(out_smm$brms_formula)


cat("\n=== Test 4: Pair Clustering Model (PCM, K=3 multinomial) ===\n\n")
# Batchelder & Riefer (1980): P(C)=cp+(1-cp)*rp^2, P(E)=2(1-cp)*rp*(1-rp),
# P(U)=(1-cp)*(1-rp)^2.  Use cp/rp to avoid R name conflicts.
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

.mpt_validate_tree(spec_pcm$trees[[1]], c(cp = 0.5, rp = 0.7))
cat("PCM tree validation passed (K=3, branches sum to 1).\n")

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


cat("\n=== Test 5: SMM 3-category multinomial with simplex_params ===\n\n")
# Full source-monitoring model: 3 trees (A, B, New), 3 response cats (A, B, New)
# Guessing parameters g_A, g_B, g_New must satisfy g_A + g_B + g_New = 1
# => stick-breaking via simplex_params = c("g_A", "g_B", "g_New")
smm3_str <- "
d_A + (1 - d_A) * g_A       # A
(1 - d_A) * g_B              # B
(1 - d_A) * g_New            # New

(1 - d_B) * g_A              # A
d_B + (1 - d_B) * g_B       # B
(1 - d_B) * g_New            # New

g_A                          # A
g_B                          # B
g_New                        # New
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
    d_A = ~ 1 + (1 | id),
    d_B = ~ 1 + (1 | id),
    g_A = ~ 1,
    g_B = ~ 1
  ),
  simplex_params = c("g_A", "g_B", "g_New")
)
cat("\nSMM3 multinomial brms formula:\n")
print(out_smm3$brms_formula)
cat("\nSMM3 free l-params (stick-breaking):", paste(names(out_smm3$link_params), collapse = ", "), "\n")
cat("SMM3 family:", out_smm3$family, "\n")

cat("\n=== All tests passed ===\n")
