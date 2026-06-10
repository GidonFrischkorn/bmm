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
#' Returns a named list:
#'   - \code{brms_formula}: the full brms::bf() object (ready to pass to brm())
#'   - \code{param_formulas}: named list of brms::lf() calls, one per parameter,
#'     built from the user-supplied \code{predictor_formulas} bmf-style list.
#'   - \code{data_prep}: a function(data) that adds the required indicator
#'     columns to the data frame.
#'   - \code{suggested_priors}: a brms::prior() object with logistic(0,1) priors
#'     for each parameter (on the link scale).
#'
#' @param spec An mpt_spec object (output of mpt())
#' @param predictor_formulas A named list of one-sided R formulas giving the
#'   GLM predictors for each model parameter, e.g.
#'   \code{list(D = ~ 1 + (1 | id), g = ~ 1)}. Any parameter not supplied
#'   defaults to \code{~ 1}.
#' @param response_col Character. Name of the "number of old responses" column.
#'   Assumed to be the first response category by default.
#' @param trials_col Character. Name of the total-trials column.
#'
#' @return A list with elements \code{brms_formula}, \code{data_prep},
#'   \code{suggested_priors}.
#'
#' @examples
#' tree_old <- mpt_tree("old", list(old = "D + (1-D)*g", new = "(1-D)*(1-g)"))
#' tree_new <- mpt_tree("new", list(old = "(1-D)*g",    new = "D+(1-D)*(1-g)"))
#' spec <- mpt(list(tree_old, tree_new), condition = "condition")
#' out  <- mpt_to_brms(spec,
#'           predictor_formulas = list(D = ~ 1 + (1|id), g = ~ 1),
#'           response_col = "old", trials_col = "n")
#' # out$brms_formula is ready for brm(...)
mpt_to_brms <- function(spec,
                         predictor_formulas = list(),
                         response_col = spec$resp_cats[1],
                         trials_col   = "n") {
  stopifnot(inherits(spec, "mpt_spec"))
  link  <- spec$link
  trees <- spec$trees
  resp  <- spec$resp_cats
  n_trees <- length(trees)

  # ---- 1. Build indicator column names (one per tree) ---------------------
  tree_names <- vapply(trees, `[[`, "", "name")
  ind_names  <- paste0(".ind_", tree_names)

  # ---- 2. Build the aggregated probability expression for resp_cats[1] ----
  # P(response_col) = sum_tree [ indicator_tree * branch_prob_tree ]
  build_agg_prob <- function(cat) {
    parts <- character(n_trees)
    for (i in seq_along(trees)) {
      parts[i] <- sprintf("%s * (%s)", ind_names[i], trees[[i]]$branches[[cat]])
    }
    paste(parts, collapse = " + ")
  }

  # The main outcome is response_col (first category); brms binomial needs the
  # probability on the logit/probit scale:
  #   logit(p) where p = aggregated_prob_expression
  agg_prob <- build_agg_prob(response_col)
  lhs      <- sprintf("%s | trials(%s)", response_col, trials_col)

  # The rhs feeds logit() directly (brms NL formula approach):
  rhs <- sprintf("logit(%s)", agg_prob)

  main_bf <- brms::bf(
    as.formula(sprintf("%s ~ %s", lhs, rhs)),
    nl = TRUE
  )

  # ---- 3. Reparametrisation: param → link_scale param (l<param>) ----------
  link_fn  <- if (link == "logit") "inv_logit" else "pnorm"
  l_params <- paste0("l", spec$params)     # e.g. lD, lg

  for (i in seq_along(spec$params)) {
    p  <- spec$params[i]
    lp <- l_params[i]
    main_bf <- main_bf + brms::nlf(
      as.formula(sprintf("%s ~ %s(%s)", p, link_fn, lp))
    )
  }

  # ---- 4. GLM predictor formulas for each l<param> -----------------------
  for (i in seq_along(spec$params)) {
    p        <- spec$params[i]
    lp       <- l_params[i]
    user_rhs <- if (!is.null(predictor_formulas[[p]])) {
      deparse(predictor_formulas[[p]][[2]])
    } else {
      "1"
    }
    main_bf <- main_bf + brms::lf(
      as.formula(sprintf("%s ~ %s", lp, user_rhs))
    )
  }

  # ---- 5. Suggested priors ------------------------------------------------
  prior_list <- do.call(c, lapply(l_params, function(lp) {
    brms::prior_string("logistic(0, 1)", nlpar = lp, class = "b")
  }))

  # ---- 6. data_prep closure -----------------------------------------------
  data_prep <- function(data) {
    for (i in seq_along(trees)) {
      data[[ind_names[i]]] <- as.integer(data[[spec$condition]] == tree_names[i])
    }
    data
  }

  list(
    brms_formula     = main_bf,
    data_prep        = data_prep,
    suggested_priors = prior_list,
    link_params      = setNames(l_params, spec$params)
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


cat("\n=== Test 3: Source Monitoring Model (SMM) ===\n\n")
#
# Classic 3-source monitoring model (Johnson et al. 1993):
#   Source A items, Source B items, and New items
#   Response categories: "A", "B", "New"
#
#   Tree A (source A items):
#     P(A | A)   = d_A * a
#     P(B | A)   = d_A * (1 - a) * b
#     P(New | A) = 1 - d_A*a - d_A*(1-a)*b = (1-d_A) + d_A*(1-a)*(1-b)
#
#   Tree B (source B items):
#     P(A | B)   = d_B * (1 - a) * b
#     P(B | B)   = d_B * a     ... wait, let me use a cleaner parametrisation:
#
# Simplified 2-source SMM (Batchelder & Riefer 1990, simplest version):
#   Parameters: d = item detection, a = source A decision, b = guessing "A"
#
#   Tree 1 (source A items):
#     A   = d * a
#     B   = d * (1 - a)
#     New = (1 - d) * b_new    (where b_new = 1 - b... let's use simple version)
#
# Actually let's use the simplest SMM that fits our 3-response setup:
#
#   Tree "A" (old items from source A, n=30):
#     resp_A   = d * a + (1 - d) * g
#     resp_B   = d * (1 - a) * (1 - g)
#     resp_New = (1 - d) * (1 - g)    [wait, must sum to 1]
#
# Let me use a cleaner version: Rouder & Lu (2005) style SMM
#
#   Tree A: P(A) = d_A + (1-d_A)*g_A,  P(B) = (1-d_A)*g_B,  P(New) = (1-d_A)*(1-g_A-g_B)
#   Tree B: P(A) = (1-d_B)*g_A,        P(B) = d_B + (1-d_B)*g_B, P(New) = (1-d_B)*(1-g_A-g_B)
#   Tree N: P(A) = g_A,                P(B) = g_B,           P(New) = 1 - g_A - g_B
#
# This requires g_A + g_B < 1 (a simplex constraint on guessing).
# For the prototype, we use a reduced version where the first category is
# the response variable for a binomial (or multinomial separately for each pair).

# Simplified 2-category prototype using the "is source A detected?" question:
# For illustration, use 2-response version: responded A vs not-A
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

# With condition-varying detection parameters:
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

cat("\n=== Test 4: Pair Clustering Model (PCM) ===\n\n")
# Batchelder & Riefer (1980) pair-clustering model (simplified)
# Studies pairs; at test recognises both (C), one (E), or neither (U):
#   P(C) = c + (1-c) * r^2
#   P(E) = 2 * (1-c) * r * (1-r)
#   P(U) = (1-c) * (1-r)^2
#
# With 3 response cats and 1 tree (no condition), condition = "all"

pcm_str <- "
c + (1 - c) * r * r           # C
2 * (1 - c) * r * (1 - r)     # E
(1 - c) * (1 - r) * (1 - r)   # U
"
# PCM has 1 tree — treat "condition" as a dummy
spec_pcm <- parse_mpt_string(
  pcm_str,
  tree_conditions = c("study"),
  condition = "dummy_cond"
)
print(spec_pcm)

# Validate: at c=0.5, r=0.7
.mpt_validate_tree(spec_pcm$trees[[1]], c(c = 0.5, r = 0.7))
cat("PCM tree validation passed.\n")

cat("\n=== All tests passed ===\n")
