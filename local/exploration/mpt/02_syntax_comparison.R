# MPT syntax design: compare 3 candidate syntaxes
# Evaluates each against: (a) parameters varying by condition,
# (b) multiple trees, (c) equality constraints (Do = Dn), and
# (d) hierarchical extensions via probit/logit reparametration.

# =============================================================================
# BACKGROUND: The 2HTM example
# =============================================================================
# Model equations per tree (MPTinR style):
#
#   Tree 1  (old items):
#     branch old_hit  = D + (1 - D) * g
#     branch old_miss = (1 - D) * (1 - g)
#
#   Tree 2  (new items):
#     branch new_fa   = (1 - D) * g
#     branch new_cr   = D + (1 - D) * (1 - g)
#
# Response categories (leaves shared across trees):
#     "old"  ← old_hit (tree1) + new_fa (tree2)
#     "new"  ← old_miss(tree1) + new_cr(tree2)
#
# Challenge for bmm: merge tree-specific branches into response-category
# probabilities that depend on a condition indicator.

# =============================================================================
# SYNTAX A: MPTinR-string → parser → bmm
# =============================================================================
#
# The user provides two pieces:
#   1. An MPTinR-compatible model string.
#   2. A named list mapping tree names to data condition values.
#
# Example:
#
#   model_str <- "
#   D + (1 - D) * g        # old_hit
#   (1 - D) * (1 - g)      # old_miss
#
#   (1 - D) * g            # new_fa
#   D + (1 - D) * (1 - g)  # new_cr
#   "
#
#   mpt_model <- mpt(
#     model    = model_str,
#     resp_map = list(old = c("old_hit", "new_fa"),
#                     new = c("old_miss", "new_cr")),
#     tree_map = list(tree1 = "old", tree2 = "new"),  # maps to condition column values
#     trees    = "condition"
#   )
#
# Pro:
#   + Most familiar to MPT researchers (re-uses existing knowledge/files)
#   + Parser auto-generates all brms nlf() formulas
#   + Equality constraints (D = Dn) handled by string substitution before parsing
# Con:
#   - String parsing is fragile (parentheses, operator precedence)
#   - Condition-varying parameters require extra annotation (not in string)
#   - User must understand resp_map abstraction

# =============================================================================
# SYNTAX B: Explicit bmm-formula with probability-scale parameters
# =============================================================================
#
# Closest to what the M3 uses. The user writes one non-linear formula
# per response category, directly giving the aggregated probability expression.
# Parameters are probability-scale; bmm adds the logit reparametration
# automatically (analogous to M3's log/logit link).
#
# Example:
#
#   mpt_model <- mpt(
#     resp_cats = c("old", "new"),
#     condition = "is_old"         # name of the 0/1 indicator column in data
#   )
#
#   mpt_formula <- bmf(
#     old ~ is_old * (D + (1 - D) * g) + (1 - is_old) * (1 - D) * g,
#     new ~ is_old * (1 - D) * (1 - g) + (1 - is_old) * (D + (1 - D) * (1 - g)),
#     D   ~ 1 + condition + (1 + condition | id),
#     g   ~ 1 + (1 | id)
#   )
#
# Pro:
#   + Most transparent to brms-literate users
#   + Condition-varying parameters follow standard GLM formula syntax
#   + No parsing needed: expressions are R formulas
#   + Equality constraints simply omit a second parameter (D = Dn → use D once)
# Con:
#   - User must manually aggregate across trees (no helper)
#   - More boilerplate for complex models (many trees → long formulas)
#   - New users must understand index-variable trick

# =============================================================================
# SYNTAX C: Tree / branch list objects
# =============================================================================
#
# User defines each tree as a list of branch probabilities, then declares
# how branches map to observable response categories. bmm builds the
# aggregated formula automatically.
#
# Example:
#
#   tree_old <- mpt_tree(
#     name     = "old",                     # maps to condition == "old"
#     branches = list(
#       old = "D + (1 - D) * g",
#       new = "(1 - D) * (1 - g)"
#     )
#   )
#
#   tree_new <- mpt_tree(
#     name     = "new",
#     branches = list(
#       old = "(1 - D) * g",
#       new = "D + (1 - D) * (1 - g)"
#     )
#   )
#
#   mpt_model <- mpt(
#     trees     = list(tree_old, tree_new),
#     condition = "condition"               # column in data giving tree label
#   )
#
#   mpt_formula <- bmf(
#     D ~ 1 + (1 | id),
#     g ~ 1 + (1 | id)
#   )
#
# Pro:
#   + Explicit tree structure; easy to read large models
#   + bmm auto-generates the aggregated probability and index variables
#   + Natural extension to multinomial (>2 categories)
# Con:
#   - Most new infrastructure (mpt_tree objects, constructor methods)
#   - Condition-varying parameters still need separate bmf() lines
#   - Furthest from MPTinR (no familiar string syntax)

# =============================================================================
# EVALUATION MATRIX
# =============================================================================
#
#                              | Syntax A | Syntax B | Syntax C |
# -----------------------------|----------|----------|----------|
# Params vary by condition     |  medium  |  easy    |  easy    |
# Multiple trees               |  easy    |  medium  |  easy    |
# Equality constraints (D=Dn)  |  easy    |  easy    |  easy    |
# Probit/logit hierarchical    |  auto    |  auto    |  auto    |
# Learning curve for MPT users |  low     |  high    |  medium  |
# Learning curve for bmm users |  high    |  low     |  medium  |
# Implementation complexity    |  high    |  low     |  medium  |
# Robustness (complex models)  |  medium  |  low     |  high    |
#
# RECOMMENDATION: Syntax C (tree/branch list) as the primary user-facing API,
# with Syntax A (MPTinR string parser) as a convenience import wrapper.
#
# Rationale:
# - Syntax C aligns with how psychologists think about MPT models (trees)
# - bmm auto-generates the index-variable trick, hiding the brms complexity
# - Syntax A allows researchers to import existing MPTinR model files unchanged
# - Syntax B can serve as the intermediate representation (what A and C emit)
#   and gives power users a path to hand-roll formulas

cat("Syntax comparison complete. See comments in this file for the evaluation.\n")
cat("Recommended approach: Syntax C as primary API + Syntax A as import wrapper.\n")
