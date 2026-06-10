# Smoke test: no brm() calls — just test the pure-R parser functions
lines <- readLines("local/exploration/mpt/03_mpt_parser.R")

# Find where the tests start (to avoid sourcing brm() calls)
test_start <- grep("=== Test 1", lines)[1]

# Source just the function definitions
tmp <- tempfile(fileext = ".R")
writeLines(lines[seq_len(test_start - 1L)], tmp)
source(tmp)

# --- Test 1: mpt_tree + mpt ---
tree_old <- mpt_tree("old", list(
  old = "D + (1 - D) * g",
  new = "(1 - D) * (1 - g)"
))
tree_new <- mpt_tree("new", list(
  old = "(1 - D) * g",
  new = "D + (1 - D) * (1 - g)"
))
spec <- mpt(list(tree_old, tree_new), condition = "condition")
cat("mpt_spec created:\n")
print(spec)

# Validate tree branch sums
.mpt_validate_tree(tree_old, c(D = 0.7, g = 0.5))
.mpt_validate_tree(tree_new, c(D = 0.7, g = 0.5))
cat("Tree validation OK\n")

# --- Test 2: parse_mpt_string ---
model_str <- paste0(
  "\n",
  "D + (1 - D) * g        # old\n",
  "(1 - D) * (1 - g)      # new\n",
  "\n",
  "(1 - D) * g            # old\n",
  "D + (1 - D) * (1 - g)  # new\n"
)
spec2 <- parse_mpt_string(
  model_str,
  tree_conditions = c("old", "new"),
  condition = "condition"
)
cat("\nParsed spec:\n")
print(spec2)
cat("\nParams match:", identical(sort(spec$params), sort(spec2$params)), "\n")

# --- Test 3: SMM ---
smm_str <- paste0(
  "\n",
  "d_A + (1 - d_A) * g        # A\n",
  "(1 - d_A) * (1 - g)        # notA\n",
  "\n",
  "(1 - d_B) * g              # A\n",
  "d_B + (1 - d_B) * (1 - g)  # notA\n"
)
spec_smm <- parse_mpt_string(smm_str,
  tree_conditions = c("sourceA", "sourceB"),
  condition = "source"
)
cat("\nSMM spec:\n")
print(spec_smm)

# --- Test 4: PCM ---
pcm_str <- paste0(
  "\n",
  "c + (1 - c) * r * r           # C\n",
  "2 * (1 - c) * r * (1 - r)     # E\n",
  "(1 - c) * (1 - r) * (1 - r)   # U\n"
)
spec_pcm <- parse_mpt_string(pcm_str,
  tree_conditions = c("study"),
  condition = "dummy"
)
cat("\nPCM spec:\n")
print(spec_pcm)
.mpt_validate_tree(spec_pcm$trees[[1]], c(c = 0.5, r = 0.7))
cat("PCM tree validation OK\n")

# --- Test 5: mpt_to_brms without actually fitting ---
# Check that the formula structure is correct
out <- mpt_to_brms(
  spec,
  predictor_formulas = list(D = ~ 1, g = ~ 1),
  response_col = "old",
  trials_col   = "n"
)
cat("\nFormula emitted by mpt_to_brms():\n")
print(out$brms_formula)
cat("\nSuggested priors:\n")
print(out$suggested_priors)

# Verify data_prep adds indicator columns
fake_data <- data.frame(
  id        = 1:4,
  condition = c("old", "new", "old", "new"),
  old       = c(40, 10, 35, 8),
  n         = 50L
)
prepped <- out$data_prep(fake_data)
cat("\ndata_prep() added columns:", setdiff(names(prepped), names(fake_data)), "\n")
stopifnot(".ind_old" %in% names(prepped))
stopifnot(".ind_new" %in% names(prepped))
stopifnot(all(prepped$.ind_old == c(1L, 0L, 1L, 0L)))
stopifnot(all(prepped$.ind_new == c(0L, 1L, 0L, 1L)))

cat("\n=== All smoke tests passed ===\n")
