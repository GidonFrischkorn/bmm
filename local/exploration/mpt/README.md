# MPT Model Prototype (Exploration — Issue #8)

Prototype of Multinomial Processing Tree (MPT) models on bmm's M3/multinomial
infrastructure. Addresses [issue #8](https://github.com/GidonFrischkorn/bmm/issues/8)
and proposes a concrete design for
[venpopov/bmm#255](https://github.com/venpopov/bmm/issues/255).

## Files

| File | Description |
|------|-------------|
| `01_reproduce_upstream_demo.R` | Reproduce the upstream 2HTM raw-brms demo; cross-validate with MPTinR (if installed) |
| `02_syntax_comparison.R`       | Evaluate 3 candidate user-facing syntaxes; recommendation with rationale |
| `03_mpt_parser.R`              | Pure-R parser prototype: `mpt_tree()`, `mpt()`, `mpt_to_brms()`, `parse_mpt_string()` |
| `04_hierarchical_demo.R`       | Hierarchical 2HTM fit using the parser output; parameter recovery checks |
| `05_design_document.md`        | Full design document: API, infrastructure reuse, implementation plan |
| `smoke_test.R`                 | Automated tests for the parser (no Stan/brm() calls; runs fast) |

## Running

```r
# Fast tests (no MCMC — runs in seconds):
Rscript local/exploration/mpt/smoke_test.R

# Full upstream demo (fits brms models — requires Stan):
Rscript local/exploration/mpt/01_reproduce_upstream_demo.R

# Hierarchical demo (fits brms models):
Rscript local/exploration/mpt/04_hierarchical_demo.R
```

## Key Findings

1. **Structural analogy confirmed**: MPT models fit naturally into the M3
   infrastructure. The index-variable trick (already in M3) generalises
   directly to multi-tree MPT condition handling.

2. **Recommended syntax**: `mpt_tree()` + `mpt()` (tree/branch list, Syntax C)
   as the primary API; `mpt_from_string()` (MPTinR string parser, Syntax A)
   as a convenience import wrapper.

3. **Formula generation works**: `mpt_to_brms()` correctly emits the
   `logit(Σ_t ind_t * P_t(response))` formula, `nlf()` reparametrisations,
   and `lf()` predictor formulas. Verified against the upstream brms demo.

4. **Hierarchical extension is straightforward**: random effects are added via
   standard `bmf()` syntax on the logit scale.

## Dependencies (for full demo scripts)

- `brms` (≥ 2.21)
- `dplyr`, `tidyr`
- `MPTinR` (optional, for cross-validation in `01_reproduce_upstream_demo.R`)
- `bayesplot` (optional, for pp_check in `04_hierarchical_demo.R`)
