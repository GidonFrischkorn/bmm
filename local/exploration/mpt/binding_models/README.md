# Binding Models — Al Hadhrami, Bartsch, & Oberauer (2025)

Source: Al Hadhrami, A., Bartsch, L. M., & Oberauer, K. (2025). A multinomial
model-based analysis of bindings in working memory. *Psychological Review*.
OSF project: https://osf.io/nu4sb/

## Files

- `unitization.eqn` — Unitization model EQN (93 lines; 12 free parameters).
  Extracted from the TreeBUGS scripts in the OSF project. Design-fixed guessing
  constants (P_G1, P_g1, etc.) are substituted as decimal literals by the
  converter in `12_eqn_converter.R`.

- `hybrid.eqn` — Hybrid model EQN (126 lines; 18 free parameters).

- `exp1_response_frequencies.csv` — Per-participant response frequency counts
  from Experiment 1 (32 participants × 30 response-category columns; 50 trials
  per participant per cue-condition tree: WC = word cue, LC = location cue,
  CC = color cue).

## Trees and response categories

Both models have three cue-condition trees (WC, LC, CC) and 10 joint response
categories per tree (two-element reports: item × feature):

- Correct_Correct, Correct_Lure, Correct_Extra
- Lure_Correct, Lure_s_Lure_s, Lure_d_Lure_d, Lure_Extra
- Extra_Correct, Extra_Lure, Extra_Extra

## Usage

See `12_eqn_converter.R` for how to load these files into the prototype.
