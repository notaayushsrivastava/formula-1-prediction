# Phase 15: pooling the grid (B0) with B1 / M1 after qualifying — FAILED the gate

**Question (from Phase 14).** After qualifying the plain grid baseline B0 beats B1 and M1
on the winner. Does a blend of B0 with B1 (or M1) beat B0 alone?

**What ran.** Linear and log pools of B0 + B1 and B0 + M1 at the quali moment, three
values of the pull toward equal weights (l2 = 0.01, 0.05, 0.2): 12 configurations, all
logged as trials of the `pool` family. Walk-forward: weights fitted on earlier races
only (warm-up 2024, gate season 2025, `pool_forward`). CPCV: the four l2 = 0.05 pools as
`PooledModel`, weights from inner out-of-fold predictions of each split's training races.
Data: f1db-only dataset (see the note in `reports/validation.md`).

**Result.**

| Pool | WF win LL 2025 (B0: 0.0960) | 95% CI vs B0, win, 2025 | CPCV median win LL (B0: 0.1185) |
|---|---|---|---|
| POOL-linear B0+B1 | 0.1063 | +0.0104 [+0.0006, +0.0196] | 0.1144 (better on 7/7 paths) |
| POOL-log B0+B1 | 0.1057 | +0.0098 [+0.0003, +0.0191] | 0.1139 (7/7) |
| POOL-linear B0+M1 | 0.1060 | +0.0100 [+0.0007, +0.0188] | 0.1141 (7/7) |
| POOL-log B0+M1 | 0.1053 | +0.0094 [+0.0001, +0.0182] | 0.1138 (7/7) |

Deflated p = 0.999 for all four (12 trials), PBO of the 12 configurations = 0.10,
no Benjamini-Hochberg discovery among the 13 gates of the phase.

**Why it failed.** The two tests disagree, and the gate is right to refuse:
- Over 2018-2025 (CPCV) the pools beat the grid on every path, by about 0.004 win LL.
- On 2025 (the gate season, walk-forward) they are clearly *worse* than the grid. In
  2025 pole won 16 of 24 races (67%); B0 gave the pole-sitter 46% on average, B1 and M1
  only 21%. Mixing in a model that spreads the win chance dilutes the grid exactly when
  the front of the grid was most reliable.

**Why B1 / M1 lose on the winner (evidence).** Plackett-Luce uses one temperature for
every place. Measured out of sample (fitted on 2018-2023, scored on 2024-2025), the best
temperature is about 2.0 for 1st place, 1.5 for 2nd-3rd, 1.1-1.2 for 4th-10th and 0.2
for 11th onwards. The single fitted scale is a compromise pulled down by the noisy back
of the field, so the front is too flat: too little probability on the favourite.

**What a later phase could try** (one family per phase): a place-dependent temperature
(sharper first stage, e.g. a "stagewise" Plackett-Luce), or a direct winner model
(softmax over the grid with its own temperature) blended only for P_win. The gate must
be rerun on the full dataset (with OpenF1) before any conclusion is final.
