# pitwall backtest

Seasons tested: **2024, 2025**. Walk-forward: every race is predicted by models trained only on earlier races, retrained every 2 races. Generated 2026-11-25 (git 04e03e2).

- LL = binary log loss averaged over driver rows (comparable to the 0.075 / 0.191 targets). Winner LL = mean of -log P_win of the actual winner. Lower is better for LL, Brier, RPS and ECE; higher for Spearman, Winner hit and Podium overlap.
- Baseline: **B0 grid slot**. At the 'quali' moment it is the grid baseline.

## Moment: pre

| Model | Win LL | Podium LL | Points LL | DNF LL | Winner LL | Win Brier | RPS | Spearman | Winner hit | Podium overlap | Win ECE | Races |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 grid slot (baseline) | 0.1472 | 0.2783 | 0.5453 | 0.3290 | 2.0046 | 0.0433 | 0.1336 | 0.5674 | 0.2708 | 0.5417 | 0.0162 | 48 |
| B1 logistic | 0.1435 | 0.2778 | 0.5167 | 0.3503 | 1.9704 | 0.0406 | 0.1282 | 0.6053 | 0.3125 | 0.6181 | 0.0181 | 48 |
| B2 LightGBM | 0.1545 | 0.3070 | 0.5146 | 0.3342 | 2.1715 | 0.0423 | 0.1310 | 0.5944 | 0.2292 | 0.4792 | 0.0056 | 48 |
| M1 strength | 0.1448 | 0.2786 | 0.5056 | 0.3510 | 1.9923 | 0.0408 | 0.1266 | 0.6229 | 0.3750 | 0.6250 | 0.0202 | 48 |
| M1-pure strength | 0.1655 | 0.3274 | 0.5509 | 0.3503 | 2.3727 | 0.0438 | 0.1383 | 0.5877 | 0.2708 | 0.5625 | 0.0214 | 48 |
| M3 simulator | 0.1667 | 0.3320 | 0.5612 | 0.3439 | 2.3952 | 0.0439 | 0.1415 | 0.5535 | 0.2917 | 0.5556 | 0.0243 | 48 |
| M2-full | 0.1475 | 0.3004 | 0.5846 | 0.6216 | 2.0444 | 0.0410 | 0.1403 | 0.6108 | 0.3542 | 0.5556 | 0.0207 | 48 |
| Target: best open model | 0.0750 | 0.1910 | — | — | — | — | — | — | — | — | — | — |

Difference vs baseline (model minus baseline; negative = better), mean per race with a 95% paired-bootstrap interval:

| Model | Win LL | Podium LL |
|---|---|---|
| B1 logistic | -0.0036 [-0.0140, +0.0063] (no clear difference) | -0.0004 [-0.0109, +0.0097] (no clear difference) |
| B2 LightGBM | +0.0073 [-0.0062, +0.0203] (no clear difference) | +0.0287 [+0.0127, +0.0447] (worse) |
| M1 strength | -0.0024 [-0.0143, +0.0090] (no clear difference) | +0.0004 [-0.0123, +0.0128] (no clear difference) |
| M1-pure strength | +0.0183 [+0.0021, +0.0352] (worse) | +0.0491 [+0.0276, +0.0715] (worse) |
| M3 simulator | +0.0195 [+0.0037, +0.0360] (worse) | +0.0536 [+0.0312, +0.0761] (worse) |
| M2-full | +0.0003 [-0.0115, +0.0119] (no clear difference) | +0.0221 [+0.0047, +0.0382] (worse) |

- B0 grid slot: win LL 0.1472 does NOT beat the 0.075 target; podium LL 0.2783 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1435 does NOT beat the 0.075 target; podium LL 0.2778 does NOT beat the 0.191 target.
- B2 LightGBM: win LL 0.1545 does NOT beat the 0.075 target; podium LL 0.3070 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1448 does NOT beat the 0.075 target; podium LL 0.2786 does NOT beat the 0.191 target.
- M1-pure strength: win LL 0.1655 does NOT beat the 0.075 target; podium LL 0.3274 does NOT beat the 0.191 target.
- M3 simulator: win LL 0.1667 does NOT beat the 0.075 target; podium LL 0.3320 does NOT beat the 0.191 target.
- M2-full: win LL 0.1475 does NOT beat the 0.075 target; podium LL 0.3004 does NOT beat the 0.191 target.

## Moment: practice

| Model | Win LL | Podium LL | Points LL | DNF LL | Winner LL | Win Brier | RPS | Spearman | Winner hit | Podium overlap | Win ECE | Races |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 grid slot (baseline) | 0.2059 | 0.4314 | 0.6913 | 0.3270 | 3.1183 | 0.0492 | 0.1742 | 0.2204 | 0.1250 | 0.2500 | 0.0218 | 48 |
| B1 logistic | 0.1438 | 0.2776 | 0.5172 | 0.3478 | 1.9759 | 0.0407 | 0.1287 | 0.5952 | 0.3125 | 0.6111 | 0.0161 | 48 |
| B2 LightGBM | 0.1531 | 0.3072 | 0.5148 | 0.3318 | 2.1455 | 0.0420 | 0.1309 | 0.5926 | 0.2292 | 0.4931 | 0.0109 | 48 |
| M1 strength | 0.1452 | 0.2788 | 0.5064 | 0.3489 | 2.0001 | 0.0409 | 0.1270 | 0.6153 | 0.3750 | 0.6111 | 0.0212 | 48 |
| M1-pure strength | 0.1659 | 0.3277 | 0.5505 | 0.3478 | 2.3791 | 0.0439 | 0.1386 | 0.5835 | 0.2708 | 0.5486 | 0.0217 | 48 |
| M3 simulator | 0.1665 | 0.3301 | 0.5579 | 0.3439 | 2.3911 | 0.0439 | 0.1406 | 0.5604 | 0.2917 | 0.5625 | 0.0243 | 48 |
| M2-full | 0.1558 | 0.3212 | 0.6002 | 0.5722 | 2.1984 | 0.0421 | 0.1448 | 0.5810 | 0.3542 | 0.5347 | 0.0264 | 48 |
| Target: best open model | 0.0750 | 0.1910 | — | — | — | — | — | — | — | — | — | — |

Difference vs baseline (model minus baseline; negative = better), mean per race with a 95% paired-bootstrap interval:

| Model | Win LL | Podium LL |
|---|---|---|
| B1 logistic | -0.0621 [-0.0816, -0.0414] (better) | -0.1538 [-0.1850, -0.1223] (better) |
| B2 LightGBM | -0.0528 [-0.0732, -0.0316] (better) | -0.1242 [-0.1557, -0.0930] (better) |
| M1 strength | -0.0607 [-0.0794, -0.0411] (better) | -0.1527 [-0.1806, -0.1242] (better) |
| M1-pure strength | -0.0401 [-0.0567, -0.0230] (better) | -0.1039 [-0.1295, -0.0787] (better) |
| M3 simulator | -0.0394 [-0.0559, -0.0225] (better) | -0.1015 [-0.1266, -0.0768] (better) |
| M2-full | -0.0501 [-0.0666, -0.0332] (better) | -0.1103 [-0.1332, -0.0871] (better) |

- B0 grid slot: win LL 0.2059 does NOT beat the 0.075 target; podium LL 0.4314 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1438 does NOT beat the 0.075 target; podium LL 0.2776 does NOT beat the 0.191 target.
- B2 LightGBM: win LL 0.1531 does NOT beat the 0.075 target; podium LL 0.3072 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1452 does NOT beat the 0.075 target; podium LL 0.2788 does NOT beat the 0.191 target.
- M1-pure strength: win LL 0.1659 does NOT beat the 0.075 target; podium LL 0.3277 does NOT beat the 0.191 target.
- M3 simulator: win LL 0.1665 does NOT beat the 0.075 target; podium LL 0.3301 does NOT beat the 0.191 target.
- M2-full: win LL 0.1558 does NOT beat the 0.075 target; podium LL 0.3212 does NOT beat the 0.191 target.

## Moment: quali

| Model | Win LL | Podium LL | Points LL | DNF LL | Winner LL | Win Brier | RPS | Spearman | Winner hit | Podium overlap | Win ECE | Races |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 grid slot (baseline) | 0.1111 | 0.2239 | 0.4809 | 0.3227 | 1.4418 | 0.0317 | 0.1149 | 0.6834 | 0.5625 | 0.7083 | 0.0132 | 48 |
| B1 logistic | 0.1294 | 0.2422 | 0.4475 | 0.5435 | 1.7235 | 0.0377 | 0.1124 | 0.7002 | 0.4167 | 0.6944 | 0.0279 | 48 |
| B2 LightGBM | 0.1277 | 0.2659 | 0.4508 | 0.3249 | 1.6995 | 0.0369 | 0.1155 | 0.6995 | 0.5000 | 0.6042 | 0.0244 | 48 |
| M1 strength | 0.1311 | 0.2445 | 0.4470 | 0.5438 | 1.7515 | 0.0381 | 0.1126 | 0.7027 | 0.4375 | 0.7014 | 0.0304 | 48 |
| M1-pure strength | 0.1655 | 0.3266 | 0.5459 | 0.5435 | 2.3721 | 0.0438 | 0.1371 | 0.5905 | 0.2708 | 0.5486 | 0.0213 | 48 |
| M3 simulator | 0.1377 | 0.2617 | 0.4717 | 0.3439 | 1.8691 | 0.0393 | 0.1179 | 0.6858 | 0.4167 | 0.7014 | 0.0332 | 48 |
| M2-full | 0.1451 | 0.3001 | 0.5694 | 0.5661 | 2.0065 | 0.0402 | 0.1358 | 0.6517 | 0.3958 | 0.6319 | 0.0379 | 48 |
| Target: best open model | 0.0750 | 0.1910 | — | — | — | — | — | — | — | — | — | — |

Difference vs baseline (model minus baseline; negative = better), mean per race with a 95% paired-bootstrap interval:

| Model | Win LL | Podium LL |
|---|---|---|
| B1 logistic | +0.0184 [+0.0038, +0.0323] (worse) | +0.0184 [+0.0030, +0.0340] (worse) |
| B2 LightGBM | +0.0166 [+0.0048, +0.0285] (worse) | +0.0422 [+0.0236, +0.0609] (worse) |
| M1 strength | +0.0200 [+0.0053, +0.0339] (worse) | +0.0207 [+0.0051, +0.0360] (worse) |
| M1-pure strength | +0.0544 [+0.0359, +0.0717] (worse) | +0.1028 [+0.0769, +0.1276] (worse) |
| M3 simulator | +0.0266 [+0.0124, +0.0404] (worse) | +0.0378 [+0.0200, +0.0544] (worse) |
| M2-full | +0.0341 [+0.0188, +0.0491] (worse) | +0.0763 [+0.0548, +0.0974] (worse) |

- B0 grid slot: win LL 0.1111 does NOT beat the 0.075 target; podium LL 0.2239 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1294 does NOT beat the 0.075 target; podium LL 0.2422 does NOT beat the 0.191 target.
- B2 LightGBM: win LL 0.1277 does NOT beat the 0.075 target; podium LL 0.2659 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1311 does NOT beat the 0.075 target; podium LL 0.2445 does NOT beat the 0.191 target.
- M1-pure strength: win LL 0.1655 does NOT beat the 0.075 target; podium LL 0.3266 does NOT beat the 0.191 target.
- M3 simulator: win LL 0.1377 does NOT beat the 0.075 target; podium LL 0.2617 does NOT beat the 0.191 target.
- M2-full: win LL 0.1451 does NOT beat the 0.075 target; podium LL 0.3001 does NOT beat the 0.191 target.

## Gate: B1 logistic

Candidate minus the best previous model (chosen per moment by win log loss), 95% paired-bootstrap interval:

| Moment | Best previous | Win LL | Podium LL | Both clearly better? |
|---|---|---|---|---|
| pre | B0 grid slot | -0.0036 [-0.0140, +0.0063] (no clear difference) | -0.0004 [-0.0109, +0.0097] (no clear difference) | no |
| practice | B0 grid slot | -0.0621 [-0.0816, -0.0414] (better) | -0.1538 [-0.1850, -0.1223] (better) | yes |
| quali | B0 grid slot | +0.0184 [+0.0038, +0.0323] (worse) | +0.0184 [+0.0030, +0.0340] (worse) | no |

**Decision:** B1 logistic is KEPT: clearly better than the best previous model on both win and podium log loss at: practice.

## Gate: B2 LightGBM

Candidate minus the best previous model (chosen per moment by win log loss), 95% paired-bootstrap interval:

| Moment | Best previous | Win LL | Podium LL | Both clearly better? |
|---|---|---|---|---|
| pre | B0 grid slot | +0.0073 [-0.0062, +0.0203] (no clear difference) | +0.0287 [+0.0127, +0.0447] (worse) | no |
| practice | B0 grid slot | -0.0528 [-0.0732, -0.0316] (better) | -0.1242 [-0.1557, -0.0930] (better) | yes |
| quali | B0 grid slot | +0.0166 [+0.0048, +0.0285] (worse) | +0.0422 [+0.0236, +0.0609] (worse) | no |

**Decision:** B2 LightGBM is KEPT: clearly better than the best previous model on both win and podium log loss at: practice.

## Gate: M1 strength

Candidate minus the best previous model (chosen per moment by win log loss), 95% paired-bootstrap interval:

| Moment | Best previous | Win LL | Podium LL | Both clearly better? |
|---|---|---|---|---|
| pre | B0 grid slot | -0.0024 [-0.0143, +0.0090] (no clear difference) | +0.0004 [-0.0123, +0.0128] (no clear difference) | no |
| practice | B0 grid slot | -0.0607 [-0.0794, -0.0411] (better) | -0.1527 [-0.1806, -0.1242] (better) | yes |
| quali | B0 grid slot | +0.0200 [+0.0053, +0.0339] (worse) | +0.0207 [+0.0051, +0.0360] (worse) | no |

**Decision:** M1 strength is KEPT: clearly better than the best previous model on both win and podium log loss at: practice.

## Gate: M1-pure strength

Candidate minus the best previous model (chosen per moment by win log loss), 95% paired-bootstrap interval:

| Moment | Best previous | Win LL | Podium LL | Both clearly better? |
|---|---|---|---|---|
| pre | B0 grid slot | +0.0183 [+0.0021, +0.0352] (worse) | +0.0491 [+0.0276, +0.0715] (worse) | no |
| practice | B0 grid slot | -0.0401 [-0.0567, -0.0230] (better) | -0.1039 [-0.1295, -0.0787] (better) | yes |
| quali | B0 grid slot | +0.0544 [+0.0359, +0.0717] (worse) | +0.1028 [+0.0769, +0.1276] (worse) | no |

**Decision:** M1-pure strength is KEPT: clearly better than the best previous model on both win and podium log loss at: practice.

## Gate: M3 simulator

Candidate minus the best previous model (chosen per moment by win log loss), 95% paired-bootstrap interval:

| Moment | Best previous | Win LL | Podium LL | Both clearly better? |
|---|---|---|---|---|
| pre | B0 grid slot | +0.0195 [+0.0037, +0.0360] (worse) | +0.0536 [+0.0312, +0.0761] (worse) | no |
| practice | B0 grid slot | -0.0394 [-0.0559, -0.0225] (better) | -0.1015 [-0.1266, -0.0768] (better) | yes |
| quali | B0 grid slot | +0.0266 [+0.0124, +0.0404] (worse) | +0.0378 [+0.0200, +0.0544] (worse) | no |

**Decision:** M3 simulator is KEPT: clearly better than the best previous model on both win and podium log loss at: practice.

## Gate: M2-full

Candidate minus the best previous model (chosen per moment by win log loss), 95% paired-bootstrap interval:

| Moment | Best previous | Win LL | Podium LL | Both clearly better? |
|---|---|---|---|---|
| pre | B0 grid slot | +0.0003 [-0.0115, +0.0119] (no clear difference) | +0.0221 [+0.0047, +0.0382] (worse) | no |
| practice | B0 grid slot | -0.0501 [-0.0666, -0.0332] (better) | -0.1103 [-0.1332, -0.0871] (better) | yes |
| quali | B0 grid slot | +0.0341 [+0.0188, +0.0491] (worse) | +0.0763 [+0.0548, +0.0974] (worse) | no |

**Decision:** M2-full is KEPT: clearly better than the best previous model on both win and podium log loss at: practice.

## Verdicts

- B0 grid slot: win LL 0.1472 does NOT beat the 0.075 target; podium LL 0.2783 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1435 does NOT beat the 0.075 target; podium LL 0.2778 does NOT beat the 0.191 target.
- B2 LightGBM: win LL 0.1545 does NOT beat the 0.075 target; podium LL 0.3070 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1448 does NOT beat the 0.075 target; podium LL 0.2786 does NOT beat the 0.191 target.
- M1-pure strength: win LL 0.1655 does NOT beat the 0.075 target; podium LL 0.3274 does NOT beat the 0.191 target.
- M3 simulator: win LL 0.1667 does NOT beat the 0.075 target; podium LL 0.3320 does NOT beat the 0.191 target.
- B0 grid slot: win LL 0.2059 does NOT beat the 0.075 target; podium LL 0.4314 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1438 does NOT beat the 0.075 target; podium LL 0.2776 does NOT beat the 0.191 target.
- B2 LightGBM: win LL 0.1531 does NOT beat the 0.075 target; podium LL 0.3072 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1452 does NOT beat the 0.075 target; podium LL 0.2788 does NOT beat the 0.191 target.
- M1-pure strength: win LL 0.1659 does NOT beat the 0.075 target; podium LL 0.3277 does NOT beat the 0.191 target.
- M3 simulator: win LL 0.1665 does NOT beat the 0.075 target; podium LL 0.3301 does NOT beat the 0.191 target.
- B0 grid slot: win LL 0.1111 does NOT beat the 0.075 target; podium LL 0.2239 does NOT beat the 0.191 target.
- B1 logistic: win LL 0.1294 does NOT beat the 0.075 target; podium LL 0.2422 does NOT beat the 0.191 target.
- B2 LightGBM: win LL 0.1277 does NOT beat the 0.075 target; podium LL 0.2659 does NOT beat the 0.191 target.
- M1 strength: win LL 0.1311 does NOT beat the 0.075 target; podium LL 0.2445 does NOT beat the 0.191 target.
- M1-pure strength: win LL 0.1655 does NOT beat the 0.075 target; podium LL 0.3266 does NOT beat the 0.191 target.
- M3 simulator: win LL 0.1377 does NOT beat the 0.075 target; podium LL 0.2617 does NOT beat the 0.191 target.
- M2-full: win LL 0.1475 does NOT beat the 0.075 target; podium LL 0.3004 does NOT beat the 0.191 target.
- M2-full: win LL 0.1558 does NOT beat the 0.075 target; podium LL 0.3212 does NOT beat the 0.191 target.
- M2-full: win LL 0.1451 does NOT beat the 0.075 target; podium LL 0.3001 does NOT beat the 0.191 target.

