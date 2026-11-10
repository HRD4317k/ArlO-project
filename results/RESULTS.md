# Results

Seeds: 5 (mean ± std). Episode = 2 h of turbine operation (240 steps x 30 s). Evaluation: 8 fixed episodes with unseen Oct-Dec 2018 wind. Simulator power-curve RMSE vs measured SCADA curve: 0.044 of rated power.

## Exp 1: single-regime baseline

| Regime | SAC return | Fixed return | SAC δ increase | Fixed δ increase | SAC failure rate | Fixed failure rate |
|---|---|---|---|---|---|---|
| R0 healthy | 85.1 ± 3.4 | 82.7 | 0.009 ± 0.000 | 0.0090 | 0.000 ± 0.000 | 0.000 |
| R1 mild | 82.5 ± 2.6 | 74.6 | 0.011 ± 0.001 | 0.0127 | 0.000 ± 0.000 | 0.000 |
| R2 moderate | 34.6 ± 13.0 | 2.4 | 0.030 ± 0.006 | 0.0393 | 0.025 ± 0.050 | 0.125 |
| R3 critical | 22.6 ± 3.8 | -5.6 | 0.021 ± 0.001 | 0.0306 | 0.125 ± 0.000 | 0.250 |

## Exp 2: cross-regime return (rows = trained on, columns = tested on)

| | test R0 | test R1 | test R2 | test R3 |
|---|---|---|---|---|
| train R0 | 85.1 | 72.1 | 13.5 | -3.5 |
| train R1 | 87.2 | 82.5 | 18.8 | 14.2 |
| train R2 | 74.0 | 78.8 | 34.6 | 15.8 |
| train R3 | 67.9 | 69.8 | 49.2 | 22.6 |

Generalisation gap (same-regime minus cross-regime return):

| | test R0 | test R1 | test R2 | test R3 |
|---|---|---|---|---|
| train R0 | 0.0 | 10.4 | 21.0 | 26.1 |
| train R1 | -2.1 | 0.0 | 15.8 | 8.4 |
| train R2 | 11.1 | 3.7 | 0.0 | 6.8 |
| train R3 | 17.2 | 12.7 | -14.6 | 0.0 |

## Exp 3-4: adaptation to R2 moderate (unseen during meta-training)

| Method | return K=0 | return K=1 | return K=5 | return K=10 | return K=20 | δ increase K=5 | failure K=0 | failure K=5 | episodes to 90 % of oracle (per seed) |
|---|---|---|---|---|---|---|---|---|---|
| (a) SAC from scratch | 8.3 ± 5.3 | -1.3 ± 22.9 | 22.2 ± 6.2 | 10.8 ± 6.9 | 22.4 ± 11.1 | 0.028 ± 0.006 | 0.000 ± 0.000 | 0.000 ± 0.000 | never, 20, never, never, 5 |
| (b) SAC pre-trained R0 + fine-tune | 13.5 ± 8.5 | 10.9 ± 11.7 | 17.3 ± 12.8 | 15.8 ± 9.7 | 19.4 ± 12.2 | 0.027 ± 0.010 | 0.075 ± 0.061 | 0.050 ± 0.061 | never, never, never, 20, 1 |
| (c) SAC joint multi-regime | 32.0 ± 15.8 | 29.9 ± 14.2 | 34.6 ± 10.4 | 32.7 ± 12.4 | 31.9 ± 9.7 | 0.028 ± 0.005 | 0.025 ± 0.050 | 0.000 ± 0.000 | 0, 0, 0, 20, 5 |
| (d) First-order MAML + SAC, Reptile update | 27.2 ± 3.3 | 24.9 ± 4.1 | 26.6 ± 3.7 | 29.9 ± 2.6 | 33.7 ± 4.2 | 0.029 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 10, 20, 0, 20, never |
| (e) FOMAML + SAC, query-gradient update | -2.6 ± 14.5 | -8.3 ± 14.9 | -3.1 ± 27.8 | 19.4 ± 11.8 | 25.5 ± 9.6 | 0.032 ± 0.005 | 0.025 ± 0.050 | 0.050 ± 0.061 | never, 20, 10, never, never |

Oracle (SAC trained on R2): 34.6; fixed controller: 2.4.

## Exp 3-4: adaptation to R3 critical (seen during meta-training)

| Method | return K=0 | return K=1 | return K=5 | return K=10 | return K=20 | δ increase K=5 | failure K=0 | failure K=5 | episodes to 90 % of oracle (per seed) |
|---|---|---|---|---|---|---|---|---|---|
| (a) SAC from scratch | -7.7 ± 1.2 | -14.2 ± 14.5 | -6.0 ± 17.1 | 9.2 ± 10.2 | 17.4 ± 6.3 | 0.025 ± 0.005 | 0.150 ± 0.050 | 0.175 ± 0.061 | 20, never, 10, never, never |
| (b) SAC pre-trained R0 + fine-tune | -3.5 ± 4.6 | 0.4 ± 9.4 | -12.5 ± 13.9 | -8.9 ± 20.3 | 2.3 ± 13.7 | 0.023 ± 0.007 | 0.200 ± 0.061 | 0.175 ± 0.061 | never, never, never, never, never |
| (c) SAC joint multi-regime | 16.5 ± 11.9 | 15.8 ± 11.3 | 11.9 ± 11.0 | 11.8 ± 9.1 | 18.3 ± 11.2 | 0.024 ± 0.004 | 0.150 ± 0.050 | 0.150 ± 0.050 | 0, 0, 20, never, 0 |
| (d) First-order MAML + SAC, Reptile update | 6.7 ± 3.7 | 7.1 ± 5.8 | 5.3 ± 7.9 | 11.3 ± 4.3 | 12.2 ± 3.0 | 0.026 ± 0.003 | 0.150 ± 0.050 | 0.125 ± 0.000 | never, never, never, never, never |
| (e) FOMAML + SAC, query-gradient update | -9.5 ± 6.4 | -8.4 ± 9.4 | -5.2 ± 6.2 | 7.1 ± 4.8 | 10.0 ± 6.6 | 0.024 ± 0.006 | 0.125 ± 0.000 | 0.150 ± 0.050 | never, never, never, never, never |

Oracle (SAC trained on R3): 22.6; fixed controller: -5.6.

## Ablation on β₂ (target R2, seeds 0-2): method (d) minus scratch

| β₂ | Δreturn K=0 | Δreturn K=1 | Δreturn K=5 | Δreturn K=10 | Δreturn K=20 | Δ(δ increase) K=0 | Δ(δ increase) K=1 | Δ(δ increase) K=5 | Δ(δ increase) K=10 | Δ(δ increase) K=20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 33.2 | 33.2 | 4.6 | 6.1 | 1.5 | +0.0079 | -0.0016 | -0.0040 | -0.0053 | -0.0019 |
| 2000 | 24.1 | 30.9 | 6.2 | 18.2 | 11.5 | +0.0077 | -0.0003 | -0.0035 | -0.0096 | -0.0041 |
| 4000 | 23.4 | 46.7 | 33.8 | 47.6 | 29.0 | +0.0034 | -0.0052 | -0.0094 | -0.0132 | -0.0075 |
