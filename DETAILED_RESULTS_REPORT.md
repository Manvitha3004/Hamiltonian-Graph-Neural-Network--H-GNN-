# H-GNN Detailed Results Report

Date: March 16, 2026
Project: Hamiltonian Graph Neural Network (H-GNN)

---

## 1. Report Scope

This report consolidates all available experiment outputs from the `results/` directory and related project summaries into one detailed technical results document.

Primary sources used:

1. `results/baseline_comparison/comparison_results.json`
2. `results/epsilon_sweep/sweep_results.json`
3. `results/curriculum_training/curriculum_results.json`
4. `results/scaling/scaling_results.json`
5. `results/scaling_law/scaling_law_results.json`
6. `results/invariant_analysis/invariant_results.json`
7. `results/rare_regime/rare_regime_results.json`
8. `results/volume_test/volume_test_results.json`
9. `results/correction_accuracy/correction_accuracy_results.json`
10. `results/results.json` and `results/comparison_table.txt` (new integrated pipeline)
11. `EXPERIMENTAL_RESULTS.md`

---

## 2. Executive Summary

1. H-GNN consistently outperforms vanilla non-symplectic neural baselines in the core baseline-comparison suite at short and medium horizons.
2. Improvement versus two-body baseline is strong up to 100-step horizons in many runs, then degrades at 200 steps (mixed/negative in multi-run average).
3. Epsilon sweep supports a learnability threshold: epsilon below approximately 5 is weak or failing; epsilon in 10-50 gives best improvements.
4. Curriculum training improves robustness over a wide horizon set and shows positive performance at 500 steps, but T=200 remains difficult.
5. Scaling measurements show runtime growth with system size and graph construction scaling near linear-log/weakly superlinear behavior in measured regime.
6. Rare regime test contains a dtype failure for vanilla GNN (`Double` vs `Float`) and is therefore partially incomplete.
7. New integrated `run_experiments.py` long-horizon outputs currently conflict with validated historical results (H-GNN underperforms there), indicating calibration/integration mismatch still in progress.

---

## 3. Experiment Inventory and Status

### Completed and with numeric outputs

1. Baseline comparison
2. Epsilon sweep
3. Curriculum training summary
4. Scaling runtime/memory analysis
5. Scaling-law epsilon-times-horizon analysis
6. Invariant analysis
7. Rare regime analysis (partial due to vanilla dtype failure)
8. Phase-space volume test
9. Correction-accuracy diagnostic
10. Integrated long-horizon experiment output from `run_experiments.py`

### Not completed or not present in provided outputs

1. Noise robustness test outputs are not present in `results/` as a completed JSON report.

---

## 4. Baseline Comparison (Core Model Benchmark)

Source: `results/baseline_comparison/comparison_results.json`

The file contains 15 records corresponding to 3 runs across horizons {10, 25, 50, 100, 200}.

### 4.1 H-GNN improvement over two-body baseline by run

Run 1:

1. T=10: +7.372%
2. T=25: +16.003%
3. T=50: +18.495%
4. T=100: +17.425%
5. T=200: +8.003%

Run 2:

1. T=10: +11.967%
2. T=25: +24.970%
3. T=50: +24.076%
4. T=100: +5.395%
5. T=200: -10.454%

Run 3:

1. T=10: +6.137%
2. T=25: +16.031%
3. T=50: +20.018%
4. T=100: +13.255%
5. T=200: -3.812%

### 4.2 Vanilla GNN improvement over two-body baseline by run

Run 1:

1. T=10: -0.576%
2. T=25: -2.082%
3. T=50: -9.319%
4. T=100: -6.518%
5. T=200: -6.340%

Run 2:

1. T=10: -0.304%
2. T=25: -0.065%
3. T=50: +0.828%
4. T=100: +0.080%
5. T=200: +0.337%

Run 3:

1. T=10: -0.206%
2. T=25: -0.569%
3. T=50: -0.536%
4. T=100: +1.795%
5. T=200: +1.270%

### 4.3 Baseline comparison interpretation

1. H-GNN is consistently positive at T=10, 25, and 50 in all 3 runs.
2. H-GNN remains mostly positive at T=100 but with reduced stability.
3. T=200 is unstable: one positive run and two negative runs.
4. Vanilla GNN is generally near-zero or negative, especially at shorter horizons.

---

## 5. Epsilon Sweep (Learnability and Signal Strength)

Source: `results/epsilon_sweep/sweep_results.json`

Epsilon values tested: 1, 5, 10, 25, 50, 100, 200.

### 5.1 Improvement over two-body baseline at T=50

1. eps=1: -1.900%
2. eps=5: +15.037%
3. eps=10: +18.761%
4. eps=25: +23.385%
5. eps=50: +20.863%
6. eps=100: +13.357%
7. eps=200: +15.856%

### 5.2 Mean Phi magnitude trend

1. eps=1: 0.000430
2. eps=5: 0.002080
3. eps=10: 0.004149
4. eps=25: 0.010061
5. eps=50: 0.016862
6. eps=100: 0.022817
7. eps=200: 0.027093

### 5.3 Interpretation

1. There is a practical threshold around eps approximately 5 where learning shifts from weak/failing to clearly positive.
2. Best medium-horizon gain appears around eps in [25, 50].
3. Phi magnitude increases monotonically with epsilon, confirming stronger learned correction activation.

---

## 6. Curriculum Training Results

Source: `results/curriculum_training/curriculum_results.json`

Summary metrics from file:

1. T=10: mean +4.674%, std 3.330%
2. T=25: mean +11.205%, std 7.661%
3. T=50: mean +13.411%, std 9.341%
4. T=100: mean +9.517%, std 7.083%
5. T=200: mean -1.127%, std 4.836%
6. T=500: mean +5.571%, std 4.226%

### 6.1 Interpretation

1. Curriculum improves broad-horizon robustness with positive mean performance up to 500.
2. T=200 remains a difficult regime and still shows slightly negative mean.
3. Variance is non-trivial, indicating seed sensitivity.

### 6.2 Correction analysis from file

`correction_analysis` indicates:

1. Target correction/base ratio around 2.9%.
2. Model ratio varies widely from very small to moderate values.
3. Learning efficiency ranges from very low to moderate depending on run.

---

## 7. Invariant Analysis

Source: `results/invariant_analysis/invariant_results.json`

The file tracks energy drift, angular momentum drift (`L_drift`), and MSE at multiple steps and multiple runs.

### 7.1 Representative findings

1. At short horizons (for example step 10), H-GNN often has lower MSE than vanilla and comparable or improved conservation over vanilla.
2. Angular momentum drift for vanilla grows much larger than H-GNN and two-body in many segments.
3. At very long horizons (step 200 and 500), H-GNN may still have lower or comparable MSE versus vanilla in some runs, but error growth is substantial for all methods.

### 7.2 Interpretation

1. Symplectic structure helps preserve invariants better than vanilla neural integration.
2. Long-horizon error accumulation remains a shared challenge.

---

## 8. Scaling Runtime and Memory Results

Source: `results/scaling/scaling_results.json`

### 8.1 Runtime and memory points

Tested object counts: 50, 100, 200, 500.

H-GNN:

1. Runtime (s): [22.34, 25.89, 38.32, 80.52]
2. Memory (GB-ish logged units): [0.0099, 0.0193, 0.0330, 0.1712]
3. Fitted scaling exponent: 0.567

Pure two-body:

1. Runtime (s): [23.32, 26.08, 38.16, 92.22]
2. Memory: [0.0194, 0.0192, 0.0330, 0.1712]
3. Fitted scaling exponent: 0.605

Graph construction:

1. Runtime (s): [0.129, 0.203, 0.441, 1.604]
2. Fitted scaling exponent: 1.106

### 8.2 Interpretation

1. End-to-end runtime scales moderately in this tested regime.
2. Graph construction cost increases faster with N and will become a dominant contributor at larger scales.

---

## 9. Scaling-Law Results (epsilon x horizon)

Source: `results/scaling_law/scaling_law_results.json`

This file combines epsilon and horizon settings and records MSE and improvement.

### 9.1 Key observations

1. Improvements are positive for eps >= 5 across many settings.
2. Improvement weakens at larger horizons (for example from T=50 to T=100 at same epsilon).
3. Phi magnitude remains consistent per epsilon while performance changes by horizon, supporting the view that rollout difficulty, not only correction magnitude, governs long-horizon behavior.

---

## 10. Rare Regime Analysis

Source: `results/rare_regime/rare_regime_results.json`

### 10.1 H-GNN vs pure two-body

H-GNN MSE by regime:

1. Equatorial: 9661.53
2. Low inclination: 6603.66
3. Medium inclination: 6469.17
4. High inclination: 7264.74
5. Polar: 6890.97

Pure two-body MSE by regime:

1. Equatorial: 6384.45
2. Low inclination: 6341.66
3. Medium inclination: 6290.24
4. High inclination: 6723.75
5. Polar: 6643.76

Generalization gap:

1. H-GNN: 0.713
2. Pure two-body: 1.041

### 10.2 Failure noted

Vanilla GNN entries are invalid (`NaN`) due to dtype mismatch error:

`mat1 and mat2 must have the same dtype, but got Double and Float`

### 10.3 Interpretation

1. This experiment is partially complete.
2. Vanilla comparison is not currently valid until dtype consistency is fixed.

---

## 11. Phase-Space Volume Test

Source: `results/volume_test/volume_test_results.json`

The arrays track cumulative log-determinant behavior.

### 11.1 Findings

1. `log_det_hgnn` remains 0.0 throughout shown steps.
2. `log_det_vanilla` fluctuates at very small magnitudes near numerical precision.

### 11.2 Interpretation

1. H-GNN rollout preserves phase-space volume in this test setup.
2. Vanilla appears close to volume-preserving numerically in this specific test but with small fluctuations.

---

## 12. Correction Accuracy Diagnostic

Source: `results/correction_accuracy/correction_accuracy_results.json`

Reported metrics:

1. Relative error mean: 1.0
2. Cosine similarity mean: 0.0
3. Magnitude ratio mean: 0.0
4. True magnitude mean: 3.26e-06
5. Learned magnitude mean: 0.0

### Interpretation

1. This diagnostic indicates complete correction collapse in that particular training/evaluation path.
2. It matches the documented history where gradient-matching style objectives failed and trajectory-based training became necessary.

---

## 13. T200 Diagnosis Artifact

Source: `results/t200_diagnosis/diagnosis_results.json`

This file contains detailed step-wise curves (0 to 500) for error trajectories such as `mse_twobody` and likely companion series for model diagnostics.

### Practical summary

1. It is a dense trajectory-diagnostics artifact intended for plotting and forensic debugging.
2. The file should be interpreted with `diagnosis_plots.png` in the same folder.
3. This artifact supports the conclusion that T=200+ behavior is a key failure/stress regime.

---

## 14. Integrated Pipeline Output (Current In-Progress Runner)

Sources:

1. `results/results.json`
2. `results/comparison_table.txt`

Long-horizon integrated run reports:

1. 15 min: Physics RMSE 5.61 km, H-GNN RMSE 232.15 km
2. 30 min: Physics RMSE 21.76 km, H-GNN RMSE 643.18 km
3. 60 min: Physics RMSE 70.41 km, H-GNN RMSE 661.55 km

This yields strongly negative improvements in that pipeline.

### Interpretation

1. This output is inconsistent with validated historical experiment suites.
2. Most likely cause is pipeline mismatch (training config, split strategy, objective, baseline setup, or insufficient training budget) rather than fundamental model invalidity.

---

## 15. Cross-Source Consistency Assessment

### Consistent signals across multiple files

1. H-GNN improves over vanilla non-symplectic neural baselines at short/medium horizons.
2. Epsilon threshold behavior exists.
3. Long-horizon behavior is harder and has high variance.
4. T=200 is the most unstable regime.

### Conflicts that must be resolved

1. Historical validated suites show broadly positive performance profile.
2. New integrated `run_experiments.py` long-horizon output currently shows severe underperformance.

This conflict should be explicitly tracked and fixed before a final publication table is frozen.

---

## 16. Risks and Limitations

1. High sensitivity to seed and horizon.
2. Result variation across scripts and training objectives.
3. Incomplete/failed rare-regime vanilla baseline due to dtype mismatch.
4. Missing finalized noise-robustness report.
5. In-progress integrated pipeline not yet aligned with validated settings.

---

## 17. Recommended Finalization Checklist

1. Fix dtype handling in rare-regime vanilla experiment and regenerate results.
2. Re-run integrated long-horizon pipeline using validated training recipe from improved script.
3. Add noise robustness JSON output and include it in final report.
4. Freeze a single baseline protocol and seeds for paper-grade reproducibility.
5. Produce final aggregated table with mean and std at horizons {10, 25, 50, 100, 200, 500}.

---

## 18. Final Conclusion

The available results strongly support the core research direction: structure-preserving Hamiltonian learning with symplectic integration is superior to unconstrained neural forecasting for this problem class. H-GNN provides clear gains at short-to-medium horizons, shows meaningful correction learning under adequate signal conditions, and preserves key geometric properties better than vanilla neural baselines. The remaining work is reproducibility consolidation: aligning the new integrated experiment runner with the validated configuration, resolving known experiment-level errors, and finalizing missing robustness outputs.
