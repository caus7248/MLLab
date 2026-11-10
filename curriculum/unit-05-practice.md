# Unit 5 — Reliable Machine-Learning Practice

**Goal:** Turn one-off experiments into reproducible evidence. **Level:** core.

| Lesson | Mini-lecture focus | Mini-lab proposal | Diagnostic / checkpoint |
|---|---|---|---|
| 5.1 Data pipeline | Splits, preprocessing, leakage, provenance | Build a dataset loader with train-only fitted transforms | Deliberately introduce and detect leakage |
| 5.2 Experiment design | Baselines, controlled variables, seeds, ablations | Reproduce an earlier optimizer or activation comparison | Quantify run-to-run variability |
| 5.3 Training instrumentation | Checkpoints, logging, curves, evaluation modes | Add a common trainer and metrics interface | Detect overfit, unstable gradients, bad stopping |
| 5.4 Compute and reproducibility | CPU/GPU, precision, profiling, dependency pinning | Profile a fixed experiment across available devices | Explain throughput vs accuracy and nondeterminism |

**Unit checkpoint:** Another learner can rerun a documented experiment and interpret discrepancies.

**Artifact planned:** Small reusable MLLab training/evaluation harness, with CI smoke test (future work).
