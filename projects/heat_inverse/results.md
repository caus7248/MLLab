# Initial experiment results

Seed 42, 1,000 training trajectories, 200 validation trajectories, 200 test trajectories per scenario. Training noise standard deviation: 0.01. These are single-seed observations, not statistical performance guarantees.

| Sensors | Test | MLP RMSE | Least-squares RMSE | MLP mean relative error | LS mean relative error |
|---|---|---:|---:|---:|---:|
| 16 | clean | 0.00129917 | 6.17344e-10 | 0.512% | 1.37e-07% |
| 16 | noisy | 0.00268726 | 0.00222673 | 0.811% | 0.562% |
| 16 | extrapolation | 0.12172 | 0.00601963 | 15% | 0.677% |
| 4 | clean | 0.000835901 | 6.17344e-10 | 0.292% | 1.37e-07% |
| 4 | noisy | 0.00383727 | 0.0035004 | 1.12% | 0.935% |
| 4 | extrapolation | 0.102537 | 0.0105988 | 12.6% | 1.25% |

Both runs completed without training convergence warnings. All five unit tests passed: initial/boundary data, PDE residual, noiseless inverse recovery, seeded data reproduction, and invalid sensor configuration.

The network beats the training-mean baseline but loses accuracy beyond its training range. Least squares benefits from the exact known forward model. Sparse observations increased noisy-test error in this run; the smaller network input also changes optimization, so this alone does not isolate an information-theoretic sensor effect.

Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0. Runs used OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1. JSON companions preserve configuration, candidate diagnostics, baseline errors, and single-run timings. Regenerate plots using the README commands.
