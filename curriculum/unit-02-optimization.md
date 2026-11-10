# Unit 2 — Optimization and Learning Dynamics

**Goal:** Understand how an objective is minimized and how numerical behavior changes with algorithm and scaling. **Level:** core.

| Lesson | Mini-lecture focus | Mini-lab proposal | Diagnostic / checkpoint |
|---|---|---|---|
| 2.1 First-order methods | Gradient descent, stochastic gradients, momentum, step sizes | Optimize a quadratic with adjustable condition number | Stability vs step size; convergence-rate plots |
| 2.2 Adaptive and quasi-Newton methods | Adam, SGD/momentum, L-BFGS; assumptions and tradeoffs | Train a fixed small regression network with matched budget | Compare loss, wall time, function/gradient evaluations |
| 2.3 Conditioning and scaling | Hessian spectrum, normalization, initialization, learning rates | Rescale features while fixing model/data semantics | Explain optimizer sensitivity |
| 2.4 Stochastic training | Mini-batches, noise, schedules, stopping | Vary batch size and learning-rate schedule | Separate optimization error from validation performance |

**Unit checkpoint:** Justify an optimizer choice for a specified compute budget and problem, without claiming universal superiority.

**Artifact planned:** Optimizer benchmark with consistent stopping rules and reproducible plots.
