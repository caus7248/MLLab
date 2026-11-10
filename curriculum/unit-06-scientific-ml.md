# Unit 6 — Scientific Machine Learning

**Goal:** Evaluate where ML complements or fails to improve numerical PDE methods. **Level:** advanced; requires relevant earlier checkpoints.

| Lesson | Mini-lecture focus | Mini-lab proposal | Diagnostic / checkpoint |
|---|---|---|---|
| 6.1 Surrogate models | Operator maps, data generation, interpolation vs extrapolation | Learn a parameter-to-solution map for a simple PDE | Compare against solver error and runtime |
| 6.2 PINNs | Residual losses, automatic differentiation, BC/IC enforcement | Solve a 1D diffusion benchmark with a PINN | Compare residual, boundary error, solution error |
| 6.3 Inverse problems | Identifiability, noise, regularization, parameter estimation | Recover one coefficient from synthetic observations | Study noise sensitivity and uncertainty |
| 6.4 Hybrid numerical/ML | Learned components vs trusted discretizations | Explore learned stencil selection on advection benchmarks | Check consistency, stability, conservation, generalization |
| 6.5 Research evaluation | Baselines, ablations, robustness, failure analysis | Compare one ML method against a conventional solver | Identify conditions where ML provides real value |

**Unit checkpoint:** Present a numerical case for or against using ML, accounting for data generation, training, inference, accuracy, and robustness.

**Artifact planned:** A PDE-linked benchmark with transparent non-ML baseline. MOLT and other NumLab solvers are optional future integration candidates, not prerequisites.
