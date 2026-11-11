# Infer heat diffusivity from snapshots

Learn the supervised-learning workflow using a PDE whose solution we know.
The goal is to understand what the network learns, how to evaluate it, and where
it fails. Beating the physics baseline is not a requirement.

## Run

From the repository root, with Python 3.10 or newer:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r projects/heat_inverse/requirements.txt
python -m projects.heat_inverse.experiment
python -m unittest discover -s tests -v
```

The experiment writes `metrics.json` and `predictions.png` under
`outputs/heat_inverse/`. It needs no GPU or downloaded dataset. Begin with
[Lesson 1](lesson_01.md) before reading through the complete experiment.

## Mathematical contract

On the dimensionless interval 0 <= x <= 1, solve

\[
u_t=\alpha u_{xx},\qquad u(0,t)=u(1,t)=0,
\]
\[
u(x,0)=\sin(\pi x)+0.3\sin(2\pi x).
\]

The exact solution is

\[
u(x,t;\alpha)=e^{-\alpha\pi^2t}\sin(\pi x)
+0.3e^{-4\alpha\pi^2t}\sin(2\pi x).
\]

Every example has one diffusivity and three snapshots at t = 0.1, 0.3, 0.6,
observed at 16 equally spaced interior sensors. Flatten these into 48 features;
the label is the scalar alpha. Positions, times, boundary values, initial state,
and amplitude are known and fixed. Independent Gaussian measurement noise has
absolute standard deviation 0.01; it is not 1% relative noise at each sensor.

Training uses 1,000 independently sampled trajectories with alpha uniform on
[0.05, 0.5]. Validation uses another 200; each test scenario uses 200 more.
Separate random streams isolate these sets. Splitting is by whole trajectory,
never by individual snapshot. Both input and target scaling fit training data
only. No test result selects the model.

## What is implemented

- A 48 -> 32 -> 32 -> 1 fully connected network with tanh hidden activations
  and a linear output. It learns a supervised inverse map; this is not a PINN.
- L-BFGS training with validation selection between two L2 penalties. Tanh and
  this optimizer are starting choices, not claims of optimality.
- Independent nonlinear least-squares estimates using the same forward model,
  sensors, observations, and noise. Search bounds [0.001, 1] deliberately cover
  all test diffusivities, including the network's extrapolation test.
- A training-mean predictor to check that learning adds information.
- Clean, noisy, and out-of-training-range evaluations. The network trains on
  noisy data even for the clean test. Predictions are not clipped; nonpositive
  estimates are counted in the report.

The least-squares estimator knows the exact physics. This synthetic setup favors
it and measures the learned approximation of an available inverse procedure.
It does not establish usefulness on an unknown physical system. An exact forward
solution deliberately removes discretization error for this first lesson.

## Mini-lecture and mini-lab sequence

| Module | Lecture focus | Lab and checkpoint |
|---|---|---|
| 1 | Examples, features, labels, inverse maps | Inspect snapshots and derive sensitivity; explain what constitutes one example |
| 2 | Empirical risk and information leakage | Trace the train/validation/test paths; explain why snapshots stay together |
| 3 | Neural network as a parameterized map | Count weights, follow scaling, compare with the mean predictor |
| 4 | Physics baseline and error metrics | Compare least squares and MLP on the same noisy trajectories |
| 5 | Distribution shift and sensor information | Run sparse sensors and several seeds; explain interpolation versus extrapolation |

Lesson 1 is written out; modules 2–5 are a roadmap for guided follow-up.

## Controlled follow-up experiments

```bash
python -m projects.heat_inverse.experiment --sensors 4 --output outputs/heat_sparse
python -m projects.heat_inverse.experiment --noise 0 --output outputs/heat_clean_training
python -m projects.heat_inverse.experiment --seed 43 --output outputs/heat_seed43
```

Changing sensor count retrains a network with the appropriate input dimension;
it does not simulate missing sensors for an already trained network. Changing
noise changes training, validation, and the noisy/OOD test noise together.
Use a separate experiment for unexpected test-time noise or missing observations.
Sensor runs use newly generated noise arrays and are not paired-noise comparisons.

Record configuration, metrics, warnings, and dependency versions. Runtime fields
are single-run measurements, not robust benchmarks; training cost is reported
separately. Repeat across seeds before drawing comparative conclusions. GPU
training, unknown initial states, finite-difference data, PINNs, and model export
are left for later modules.

See [the initial run](results.md) for measured results and limits.
