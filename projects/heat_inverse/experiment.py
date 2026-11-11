"""CPU experiment: python -m projects.heat_inverse.experiment --help."""
import argparse
import json
import platform
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import minimize_scalar
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def forward(alpha, x, times):
    """Return (trajectory, time, sensor) values for a fixed known initial state."""
    a = np.asarray(alpha).reshape(-1, 1, 1)
    x = np.asarray(x)[None, None, :]
    t = np.asarray(times)[None, :, None]
    return (np.exp(-a * np.pi**2 * t) * np.sin(np.pi * x)
            + 0.3 * np.exp(-4 * a * np.pi**2 * t) * np.sin(2 * np.pi * x))


def dataset(rng, n, bounds, x, times, sigma):
    alpha = rng.uniform(*bounds, size=n)
    clean = forward(alpha, x, times)
    observations = clean + rng.normal(0, sigma, size=clean.shape)
    return observations.reshape(n, -1), alpha


def fit_physics(observations, x, times):
    """Fit each trajectory independently; bounds cover both ID and OOD tests."""
    predictions = []
    for row in observations:
        result = minimize_scalar(
            lambda a: np.mean((forward([a], x, times).ravel() - row)**2),
            bounds=(0.001, 1.0), method="bounded", options={"xatol": 1e-12})
        if not result.success:
            raise RuntimeError(result.message)
        predictions.append(result.x)
    return np.asarray(predictions)


def scores(truth, prediction):
    error = prediction - truth
    return {"mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mean_relative_error": float(np.mean(np.abs(error) / truth))}


def run(seed=42, sensors=16, train_size=1000, test_size=200, noise=0.01):
    if sensors < 1 or train_size < 20 or test_size < 1 or noise < 0:
        raise ValueError("Require sensors >= 1, train_size >= 20, test_size >= 1, noise >= 0")
    # Separate RNG streams keep test data independent of training size and selection.
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    x = np.linspace(0, 1, sensors + 2)[1:-1]
    times = np.array([0.1, 0.3, 0.6])
    train_x, train_y = dataset(streams[0], train_size, (0.05, 0.5), x, times, noise)
    val_x, val_y = dataset(streams[1], 200, (0.05, 0.5), x, times, noise)
    candidates = []
    start = perf_counter()
    for penalty in (1e-4, 1e-2):
        # All scalers are fitted on training data only. Predict returns physical alpha.
        model = TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(32, 32), activation="tanh", solver="lbfgs",
                alpha=penalty, max_iter=1000, random_state=seed)),
            transformer=StandardScaler())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, train_y)
        candidates.append((scores(val_y, model.predict(val_x))["rmse"], penalty, model,
                           [str(w.message) for w in caught]))
    val_rmse, penalty, model, _ = min(candidates, key=lambda item: item[0])
    train_seconds = perf_counter() - start
    metrics = {"seed": seed, "sensors": sensors, "times": times.tolist(),
               "train_size": train_size, "validation_size": 200, "test_size": test_size,
               "training_noise_std": noise, "selected_penalty": penalty,
               "validation_rmse": val_rmse, "training_selection_seconds": train_seconds,
               "candidate_diagnostics": [{"penalty": p, "validation_rmse": v, "warnings": w}
                                         for v, p, _, w in candidates],
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "sklearn": sklearn.__version__},
               "scenarios": {}}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    scenarios = [("clean", (0.05, 0.5), 0), ("noisy", (0.05, 0.5), noise),
                 ("extrapolation", (0.55, 0.8), noise)]
    for ax, rng, (name, bounds, sigma) in zip(axes, streams[2:], scenarios):
        features, truth = dataset(rng, test_size, bounds, x, times, sigma)
        start = perf_counter()
        neural = model.predict(features)
        neural_seconds = perf_counter() - start
        start = perf_counter()
        physics = fit_physics(features, x, times)
        physics_seconds = perf_counter() - start
        metrics["scenarios"][name] = {
            "alpha_bounds": bounds, "noise_std": sigma,
            "mlp": scores(truth, neural), "least_squares": scores(truth, physics),
            "training_mean": scores(truth, np.full_like(truth, train_y.mean())),
            "mlp_prediction_seconds": neural_seconds,
            "least_squares_prediction_seconds": physics_seconds,
            "nonpositive_mlp_predictions": int(np.sum(neural <= 0))}
        ax.scatter(truth, neural, s=10, alpha=0.6, label="MLP")
        ax.scatter(truth, physics, s=10, alpha=0.5, label="Least squares")
        ax.plot(bounds, bounds, "k--", linewidth=1)
        ax.set(title=name, xlabel="True diffusivity", ylabel="Estimated diffusivity")
        ax.legend()
    return metrics, fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sensors", type=int, default=16)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("outputs/heat_inverse"))
    args = parser.parse_args()
    metrics, fig = run(args.seed, args.sensors, args.train_size, args.test_size, args.noise)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fig.savefig(args.output / "predictions.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
