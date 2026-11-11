# Lesson 1 — What exactly is the network learning?

## Mini-lecture

You already know the forward problem: choose alpha and compute a solution.
Here the observation operator samples that solution at fixed space-time points:

\[
G:\alpha\longmapsto (u(x_j,t_k;\alpha))_{k,j}\in\mathbb R^{48}.
\]

We observe y = G(alpha) + epsilon and seek alpha. A conventional fit solves

\[
\widehat\alpha_{LS}=\arg\min_{0.001\leq a\leq1}\|G(a)-y\|_2^2.
\]

The network instead represents a family of maps f_theta: R^48 -> R. Training
adjusts theta using many known pairs (y_i, alpha_i). Once trained, evaluation
is a composition of matrix-vector products and componentwise nonlinearities.
There is no per-observation inverse optimization during prediction.

One training example is **one entire observed trajectory**, represented by three
snapshots. It has 48 features and one label. A batch of N examples has feature
shape (N, 48) and target shape (N,). The neural-network weights are learned
parameters; diffusivity is the physical parameter being predicted.

The training objective is squared error in the standardized target, with L2
regularization of network weights. Validation compares predictions in physical
alpha units. These are two distinct operations: fitting weights and choosing
between candidate regularization strengths. Test data enter neither operation.

Why should inversion be possible? At the center x = 1/2, the second sine mode
vanishes, leaving u(1/2,t) = exp(-alpha*pi^2*t). For a positive noiseless value,

\[
\alpha=-\frac{\log u(1/2,t)}{\pi^2t}.
\]

This establishes identifiability in a simple observation setting, although the
default even-sized sensor grid does not include the center. It also gives a
sanity check: the network is approximating an inverse relation, not discovering
the heat equation. With noise, log inversion can amplify small absolute errors
when the observed value is small, and negative observations cannot be logged.
Our baseline fits untransformed observations across all sensors and times.

## Mini-lab (about 20 minutes)

1. Before training, predict how increasing alpha changes the snapshots.
2. Run this from the repository root:

   ```python
   import numpy as np
   from projects.heat_inverse.experiment import forward

   x = np.linspace(0, 1, 18)[1:-1]
   times = [0.1, 0.3, 0.6]
   snapshots = forward([0.1, 0.4], x, times)
   print(snapshots.shape)  # (2, 3, 16)
   print(snapshots[:, -1, :])
   center = forward([0.2], [0.5], [0.3]).item()
   print(-np.log(center) / (np.pi**2 * 0.3))
   ```

3. Derive dG/dalpha. For the first mode its magnitude contains
   pi^2*t*exp(-alpha*pi^2*t). Explain why t = 0 has no diffusivity information,
   while very late times may also be poor with fixed absolute noise.
4. Run the complete experiment and inspect `predictions.png`. Identify which
   panel is a test of interpolation and which is a test of extrapolation.
5. Read only `forward`, `dataset`, and `fit_physics` first. Trace the shapes
   before studying the network's training code.

## Checkpoint

Explain aloud: What is one example? What is known at prediction time? What is
learned during training? Why does splitting snapshots from one trajectory across
training and testing compromise an assessment of generalization to new trajectories?

**Pause here for discussion.** A productive next step is to sketch f_theta and
its loss together before modifying architecture or optimizer settings.
