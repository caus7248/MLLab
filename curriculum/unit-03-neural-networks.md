# Unit 3 — Neural-Network Fundamentals

**Goal:** Understand representation, differentiation, and training stability. **Level:** core.

| Lesson | Mini-lecture focus | Mini-lab proposal | Diagnostic / checkpoint |
|---|---|---|---|
| 3.1 Function approximation | Affine layers, depth, width, ReLU/tanh/SiLU | Fit smooth and nonsmooth targets using matched small MLPs | Compare function and derivative errors |
| 3.2 Backpropagation | Chain rule, computational graphs, vector-Jacobian products | Implement a tiny NumPy MLP and compare to PyTorch | Finite-difference gradient checks |
| 3.3 Initialization and gradients | Variance propagation, saturation, vanishing/exploding gradients | Sweep depth and initialization | Inspect gradient norms across layers |
| 3.4 Normalization and regularization | Feature scaling, weight decay, dropout, normalization | Change one intervention at a time | Distinguish training stabilization from generalization |

**Unit checkpoint:** Derive one layer's backward pass and identify why a chosen activation may fail for derivative-based objectives. ReLU's second derivative is zero almost everywhere, an important limitation for strong-form second-order PINNs.

**Artifact planned:** Tested NumPy/PyTorch parity example and activation benchmark.
