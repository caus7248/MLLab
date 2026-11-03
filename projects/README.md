# ML experiments

Keep each experiment in its own numbered directory, such as `001-function-approximation/`.

Each project should include:

- A README stating the question, mathematical model, and baseline.
- Reproducible setup and run commands, with dependency versions recorded.
- Training and evaluation code, with separate training and test data.
- Metrics, plots, random seeds, and a short interpretation of results.

Record hardware and runtime when comparing CPU and GPU performance. Keep datasets and model weights outside Git; document their source, license, and download procedure.

Possible starting directions (not yet selected):

1. Function approximation: compare a small neural network against polynomial approximation on a known function, including error away from training points.
2. Small image classifier: follow a complete training, validation, and evaluation workflow.
3. Existing language model: establish an inference baseline before experimenting with weights.
