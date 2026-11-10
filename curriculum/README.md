# MLLab: Modular Machine Learning Curriculum

**Status:** Blueprint v0.1 — curriculum design, not delivered lessons.

## Purpose
Develop practical and mathematical fluency in machine learning, building toward transformers and scientific ML. The intended learner has experience in numerical analysis, Python and some PyTorch; mathematical prerequisites can be reviewed selectively.

## Learning contract
- Each lesson combines a **15–20 minute mini-lecture**, **20–40 minute mini-lab**, **~10 minute diagnostics**, and a **completion checkpoint**.
- Lessons may be marked **skip / review / complete / deepen**. Skipping is an informed choice, not a failure.
- No advancement requirement is based solely on code running. Explain the mechanism, evidence, and failure modes.
- Labs should be reproducible, small, and reusable. Record seeds, data splits, baselines, error measures, and runtime.
- Separate training objective, evaluation metric, and generalization evidence.
- Units are a suggested dependency order, not a rigid calendar.

## Units
| Unit | Focus | Deliverable |
|---|---|---|
| [1](unit-01-foundations.md) | Statistical learning fundamentals | Controlled generalization experiment |
| [2](unit-02-optimization.md) | Optimization and learning dynamics | Optimizer comparison with diagnostics |
| [3](unit-03-neural-networks.md) | Neural-network fundamentals | Verified network and gradient tests |
| [4](unit-04-architectures.md) | CNNs, sequences, attention, transformers | Architecture comparison |
| [5](unit-05-practice.md) | Reproducible ML engineering | Reusable experiment harness |
| [6](unit-06-scientific-ml.md) | ML for differential equations | PDE-linked benchmark and limitations |

## Implementation convention
Use Python/NumPy for transparent baselines; introduce scikit-learn for statistical baselines and PyTorch for automatic differentiation and scalable architectures. Avoid introducing frameworks merely to make a toy example work. Each lab's code and results will be separate future contributions.

## Lesson specification
See [lesson template](lesson-template.md). Lesson IDs (e.g., 2.2) are stable references; detailed lectures, code, and numerical results are deliberately not included in this PR.

## Suggested progression
Start with a short self-assessment against Unit 1 objectives. Skip familiar introductory topics if justified; retain the generalization and diagnostics checkpoints. Then work through optimization and neural-network mechanisms before specializing in transformers or PDE applications.

## Reference
The educational repository [w3cj/how-llms-work](https://github.com/w3cj/how-llms-work) is optional supplementary material, not a dependency or source of code copied into MLLab.
