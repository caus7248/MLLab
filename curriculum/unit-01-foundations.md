# Unit 1 — Foundations of Machine Learning

**Goal:** Distinguish fitting an objective from learning a relationship that generalizes. **Level:** foundational; lessons may be skipped after self-assessment.

| Lesson | Mini-lecture focus | Mini-lab proposal | Diagnostic / checkpoint |
|---|---|---|---|
| 1.1 Learning as estimation | Data-generating process, hypotheses, empirical risk, inductive bias | Fit a known smooth function with controlled noise | Explain training vs population error and assumptions |
| 1.2 Loss and likelihood | MSE, cross-entropy, likelihood, evaluation metrics | Compare squared and robust regression losses under outliers | Show why metric and objective can disagree |
| 1.3 Generalization | Train/validation/test, capacity, leakage, regularization | Polynomial degrees on noisy samples with a held-out test set | Plot training/validation error; detect leakage |
| 1.4 Statistical reliability | Sampling variation, uncertainty, cross-validation, baseline selection | Repeat splits and seeds; summarize error distributions | Explain why a single successful run is weak evidence |

**Unit checkpoint:** Design a clean comparison without tuning on the test set. Explain what can and cannot be inferred from validation performance.

**Artifact planned:** A reproducible regression benchmark and short diagnostic report (future implementation).
