# Unit 4 — Architectures and Inductive Bias

**Goal:** Match architecture to structure in the input and prediction task. **Level:** core to advanced.

| Lesson | Mini-lecture focus | Mini-lab proposal | Diagnostic / checkpoint |
|---|---|---|---|
| 4.1 Convolution | Locality, translation equivariance, receptive field | MLP vs CNN on small image patterns | Test shifted inputs and parameter efficiency |
| 4.2 Sequences | Autoregression, recurrence, context limitations | Compare sequence baselines on synthetic dependencies | Evaluate longer-than-training sequences |
| 4.3 Attention | Queries, keys, values, masking, positional information | Implement small scaled dot-product attention | Check mask behavior and permutation sensitivity |
| 4.4 Transformer blocks | Residuals, normalization, feed-forward blocks, causal prediction | Train a tiny character/token predictor | Compare training and held-out loss; inspect generation |
| 4.5 Architecture selection | CNN, recurrent, transformer, and simpler baselines | Compare two architectures on one controlled task | Explain whether added complexity is justified |

**Unit checkpoint:** Select an architecture for a task and defend its inductive bias, data requirements, and computational tradeoffs.

**Optional reference:** w3cj/how-llms-work for tokenization, embeddings, and a teaching-scale transformer.
