# L6.1 : Introduction to Transformer — Self-Attention and Multi-Head Attention

> In this lecture, I learned the core building block of the Transformer: self-attention.

https://youtu.be/RI1YX4xgAAI?si=al1dVUPQjJzJB3E7

### Scaled Dot-Product Attention

`$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$`

- $Q$ = Queries, $K$ = Keys, $V$ = Values (all derived from the same input).
- Scaling by $\sqrt{d_k}$ prevents extremely small gradients in softmax.

### Multi-Head Attention

`$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$`

`$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$`

- Each head attends to **different subspaces** of the representation.
- Allows the model to capture **multiple types of relationships** simultaneously.

### Advantages Over RNNs
- **Parallelizable** — all positions processed simultaneously.
- Captures **long-range dependencies** directly via attention.
- No vanishing gradient problem across positions.
[[Next lecture notes-L6.2|L6.2 intro]]
