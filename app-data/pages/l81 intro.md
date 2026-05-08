# L8.1 : Advanced Attention Mechanisms — Part I

> In this lecture, I learned about efficient attention variants designed to handle long sequences.

https://youtu.be/nH30ZshilMM?si=QL_ui9ETSV-oEdC7
### Problem with Standard Attention
- Complexity: $O(n^2)$ in both **time and memory** w.r.t. sequence length $n$.
- Becomes infeasible for very long sequences.

### Sparse Attention
- Attend only to a **subset** of positions instead of all.
- E.g., **local attention** (fixed window), **strided attention**, **global tokens**.

### Linear Attention
- Approximates full attention in $O(n)$ using kernel trick:
`$\text{Attention}(Q, K, V) \approx \phi(Q)(\phi(K)^T V)$`
where `$\phi$` is a feature map approximating the softmax kernel.

### Longformer Attention
- Combines **sliding window attention** (local) + **global attention** (for special tokens).
- Complexity: `$O(n \cdot w)$` where $w$ is the window size.

### BigBird Attention
- Uses **random attention + local window + global tokens**.
- Proven to be a **universal approximator** and Turing complete.

[[Next lecture notes-L8.2|L8.2 intro]]
