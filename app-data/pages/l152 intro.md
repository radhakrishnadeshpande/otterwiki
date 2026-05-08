# L15.2 : Efficient LLM Decoding — Part II

> In this lecture, I learned advanced decoding strategies to reduce latency and computational cost.

https://youtu.be/JVUWCv_vMFU?si=Q-N4yWCA1N_ZLEIz

### Speculative Decoding
- Use a **small draft model** to generate $k$ tokens quickly.
- Verify all $k$ tokens with the large model **in parallel** in one forward pass.
- Accept tokens up to the first mismatch; retry from there.
- Reduces wall-clock time without changing output distribution.

**Acceptance probability:**
`$\alpha = \min\left(1, \frac{p(x)}{q(x)}\right)$`
where $p$ = large model prob, $q$ = draft model prob.

### Continuous Batching
- Dynamically add new requests to a batch as old requests finish.
- Maximizes **GPU utilization** vs. static batching.

### PagedAttention (vLLM)
- Manages KV cache using **virtual memory paging** (like OS memory).
- Eliminates memory fragmentation; enables higher batch sizes.

### Flash Attention
- Reorders attention computation to be **IO-bound aware**:
  - Tiles the computation to fit in **SRAM** (fast) rather than HBM (slow).
  - Reduces HBM reads/writes from $O(n^2)$ to $O(n)$.

[[Next lecture notes-L16.1|L16.1 intro]]
