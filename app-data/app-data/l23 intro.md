# L23 : An Alternate Formulation of Transformers

> In this lecture, I explored alternative mathematical frameworks for understanding Transformers.

https://youtu.be/WdMQelGUZdU?si=lTx1fzlz4PkkIA3V

### Linear Attention / Kernel Reformulation

Standard attention: $\text{Attn}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d})V$

With kernel decomposition $\text{softmax}(qk^T) \approx \phi(q)\phi(k)^T$:
`$\text{Attn}(Q,K,V) \approx \phi(Q)(\phi(K)^TV) \quad \text{— computable in } O(n)$`

### Transformers as RNNs (SSM Perspective)
- Linear attention can be re-expressed as a **recurrent model**:
`$s_t = s_{t-1} + k_t^T v_t \quad \text{(state update)}$`
`$o_t = q_t s_t \quad \text{(output)}$`
- This connects Transformers to **State Space Models (SSMs)** like Mamba.

### Mamba (Selective SSM)
`$h_t = \bar{A} h_{t-1} + \bar{B} x_t$`
`$y_t = C h_t$`
- Parameters $\bar{A}, \bar{B}, C$ are **input-dependent** (selective).
- Linear complexity; strong alternative to Transformer for sequences.

### Attention as Associative Memory
- The attention operation can be viewed as **querying a Hopfield network**.
- Key-value pairs form an associative memory; queries retrieve stored patterns.



[[Next lecture notes-L24|L24 intro]]
