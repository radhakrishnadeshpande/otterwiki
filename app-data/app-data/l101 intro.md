# L10.1 : Mixture of Experts — Part I

> In this lecture, I learned about the MoE architecture that enables massive scaling without proportional compute increase.

https://youtu.be/wbG71yjomgY?si=O7Yfhh3XCmnBLMn4

- **Mixture of Experts (MoE)**: Only a subset of "expert" networks is activated for each input token.
- Allows **huge model capacity** (parameters) with manageable **inference FLOPs**.

### MoE Architecture
- Replace the FFN in each Transformer layer with $E$ expert FFNs.
- A **routing network** (gating function) selects the top-$k$ experts per token.

### Gating / Routing

**Top-K Sparse Gating:**
`$G(x) = \text{TopK}(\text{softmax}(x \cdot W_g), k)$`

**Output:**
`$y = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x)$`

### Load Balancing Loss
To prevent **expert collapse** (all tokens routed to same experts):
`$L_{aux} = \alpha \sum_{i=1}^{E} f_i \cdot p_i$`
where $f_i$ = fraction of tokens dispatched to expert $i$, $p_i$ = average routing probability.

[[Next lecture notes-L10.2|L10.2 intro]]
