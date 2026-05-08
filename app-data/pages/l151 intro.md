# L15.1 : Efficient LLM Decoding — Part I

> In this lecture, I learned about the inference bottleneck in LLMs and strategies for faster decoding.

https://youtu.be/Ry8Bv2otZV0?si=MqRFztIw9iggmCaG

### Autoregressive Decoding Bottleneck
- LLMs generate one token at a time: $O(n^2)$ memory for KV cache over sequence length $n$.
- **KV Cache**: Store past key-value pairs to avoid recomputation.

### KV Cache Memory
`$\text{KV Cache Size} = 2 \times n_{layers} \times n_{heads} \times d_{head} \times L \times \text{bytes per element}$`

where $L$ = sequence length.

### Greedy Decoding
`$w_t = \arg\max_{w} P(w \mid w_1, \ldots, w_{t-1})$`

### Beam Search
- Keep top-$k$ hypotheses at each step; expand each and keep top-$k$ again.
- **Score**: $\log P(w_1, \ldots, w_t) = \sum_{i=1}^{t} \log P(w_i \mid w_{<i})$

### Sampling Methods

**Temperature Sampling:**
`$P'(w_i) = \frac{\exp(\log P(w_i)/T)}{\sum_j \exp(\log P(w_j)/T)}$`

**Top-k Sampling:** Sample from top-$k$ most probable tokens.

**Top-p (Nucleus) Sampling:** Sample from the smallest set of tokens whose cumulative probability ≥ $p$.

[[Next lecture notes-L15.2|L15.2 intro]]
