# L8.2 : Advanced Attention Mechanisms — Part II

> In this lecture, I continued learning about advanced attention mechanisms, focusing on KV cache and grouped query attention.

https://youtu.be/wKeVkVMCS6U?si=gzzLTNFKrNWUe7xe

### Multi-Query Attention (MQA)
- Multiple query heads share a **single key-value head**.
- Reduces **KV cache memory** significantly.

### Grouped Query Attention (GQA)
- A middle ground: multiple queries share a **group** of KV heads.
`$\text{Num KV heads} = \frac{\text{Num Q heads}}{G}$`
- Used in **LLaMA 2, Mistral** etc. — balances quality and efficiency.

### Rotary Positional Embedding (RoPE)
- Encodes position by **rotating** the Query and Key vectors:
`$q_m^T k_n = \text{Re}[(W_q x_m) \odot e^{im\theta}]^* [(W_k x_n) \odot e^{in\theta}]$`
- Decays with relative distance, capturing relative positions naturally.
- Used in **GPT-NeoX, LLaMA, PaLM**.

### ALiBi (Attention with Linear Biases)
- Adds a **linear bias** based on relative distance to attention scores.
`$\text{softmax}(q_i k^T + m \cdot [-(i-1), \ldots, -1, 0])$`
- Extrapolates well to longer sequences at inference.

[[Next lecture notes-L9|L9 intro]]
