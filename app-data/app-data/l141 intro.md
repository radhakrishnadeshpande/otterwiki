# L14.1 : Parameter Efficient Fine-Tuning (PEFT)

> In this lecture, I learned how to adapt large pre-trained LLMs to downstream tasks without updating all parameters.

https://youtu.be/SHY034_kMs0?si=R3Mfr2cSzHVYlqL6

- Full fine-tuning is **expensive** (billions of parameters, large memory).
- PEFT methods update only a **small subset** of parameters.

### LoRA (Low-Rank Adaptation)

`$W = W_0 + \Delta W = W_0 + BA$`

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

- Only $A$ and $B$ are trained; $W_0$ is frozen.
- **Trainable parameters**: $r(d + k)$ vs. $dk$ for full fine-tuning.
- At inference: $W_{merged} = W_0 + BA$ — **no additional latency**.

### Prefix Tuning
- Prepend trainable **soft tokens** (prefix) to the input sequence.
- Only the prefix parameters are updated.

### Prompt Tuning
- Similar to prefix tuning but only adds tokens to the **input layer** (not every layer).
- Very parameter-efficient for large models.

### Adapter Layers
- Insert small **bottleneck FFN layers** inside each Transformer block.
- Only adapters are trained.

### Key Formula (LoRA parameter reduction)
`$\text{Compression ratio} = \frac{dk}{r(d+k)} \approx \frac{d}{2r} \text{ (when } d = k\text{)}$`

[[Next lecture notes-L14.2|L14.2 intro]]
