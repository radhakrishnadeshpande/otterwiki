# L14.2 : Quantization, Pruning & Distillation

> In this lecture, I learned compression techniques that reduce the memory footprint and speed up LLM inference.

https://youtu.be/FKhjxjjupLA?si=LUnr4PDij_RB_40W

### Quantization

Represent weights/activations in **lower bit precision**:
`$w_q = \text{round}\left(\frac{w}{\Delta}\right), \quad \Delta = \frac{w_{max} - w_{min}}{2^b - 1}$`

- **FP32 → INT8**: 4× memory reduction.
- **FP32 → INT4**: 8× memory reduction.
- **GPTQ**: Post-training quantization with second-order information.
- **QLoRA**: Fine-tuning in 4-bit with LoRA adapters.

### Pruning

**Magnitude Pruning:** Remove weights with smallest absolute values.
`$\text{mask}(w) = \begin{cases} 0 & \text{if } |w| < \tau \\ 1 & \text{otherwise} \end{cases}$`

- **Structured pruning**: Remove entire heads/layers.
- **Unstructured pruning**: Remove individual weights (requires sparse hardware).

### Knowledge Distillation

Train a small **student** model to mimic a large **teacher** model:

`$\mathcal{L}_{KD} = (1 - \alpha)\mathcal{L}_{CE}(y, y_{student}) + \alpha \mathcal{L}_{KL}(p_{teacher}^T \| p_{student}^T)$`

**Soft Targets (temperature scaling):**
`$p_i^T = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$`

Higher temperature $T$ produces **softer probability distributions** that carry more information.


[[Next lecture notes-L15.1|L15.1 intro]]
