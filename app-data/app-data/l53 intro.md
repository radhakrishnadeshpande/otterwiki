# L5.3 : Neural Language Models — Sequence-to-Sequence and Attention

> In this lecture, I learned about the Seq2Seq architecture and why attention was introduced.

https://youtu.be/K3onh9eo95A?si=aPtp4_4fmHSDVXkh
### Seq2Seq (Encoder-Decoder)
- **Encoder** reads the input and compresses it into a fixed-length **context vector** $c$.
- **Decoder** generates the output sequence from $c$.
- Problem: Fixed-length bottleneck loses information for long sequences.

### Bahdanau Attention (Additive Attention)

`$\alpha_{ts} = \frac{\exp(e_{ts})}{\sum_{s'=1}^{S} \exp(e_{ts'})}$`

`$e_{ts} = v_a^T \tanh(W_a h_{t-1} + U_a \bar{h}_s)$`

`$c_t = \sum_{s=1}^{S} \alpha_{ts} \bar{h}_s$`

- $\alpha_{ts}$ = **attention weight** of decoder step $t$ over encoder step $s$.
- $c_t$ = **context vector** as a weighted sum of encoder hidden states.
- The model **learns where to look** in the input for each output step.
[[Next lecture notes-L6.1|L6.1 intro]]
