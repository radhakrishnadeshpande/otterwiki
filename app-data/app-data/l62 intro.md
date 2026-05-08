# L6.2 : Introduction to Transformer — Positional Encoding and Layer Normalization

> In this lecture, I learned how Transformers encode position and stabilize training.

https://youtu.be/Y9iYWoUf22Q?si=xDgHlwjEqJSHzara

### Positional Encoding (Sinusoidal)

`$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$`

`$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$`

- Since Transformers have no recurrence, **positional encodings** inject position information.
- Added to input embeddings: $x' = x + PE$

### Feed-Forward Network (FFN) in Each Layer
`$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$`

### Layer Normalization
`$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta$`

- Applied **before** each sub-layer in modern Transformers (Pre-LN variant).
- More stable training than Batch Normalization for sequences.

### Residual Connections
`$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$`

[[Next lecture notes-L7.1|L7.1 intro]]
