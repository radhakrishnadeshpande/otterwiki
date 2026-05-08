# L3.2 : Introduction to Language Models — Advanced Smoothing and Evaluation

> In this lecture, I learned how to handle unseen N-grams using smoothing and how to evaluate language models.

https://youtu.be/0BCs3L4a9pY?si=2cP_FKvYcwkvUu1d

### Smoothing Techniques

**Laplace (Add-1) Smoothing:**
`$P_{add1}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$`

**Add-k Smoothing:**
`$P_{addk}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + kV}$`

**Kneser-Ney Smoothing** (most effective in practice):
`$P_{KN}(w_i \mid w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - d, 0)}{C(w_{i-1})} + \lambda(w_{i-1}) P_{KN}(w_i)$`

### Evaluation: Perplexity

`$\text{Perplexity}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N}$`

`$= \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i \mid w_1, \ldots, w_{i-1})}}$`

- **Lower perplexity = better model**.
- Perplexity measures how "surprised" the model is by a test set.

[[Next lecture notes-L4|L4 intro]]
