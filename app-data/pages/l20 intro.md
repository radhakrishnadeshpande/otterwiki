# L20 : Long Context LLMs — Challenges & Solutions

> In this lecture, I learned the challenges of handling very long sequences and current solutions.

https://youtu.be/VEQU7OQtJws?si=b6ouBCiKZvko_IAz

### The Long Context Problem
- Standard Transformers: $O(n^2)$ attention memory/compute.
- **Context window**: GPT-4 (128K), Claude (200K), Gemini (1M tokens).

### Position Extrapolation Problem
- Models trained on short sequences **struggle on longer sequences** due to out-of-distribution positional encodings.

### RoPE Extension Strategies

**YaRN (Yet another RoPE extensioN):**
`$\theta_i' = \frac{\theta_i}{\text{scale\_factor}^{2i/d}}$`
- Scales RoPE base to extrapolate to longer contexts.

**LongRoPE**: Searches for optimal positional interpolation non-uniformly.

### Sliding Window Attention (Mistral)
- Each token attends only to the **previous $W$ tokens** (window).
- Combined with **rolling KV cache**.

### Long Context Evaluation Benchmarks
- **SCROLLS, LongBench, "Needle in a Haystack"** test.
- The "Needle" test: retrieve a specific piece of information buried in a very long document.



[[Next lecture notes-L21|L21 intro]]
