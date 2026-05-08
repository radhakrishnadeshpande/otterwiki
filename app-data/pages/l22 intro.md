# L22 : Self-Evolving LLMs

> In this lecture, I learned how LLMs can improve themselves using their own generated data.

https://youtu.be/NXUumIVmgeQ?si=Hu2KoUk9qV9vyMaI

### The Self-Improvement Paradigm
- Use the LLM to **generate training data for itself**.
- Iteratively: Generate → Filter/Verify → Fine-tune → Repeat.

### Self-Play Fine-Tuning (SPIN)
- Generate "fake" data using the current model.
- Train to distinguish **real human data** from **self-generated data**.
- Improves alignment without additional human labels.

### Constitutional AI (Iterative Self-Critique)
1. Model generates a response.
2. Model critiques its response using principles.
3. Model revises based on the critique.
4. Revised data is used for fine-tuning.

### STaR (Self-Taught Reasoner)
- Generate reasoning traces; keep those that lead to correct answers.
- Fine-tune on these verified traces.
`$\mathcal{D}_{new} = \{(x, r, y) : \text{LLM}(x) = r, \text{answer}(r) = y_{correct}\}$`

### ReST (Reinforced Self-Training)
- **Grow step**: Sample many outputs; filter by reward.
- **Improve step**: Fine-tune on filtered data.
- Repeats iteratively.


[[Next lecture notes-L23|L23 intro]]
