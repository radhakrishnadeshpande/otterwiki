# L12.1 : Pre-training of Causal LMs and In-context Learning

> In this lecture, I learned how causal LMs are pre-trained and how they generalize via in-context learning.

https://youtu.be/xGL15KimKL4?si=4vfykGH1lQC0TNHQ

### Causal LM Pre-training
- Trained on massive text corpora to predict the next token autoregressively.
- **No task-specific labels needed** — self-supervised.

### In-Context Learning (ICL)
- At inference, provide **examples directly in the prompt** — no gradient updates.
- The model "learns" from examples in the context window.

**Zero-shot:** Task description only, no examples.
**Few-shot:** Task description + $k$ input-output examples in the prompt.

### Why Does ICL Work?
- Hypotheses:
  - Models locate relevant "task" patterns in pre-training data.
  - ICL performs **implicit Bayesian inference** over possible tasks.
  - Attention acts like **gradient descent** in the forward pass.

### Chain-of-Thought (CoT) Prompting
- Provide reasoning steps in examples:
  > "Q: Roger has 5 balls. He buys 2 more cans of 3 balls each. How many? A: Roger starts with 5, buys 2×3=6, so 5+6=**11**."
- Significantly improves reasoning on complex tasks.

[[Next lecture notes-L12.2|L12.2 intro]]
