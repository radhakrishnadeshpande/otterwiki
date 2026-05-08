# L24 : Interpretability — Demystifying the Black-Box LMs

> In this lecture, I learned methods to understand what is happening inside LLMs.

https://youtu.be/IJsigZ7tM94?si=OAYNXZKLqNAmmKXr

### Why Interpretability?
- LLMs are opaque — we don't fully understand **what** they store or **how** they reason.
- Important for: debugging, safety, bias detection, trust.

### Probing Classifiers
- Train a simple classifier on **internal representations** to check if a concept is encoded.
`$f: h_l \rightarrow y_{concept}$`
- High accuracy → the representation encodes that concept.

### Attention Visualization
- Visualize attention weights to see **which tokens the model attends to**.
- Limitation: Attention ≠ explanation (attention can be misleading).

### Causal Tracing (Rome)
- **Corrupt** one token's activations and measure the effect on the output.
- Identifies which **layers and positions** are causally responsible for a fact.

### Circuit Discovery
- Identify **minimal subgraphs** of the network (circuits) responsible for specific behaviors.
- E.g., the "induction head" circuit for in-context learning.

### Sparse Autoencoders (SAE) — Mechanistic Interpretability
- Decompose residual stream activations into **sparse, interpretable features**:
`$x \approx W_{dec} \cdot \text{ReLU}(W_{enc} x + b)$`
- Each feature corresponds to a **human-interpretable concept** (e.g., "Python code", "legal text").

### Logit Lens
- Project intermediate layer activations **directly onto the vocabulary** to see what the model "thinks" at each layer.

### Key Takeaways
- Interpretability is an **active research area** — no complete understanding yet.
- Mechanistic interpretability aims for **circuit-level** understanding.
- Techniques include probing, causal tracing, attention analysis, and SAEs.

[[Next lecture end_course_all|End_Illm]]
