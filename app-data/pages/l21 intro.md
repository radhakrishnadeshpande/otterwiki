# L21 : Knowledge Editing

> In this lecture, I learned how to update or correct specific knowledge in a trained LLM without full retraining.

- https://youtu.be/UMx1aSFi1xs?si=beO42IW8vfLfk3Nu -21.1

- https://youtu.be/-D5JpS-L-VI?si=KDcBkQjvA65s_ozs -21.2

- https://youtu.be/bMVKbSfwBW4?si=uQVp4fsW8_ipuoHV -21.3

- https://youtu.be/wU4iHrzMNnk?si=K7eUIIn1p3rFe7AL -21.4

### Why Knowledge Editing?
- Facts change over time (e.g., leaders, prices, events).
- Full retraining is expensive.
- Need to correct specific **factual errors** or **toxic content**.

### Desired Properties
- **Reliability**: The edited fact is correctly recalled.
- **Generality**: Related phrasings of the fact are also updated.
- **Locality**: Unrelated facts are not affected.

### Locate-Then-Edit Methods

**ROME (Rank-One Model Editing):**
- Identifies which **MLP layer** stores a fact using causal tracing.
- Edits that layer's weights using a **rank-one update**:
`$W_{new} = W_{old} + \frac{(v^* - W_{old} k)k^T}{k^T k}$`
where $k$ is the key vector and $v^*$ is the new value vector.

**MEMIT**: Extends ROME to edit thousands of facts simultaneously.

### Meta-Learning Based (MEND)
- Trains a **hypernetwork** that takes gradient updates as input and produces targeted weight edits.

### Retrieval-Based Editing
- Store edits in an **external memory**; retrieve at inference time.
- No weight modification needed.



[[Next lecture notes-L22|L22 intro]]
