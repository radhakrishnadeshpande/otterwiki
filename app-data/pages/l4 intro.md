# L4.1 : Word Representation — Word2Vec

> In this lecture, I learned how words can be represented as dense vectors using neural methods.

https://youtu.be/VS5N6vuqYfA?si=J_srLe4SnZty6-Oo
- **One-hot encoding** is sparse and has no notion of similarity — replaced by **word embeddings**.
- **Word2Vec** learns dense, low-dimensional vectors that encode semantic similarity.
- Words with similar meanings have **similar vector representations** (measured by cosine similarity).

### Two Architectures of Word2Vec

**CBOW (Continuous Bag of Words):** Predict the center word from context words.
`$\text{Maximize: } P(w_t \mid w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c})$`

**Skip-gram:** Predict context words from the center word.
`$\text{Maximize: } \sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j} \mid w_t)$`

### Softmax Objective (Skip-gram)
`$P(w_O \mid w_I) = \frac{\exp(v_{w_O}^T v_{w_I})}{\sum_{w=1}^{W} \exp(v_w^T v_{w_I})}$`

### Key Tricks
- **Negative Sampling**: Instead of full softmax, sample a few negative words.
- **Subsampling of frequent words**: Downsample very frequent words like "the".

### Properties
- **Analogy reasoning**: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$
- **Cosine similarity** used to measure word closeness.
[[Next lecture notes-L4.2|L4.2 intro]]
