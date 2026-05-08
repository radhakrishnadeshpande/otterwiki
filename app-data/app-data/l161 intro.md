# L16.1 : Retrieval-Based Language Models — Part I

> In this lecture, I learned how external knowledge retrieval can augment LLMs.

https://youtu.be/JDULo8qMARg?si=QGAy2NaiAeZnqRjx

### Motivation
- LLMs have a **knowledge cutoff** and may hallucinate facts.
- **Retrieval Augmented Generation (RAG)**: Retrieve relevant documents at query time and include them in the context.

### RAG Pipeline
1. **Index**: Embed all documents into a vector store.
2. **Retrieve**: For a query $q$, retrieve top-$k$ relevant documents.
3. **Generate**: Condition the LLM on the retrieved documents + query.

### Dense Retrieval
- Encode query $q$ and document $d$ with encoders $E_Q$, $E_D$:
`$\text{sim}(q, d) = E_Q(q)^T E_D(d)$`
- Use **FAISS** or similar ANN index for fast retrieval.

### Sparse Retrieval (BM25)
`$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d)(k_1 + 1)}{f(t, d) + k_1(1 - b + b \cdot |d|/\text{avgdl})}$`

- $f(t, d)$ = term frequency of term $t$ in document $d$.
- $k_1$ and $b$ are tuning parameters.

[[Next lecture notes-L16.2|L16.2 intro]]
