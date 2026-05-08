# L16.2 : Retrieval-Based Language Models — Part II

> In this lecture, I explored advanced RAG architectures and end-to-end retrieval-augmented models.

https://youtu.be/6Nwm0hA9RDA?si=t-ZLBuRAJCaQSw3C

### Fusion-in-Decoder (FiD)
- Encode each retrieved passage **independently** with the encoder.
- Concatenate all encoded representations for the decoder.
- Scales to many retrieved documents.

### REALM & RAG (Lewis et al.)
- **End-to-end trainable** retrieval + generation.
`$P(y \mid x) = \sum_{z \in \mathcal{Z}} P(z \mid x) P(y \mid x, z)$`
- $z$ = retrieved document, $\mathcal{Z}$ = top-$k$ retrieved docs.

### Advanced RAG Strategies
- **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical document first, then retrieve similar real documents.
- **FLARE**: Actively retrieves when the model is uncertain (low token probabilities).
- **RAPTOR**: Hierarchical document summarization for multi-level retrieval.
- **Re-ranking**: Use a cross-encoder to re-rank retrieved candidates.

[[Next lecture notes-L17.1|L17.1 intro]]
