# L9 : Tokenization Strategies

> In this lecture, I learned how raw text is converted into tokens, which is the first step for all LLMs.

https://youtu.be/NiFqrK9wv4k?si=xkh2TeD5Seg3T89l

### Why Tokenization Matters
- Models cannot process raw characters directly at scale.
- Tokenization balances **vocabulary size** vs. **sequence length**.

### Byte Pair Encoding (BPE)
1. Start with character-level vocabulary.
2. Iteratively **merge** the most frequent adjacent pair.
3. Repeat until vocabulary size $V$ is reached.
- Used by: **GPT-2, GPT-3, RoBERTa**

### WordPiece
- Similar to BPE but merges pairs that **maximize likelihood** of training data.
- Used by: **BERT**

### SentencePiece
- Language-agnostic; treats input as a **raw character stream** (no pre-tokenization).
- Supports BPE and Unigram LM.
- Used by: **T5, LLaMA, Gemma**

### Unigram Language Model Tokenization
- Starts with a large vocabulary and **prunes** tokens:
`$\mathcal{L} = -\sum_{s \in D} \log P(s), \quad P(s) = \prod_{i} p(x_i)$`
### Key Trade-offs
| Granularity | Vocabulary | Sequence Length |
|-------------|-----------|----------------|
| Character | Small | Very Long |
| Subword (BPE) | Medium | Medium |
| Word | Large | Short |

[[Next lecture notes-L10.1|L10.1 intro]]
