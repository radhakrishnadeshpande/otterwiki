# L7 : Pre-training Strategies

> In this lecture, I learned how LLMs are pre-trained using different objectives.

https://youtu.be/QxFI3T_tvNw?si=VqveRabWGLMZMIt1

### Types of Pre-training Objectives

**Causal Language Modeling (CLM) — used by GPT:**
`$\mathcal{L} = -\sum_{t=1}^{T} \log P(w_t \mid w_1, \ldots, w_{t-1})$`
- Predicts the **next token** (left-to-right/autoregressive).

**Masked Language Modeling (MLM) — used by BERT:**
`$\mathcal{L}_{MLM} = -\sum_{i \in \mathcal{M}} \log P(w_i \mid w_{\backslash \mathcal{M}})$`
- Randomly masks ~15% of tokens; model predicts them using **both left and right context**.

**Next Sentence Prediction (NSP) — used by BERT:**
- Binary classification: Is sentence B the actual next sentence after sentence A?

**Span Corruption — used by T5:**
- Replaces contiguous spans of tokens with sentinel tokens; model predicts the original spans.

### Key Models
| Model | Architecture | Objective |
|-------|-------------|-----------|
| GPT | Decoder-only | CLM |
| BERT | Encoder-only | MLM + NSP |
| T5 | Encoder-Decoder | Span Denoising |
| XLNet | Decoder (permutation) | Permutation LM |
[[Next lecture notes-L8.1|L8.1 intro]]
