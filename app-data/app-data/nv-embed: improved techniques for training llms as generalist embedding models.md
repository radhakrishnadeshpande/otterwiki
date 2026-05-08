# NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models

> **arXiv:** [arXiv:2405.17428](https://arxiv.org/abs/2405.17428)
> **PDF:** [View PDF](https://arxiv.org/pdf/2405.17428)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2405.17428v3)
> **ICLR 2025 Proceedings:** [Paper Page](https://proceedings.iclr.cc/paper_files/paper/2025/file/c4bf73386022473a652a18941e9ea6f8-Paper-Conference.pdf)
> **OpenReview:** [ICLR 2025 Submission](https://openreview.net/forum?id=lgsyLSsDRe)
> **HuggingFace (v1):** [nvidia/NV-Embed-v1](https://huggingface.co/nvidia/NV-Embed-v1)
> **HuggingFace (v2):** [nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)
> **HuggingFace Papers:** [HF Paper Page](https://huggingface.co/papers/2405.17428)
> **Semantic Scholar:** [Paper Details](https://www.semanticscholar.org/paper/NV-Embed:-Improved-Techniques-for-Training-LLMs-as-Lee-Roy/cde48b24264e44355abb0e548a2cf7c70bb072b4)


---

## Introduction

Text embedding models — which map variable-length text sequences into fixed-dimensional dense vectors — are a foundational building block of modern information retrieval, semantic search, and RAG systems. For years, this landscape was dominated by bidirectional encoder models (BERT, T5, E5) that use masked self-attention to produce rich contextual representations. The encoder architecture is naturally suited to embedding: every token can attend to every other token, producing holistic sequence representations. Decoder-only LLMs (GPT, Mistral, LLaMA), while far superior on generative tasks, were considered structurally ill-suited for embedding — their causal (unidirectional) attention mask, designed for autoregressive generation, prevents each token from attending to future tokens, fundamentally limiting the representational power of any pooled embedding derived from such a model.

NV-Embed challenges this assumption with three targeted architectural and training interventions. First, the causal attention mask is removed during contrastive training, enabling full bidirectional attention for embedding tasks. Second, a **latent attention layer** — a compact cross-attention module with learnable latent query vectors — replaces mean pooling and EOS-token embedding as the pooling mechanism, producing richer, more expressive sequence representations. Third, a **two-stage contrastive instruction-tuning** procedure sequentially specializes the model on retrieval tasks before generalizing it to the full spectrum of embedding tasks. The result — NV-Embed-v1 and NV-Embed-v2 — achieved No. 1 on the MTEB leaderboard across 56 tasks, using only publicly available training data with no proprietary GPT-4 synthetic data.

---

## Problem Statement

1. **Causal attention limits decoder LLM embedding quality** — the unidirectional attention mask of decoder-only LLMs prevents full contextual interaction across tokens, constraining representation quality for embedding tasks.
2. **Mean pooling and EOS-token embedding are inadequate** — naive pooling strategies that reduce a sequence of token embeddings to a single vector lose structural and positional information critical for fine-grained retrieval.
3. **Task conflict between retrieval and non-retrieval objectives** — training jointly on retrieval tasks (which benefit from in-batch negatives) and non-retrieval tasks (classification, clustering, STS) can produce "zigzag" gradient updates that degrade both task types.
4. **Dependency on proprietary synthetic data** — prior SOTA embedding models (E5-mistral, SFR-Embedding) relied on GPT-4-generated training data unavailable to the research community, limiting reproducibility.

---

## Contributions

1. **Proposes a latent attention layer** as a novel pooling mechanism for decoder-only LLMs, consistently outperforming mean pooling and EOS-token embedding across all MTEB task categories.
2. **Removes the causal attention mask** during contrastive training to enable bidirectional attention and richer token interactions for embedding.
3. **Introduces two-stage contrastive instruction-tuning** — Stage 1 on retrieval tasks with in-batch and hard negatives; Stage 2 blending non-retrieval datasets — with well-blended batching to eliminate zigzag gradient updates.
4. **Achieves MTEB No. 1 using only public data** — no GPT-4 synthetic data required, establishing a new reproducibility benchmark.
5. **Provides model compression analysis** — demonstrates that pruning + quantization of NV-Embed-v2 outperforms smaller embedding models (LLaMA-3.2-3B, Qwen2.5-3B, Minitron-4B) while reducing deployment cost.

---

## Architecture: Latent Attention Layer

### Problem with Existing Pooling Strategies

```
Mean Pooling:
  e = (1/T) Σ h_t    ← averages all token hidden states equally
  ⚠️ Treats all tokens as equally important; loses positional structure

EOS Token Embedding:
  e = h_T            ← uses only the final token's hidden state
  ⚠️ Decoder LLMs optimize h_T for next-token prediction, not embedding

Latent Attention Layer (NV-Embed):
  e = CrossAttention(Q=Z, K=H, V=H)   ← trainable latent queries Z attend over H
  ✅ Learns task-relevant aggregation; flexible and expressive
```

### Latent Attention Layer (Formal)
```math
Let \mathbf{H} = [h_1, h_2, \ldots, h_T] \in \mathbb{R}^{T \times d} be the hidden states from the LLM for an input of T tokens.
```
A set of M trainable latent query vectors \mathbf{Z} \in \mathbb{R}^{M \times d} attends over \mathbf{H} via cross-attention:

```math
\mathbf{A} = \text{softmax}\!\left(\frac{\mathbf{Z}\,\mathbf{W}_Q \cdot (\mathbf{H}\,\mathbf{W}_K)^\top}{\sqrt{d_k}}\right)
```
```math
\mathbf{E} = \mathbf{A} \cdot \mathbf{H}\,\mathbf{W}_V \in \mathbb{R}^{M \times d}
```
The final pooled embedding is obtained by flattening or mean-pooling over the M latent output vectors:
```math
\mathbf{e} = \text{MLP}(\text{flatten}(\mathbf{E})) \in \mathbb{R}^{d_{\text{emb}}}
```
where \mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V are projection matrices and d_k is the key dimension. Unlike standard mean pooling, \mathbf{Z} is learned during training, allowing the model to adaptively focus on relevant tokens.

### Causal Mask Removal

Standard decoder attention for position i:
```math
a_{ij} = \begin{cases} \text{softmax}(q_i^\top k_j / \sqrt{d}) & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
```
NV-Embed removes the future-masking constraint during embedding:
```math
a_{ij} = \text{softmax}(q_i^\top k_j / \sqrt{d}) \quad \forall\, j \in [1, T]
```
This allows each token to attend to all others bidirectionally, matching the representational capacity of encoder models like BERT.

---

## Training: Two-Stage Contrastive Instruction-Tuning

### Contrastive Training Objective (InfoNCE)

For a query q with positive passage p^+ and N-1 in-batch negatives \{p^-_i\}_{i=1}^{N-1}:
```math
\mathcal{L}_{\text{contrast}} = -\log \frac{e^{\text{sim}(q,\, p^+)/\tau}}{\displaystyle e^{\text{sim}(q,\, p^+)/\tau} + \sum_{i=1}^{N-1} e^{\text{sim}(q,\, p^-_i)/\tau}}

where \text{sim}(q, p) = \dfrac{\mathbf{e}_q^\top \mathbf{e}_p}{\|\mathbf{e}_q\| \|\mathbf{e}_p\|} is cosine similarity and \tau is temperature.
```

### Hard Negative Mining (Positive-Aware)

Standard hard negative mining retrieves passages with high query similarity regardless of positive relevance — risking false negatives. NV-Embed applies **positive-aware hard-negative mining** from NV-Retriever:
```math
\text{HardNeg}(q) = \{p : \text{sim}(q,p) \text{ is high} \;\land\; \text{relevance}(q,p) < \text{relevance}(q, p^+)\}
```
Passages with positive relevance scores above a threshold are excluded from the negative pool, reducing false negative contamination.

### Two-Stage Training Pipeline

```
Stage 1 — Retrieval-Focused Contrastive Training
  Data:       Retrieval datasets only (MSMARCO, HotpotQA, NQ, etc.)
  Negatives:  In-batch negatives + positive-aware hard negatives
  Batching:   Well-blended cross-task batches (not task-homogeneous)
  Objective:  InfoNCE contrastive loss with instruction prefixes
  Purpose:    Learn strong dense retrieval representations

Stage 2 — Non-Retrieval Blending
  Data:       Stage 1 retrieval data + classification, clustering, STS, pair-classification
  Negatives:  Same hard-negative strategy
  Batching:   Blended across all task types (avoids zigzag gradient updates)
  Purpose:    Generalize to full MTEB task spectrum without sacrificing retrieval quality
```

> **Why blended batching?** Task-homogeneous batching (used by SFR-Embedding-Mistral) constructs each batch from a single task type. This can produce "zigzag" gradient updates — updates from one task undo progress on another. Well-blended batches produce smoother optimization and higher final MTEB scores.

---

## Experimental Setup

| Property | Details |
|----------|---------|
| **Base LLM** | Mistral-7B |
| **Embedding Dimension** | 4096 |
| **Benchmark** | MTEB (56 tasks) · BEIR (15 retrieval tasks) · AIR Benchmark |
| **Training Data** | Publicly available datasets only — no GPT-4 synthetic data |
| **Task Types** | Retrieval · Reranking · Classification · Clustering · STS · PairClassification |
| **Evaluation Protocol** | Zero-shot (no task-specific fine-tuning at eval) |

---

## Key Results

| Model | MTEB Score (56 tasks) | BEIR Score | Date |
|-------|----------------------|------------|------|
| NV-Embed-v1 | **69.32** (No. 1) | — | May 24, 2024 |
| NV-Embed-v2 | **72.31** (No. 1, Retrieval: 62.65) | — | Aug 30, 2024 |
| E5-mistral-7B | ~66.6 | — | Prior SOTA |
| LLM2Vec | 65.01 | — | Public-data prior SOTA |
| bge-en-icl | 71.24 | — | ICL-augmented baseline |

- **NV-Embed-v2 No. 1 on MTEB Retrieval** (15 tasks, score 62.65) — critical for RAG applications.
- **No. 1 on AIR Benchmark Long Doc section** — out-of-domain retrieval generalization.
- **Second-highest on AIR Benchmark QA section**.
- **Model compression**: Pruned + quantized NV-Embed-v2 outperforms LLaMA-3.2-3B, Qwen2.5-3B, and Minitron-4B embedding models at smaller deployment sizes.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **English-only** | NV-Embed is trained and evaluated exclusively on English tasks; multilingual embedding capability is not addressed. |
| **Mistral-7B backbone dependency** | The architecture and training procedure are validated on Mistral-7B; transferability to other decoder backbones (LLaMA, Qwen, Gemma) is not fully evaluated. |
| **Latent query count sensitivity** | The number of latent query vectors $M$ is a design hyperparameter whose sensitivity is not extensively ablated in the paper. |
| **Inference overhead vs. mean pooling** | The latent attention layer adds a cross-attention forward pass over the latent queries at inference time — minor but nonzero overhead compared to mean pooling or EOS-token extraction. |
| **MTEB leaderboard drift** | MTEB rankings change continuously; newer models (e.g., KaLM-Embedding-V2) have since surpassed NV-Embed-v2 on some tasks. |

---

## Companion Paper

> **NV-Retriever** (Moreira et al., 2024) — [*Improving Text Embedding Models with Effective Hard-Negative Mining*](https://arxiv.org/abs/2407.15831)
> Introduces the **positive-aware hard-negative mining** technique applied in NV-Embed-v2. Provides detailed ablation of false-negative filtering strategies and their effect on retrieval accuracy across BEIR benchmarks.

---

## Quick Start (Inference)

```python
# Install dependencies
# pip install transformers torch sentence-transformers

import torch
from transformers import AutoTokenizer, AutoModel

model_name = "nvidia/NV-Embed-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# Instruction prefixes (required for retrieval tasks)
query_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
passage_prefix = ""

queries = ["What is retrieval-augmented generation?"]
passages = [
    "Retrieval-augmented generation (RAG) is a technique that combines information retrieval with text generation.",
    "The Eiffel Tower is located in Paris, France."
]

# Tokenize
def embed(texts, prefix="", max_length=512):
    texts = [prefix + t for t in texts]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**encoded)
    # NV-Embed uses the last hidden state mean pooling by default via trust_remote_code
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # L2 normalize
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings

query_embs = embed(queries, prefix=query_prefix)
passage_embs = embed(passages, prefix=passage_prefix)

# Cosine similarity (already normalized → dot product)
scores = (query_embs @ passage_embs.T)
print("Similarity scores:", scores)
# → tensor([[0.82, 0.11]])  (first passage is highly relevant)
```

### Using with Sentence-Transformers

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nvidia/NV-Embed-v2", trust_remote_code=True)

query_instruction = "Instruct: Given a web search query, retrieve relevant passages\nQuery: "

queries = [query_instruction + "What is RAG?"]
passages = [
    "RAG combines retrieval with generation for knowledge-intensive tasks.",
    "Python is a popular programming language."
]

query_embs = model.encode(queries, normalize_embeddings=True)
passage_embs = model.encode(passages, normalize_embeddings=True)

scores = query_embs @ passage_embs.T
print(scores)
```

### Task-Specific Instruction Prefixes

NV-Embed uses instruction prefixes to condition embeddings on task type. Key examples from the paper:

```python
INSTRUCTIONS = {
    # Retrieval (asymmetric: query gets instruction, passage does not)
    "retrieval_query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
    "retrieval_passage": "",  # No prefix for passages

    # Semantic Textual Similarity (symmetric)
    "sts": "Instruct: Retrieve semantically similar text\nQuery: ",

    # Classification
    "classification": "Instruct: Classify the sentiment of the text\nQuery: ",

    # Clustering
    "clustering": "Instruct: Identify the topic or theme of the given text\nQuery: ",
}
```

> For the full instruction table across all 56 MTEB tasks, see Table 7 in the paper or `instructions.json` in the HuggingFace model repository.

---

## Citation

```bibtex
@inproceedings{lee2025nvembed,
  title     = {{NV-Embed}: Improved Techniques for Training {LLM}s as Generalist Embedding Models},
  author    = {Chankyu Lee and Rajarshi Roy and Mengyao Xu and Jonathan Raiman and Mohammad Shoeybi and Bryan Catanzaro and Wei Ping},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025},
  url       = {https://openreview.net/forum?id=lgsyLSsDRe}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training (Wang et al., 2022) | [arXiv:2212.03533](https://arxiv.org/abs/2212.03533) |
| E5-mistral: Improving Text Embeddings with LLMs (Wang et al., 2023) | [arXiv:2401.00368](https://arxiv.org/abs/2401.00368) |
| LLM2Vec: Large Language Models as Text Encoders (BehnamGhader et al., 2024) | [arXiv:2404.05961](https://arxiv.org/abs/2404.05961) |
| NV-Retriever: Hard-Negative Mining for Embedding Models (Moreira et al., 2024) | [arXiv:2407.15831](https://arxiv.org/abs/2407.15831) |
| MTEB: Massive Text Embedding Benchmark (Muennighoff et al., 2022) | [arXiv:2210.07316](https://arxiv.org/abs/2210.07316) |
| Dense Passage Retrieval for Open-Domain QA (Karpukhin et al., 2020) | [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) |
| SimCSE: Simple Contrastive Learning of Sentence Embeddings (Gao et al., 2021) | [arXiv:2104.08821](https://arxiv.org/abs/2104.08821) |
| Sentence-BERT: Sentence Embeddings using Siamese BERT (Reimers & Gurevych, 2019) | [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) |
| MIRACL: Multilingual Retrieval (Zhang et al., 2023) | [arXiv:2210.09984](https://arxiv.org/abs/2210.09984) |
