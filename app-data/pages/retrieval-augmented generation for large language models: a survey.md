# Retrieval-Augmented Generation for Large Language Models: A Survey

> **arXiv:** [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
> **PDF:** [View PDF](https://arxiv.org/pdf/2312.10997)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2312.10997v5)
> **HuggingFace Papers:** [HF Paper Page](https://huggingface.co/papers/2312.10997)
> **Semantic Scholar:** [Paper Details](https://www.semanticscholar.org/paper/Retrieval-Augmented-Generation-for-Large-Language-A-Gao-Xiong/46f9f7b8f88f72e12cbdb21e3311f995eb6e65c5)


---


## Introduction

Large Language Models have demonstrated remarkable capabilities across NLP tasks, but their deployment in real-world, knowledge-intensive settings reveals fundamental limitations: they hallucinate facts not in their parametric memory, cannot incorporate knowledge acquired after their training cutoff, and produce outputs whose factual basis is opaque and difficult to verify. These are not incidental failure modes — they are structural consequences of relying entirely on frozen parametric weights for knowledge storage.

Retrieval-Augmented Generation (RAG) addresses these limitations by decoupling knowledge storage from model parameters. Rather than encoding all required knowledge at training time, a RAG system retrieves relevant information from an external, updateable knowledge base at inference time and supplies it as context for generation. This approach allows models to remain current without retraining, produce verifiable outputs grounded in cited sources, and handle domain-specific knowledge unavailable during pretraining.

This survey is the first comprehensive review of RAG systems in the LLM era. It introduces the now-standard **Naive RAG → Advanced RAG → Modular RAG** progression as a unifying taxonomy, systematically reviews the tripartite technical foundation (Retrieval, Generation, Augmentation), surveys evaluation frameworks, and outlines open challenges. It has become the foundational reference for the RAG research community, widely cited as the standard entry point to the field.

---

## Problem Statement

1. **Hallucination** — LLMs generate plausible but factually incorrect content when the required knowledge is absent or weak in their parametric memory.
2. **Knowledge staleness** — parametric knowledge is frozen at training cutoff; models cannot incorporate events, facts, or domain updates that postdate pretraining without retraining.
3. **Non-transparent reasoning** — purely parametric generation provides no mechanism for users to verify the factual basis of outputs or trace claims to sources.
4. **Domain adaptation cost** — fine-tuning large models for every domain or knowledge update is computationally prohibitive; a retrieval-based approach offers a lower-cost alternative.

---

## Contributions

1. **Introduces the Naive / Advanced / Modular RAG taxonomy** — a three-tier progression that has become the standard organizing framework for the RAG research community.
2. **Systematic review of the tripartite RAG foundation**: Retrieval, Generation, and Augmentation techniques with their respective state-of-the-art methods.
3. **RAG technology tree** tracing RAG research across pre-training, fine-tuning, and inference stages of LLM development.
4. **Evaluation framework and benchmark survey** — covers metrics, datasets, and automated evaluation tools for RAG systems.
5. **Open challenges and future directions** identifying the key unsolved problems in RAG system design.

---

## The Three RAG Paradigms

### Paradigm 1 — Naive RAG

The earliest and simplest RAG pattern, widely deployed before 2023:

```
Pipeline:
  1. Indexing:   Chunk documents → embed with encoder → store in vector DB
  2. Retrieval:  Embed query → retrieve top-K chunks via ANN search
  3. Generation: Concatenate [query + retrieved chunks] → feed to LLM → output answer

Key Properties:
  ✅ Simple to implement; works well on factual QA with short queries
  ❌ Retrieval precision depends entirely on query-chunk embedding alignment
  ❌ No feedback loop — retrieval errors propagate silently to generation
  ❌ Chunking is naive (fixed-size or sentence-level); semantic coherence not guaranteed
  ❌ Context window stuffing: all top-K chunks concatenated regardless of relevance
```

**Failure modes:** Lost-in-the-middle effect (relevant content at middle of context is ignored), hallucination despite retrieval (when retrieved chunks are irrelevant or insufficient), and over-retrieval noise.

---

### Paradigm 2 — Advanced RAG

Addresses Naive RAG's failure modes through targeted improvements at each pipeline stage:

```
Pre-Retrieval Enhancements:
  ├── Query rewriting / expansion        (HyDE, query2doc, step-back prompting)
  ├── Query decomposition                (break multi-hop queries into sub-questions)
  └── Metadata filtering                 (filter by date, domain, source before retrieval)

Retrieval Enhancements:
  ├── Hybrid retrieval                   (sparse BM25 + dense vector retrieval combined)
  ├── Re-ranking                         (cross-encoder re-ranker on top-K candidates)
  ├── Iterative / multi-step retrieval   (retrieve → reason → retrieve again)
  └── Parent document retrieval          (retrieve small chunks; expand to parent context)

Post-Retrieval Enhancements:
  ├── Context compression                (remove irrelevant sentences from retrieved chunks)
  ├── Context reordering                 (address lost-in-the-middle by reordering chunks)
  └── Source attribution                 (link claims in output back to retrieved passages)
```

---

### Paradigm 3 — Modular RAG

The most flexible and extensible paradigm — RAG as a composable system of interchangeable modules:

```
Core Modules (each independently replaceable):
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │    Search    │ → │    Memory    │ → │    Fusion    │ → │  Generation  │
  │   Module     │   │   Module     │   │   Module     │   │   Module     │
  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
         ↑                                      ↑
  ┌──────────────┐                    ┌──────────────┐
  │    Router    │                    │   Re-ranking │
  │   Module     │                    │   Module     │
  └──────────────┘                    └──────────────┘

Enables:
  ✅ Adaptive retrieval (decide whether to retrieve at all — routing)
  ✅ Multiple knowledge sources (web, DB, KG, structured + unstructured)
  ✅ Plug-and-play module replacement without retraining the core LLM
  ✅ Pipeline orchestration frameworks (LangChain, LlamaIndex, DSPy)
```

---

## Tripartite Technical Foundation

### 1. Retrieval Techniques

| Method | Type | Description |
|--------|------|-------------|
| BM25 | Sparse | TF-IDF-based lexical matching; fast and robust for exact keyword queries |
| DPR / bi-encoder | Dense | Dual encoder; query and passage in shared embedding space |
| Hybrid retrieval | Sparse + Dense | Combines BM25 and dense retrieval scores; best of both worlds |
| HyDE | Query expansion | Generate hypothetical document → embed it → retrieve similar real documents |
| Iterative retrieval | Multi-step | Alternate reasoning and retrieval; each reasoning step generates new queries |
| Knowledge Graph | Structured | Retrieve relational triples; supports multi-hop logical queries |

### 2. Generation Techniques

| Technique | Description |
|-----------|-------------|
| In-context RAG | Retrieved passages prepended to prompt; no model modification required |
| RAG Fine-tuning | LLM fine-tuned to better utilize retrieved context (e.g., REALM, Atlas) |
| Retrieval-aware pretraining | Retrieval integrated during pretraining (RETRO architecture) |
| Generator with attribution | LLM trained to cite sources inline (Self-RAG, RARR) |

### 3. Augmentation Techniques

| Stage | Examples |
|-------|---------|
| Pre-retrieval | Query expansion, query decomposition, hypothetical document embedding (HyDE) |
| In-retrieval | Re-ranking, hybrid fusion (RRF), maximal marginal relevance (MMR) |
| Post-retrieval | Context compression, RECOMP, Selective Context, context reordering |

---

## RAG Evaluation Framework

### Evaluation Dimensions

| Dimension | Metrics | Tools |
|-----------|---------|-------|
| **Retrieval Quality** | Recall@K, MRR, NDCG | BEIR, MTEB |
| **Generation Faithfulness** | Faithfulness score, NLI entailment | RAGAS, TruLens |
| **Answer Relevance** | Answer relevance score, ROUGE, BLEU | RAGAS, ARES |
| **Context Precision** | What fraction of retrieved context is used | RAGAS |
| **Context Recall** | What fraction of required context was retrieved | RAGAS |
| **End-to-End QA Accuracy** | Exact match, F1 | KILT, Natural Questions |

### Key Evaluation Benchmarks

| Benchmark | Type | Link |
|-----------|------|------|
| [Natural Questions (NQ)](https://ai.google.com/research/NaturalQuestions) | Open-domain QA | Google Research |
| [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) | Trivia QA | UW NLP |
| [HotpotQA](https://hotpotqa.github.io/) | Multi-hop QA | Stanford / CMU |
| [KILT](https://arxiv.org/abs/2009.02252) | Knowledge-intensive tasks | Facebook AI |
| [BEIR](https://arxiv.org/abs/2104.08663) | Retrieval generalization | Multiple domains |
| [MS MARCO](https://microsoft.github.io/msmarco/) | Passage retrieval | Microsoft |
| [RAGAS](https://arxiv.org/abs/2309.15217) | RAG-specific evaluation | Automated framework |

---

## Key Arguments

- **RAG vs. Fine-tuning is a false dichotomy** — RAG and fine-tuning address different failure modes. RAG supplies missing factual knowledge; fine-tuning adapts style, format, and task-specific behavior. Combining both is often optimal.
- **Chunking strategy is a bottleneck** — naive fixed-size chunking degrades retrieval quality on complex documents; semantic, hierarchical, or task-aware chunking yields substantial improvements.
- **Retrieval precision matters more than recall for generation quality** — irrelevant retrieved content actively degrades generation by introducing noise; high-precision retrieval with re-ranking typically outperforms high-recall retrieval without filtering.
- **Context window position effects are significant** — the "lost in the middle" phenomenon shows LLM attention is biased toward the beginning and end of context; retrieved passage placement within the prompt affects generation quality.
- **Evaluation frameworks lag behind system complexity** — end-to-end QA accuracy metrics do not capture retrieval quality, faithfulness, or attribution quality independently.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Coverage cutoff** | Revised March 2024; significant post-2024 advances — agentic RAG, RL-trained retrievers, reasoning-integrated RAG, multimodal RAG — are not covered |
| **Evaluation standardization** | The survey calls for but does not resolve the lack of standardized evaluation protocols across retrieval quality, faithfulness, and generation accuracy |
| **Retrieval latency at scale** | RAG introduces retrieval latency; for real-time applications, this overhead must be managed through caching, async retrieval, or approximate indexing |
| **Chunking and indexing are underspecified** | The survey acknowledges chunking as important but does not provide a systematic treatment of chunking strategies (see MoC for this) |
| **Preprint status** | Not peer reviewed in this form; conclusions reflect authors' synthesis |

---

## Ecosystem & Tooling

| Tool / Framework | Role | Link |
|-----------------|------|------|
| LangChain | RAG pipeline orchestration | [langchain.com](https://www.langchain.com/) |
| LlamaIndex | Document indexing + RAG | [llamaindex.ai](https://www.llamaindex.ai/) |
| FAISS | Dense vector indexing (Facebook AI) | [GitHub](https://github.com/facebookresearch/faiss) |
| Chroma | Open-source vector DB | [trychroma.com](https://www.trychroma.com/) |
| Weaviate | Vector DB with hybrid retrieval | [weaviate.io](https://weaviate.io/) |
| RAGAS | RAG evaluation framework | [ragas.io](https://ragas.io/) |
| DSPy | Programmatic LLM + retriever pipeline | [GitHub](https://github.com/stanfordnlp/dspy) |

---

## Citation

```bibtex
@misc{gao2024retrievalaugmentedgenerationlargelanguage,
  title         = {Retrieval-Augmented Generation for Large Language Models: A Survey},
  author        = {Yunfan Gao and Yun Xiong and Xinyu Gao and Kangxiang Jia and Jinliu Pan and Yuxi Bi and Yi Dai and Jiawei Sun and Meng Wang and Haofen Wang},
  year          = {2024},
  eprint        = {2312.10997},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2312.10997}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| Original RAG Paper (Lewis et al., 2020) | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| Self-RAG: Learn to Retrieve, Generate, Critique (Asai et al., 2023) | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) |
| FLARE: Active Retrieval Augmented Generation (Jiang et al., 2023) | [arXiv:2305.06983](https://arxiv.org/abs/2305.06983) |
| HyDE: Precise Zero-Shot Dense Retrieval (Gao et al., 2022) | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) |
| GraphRAG: From Local to Global (Edge et al., 2024) | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) |
| RAGAS: Automated Evaluation of RAG (Es et al., 2023) | [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) |
| Lost in the Middle: Long-Context LLMs (Liu et al., 2023) | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) |
| MoC: Mixtures of Text Chunking Learners (Zhao et al., 2025) | [ACL Anthology](https://aclanthology.org/2025.acl-long.258/) |
| XRAG: Cross-lingual RAG Benchmark (Liu et al., 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.849/) |
| Towards Agentic RAG with Deep Reasoning (Li et al., 2025) | [arXiv:2507.09477](https://arxiv.org/abs/2507.09477) |
| A Survey on RAG Meeting LLMs (Fan et al., 2024) | [ACM KDD 2024](https://dl.acm.org/doi/10.1145/3637528.3671470) |
| NV-Embed: LLMs as Generalist Embedding Models (Lee et al., 2025) | [ICLR 2025](https://openreview.net/forum?id=lgsyLSsDRe) |
