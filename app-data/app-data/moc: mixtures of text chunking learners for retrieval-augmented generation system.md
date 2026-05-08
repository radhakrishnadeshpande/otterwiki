# MoC: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System

> **arXiv:** [arXiv:2503.09600](https://arxiv.org/abs/2503.09600)
> **ACL Anthology:** [ACL 2025 Main — Long Papers](https://aclanthology.org/2025.acl-long.258/)
> **PDF:** [View PDF](https://arxiv.org/pdf/2503.09600)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2503.09600v2)
> **GitHub:** [IAAR-Shanghai/Meta-Chunking/MoC](https://github.com/IAAR-Shanghai/Meta-Chunking/tree/main/MoC)
> **HuggingFace Papers:** [HF Paper Page](https://huggingface.co/papers/2503.09600)
> **Semantic Scholar:** [Paper Details](https://api.semanticscholar.org/arXiv:2503.09600)

---

## Introduction

Text chunking — the segmentation of documents into discrete units for indexing and retrieval — is a foundational but largely overlooked component of RAG pipelines. Standard approaches fall into two camps: rule-based methods that split on fixed token counts or sentence boundaries, and semantic chunking methods that use embedding similarity to determine split points. Both fail on complex documents because they lack the contextual understanding needed to recognize meaningful semantic boundaries, particularly in texts with nuanced reasoning, long-range dependencies, or domain-specific structure.

LLM-based chunking addresses this by exploiting the reasoning capabilities of large models to identify semantically coherent boundaries. However, it introduces a new trade-off: LLM inference is expensive, and applying a single LLM chunker uniformly across all document segments is computationally wasteful when simpler segments could be handled by lightweight methods. MoC resolves this tension with a **granularity-aware routing strategy** — dispatching each segment to the most appropriate chunker from a mixture, matching computational cost to segment complexity and ensuring that LLM capacity is allocated only where it provides the most benefit.

---

## Problem Statement

1. **Chunking is critical but overlooked in RAG** — retrieval quality and downstream generation both depend heavily on how documents are segmented, yet chunking receives far less attention than retriever or generator design.
2. **Traditional chunking (fixed-size, sentence-level) fails on complex text** — it ignores semantic coherence, producing chunks that split mid-argument or merge unrelated content.
3. **Semantic chunking has limited contextual reasoning** — embedding similarity signals do not capture logical or argumentative structure within documents.
4. **LLM-based chunking is accurate but uniformly expensive** — applying a full LLM to every document segment wastes compute on simple sections while providing no added benefit over lightweight methods.
5. **No quantitative metric exists for chunking quality** — prior work lacks direct evaluation of chunking output, making it impossible to systematically compare chunking strategies.

---

## Contributions

1. **Introduces two direct chunking quality metrics** — *Boundary Clarity* (how well chunk boundaries align with semantic transitions) and *Chunk Stickiness* (how coherent the content within each chunk is) — enabling quantitative chunking evaluation for the first time.
2. **Demonstrates failure modes of traditional and semantic chunking** using the dual-metric framework, establishing the need for LLM integration.
3. **Proposes the granularity-aware MoC framework**, a three-stage pipeline that routes document segments to appropriate chunkers based on complexity.
4. **Guides chunkers to output structured regular expressions** rather than direct segmentations, which are then applied to extract chunks from original documents.

---

## Framework: MoC

MoC operates as a three-stage granularity-aware pipeline:

```
Stage 1 — Granularity Assessment
  Each document segment is assessed for complexity
  Simple segments → routed to lightweight chunkers (rule-based or semantic)
  Complex segments → routed to LLM-based chunkers

Stage 2 — Mixture-of-Chunkers Dispatch
  A router selects the appropriate chunker from the mixture for each segment
  Chunkers produce structured lists of chunking regular expressions (not direct splits)

Stage 3 — Chunk Extraction
  Regular expressions are applied to the original document text
  Final chunks are extracted and indexed for retrieval
```

### Key Design Choice: Regex-Based Output

Rather than directly segmenting text, MoC guides chunkers to produce **chunking regular expressions** that are then applied to extract chunks. This decouples the chunking decision from the extraction step, improving consistency and reproducibility across chunker types.

---

## Evaluation

| Property | Details |
|----------|---------|
| **Chunking Metrics** | Boundary Clarity · Chunk Stickiness (both introduced in this paper) |
| **Downstream Task** | Open-domain question answering in RAG settings |
| **Datasets** | Wikipedia-based corpora · standard RAG QA benchmarks |
| **Baselines** | Fixed-size chunking · Sentence-level chunking · Semantic chunking · LLM-only chunking |

---

## Key Findings

- **Traditional and semantic chunking show measurable deficits** on both Boundary Clarity and Chunk Stickiness for complex documents, quantitatively confirming what practitioners had observed empirically.
- **LLM-based chunking alone improves quality** but is computationally prohibitive when applied uniformly across all segments.
- **MoC matches or exceeds full LLM chunking quality** while substantially reducing inference cost by routing simple segments away from the LLM.
- **Improved chunking quality translates directly to downstream RAG performance**, demonstrating that chunking is a bottleneck component deserving dedicated optimization.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Increased system complexity** | MoC introduces a routing layer and multiple chunker models on top of the RAG pipeline, requiring additional engineering effort to deploy and maintain. |
| **Computational overhead** | Training and running multiple chunkers, plus the granularity routing mechanism, adds latency and memory cost compared to simple fixed chunking. |
| **Router design sensitivity** | The quality of routing directly determines whether MoC gains are realized; a poorly calibrated router may send complex segments to lightweight chunkers or waste LLM capacity on simple ones. |
| **Regex brittleness** | Chunking via generated regular expressions may fail on text with unusual formatting, multilingual content, or highly irregular structure. |
| **Benchmark scope** | Evaluation focuses on standard English QA benchmarks; performance on domain-specific, multilingual, or multimodal documents is not established. |

---

## Predecessor Work

MoC builds directly on the same team's prior **Meta-Chunking** framework:

> Zhao et al. (2024) — [*Meta-Chunking: Learning Text Segmentation and Semantic Completion via Logical Perception*](https://arxiv.org/abs/2410.12788)
> Uses perplexity-based and margin-sampling chunking techniques with global information compensation; MoC extends this line with mixture-based routing and the new dual-metric evaluation framework.

---

## Citation

```bibtex
@inproceedings{zhao-etal-2025-moc,
  title     = "{M}o{C}: Mixtures of Text Chunking Learners for Retrieval-Augmented Generation System",
  author    = "Zhao, Jihao and Ji, Zhiyuan and Fan, Zhaoxin and Wang, Hanyu and Niu, Simin and Tang, Bo and Xiong, Feiyu and Li, Zhiyu",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month     = jul,
  year      = "2025",
  address   = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  pages     = "5172--5189",
  doi       = "10.18653/v1/2025.acl-long.258",
  url       = "https://aclanthology.org/2025.acl-long.258/"
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020) | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| Meta-Chunking: Text Segmentation via Logical Perception (Zhao et al., 2024) | [arXiv:2410.12788](https://arxiv.org/abs/2410.12788) |
| Is Semantic Chunking Worth the Computational Cost? (Qu et al., 2024) | [arXiv:2410.13070](https://arxiv.org/abs/2410.13070) |
| ChunkRAG: LLM-Chunk Filtering for RAG Systems (2024) | [arXiv:2410.19572](https://arxiv.org/abs/2410.19572) |
| Dense Passage Retrieval for Open-Domain QA (Karpukhin et al., 2020) | [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) |
| CRUD-RAG: Comprehensive Chinese RAG Benchmark (Lyu et al., 2024) | [arXiv:2401.17043](https://arxiv.org/abs/2401.17043) |
| Enhancing RAG with Hierarchical Text Segmentation Chunking (2025) | [arXiv:2506.07916](https://arxiv.org/abs/2506.07916) |
| Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023) | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) |
