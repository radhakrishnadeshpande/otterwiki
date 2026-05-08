# DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation

> **arXiv:** [arXiv:2506.01954](https://arxiv.org/abs/2506.01954)
> **ACL Anthology:** [ACL 2025 Main — Long Papers](https://aclanthology.org/2025.acl-long.358/)
> **PDF:** [View PDF](https://aclanthology.org/2025.acl-long.358.pdf)
> **GitHub (Code):** [VILA-Lab/DRAG](https://github.com/VILA-Lab/DRAG)
> **Semantic Scholar:** [Paper Details](https://www.semanticscholar.org/paper/DRAG%3A-Distilling-RAG-for-SLMs-from-LLMs-to-Transfer-Chen-Myrzakhan/drag)

---

## Authors

| Name | Affiliation |
|------|-------------|
| Jennifer Chen *(equal contribution)* | VILA Lab, MBZUAI · McGill University |
| Aidar Myrzakhan *(equal contribution)* | VILA Lab, Mohamed bin Zayed University of AI |
| Yaxin Luo | VILA Lab, Mohamed bin Zayed University of AI |
| Hassaan Muhammad Khan | VILA Lab, MBZUAI · NUST |
| Sondos Mahmoud Bsharat | VILA Lab, Mohamed bin Zayed University of AI |
| Zhiqiang Shen *(corresponding)* | VILA Lab, Mohamed bin Zayed University of AI |

**Published:** ACL 2025 Main Conference (Volume 1: Long Papers), Vienna, Austria (July 2025)
**arXiv Submitted:** June 2, 2025 · **Pages:** 7240–7260

---

## Introduction

Retrieval-Augmented Generation (RAG) has become a foundational technique in modern NLP, enabling LLMs to access external factual knowledge beyond their parametric memory. While large-scale RAG systems achieve strong factual consistency, they are computationally expensive and impractical for deployment in resource-constrained settings. At the same time, small language models (SLMs) — which are inherently more efficient — lack the knowledge capacity and reasoning depth to benefit from retrieval in the same way large models do.

A further complication is that even large RAG systems suffer from hallucination, particularly when retrieved context contains noise or semantically inconsistent content. The retrieved passages may be irrelevant, contradictory, or misaligned with the query, causing the LLM to produce plausible-sounding but factually incorrect outputs. This **embedding and semantic inconsistency** problem is especially acute in SLMs, which have neither the parametric knowledge to override noisy retrieval nor the capacity to distinguish relevant from irrelevant retrieved content.

DRAG addresses this by re-framing the problem: rather than running RAG at inference time on an SLM, use a powerful LLM-based RAG system as a *teacher* at training time. The teacher generates structured, ranked evidence and a relational knowledge graph from retrieved content; the SLM then trains on this distilled signal, acquiring both factual knowledge and the ability to reason over structured evidence without requiring the teacher at test time.

---

## Problem Statement

1. **SLMs lack knowledge capacity** — small models trained with standard fine-tuning cannot store or reason over the breadth of factual knowledge required for knowledge-intensive tasks.
2. **Hallucination in RAG systems** — even with retrieval, LLMs (and especially SLMs) are susceptible to hallucination when retrieved content is noisy, irrelevant, or semantically inconsistent with the query.
3. **Computational cost of large RAG** — deploying full LLM-based RAG at inference time is prohibitive for edge and low-resource deployment scenarios.
4. **Retrieval noise problem** — standard retrieval pipelines pass all retrieved content directly to the model without filtering or ranking, amplifying the signal-to-noise problem for weaker models.

---

## Contributions

1. **Introduces DRAG**, a novel teacher–student knowledge distillation framework that transfers RAG capabilities from LLMs to SLMs using structured evidence and knowledge graphs.
2. **Proposes evidence ranking + graph-based distillation**, combining cosine similarity scoring with LLM-based relevance scoring to filter and rank retrieved evidence before graph construction.
3. **Introduces a privacy leakage benchmark** alongside a privacy risk mitigation mechanism embedded in the DRAG pipeline.
4. **Comprehensive evaluation** across five benchmark types, five teacher LLMs, and seven student SLMs demonstrating consistent state-of-the-art performance.

---

## Framework Overview

DRAG operates as a four-stage pipeline during training. At inference time, only the trained SLM is required — the teacher LLM is not needed.

```
Stage 1 — Evidence Generation
  Given query q, teacher LLM generates N diverse textual evidence snippets D = {d₁, d₂, ..., dₙ}
  (The LLM acts as a more targeted retriever than a standard retrieval system for weaker SLMs)

Stage 2 — Evidence Ranking & Filtering
  Each dᵢ is scored via:
    (a) Cosine similarity against query embedding
    (b) LLM-based relevance scoring
  → Filtered subset D_filtered ⊂ D retaining top-K evidence pieces

Stage 3 — Graph RAG Construction
  Filtered evidence is converted into a structured knowledge graph G = (V, E)
  Nodes = entities/concepts; Edges = relationships extracted from evidence
  → Graph uses ~18.1% fewer tokens than raw evidence on average

Stage 4 — SLM Distillation & Evaluation
  SLM is trained on (query, distilled evidence / graph, teacher answer) tuples
  At inference: SLM uses distilled context to answer without teacher
```

### Three DRAG Configurations

| Config | Description |
|--------|-------------|
| **DRAG_E** (Evidence-based) | SLM trained on ranked evidence only |
| **DRAG_G** (Graph-based) | SLM trained on knowledge graph only |
| **DRAG_C** (Combined) | SLM trained on both evidence + graph |

> Ablations show that **DRAG_E alone often matches or exceeds DRAG_C**, with the combined configuration sometimes being redundant. Around **K = 15 evidence pieces** provides the best accuracy-cost tradeoff.

---

## Experimental Setup

| Property | Details |
|----------|---------|
| **Task Types** | Multi-choice QA · Open-ended QA · Fact verification |
| **Benchmarks** | [ARC-Challenge](https://arxiv.org/abs/1803.05457), [MedMCQA](https://arxiv.org/abs/2203.14371), [GPQA](https://arxiv.org/abs/2311.12022), [MMLU](https://arxiv.org/abs/2009.03300), [Open-LLM-Leaderboard](https://arxiv.org/abs/2406.07545), [AVERITEC](https://arxiv.org/abs/2305.13117) |
| **Teacher LLMs** | GPT-4o, DeepSeek-V3, Gemini Flash 1.5, Claude 3.5 Sonnet, LLaMA-3.3-70B |
| **Student SLMs** | Gemma-2-2B-it, Phi-3.5-mini, Qwen2.5-3B, LLaMA-3.2-3B, Qwen2.5-7B, LLaMA-3.1-8B, Gemma-2-9B-it |
| **Baselines** | Naive RAG, GraphRAG, MiniRAG, SimRAG |

---

## Key Results

- **DRAG outperforms MiniRAG by up to 27.7%** on ARC-Challenge using the same backbone models.
- **+8.2% on MMLU and +11.7% on ARC-C** vs. SimRAG with the same LLaMA-3.1-8B-Instruct backbone.
- On **ARC-Challenge**, DRAG achieves up to **94.1% accuracy** with student models.
- On **Open-LLM Leaderboard**, Gemma-2-9B-it improves from 46.44% → **53.45%** and Qwen2.5-7B from 44.67% → **52.36%** with evidence-based distillation.
- **Teacher model quality matters**: GPT-4o as teacher yields the best student performance, confirming that evidence quality and consistency are critical.
- **Graph efficiency**: Graph representation requires ~18.1% fewer tokens than raw evidence while maintaining comparable or better factual accuracy.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Teacher LLM required at training** | DRAG requires inference from a large teacher LLM during the evidence/graph generation phase, adding training-time cost. |
| **Distillation knowledge loss** | Despite structured supervision, some complex multi-hop reasoning capabilities of the teacher LLM may not fully transfer to the student. |
| **Embedding & semantic inconsistency** | Retrieved content can still contain noisy or query-irrelevant passages; the ranking pipeline reduces but does not eliminate this problem, and residual noise can confuse the SLM. |
| **Evidence generation as pseudo-retrieval** | DRAG uses the teacher LLM to *generate* evidence rather than retrieve it from a grounded corpus, which may introduce factual confabulation in the distillation signal itself. |
| **Two-stage inference cost (DRAG_C)** | The combined configuration processes both evidence and graph, which increases token usage and may negate some efficiency gains of SLM deployment. |

---

## Citation

```bibtex
@inproceedings{chen-etal-2025-drag,
  title     = "{DRAG}: Distilling {RAG} for {SLM}s from {LLM}s to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation",
  author    = "Chen, Jennifer and Myrzakhan, Aidar and Luo, Yaxin and Khan, Hassaan Muhammad and Bsharat, Sondos Mahmoud and Shen, Zhiqiang",
  booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  month     = jul,
  year      = "2025",
  address   = "Vienna, Austria",
  publisher = "Association for Computational Linguistics",
  pages     = "7240--7260",
  url       = "https://aclanthology.org/2025.acl-long.358"
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020) | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| GraphRAG: From Local to Global (Edge et al., 2024) | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) |
| GNN-RAG: Graph Neural Retrieval for LLM Reasoning (Mavromatis & Karypis, 2024) | [arXiv:2405.20139](https://arxiv.org/abs/2405.20139) |
| MiniRAG: Towards Extremely Simple RAG for SLMs (Fan et al., 2025) | [arXiv:2501.06394](https://arxiv.org/abs/2501.06394) |
| SimRAG: Self-Improving Retrieval-Augmented Generation (Sharma et al., 2024) | [arXiv:2410.17952](https://arxiv.org/abs/2410.17952) |
| MMLU: Measuring Massive Multitask Language Understanding (Hendrycks et al., 2020) | [arXiv:2009.03300](https://arxiv.org/abs/2009.03300) |
| AVERITEC: Real-world Claim Verification (Schlichtkrull et al., 2023) | [arXiv:2305.13117](https://arxiv.org/abs/2305.13117) |
| GPQA: A Graduate-Level QA Benchmark (Rein et al., 2024) | [arXiv:2311.12022](https://arxiv.org/abs/2311.12022) |
| Open-LLM-Leaderboard (Myrzakhan et al., 2024) | [arXiv:2406.07545](https://arxiv.org/abs/2406.07545) |
| Knowledge Distillation Survey (Gou et al., 2021) | [arXiv:2006.05525](https://arxiv.org/abs/2006.05525) |
