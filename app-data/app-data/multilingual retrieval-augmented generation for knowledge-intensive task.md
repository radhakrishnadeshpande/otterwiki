# Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Task

> **arXiv:** [arXiv:2504.03616](https://arxiv.org/abs/2504.03616)
> **PDF:** [View PDF](https://arxiv.org/pdf/2504.03616)
> **Papers with Code:** [paperswithcode.com](https://paperswithcode.com/paper/multilingual-retrieval-augmented-generation)
> **Semantic Scholar:** [Paper Details](https://api.semanticscholar.org/arXiv:2504.03616)
> **ResearchGate:** [Full Text](https://www.researchgate.net/publication/390545745_Multilingual_Retrieval-Augmented_Generation_for_Knowledge-Intensive_Task)

---

## Authors

| Name | Affiliation |
|------|-------------|
| Leonardo Ranaldi | University of Edinburgh |
| Barry Haddow | University of Edinburgh |
| Alexandra Birch | University of Edinburgh |

**Submitted:** April 4, 2025
**Category:** Computation and Language (cs.CL); Artificial Intelligence (cs.AI)

---

## Intro

This paper investigates multilingual RAG for open-domain QA, proposes and evaluates three RAG strategies (tRAG, MultiRAG, CrossRAG), and finds that **CrossRAG** — which translates retrieved documents into a common language before generation — significantly outperforms alternatives across both high-resource and low-resource languages.

---

## Problem Statement

LLMs often struggle with **knowledge-intensive tasks across multiple languages**, for two key reasons:

1. RAG systems are overwhelmingly optimized for English monolingual settings; their effectiveness in multilingual tasks remains largely unexplored.
2. LLMs face knowledge gaps in non-English languages due to imbalanced training data and predominantly English knowledge sources, making retrieval-augmented approaches critical but underspecified for multilingual use.

---

## Contributions

1. **Introduces a multilingual RAG framework** for open-domain question answering across multiple languages.
2. **Proposes and compares three RAG strategies**: tRAG, MultiRAG, and the novel CrossRAG.
3. **Demonstrates improved multilingual knowledge access** via cross-lingual document translation before generation.

---

## Proposed Approaches

### RAG Strategy Comparison

```
tRAG (Translation RAG):
  Question (e.g., Arabic) ──► Translate to English ──► Retrieve English docs ──► Generate answer
  ⚠️ Useful but suffers from limited coverage.

MultiRAG (Multilingual RAG):
  Question (e.g., Arabic) ──► Retrieve across all languages ──► Generate answer
  ⚠️ Improves retrieval range but introduces inconsistencies from cross-lingual variation.

CrossRAG (Cross-lingual RAG):  ← Proposed method
  Question (e.g., Arabic) ──► Retrieve across all languages ──► Translate docs to English ──► Generate answer
  ✅ Best performance — normalizes retrieved evidence before generation.
```

### Key Design Choice

CrossRAG separates the **retrieval** problem (find relevant documents in any language) from the **generation** problem (produce a correct answer from consistent-language evidence), which is the key insight enabling its performance gains.

---

## Experimental Setup

| Property | Details |
|----------|---------|
| **Task** | Open-domain multilingual QA (knowledge-intensive) |
| **Datasets** | [MLQA](https://arxiv.org/abs/1910.07475), [TyDi QA](https://aclanthology.org/2020.tacl-1.30/), Wikipedia-based multilingual corpora |
| **Language Coverage** | High-resource and low-resource languages |
| **Models Evaluated** | Llama-3-8B, Command-R, GPT-4o |
| **Baselines** | No-RAG (LLM-only), tRAG, MultiRAG |

---

## Key Findings

1. **Retrieval significantly improves factual correctness** — LLMs without retrieval perform substantially worse on knowledge-intensive multilingual QA.
2. **tRAG** is a useful baseline but limited coverage of translation pipelines constrains its effectiveness.
3. **MultiRAG** increases retrieval scope but introduces cross-lingual inconsistencies that degrade generation quality.
4. **CrossRAG achieves the best performance** across both high-resource (HR) and low-resource (LR) languages, with consistent gains over no-RAG baselines.
5. **Multilingual embedding and indexing quality are crucial** — retrieval quality directly determines generation quality in all three strategies.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Retrieval dependency** | Performance is bounded by retrieval quality and coverage of multilingual corpora; poor retrieval hurts all strategies. |
| **Translation bottleneck in CrossRAG** | CrossRAG's document translation step introduces latency and potential translation errors that can propagate to generation. |
| **Embedding quality gap** | Multilingual embedding spaces are less mature than English-only ones, which can limit recall in low-resource languages. |
| **No code released** | No implementation is publicly available at the time of writing. |

---

## Follow-up Work

The same authors extended this line of research with **Dialectic-RAG (D-RAG)**, a modular approach that adds argumentative reasoning over conflicting multilingual evidence:

> Ranaldi et al. (EMNLP 2025) — [*Improving Multilingual Retrieval-Augmented Language Models through Dialectic Reasoning Argumentations*](https://aclanthology.org/2025.emnlp-main.461/)
> arXiv: [2504.04771](https://arxiv.org/abs/2504.04771)

---

## Citation

```bibtex
@misc{ranaldi2025multilingualretrievalaugmentedgenerationknowledgeintensive,
  title     = {Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Task},
  author    = {Leonardo Ranaldi and Barry Haddow and Alexandra Birch},
  year      = {2025},
  eprint    = {2504.03616},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url       = {https://arxiv.org/abs/2504.03616}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020) | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| MLQA: Evaluating Cross-Lingual Extractive QA (Lewis et al., 2020) | [ACL Anthology](https://aclanthology.org/2020.acl-main.653/) |
| TyDi QA: A Benchmark for Information-Seeking QA (Clark et al., 2020) | [TACL](https://aclanthology.org/2020.tacl-1.30/) |
| Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023) | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) |
| NoMIRACL: Knowing When You Don't Know (Chen et al., 2024) | [arXiv:2312.11361](https://arxiv.org/abs/2312.11361) |
| MIRACL: Multilingual Information Retrieval (Zhang et al., 2023) | [arXiv:2210.09984](https://arxiv.org/abs/2210.09984) |
| D-RAG: Dialectic Reasoning for Multilingual RAG (Ranaldi et al., 2025) | [ACL Anthology](https://aclanthology.org/2025.emnlp-main.461/) |
| XRAG: Cross-lingual RAG Benchmark (Liu et al., 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.849/) |
