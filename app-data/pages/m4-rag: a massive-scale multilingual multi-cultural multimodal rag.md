# M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG

> **arXiv:** [arXiv:2512.05959](https://arxiv.org/abs/2512.05959)
> **PDF:** [View PDF](https://arxiv.org/pdf/2512.05959)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2512.05959v1)
> **Semantic Scholar:** [Paper Details](https://api.semanticscholar.org/arXiv:2512.05959)
> **ResearchGate:** [Full Text](https://www.researchgate.net/publication/398430203_M4-RAG_A_Massive-Scale_Multilingual_Multi-Cultural_Multimodal_RAG)

---

## Authors

| Name | Affiliation |
|------|-------------|
| David Anugraha | Stanford University |
| Patrick Amadeus Irawan | MBZUAI |
| Anshul Singh | Indian Institute of Science |
| En-Shiun Annie Lee | Ontario Tech University · University of Toronto |
| Genta Indra Winata | Graduate Center, CUNY |

**Submitted:** December 8, 2024
**Category:** Computation and Language (cs.CL)

---

## Introduction

Vision-language models (VLMs) have achieved strong performance in visual question answering (VQA), yet they remain constrained by static training data. RAG mitigates this limitation by enabling access to up-to-date, culturally grounded, and multilingual information. However, real-world information access is inherently both multilingual and multimodal, and current RAG systems and benchmarks rarely assess this combined setting.

Existing multilingual and multimodal benchmarks address either cross-lingual or cross-modal challenges in isolation, but not together. This leaves open critical challenges: how well do systems align cross-lingual retrieval with multimodal representations? Do they degrade equitably across languages? Can they ground responses in culturally specific visual content? M4-RAG is designed to fill this gap with a benchmark that is simultaneously multilingual, multicultural, and multimodal at massive scale.

---

## Problem Statement

1. **Traditional RAG is English-centric and text-only** — existing systems and benchmarks do not evaluate retrieval and generation when queries and evidence are multilingual and multimodal.
2. **VLMs rely on static parametric knowledge** — without retrieval, they cannot access up-to-date or culturally specific information outside their training distribution.
3. **Cross-lingual and cross-modal alignment is underexplored** — existing work addresses these axes separately; no benchmark systematically studies their interaction.
4. **Cultural and dialectal nuance is ignored** — most benchmarks treat languages as monolithic, failing to capture regional dialect variation that affects retrieval correctness.

---

## Contributions

1. **Introduces M4-RAG**, a massive-scale benchmark covering **42 languages** and **56 regional dialects and registers** with over **80,000 culturally diverse image–question pairs**.
2. **Builds a controlled retrieval environment** with millions of curated multilingual documents to balance realism with reproducibility.
3. **Provides the first systematic comparison** of model performance under cross-lingual and cross-modal conditions simultaneously.
4. **Identifies key failure modes** in existing RAG systems when language or modality of query and retrieved context differ.

---

## Benchmark Overview

| Property | Details |
|----------|---------|
| **Languages** | 42 languages + 56 regional dialects and registers |
| **Scale** | 80,000+ culturally diverse image–question pairs |
| **Task** | Retrieval-augmented VQA (multimodal) |
| **Retrieval Modes** | Text–text · Text–image (multimodal) |
| **Retrieval Corpus** | Millions of curated multilingual documents |
| **Evaluation** | VLM-as-a-judge with reasoning rubric (macro-averaged accuracy) |
| **Datasets Used** | [CVQA](https://arxiv.org/abs/2406.05967), [WorldCuisines](https://arxiv.org/abs/2410.12705) |

---

## Key Research Questions

The benchmark is designed to answer four core questions:

1. **Multimodal retrieval** — does incorporating visual or multilingual context improve model accuracy, and under what conditions does it fail?
2. **Model scaling** — do larger models benefit less from retrieval due to stronger reliance on parametric knowledge?
3. **Language equity** — do RAG systems offer equitable support for non-English users, or do language mismatches introduce systematic degradation?
4. **Retrieval quality vs. generation quality** — how strongly does retrieval quality bound downstream generation performance?

---

## Key Findings

- **Small models (<7B)** show substantial gains from multimodal retrieval; **large models (>30B)** show diminishing returns or even performance degradation, suggesting over-reliance on parametric knowledge.
- Existing RAG models **degrade substantially** when queries or retrieved contexts differ in language or modality, underscoring the need for stronger cross-lingual and cross-modal alignment.
- **Retrieval quality directly bounds generation quality** — the correction rate (incorrect → correct with RAG) and correctness retention rate (correct → still correct with RAG) both vary strongly with retrieval quality.
- **Dialect-level granularity matters** — culturally correct retrieval often depends on recognizing dialect-specific nuances that coarser language-level models miss.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Infrastructure complexity** | Building and maintaining a massive multilingual multimodal retrieval corpus requires significant storage, indexing, and compute infrastructure. |
| **High computational cost** | Evaluating VLMs across 42 languages and two retrieval modalities at scale is expensive; only a subset of models can be evaluated in practice. |
| **Two-modality scope** | The benchmark focuses on text and image modalities; audio, video, and other modalities are not included. |
| **VLM-as-a-judge evaluation bias** | Evaluation relies on a VLM judge with a reasoning rubric; this may introduce model-specific biases in scoring. |
| **Static retrieval corpus** | The controlled retrieval corpus is curated at a fixed point in time, which may not reflect the dynamic nature of real-world retrieval systems. |

---

## Citation

```bibtex
@misc{anugraha2024m4rag,
  title         = {M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG},
  author        = {David Anugraha and Patrick Amadeus Irawan and Anshul Singh and En-Shiun Annie Lee and Genta Indra Winata},
  year          = {2024},
  eprint        = {2512.05959},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2512.05959}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020) | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| CVQA: Culturally-Diverse Multilingual Visual QA Benchmark (Romero et al., 2024) | [arXiv:2406.05967](https://arxiv.org/abs/2406.05967) |
| WorldCuisines: Multilingual Multicultural VQA (Winata et al., 2025) | [arXiv:2410.12705](https://arxiv.org/abs/2410.12705) |
| mmE5: Improving Multimodal Multilingual Embeddings (Chen et al., 2024) | [arXiv:2410.05738](https://arxiv.org/abs/2410.05738) |
| Pangea: Fully Open Multilingual Multimodal LLM for 39 Languages (Yue et al., 2024) | [arXiv:2410.16153](https://arxiv.org/abs/2410.16153) |
| XRAG: Cross-lingual RAG Benchmark (Liu et al., 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.849/) |
| A Survey of Multimodal RAG (Zhao et al., 2025) | [arXiv:2504.08748](https://arxiv.org/abs/2504.08748) |
| Prometheus-Vision: VLM as a Judge (Lee et al., 2024) | [ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.25/) |
