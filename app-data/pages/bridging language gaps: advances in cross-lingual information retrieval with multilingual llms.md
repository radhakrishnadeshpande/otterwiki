# Bridging Language Gaps: Advances in Cross-Lingual Information Retrieval with Multilingual LLMs

> **arXiv:** [arXiv:2510.00908](https://arxiv.org/abs/2510.00908)
> **PDF:** [View PDF](https://arxiv.org/pdf/2510.00908)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2510.00908v1)
> **ResearchGate:** [Full Text](https://www.researchgate.net/publication/396094641_Bridging_Language_Gaps_Advances_in_Cross-Lingual_Information_Retrieval_with_Multilingual_LLMs)
> **Semantic Scholar:** [Paper Details](https://www.semanticscholar.org/paper/Bridging-Language-Gaps:-Advances-in-Cross-Lingual-Goworek-Macmillan-Scott/d9effe8369fa2a159fa1e55a076ed05e5510bcd5)

---

## Introduction

Cross-lingual information retrieval (CLIR) addresses the challenge of retrieving relevant documents written in languages different from that of the original query. The need for CLIR traces back to the early internet era, when English accounted for nearly 80% of all web content despite being the native language of only a small fraction of global users. This linguistic imbalance has persisted into the LLM era — most modern NLP systems and retrieval corpora remain skewed toward a small set of high-resource languages, leaving speakers of low-resource languages with substantially degraded access to information.

Historically, CLIR was framed as monolingual retrieval augmented by machine translation — treating retrieval methods and cross-lingual capabilities as separate concerns. Both monolingual and cross-lingual retrieval pipelines typically follow a multi-stage flow: query expansion, ranking, re-ranking, and increasingly, answer generation. The emergence of multilingual LLMs and cross-lingual embeddings has shifted the paradigm away from explicit translation toward embedding-based approaches that align semantic representations across languages in a shared space. This survey provides a comprehensive overview of that evolution — from early translation-based methods through state-of-the-art embedding-driven and generative techniques — and maps the open challenges, available resources, and future directions for building retrieval systems that are robust, inclusive, and adaptable across languages.

---

## Problem Statement

1. **Traditional CLIR relies on translation as a crutch** — framing retrieval as monolingual retrieval plus MT treats the two components in isolation, adding computational overhead, noise from translation errors, and failure modes for low-resource language pairs.
2. **High-resource language bias** — LLMs and multilingual embedding models are disproportionately trained on English-centric data; cross-lingual alignment degrades substantially for low-resource and morphologically complex languages.
3. **Weak cross-lingual semantic alignment** — standard multilingual embeddings often fail to align semantically equivalent content across languages, particularly for distant language pairs (e.g., Arabic–Chinese, Swahili–English).
4. **Fragmented literature** — advances in multilingual NLP, dense retrieval, and generative LLMs have each contributed to CLIR, but no unified survey existed connecting these threads.

---

## Contributions

1. **Structured survey of CLIR evolution** from translation-based methods (pre-2018) through multilingual pretrained encoders (2018–2022) to embedding-driven and generative LLM-based approaches (2022–2025).
2. **Analysis of core CLIR methods** — embedding alignment, multilingual pretraining, multilingual LLMs, and retrieval architectures — with explicit coverage of their strengths and trade-offs.
3. **Review of multilingual NLP advances** and their specific relevance to cross-lingual retrieval performance.
4. **Survey of datasets, evaluation protocols, and metrics**, including a discussion of current limitations and opportunities for improvement.
5. **Outline of open challenges and future directions** for building retrieval systems that are robust and inclusive across languages.

---

## CLIR Pipeline Overview

The survey organizes the CLIR pipeline into four stages, each of which has been reshaped by multilingual LLMs:

```
Stage 1 — Query Expansion
  Augments the query to address brevity, ambiguity, vocabulary mismatch, and spelling errors.
  mLLMs have shifted this from synonym augmentation to pseudo-document generation.

Stage 2 — Ranking (First-Stage Retrieval)
  Retrieves a candidate set from the document corpus.
  Methods: sparse (BM25), dense (bi-encoder), hybrid.
  Cross-lingual challenge: bridging the language gap without explicit translation.

Stage 3 — Re-Ranking
  Refines the candidate set using cross-encoder models.
  High quality but computationally expensive; performance depends on initial retrieval quality.

Stage 4 — Answer Generation (increasingly integrated)
  Uses the retrieved context to generate a final answer.
  Multilingual RAG settings; language-of-response alignment challenges.
```

---

## Methods Surveyed

| Approach | Description | Representative Work |
|----------|-------------|-------------------|
| **Translation-based** | Translate query or documents to a pivot language before retrieval | MT + BM25 pipelines |
| **Multilingual Sparse Retrieval** | Extend BM25 to multilingual corpora via multilingual tokenization | mBM25 variants |
| **Cross-lingual Dense Retrieval** | Bi-encoder models with shared multilingual embedding space | mDPR, mE5, LaBSE |
| **Contrastive Learning** | Train encoders with cross-lingual contrastive objectives to improve alignment | XLM-R fine-tuning, MCONTRIEVER |
| **Generative Re-Ranking** | Use sequence-to-sequence or decoder models for re-ranking | monoT5-based multilingual rankers |
| **LLM-based Query Expansion** | Generate pseudo-documents or expansions via mLLMs before retrieval | HyDE, multilingual variants |
| **End-to-End Generative CLIR** | Unified retrieval-and-generation without a separate re-ranking stage | RAG-based multilingual systems |

---

## Datasets and Evaluation Resources

| Dataset | Coverage | Link |
|---------|----------|------|
| MIRACL | 18 languages, Wikipedia-based | [arXiv:2210.09984](https://arxiv.org/abs/2210.09984) |
| MLQA | 7 languages, extractive QA | [ACL 2020](https://aclanthology.org/2020.acl-main.653/) |
| TyDi QA | 11 typologically diverse languages | [TACL 2020](https://aclanthology.org/2020.tacl-1.30/) |
| XOR QA | Cross-lingual open-retrieval QA | [NAACL 2021](https://aclanthology.org/2021.naacl-main.466/) |
| CLEF CLIR Track | Long-running multilingual IR benchmark | [CLEF](https://www.clef-initiative.eu/) |
| HC4 | Hard CLIR benchmark (4 languages) | [arXiv:2201.09992](https://arxiv.org/abs/2201.09992) |

---

## Key Arguments

- **Embedding-based approaches have largely superseded translation-based methods** for high-resource language pairs, but translation still plays a role for distant or low-resource pairs where multilingual embeddings remain poorly aligned.
- **LLM-based query expansion** (pseudo-document generation) improves dense retrieval by bridging the gap between short queries and long documents; query length is the key factor determining which expansion strategy is most effective.
- **Contrastive learning mitigates language bias** and yields substantial improvements for encoders with weak initial cross-lingual alignment.
- **Performance degrades significantly for low-resource languages** — systems must explicitly address the bias toward high-resource languages in both pretraining and evaluation.
- **Re-ranking quality is bounded by first-stage retrieval** — cross-encoder re-rankers are effective but cannot recover from a poor candidate set.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **High-resource language bias** | Performance varies significantly depending on language coverage and quality of multilingual pretraining data; low-resource languages consistently underperform. |
| **Evaluation benchmark coverage** | Most benchmarks concentrate on a small subset of well-resourced languages; evaluation for truly low-resource and morphologically complex languages remains limited. |
| **Rapid pace of change** | As a preprint survey of a fast-moving field, the coverage of methods and models reflects the state as of October 2025; newer architectures may not be included. |
| **Preprint status** | Not yet peer reviewed; the survey's framing and conclusions reflect the authors' synthesis rather than community consensus. |

---

## Follow-up Work (Same Authors)

The same Alan Turing Institute group released two companion papers in November 2025 that extend the survey's empirical coverage:

| Paper | arXiv |
|-------|-------|
| *What Drives Cross-lingual Ranking? Retrieval Approaches with Multilingual Language Models* — Systematic evaluation of document translation, dense retrieval, contrastive learning, and re-ranking across three CLIR benchmarks. | [arXiv:2511.19324](https://arxiv.org/abs/2511.19324) |
| *Generative Query Expansion with Multilingual LLMs for Cross-Lingual Information Retrieval* — Evaluates mLLMs and fine-tuned variants across generative expansion strategies; finds query length drives strategy effectiveness. | [arXiv:2511.19325](https://arxiv.org/abs/2511.19325) |

---

## Citation

```bibtex
@misc{goworek2025bridginglanguagegapsadvances,
  title         = {Bridging Language Gaps: Advances in Cross-Lingual Information Retrieval with Multilingual LLMs},
  author        = {Roksana Goworek and Olivia Macmillan-Scott and Eda B. Özyiğit},
  year          = {2025},
  eprint        = {2510.00908},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2510.00908}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020) | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| MIRACL: Multilingual Information Retrieval (Zhang et al., 2023) | [arXiv:2210.09984](https://arxiv.org/abs/2210.09984) |
| TyDi QA: Information-Seeking QA (Clark et al., 2020) | [TACL 2020](https://aclanthology.org/2020.tacl-1.30/) |
| XOR QA: Cross-lingual Open-Retrieval QA (Asai et al., 2021) | [NAACL 2021](https://aclanthology.org/2021.naacl-main.466/) |
| mDPR: Multilingual Dense Passage Retrieval (Zhang et al., 2021) | [arXiv:2108.11897](https://arxiv.org/abs/2108.11897) |
| LaBSE: Language-Agnostic BERT Sentence Embedding (Feng et al., 2020) | [arXiv:2007.01852](https://arxiv.org/abs/2007.01852) |
| HyDE: Precise Zero-Shot Dense Retrieval (Gao et al., 2022) | [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) |
| XRAG: Cross-lingual RAG Benchmark (Liu et al., 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.849/) |
| Multilingual RAG for Knowledge-Intensive Tasks (Ranaldi et al., 2025) | [arXiv:2504.03616](https://arxiv.org/abs/2504.03616) |
