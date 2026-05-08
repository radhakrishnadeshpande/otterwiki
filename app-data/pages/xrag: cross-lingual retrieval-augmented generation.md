# XRAG: Cross-lingual Retrieval-Augmented Generation


> **Paper:** [XRAG: Cross-lingual Retrieval-Augmented Generation](https://arxiv.org/abs/2505.10089)
> **ACL Anthology:** [EMNLP 2025 Findings](https://aclanthology.org/2025.findings-emnlp.849/)
> **HuggingFace Dataset:** [AmazonScience/XRAG](https://huggingface.co/datasets/AmazonScience/XRAG)
> **Semantic Scholar:** [Paper Details](https://www.semanticscholar.org/paper/XRAG:-Cross-lingual-Retrieval-Augmented-Generation-Liu-Trenous/cc844e3377f4af175881707774f266d140741ca9)

---

## Authors

| Name | Affiliation |
|------|-------------|
| Wei Liu | Heidelberg Institute for Theoretical Studies (HITS) |
| Sony Trenous | Amazon |
| Leonardo F. R. Ribeiro | Amazon |
| Bill Byrne | Amazon / University of Cambridge |
| Felix Hieber | Amazon |

**Published:** EMNLP 2025 Findings, Suzhou, China (November 4–9, 2025)
**ArXiv Submission:** May 15, 2025

---

## TL;DR

XRAG is a benchmark for evaluating LLM generation capabilities in **cross-lingual RAG** settings — where the user's query language does not match the language of retrieved documents. Built from real news articles post-June 2024 (after common LLM knowledge cutoffs), it exposes two previously unreported failure modes in state-of-the-art LLMs.

---

## Problem Statement

1. **No adequate benchmark exists** for cross-lingual RAG, especially for scenarios where retrieved evidence may be in a mixed-language pool or not in the user's language.

2. **Existing cross-lingual QA datasets are insufficient** — datasets like [XOR QA](https://arxiv.org/abs/2010.11856) and [XQA](https://aclanthology.org/P19-1227/) are limited in cross-lingual coverage and often answerable without retrieval (e.g., [Chirkova et al., 2024](https://arxiv.org/abs/2405.05086) shows 47.5% of XORQA questions can be answered by Command-R *without* any retrieval). They do not measure true cross-lingual RAG capability.

---

## Contributions

1. **Introduce XRAG** — a benchmark for two distinct cross-lingual RAG scenarios (monolingual retrieval and multilingual retrieval).
2. **Propose a news-based, LLM-driven construction pipeline** to generate challenging cross-document QA that genuinely requires external documents.
3. **Identify and characterize two new challenges** in cross-lingual RAG: *Response Language Correctness* and *Cross-language Reasoning*.

---

## Benchmark Overview

| Property | Details |
|----------|---------|
| **Source** | [News Crawl](http://data.statmt.org/news-crawl/) articles (English, German, Spanish, Chinese, Arabic) |
| **Timeframe** | June 1, 2024 → November 30, 2024 (post LLM knowledge cutoff) |
| **Languages** | Arabic (ar), Chinese (zh), German (de), Spanish (es) + English |
| **Scenarios** | Monolingual Retrieval · Multilingual Retrieval |
| **Dev Set** | 2,985 samples (2,336 monolingual + 649 multilingual) |
| **Instance Structure** | 1 question · 1 gold answer · 2 supporting articles · 6 distracting articles |
| **Evaluation** | LLM-as-a-Judge panel (inter-annotator κ = 0.71 vs. humans) |

### Two RAG Scenarios

```
Monolingual Retrieval:
  Question (e.g., German) ──► English-only retrieved docs ──► Answer in German

Multilingual Retrieval:
  Question (e.g., German) ──► English + German retrieved docs ──► Answer in German
```

---

## Key Findings

### Without vs. With Retrieval (GPT-4o)

| Setting | Score |
|---------|-------|
| Without retrieval | 6.30 |
| With retrieval | 75.40 |

> Questions are designed to be genuinely hard and **not answerable without retrieval**, even in English. This large gap validates that XRAG tests real retrieval-augmented generation rather than LLM memorization.

### Two Core Challenges Uncovered

**Challenge 1 — Response Language Correctness (Monolingual Retrieval)**
All five evaluated LLMs struggle to respond in the question's language when retrieved documents are in English. Models tend to "drift" and respond in the retrieval language (English) instead of the user's query language (e.g., German, Arabic).

**Challenge 2 — Cross-Language Reasoning (Multilingual Retrieval)**
When retrieved documents contain both English and the question language, the bottleneck is not *generating* non-English text — it is *reasoning across* evidence that spans multiple languages simultaneously.

### Human vs. LLM Gap

A significant gap between human performance and LLM performance demonstrates that XRAG provides a meaningful and non-trivial reasoning challenge, even before factoring in cross-lingual complexity.

---

## Models Evaluated

Five LLMs were evaluated (with legal/licensing constraints limiting broader coverage):

- GPT-4o
- Claude Sonnet 3.5
- Mistral-Large
- Command-R+
- Amazon Nova Pro

---

## Limitations

| Limitation | Details |
|------------|---------|
| **LLM-judge dependency** | Evaluation relies on an LLM-judge panel; while κ = 0.71 vs. humans, it may introduce bias or miss nuanced correctness. |
| **Two-language mixture only** | Multilingual retrieval setting uses English + one question language (not 3+ languages simultaneously). |
| **Limited model coverage** | Only five models evaluated due to legal and licensing concerns. |
| **Fixed distractor design** | Analysis of varying distractor count is noted as future work. |

---

## Usage Notes

> XRAG is a **benchmark paper**, not a retrieval model paper. The focus is on RAG *generation* rather than retrieval itself. Instances are constructed with known supporting documents + distractors (simulating imperfect retrieval). When comparing to end-to-end systems, be explicit about whether you are evaluating retrieval, generation, or the full pipeline.

---

## Citation

```bibtex
@inproceedings{liu-etal-2025-xrag,
  title     = "{XRAG}: Cross-lingual Retrieval-Augmented Generation",
  author    = "Liu, Wei and Trenous, Sony and Ribeiro, Leonardo F. R. and Byrne, Bill and Hieber, Felix",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
  month     = nov,
  year      = "2025",
  address   = "Suzhou, China",
  publisher = "Association for Computational Linguistics",
  pages     = "15669--15690",
  url       = "https://aclanthology.org/2025.findings-emnlp.849"
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| XOR QA: Cross-lingual Open-Retrieval QA (Asai et al., 2021) | [ACL Anthology](https://aclanthology.org/2021.naacl-main.466/) |
| CRAG: Comprehensive RAG Benchmark (Yang et al., 2024) | [NeurIPS 2024](https://arxiv.org/abs/2406.04744) |
| Chirkova et al. (2024) — Cross-lingual LLMs in RAG | [arXiv](https://arxiv.org/abs/2405.05086) |
| Self-RAG (Asai et al., 2023) | [arXiv](https://arxiv.org/abs/2310.11511) |
| MEGA: Multilingual Eval of Generative AI (Ahuja et al., 2023) | [arXiv](https://arxiv.org/abs/2303.12528) |
| RAGEval (Ding et al., 2025) | [ACL 2025](https://arxiv.org/abs/2408.01262) |
| LLM-as-a-Judge / MT-Bench (Zheng et al., 2023) | [NeurIPS 2023](https://arxiv.org/abs/2306.05685) |
