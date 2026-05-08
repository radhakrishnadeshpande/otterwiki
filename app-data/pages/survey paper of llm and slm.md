# A Survey of Context Engineering for Large Language Models
> **arXiv:** [arXiv:2507.13334](https://arxiv.org/abs/2507.13334)
> **PDF:** [View PDF](https://arxiv.org/pdf/2507.13334)
> **GitHub (Awesome List):** [Meirtz/Awesome-Context-Engineering](https://github.com/Meirtz/Awesome-Context-Engineering)
> **HuggingFace Papers:** [HF Paper Page](https://huggingface.co/papers/2507.13334)
> **Semantic Scholar:** [Paper Details](https://api.semanticscholar.org/arXiv:2507.13334)

---

## Authors

| Name | Affiliation |
|------|-------------|
| Lingrui Mei *(equal, corresponding)* | Institute of Computing Technology, CAS · UCAS |
| Jiayu Yao *(equal)* | Institute of Computing Technology, CAS · UCAS |
| Yuyao Ge *(equal)* | Institute of Computing Technology, CAS · UCAS |
| Yiwei Wang | University of California, Merced |
| Baolong Bi *(equal)* | Institute of Computing Technology, CAS · UCAS |
| Yujun Cai | The University of Queensland |
| Jiazhi Liu | Institute of Computing Technology, CAS |
| Mingyu Li | Institute of Computing Technology, CAS |
| Zhong-Zhi Li | University of Chinese Academy of Sciences |
| Duzhen Zhang | University of Chinese Academy of Sciences |
| Chenlin Zhou | Peking University |
| Jiayi Mao | Tsinghua University |
| Tianze Xia | University of Chinese Academy of Sciences |
| Jiafeng Guo *(corresponding)* | Institute of Computing Technology, CAS · UCAS |
| Shenghua Liu *(corresponding)* | Institute of Computing Technology, CAS · UCAS |

**Submitted:** July 17, 2025 (v1); Updated July 21, 2025 (v2)
**Scope:** 165 pages · 1,400+ papers surveyed
**Category:** Computation and Language (cs.CL)

> ⚠️ **Note:** This is a preprint. It has not yet undergone formal peer review. Treat it as a useful synthesis rather than peer-reviewed consensus. The field is evolving rapidly; post-July 2025 architectures may not be covered.

---

## Introduction

As LLMs evolve from simple instruction-following systems into the core reasoning engines of complex, multi-step AI applications, the methods used to interact with them must also evolve. The term *prompt engineering*, while foundational, is no longer sufficient to describe the full scope of designing, managing, and optimizing the information that modern AI systems require. These systems do not operate on a single, static string of text — they leverage dynamic, structured, and multifaceted information streams that span retrieved documents, tool outputs, memory traces, agent observations, and interaction histories.

This survey formalizes this broader challenge as **Context Engineering**: the systematic optimization of *information payloads* supplied to LLMs at inference time. It frames context not as a static prompt string but as a structured, dynamic composition of informational components, and treats its assembly as an optimization problem over utility, cost, coherence, and recency. The survey traces the evolution of context engineering practice from foundational RAG systems (2020) through memory-augmented agents and tool-integrated reasoning systems (2023–2025), synthesizing over 1,400 papers into a unified taxonomy and roadmap.

---

## Problem Statement

1. **Fragmented literature** — techniques for context management are scattered across prompting, RAG, memory, tool use, and multi-agent systems, with no unified conceptual framework connecting them.
2. **Prompt engineering is insufficient** — treating context as a monolithic, static string fails to capture the dynamic, compositional nature of modern LLM system inputs.
3. **LLM performance is inference-time context-dependent** — the survey argues that model capability is jointly determined by pretraining *and* the quality, structure, length, coherence, and self-consistency of the context provided at inference time.
4. **No systems discipline for context** — there is no principled methodology for selecting, filtering, compressing, ordering, and refreshing context across retrieval, memory, and tool pipelines.

---

## Contributions

1. **Formalizes Context Engineering** as a discipline distinct from prompt engineering, grounded in a probabilistic model of autoregressive LLM generation.
2. **Presents a two-layer taxonomy** decomposing the field into foundational components and system-level implementations.
3. **Synthesizes 1,400+ papers** spanning RAG, memory systems, tool-integrated reasoning, and multi-agent architectures.
4. **Identifies open challenges** in evaluation, theory, scalability, and robustness of context pipelines.
5. **Provides an evolution timeline** of context engineering implementations from 2020 to 2025.

---

## Taxonomy Overview

The survey organizes Context Engineering into two layers:

### Layer 1 — Foundational Components

| Component | Description |
|-----------|-------------|
| **Context Retrieval & Generation** | How relevant information is found or produced — dense/sparse retrieval, query rewriting, synthetic context generation |
| **Context Processing** | How raw retrieved or generated content is filtered, ranked, compressed, and structured before being placed in the context window |
| **Context Management** | How context is maintained, updated, and coordinated over time — window management, caching, eviction, freshness |

### Layer 2 — System Implementations

| Implementation | Description |
|----------------|-------------|
| **Retrieval-Augmented Generation (RAG)** | Grounding LLM outputs in retrieved external documents; covers naive, advanced, and modular RAG variants |
| **Memory Systems** | Persistent and episodic memory for long-horizon interactions; covers in-context, external, and parametric memory forms |
| **Tool-Integrated Reasoning** | Equipping LLMs with access to APIs, code interpreters, search engines, and calculators; covers ReAct and tool-use frameworks |
| **Multi-Agent Systems** | Coordinating multiple LLM agents with shared or partitioned context; covers communication protocols and orchestration patterns |

---

## Key Arguments

- **Context is not just a prompt** — it is a structured information payload comprising system instructions, retrieved passages, tool outputs, memory traces, and interaction history, each with different recency, reliability, and relevance properties.
- **Model performance is context-pipeline-dependent** — gains from better context engineering can rival or exceed gains from scaling model parameters, particularly for knowledge-intensive and long-horizon tasks.
- **Retrieval, memory, and tool use are unified under one framework** — prior work treats these as separate research threads; context engineering provides a common abstraction.
- **Evaluation is underdeveloped** — the survey identifies the lack of standardized metrics for context pipeline quality as one of the field's major open gaps.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Taxonomy may age quickly** | Context engineering is a fast-moving area; components and system designs described here may be superseded by post-July 2025 architectures not yet covered. |
| **Risk of overgeneralization** | Survey-level synthesis inevitably abstracts over heterogeneous systems with different assumptions, scales, and failure modes. |
| **Boundary fuzziness** | "Context engineering" is still an emerging term; its boundaries with prompt engineering, RAG, and agent design remain partially undefined. |
| **Preprint status** | The paper has not yet been peer reviewed; some claims and framings reflect the authors' perspective rather than community consensus. |
| **No new empirical results** | As a survey, it does not provide new benchmark numbers or experimental validation of its taxonomy. |

---

## Closely Related Areas

- [Prompt Engineering](https://arxiv.org/abs/2107.13586) — the predecessor discipline this survey explicitly supersedes
- [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401) — a central implementation within the taxonomy
- [LLM Agent Surveys](https://arxiv.org/abs/2308.11432) — overlapping coverage on tool use and multi-agent coordination
- [Memory-Augmented LLMs](https://arxiv.org/abs/2301.04589) — addressed within the memory systems component
- [Instruction Tuning](https://arxiv.org/abs/2109.01652) — related to how models are trained to respond to structured context

---

## Citation

```bibtex
@misc{mei2025surveycontextengineeringlarge,
  title         = {A Survey of Context Engineering for Large Language Models},
  author        = {Lingrui Mei and Jiayu Yao and Yuyao Ge and Yiwei Wang and Baolong Bi and Yujun Cai and Jiazhi Liu and Mingyu Li and Zhong-Zhi Li and Duzhen Zhang and Chenlin Zhou and Jiayi Mao and Tianze Xia and Jiafeng Guo and Shenghua Liu},
  year          = {2025},
  eprint        = {2507.13334},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2507.13334}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP (Lewis et al., 2020) | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| A Survey on LLM-based Autonomous Agents (Wang et al., 2023) | [arXiv:2308.11432](https://arxiv.org/abs/2308.11432) |
| ReAct: Synergizing Reasoning and Acting in LLMs (Yao et al., 2023) | [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) |
| MemGPT: Towards LLMs as Operating Systems (Packer et al., 2023) | [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) |
| Self-RAG: Learning to Retrieve, Generate, and Critique (Asai et al., 2023) | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) |
| GraphRAG: From Local to Global (Edge et al., 2024) | [arXiv:2404.16130](https://arxiv.org/abs/2404.16130) |
| Lost in the Middle: How LLMs Use Long Contexts (Liu et al., 2023) | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) |
| RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (Sarthi et al., 2024) | [arXiv:2401.18059](https://arxiv.org/abs/2401.18059) |
| Awesome-Context-Engineering GitHub Repository (Mei et al., ongoing) | [GitHub](https://github.com/Meirtz/Awesome-Context-Engineering) |
