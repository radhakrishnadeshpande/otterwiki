# Towards Agentic RAG with Deep Reasoning: A Survey of RAG-Reasoning Systems in LLMs

> **arXiv:** [arXiv:2507.09477](https://arxiv.org/abs/2507.09477)
> **PDF:** [View PDF](https://arxiv.org/pdf/2507.09477)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2507.09477v2)
> **Published:** EMNLP 2025
> **GitHub (Awesome List):** [DavidZWZ/Awesome-RAG-Reasoning](https://github.com/DavidZWZ/Awesome-RAG-Reasoning)
> **HuggingFace Papers:** [HF Paper Page](https://huggingface.co/papers/2507.09477)
> **Semantic Scholar:** [Paper Details](https://www.semanticscholar.org/paper/Towards-Agentic-RAG-with-Deep-Reasoning:-A-Survey-Li-Zhang/746575f830e07545d2fca63ccb408af0720217aa)

---


## Introduction

Retrieval-Augmented Generation (RAG) improves LLM factuality by grounding generation in retrieved external knowledge — but standard RAG pipelines are single-pass and static: query in, retrieve once, generate answer. This architecture is brittle on tasks that require *multi-step inference*, where the answer depends on chaining evidence across multiple documents or reasoning steps. The model cannot ask follow-up retrieval questions, revise its query based on what it found, or recognize when retrieved content is insufficient. Conversely, purely reasoning-oriented approaches — chain-of-thought, self-consistency, extended thinking models like DeepSeek-R1 — reason powerfully but hallucinate facts they do not actually possess.

The emerging paradigm of **Agentic RAG with Deep Reasoning** fuses both: LLMs iteratively interleave retrieval and reasoning, using each reasoning step to determine *what* to retrieve next and each retrieval to supply premises the model's parametric knowledge cannot provide. This survey is, to the authors' knowledge, the first to comprehensively map this integrated RAG-Reasoning space. It organizes the field into three perspectives — **Reasoning-Enhanced RAG**, **RAG-Enhanced Reasoning**, and **Synergized RAG-Reasoning** — and covers methods, datasets, benchmarks, and open challenges from the first wave of DeepSeek-R1-era agentic reasoning systems through mid-2025.

---

## Problem Statement

1. **Standard RAG fails on multi-step inference** — single-pass retrieve-then-generate pipelines cannot handle tasks where intermediate reasoning must guide retrieval of missing premises.
2. **Pure reasoning models hallucinate facts** — frontier reasoning LLMs (o1, DeepSeek-R1) can construct long, correct reasoning chains but fabricate factual content not in their parametric memory.
3. **RAG and reasoning are developed in isolation** — the two communities have produced strong methods independently but no unified framework connects them as an integrated system.
4. **No survey exists for this intersection** — prior surveys cover either RAG or reasoning separately; none synthesizes the emerging class of systems that deeply interleave both.

---

## Contributions

1. **First survey unifying RAG and deep reasoning** under a coherent reasoning-retrieval perspective, covering Reasoning-Enhanced RAG, RAG-Enhanced Reasoning, and Synergized frameworks.
2. **Three-pillar taxonomy** of RAG-Reasoning systems with categorized methods, representative systems, and evaluation benchmarks.
3. **Coverage of agentic and RL-trained systems** post-DeepSeek-R1, including search-augmented reasoning models and iterative retrieval agents.
4. **Open challenges and future directions** across four dimensions: effectiveness, multimodal adaptation, trustworthiness, and human-centricity.
5. **Living awesome list** at GitHub tracking new papers continuously.

---

## Taxonomy: Three Pillars

### Pillar 1 — Reasoning-Enhanced RAG

*Using reasoning capabilities to improve each stage of the RAG pipeline.*

```
Query Understanding & Reformulation
  ├── Query decomposition: break complex queries into sub-questions
  ├── Query expansion: generate hypothetical answers or related terms
  └── Adaptive routing: decide whether retrieval is needed at all

Document Processing & Retrieval
  ├── Reasoning-guided re-ranking: use CoT to reorder retrieved candidates
  ├── Iterative retrieval: use reasoning to identify missing evidence and re-query
  └── Evidence filtering: reason over retrieved passages to remove noise

Answer Generation
  ├── Chain-of-thought grounded generation
  ├── Faithful synthesis: attribute claims to retrieved sources
  └── Self-verification and refinement over retrieved evidence
```

**Representative methods:** Adaptive-RAG · Self-RAG · FLARE · IRCoT · ReAct · Self-Ask

---

### Pillar 2 — RAG-Enhanced Reasoning

*Using retrieved knowledge to supply missing premises and expand context for complex inference.*

```
Knowledge Types Retrieved:
  ├── Factual knowledge: entity facts, world knowledge
  ├── Procedural knowledge: how-to steps, domain procedures
  ├── Relational knowledge: knowledge graphs, structured relations
  └── Episodic knowledge: previous interactions, memory traces

Retrieval Targets:
  ├── Relevant passages (unstructured)
  ├── Knowledge graph triples (structured)
  ├── Code / tool documentation
  └── Multimodal content (images, tables)

Reasoning Improvement:
  ├── Supplies missing factual premises → reduces hallucination
  ├── Expands narrow parametric context → enables multi-hop reasoning
  └── Grounds intermediate claims → improves faithfulness
```

**Representative methods:** KG-RAG · StructRAG · Graph-of-Thought · RARE · AlignRAG

---

### Pillar 3 — Synergized RAG-Reasoning (Agentic)

*Iterative interleaving of retrieval and reasoning by an autonomous agent.*

```
Core Loop:
  [Observe context]
      ↓
  [Reason: identify knowledge gaps]
      ↓
  [Act: issue retrieval query / tool call]
      ↓
  [Retrieve: get external evidence]
      ↓
  [Integrate: update reasoning state]
      ↓
  [Terminate or loop again]

Implementation Strategies:
  ├── Prompt-based (no training):
  │     ReAct · Self-Ask · Search-o1 · WebThinker
  │     Lightweight; relies on instruction following
  │
  └── Training-based (RL or supervised):
        RL-trained search agents; RLVR on retrieval-augmented tasks
        Higher capability ceiling; requires retrieval-aware training signal
```

**Representative methods:** Search-o1 · WebThinker · AirRAG · Open-RAG · RAG-Critic · EviOmni

---

## Benchmarks & Datasets

| Benchmark | Task Type | Notes |
|-----------|-----------|-------|
| [HotpotQA](https://hotpotqa.github.io/) | Multi-hop QA | Requires 2-hop reasoning across Wikipedia articles |
| [MuSiQue](https://arxiv.org/abs/2108.00573) | Multi-hop QA | 2–4 hop compositional questions |
| [2WikiMultiHopQA](https://arxiv.org/abs/2011.01060) | Multi-hop QA | Cross-document reasoning |
| [FRAMES](https://arxiv.org/abs/2409.12941) | Factual reasoning | Multi-document synthesis |
| [BRIGHT](https://arxiv.org/abs/2407.12883) | Complex retrieval | Reasoning-intensive retrieval without keyword overlap |
| [KILT](https://arxiv.org/abs/2009.02252) | Knowledge-intensive | Unified benchmark: QA, fact verification, slot filling |
| [WebQSP](https://www.microsoft.com/en-us/research/publication/the-value-of-semantic-parse-labeling-for-knowledge-base-question-answering/) | KB-QA | Knowledge graph question answering |
| [AIME / MATH](https://arxiv.org/abs/2103.03874) | Math reasoning | Hard math requiring multi-step inference |

---

## Key Arguments

- **Static RAG is a reasoning bottleneck** — single-pass retrieval cannot handle tasks where each reasoning step reveals new information needs; only iterative search-reason loops can address this.
- **Retrieval grounds reasoning; reasoning directs retrieval** — the two capabilities are mutually reinforcing: retrieval reduces hallucination; reasoning enables targeted, adaptive search.
- **RL-trained agentic systems represent the frontier** — systems trained with reinforcement learning to interleave search and reasoning (post-DeepSeek-R1) achieve state-of-the-art on knowledge-intensive benchmarks.
- **Multimodality is the next frontier** — current RAG-Reasoning work is predominantly text-only; extending to image, table, and code retrieval is identified as a critical open challenge.
- **Trustworthiness requires source attribution** — agentic systems that reason over retrieved evidence must produce verifiable, attributable outputs; faithfulness to sources is a central open challenge.

---

## Companion Paper

The same group simultaneously released an agent-focused extension covering the paradigm shift from web search to autonomous deep research:

> Zhang, Li et al. (2025) — [*From Web Search towards Agentic Deep Research: Incentivizing Search with Reasoning Agents*](https://arxiv.org/abs/2506.18959) [arXiv:2506.18959]
> Traces the evolution from static web search to interactive, agent-based systems; introduces a test-time scaling law formalizing the impact of computational depth on reasoning and search quality.

---

## Open Challenges

| Challenge | Description |
|-----------|-------------|
| **Effective multi-hop retrieval** | Retrieving across long reasoning chains with dynamic, intermediate queries remains unsolved at scale |
| **Multimodal RAG-Reasoning** | Extending iterative retrieval to image, table, code, and audio modalities |
| **Trustworthiness and attribution** | Ensuring generated claims are faithfully grounded in retrieved sources |
| **RL reward design for retrieval** | Designing dense, fine-grained reward signals for retrieval-augmented RL training |
| **Human-centricity** | Aligning agentic search-reason systems with human intent, preferences, and interaction patterns |
| **Efficiency at deployment scale** | Iterative retrieval + reasoning loops are expensive; reducing latency without sacrificing quality |

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Coverage cutoff** | Submitted July 2025; rapidly evolving field means newer post-July 2025 systems are not covered |
| **English-centric** | Most surveyed methods and benchmarks are English-only; multilingual agentic RAG is underrepresented |
| **Empirical comparison absent** | As a survey, it categorizes and synthesizes rather than providing new controlled comparisons across methods |
| **Taxonomy boundary fuzziness** | The three-pillar framing (Reasoning-Enhanced RAG, RAG-Enhanced Reasoning, Synergized) has fuzzy boundaries — many systems span multiple categories |
| **Preprint status** | EMNLP camera-ready version pending; final classification and framing may differ |

---

## Citation

```bibtex
@article{li2025towards,
  title   = {Towards Agentic {RAG} with Deep Reasoning: A Survey of {RAG}-Reasoning Systems in {LLM}s},
  author  = {Li, Yangning and Zhang, Weizhi and Yang, Yuyao and Huang, Wei-Chieh and Wu, Yaozu and Luo, Junyu and Bei, Yuanchen and Zou, Henry Peng and Luo, Xiao and Zhao, Yusheng and others},
  journal = {arXiv preprint arXiv:2507.09477},
  year    = {2025},
  url     = {https://arxiv.org/abs/2507.09477}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| RAG: Retrieval-Augmented Generation (Lewis et al., 2020) | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) |
| Self-RAG: Learning to Retrieve, Generate, Critique (Asai et al., 2023) | [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) |
| ReAct: Synergizing Reasoning and Acting (Yao et al., 2023) | [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) |
| IRCoT: Interleaving Retrieval with CoT Reasoning (Trivedi et al., 2022) | [arXiv:2212.10509](https://arxiv.org/abs/2212.10509) |
| Search-o1: Agentic Search with Reasoning (Li et al., 2025) | [arXiv:2501.05366](https://arxiv.org/abs/2501.05366) |
| DeepSeek-R1: Incentivizing Reasoning via RL (DeepSeek-AI, 2025) | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| Agentic RAG: A Survey on Agentic RAG (Singh et al., 2025) | [arXiv:2501.09136](https://arxiv.org/abs/2501.09136) |
| From Web Search to Agentic Deep Research (Zhang et al., 2025) | [arXiv:2506.18959](https://arxiv.org/abs/2506.18959) |
| XRAG: Cross-lingual RAG Benchmark (Liu et al., 2025) | [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.849/) |
| A Survey on Context Engineering for LLMs (Mei et al., 2025) | [arXiv:2507.13334](https://arxiv.org/abs/2507.13334) |
| BRIGHT: Reasoning-Intensive Retrieval Benchmark (Su et al., 2024) | [arXiv:2407.12883](https://arxiv.org/abs/2407.12883) |
| HotpotQA: Multi-hop QA (Yang et al., 2018) | [arXiv:1809.09600](https://arxiv.org/abs/1809.09600) |
