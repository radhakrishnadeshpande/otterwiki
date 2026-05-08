# A Survey on Knowledge Distillation of Large Language Models

> **arXiv:** [arXiv:2402.13116](https://arxiv.org/abs/2402.13116)
> **PDF:** [View PDF](https://arxiv.org/pdf/2402.13116)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2402.13116v4)
> **GitHub (Awesome List):** [Tebmer/Awesome-Knowledge-Distillation-of-LLMs](https://github.com/Tebmer/Awesome-Knowledge-Distillation-of-LLMs)
> **Semantic Scholar:** [Paper Details](https://api.semanticscholar.org/arXiv:2402.13116)

---


## Introduction

The emergence of powerful proprietary LLMs such as GPT-4, Claude, and Gemini has set a new ceiling for AI capability, but also created a structural divide: these models are computationally expensive, closed-source, and inaccessible to most researchers and practitioners. Knowledge Distillation (KD) has emerged as the primary mechanism for bridging this divide — enabling the transfer of advanced capabilities from large teacher models (proprietary or open-source) to smaller, more efficient student models that can be deployed at lower cost.

The landscape of KD for LLMs has evolved considerably from classical KD frameworks originally designed for classification models. In the LLM era, KD now serves three distinct roles: (1) transferring advanced capabilities from proprietary models like GPT-4 to open-source counterparts such as LLaMA and Mistral, democratizing access to frontier-level capabilities; (2) compressing large open-source models into smaller, resource-efficient versions; and (3) enabling self-improvement, where open-source LLMs act as teachers for their own fine-tuned variants. The intersection of KD with data augmentation has become particularly prominent — rather than distilling logits or hidden states alone, many modern approaches use the teacher LLM to *generate* training data, expanding small seed datasets into rich skill-specific corpora. This survey provides the first comprehensive taxonomy of KD in the LLM era, structured around algorithms, skill distillation, and domain-specific verticalization.

---

## Problem Statement

1. **Capability gap between proprietary and open-source LLMs** — leading proprietary models possess emergent capabilities (instruction following, reasoning, alignment) that smaller open-source models lack, limiting equitable access to state-of-the-art AI.
2. **Full fine-tuning is resource-prohibitive** — training or fully fine-tuning billion-parameter models requires compute unavailable to most research groups.
3. **Classical KD methods are misaligned with generative LLMs** — traditional KD approaches designed for classification (logit matching, hidden state alignment) do not transfer cleanly to open-ended generation settings.
4. **Fragmented literature** — KD techniques for LLMs span supervision strategies, data augmentation, divergence minimization, RL-based alignment, and domain-specific fine-tuning, with no unified framework connecting them.

---

## Contributions

1. **Defines three roles of KD in LLMs** — capability transfer from proprietary models, model compression, and self-improvement — as a unifying framework for the field.
2. **Presents a three-pillar taxonomy** structured around KD Algorithms, Skill Distillation, and Verticalization Distillation.
3. **Synthesizes KD algorithms** — covering knowledge elicitation strategies (labeling, expansion, curation) and distillation learning approaches (SFT, divergence minimization, RL, rank optimization).
4. **Surveys skill distillation** across context following, alignment, reasoning, multi-modality, NLU, and RAG capability.
5. **Reviews domain-specific verticalization** across law, medical/healthcare, finance, science, and other specialized fields.
6. **Addresses legal considerations** around using proprietary LLM outputs for distillation, with a firm advocacy for compliance with model provider terms of use.

---

## Taxonomy Overview

The survey is organized around three foundational pillars:

### Pillar 1 — KD Algorithms

```
Knowledge Elicitation (how to extract knowledge from teacher LLMs):
  ├── Labeling       — Teacher generates labels/annotations for student training
  ├── Expansion      — Teacher generates more data from a small seed (data augmentation)
  ├── Curation       — Teacher selects or filters high-quality training examples
  ├── Feature        — Transfer intermediate representations or attention patterns
  ├── Feedback       — Teacher provides scalar or ranking-based reward signals
  └── Self-Knowledge — Model uses its own generations to improve itself

Distillation Algorithms (how students learn from teacher knowledge):
  ├── Supervised Fine-Tuning (SFT)        — Standard next-token prediction on teacher outputs
  ├── Divergence & Similarity Minimization — Forward/reverse KLD, JS divergence (see MiniLLM)
  ├── Reinforcement Learning              — PPO, GRPO with teacher-derived reward signals
  └── Rank Optimization                  — DPO, RLHF-style preference learning
```

### Pillar 2 — Skill Distillation

| Skill | Description |
|-------|-------------|
| **Context Following** | Instruction following, in-context learning |
| **Alignment** | Thinking patterns, persona/preference modeling, value alignment |
| **Reasoning** | Chain-of-Thought, mathematical, commonsense, multi-step reasoning |
| **Planning** | Tool use, agent-style task decomposition |
| **NLU** | Text classification, NER, relation extraction |
| **RAG Capability** | Retrieval-augmented generation skill transfer |
| **Multi-Modality** | Vision-language, multimodal instruction following |
| **Multilingual** | Cross-lingual capability distillation |

### Pillar 3 — Verticalization Distillation

Domain-specific KD for: **Law · Medical & Healthcare · Finance · Science · Code · Education**

---

## Key Arguments

- **Data augmentation has become the dominant KD paradigm for LLMs** — rather than matching intermediate representations, most modern approaches use teacher LLMs to generate rich skill-specific training datasets from small seeds.
- **KD narrows (and in some cases closes) the proprietary–open-source gap** — Alpaca, Vicuna, WizardLM, and related models demonstrate that targeted distillation from GPT-class teachers produces student models that rival much larger open-source baselines.
- **Self-improvement is an emerging third paradigm** — beyond teacher-student transfer and compression, open-source LLMs using themselves as teachers (via self-play, self-refinement, or self-distillation) represent a growing research direction.
- **Legal compliance is non-negotiable** — the survey explicitly flags the legal constraints on using GPT-4, ChatGPT, LLaMA, and similar models' outputs for commercial distillation purposes.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Rapid pace of field** | KD for LLMs is evolving extremely quickly; the v4 update (October 2024) is the most recent, and significant post-2024 methods may not be covered. |
| **Coverage scope** | Despite surveying 300+ papers, the field is vast; the authors acknowledge the survey may not encompass every recent development. |
| **Legal landscape variability** | Model provider terms of use change frequently; the survey's legal guidance reflects conditions at time of writing. |
| **Preprint status** | Not yet peer reviewed; taxonomy design and framing reflect the authors' perspective. |
| **Evaluation diversity** | Distillation quality is measured inconsistently across the literature; no unified evaluation protocol exists. |

---

## Citation

```bibtex
@misc{xu2024survey,
  title         = {A Survey on Knowledge Distillation of Large Language Models},
  author        = {Xiaohan Xu and Ming Li and Chongyang Tao and Tao Shen and Reynold Cheng and Jinyang Li and Can Xu and Dacheng Tao and Tianyi Zhou},
  year          = {2024},
  eprint        = {2402.13116},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2402.13116}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| Distilling the Knowledge in a Neural Network (Hinton et al., 2015) | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) |
| MiniLLM: Knowledge Distillation of LLMs via Reverse KLD (Gu et al., 2024) | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
| GKD: Generalized Knowledge Distillation for Auto-regressive Seq Models (Agarwal et al., 2023) | [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) |
| Alpaca: Instruction-Following LLaMA (Taori et al., 2023) | [GitHub](https://github.com/tatsu-lab/stanford_alpaca) |
| WizardLM: Empowering LLMs with Complex Instructions (Xu et al., 2023) | [arXiv:2304.12244](https://arxiv.org/abs/2304.12244) |
| DRAG: Distilling RAG for SLMs from LLMs (Chen et al., 2025) | [ACL Anthology](https://aclanthology.org/2025.acl-long.358/) |
| Self-Instruct: Aligning LMs with Self-Generated Instructions (Wang et al., 2023) | [arXiv:2212.10560](https://arxiv.org/abs/2212.10560) |
| Survey on KD for LLMs: Methods, Evaluation, Application (2024) | [arXiv:2407.01885](https://arxiv.org/abs/2407.01885) |
| PEFT A2Z: Parameter-Efficient Fine-Tuning Survey (Prottasha et al., 2025) | [arXiv:2504.14117](https://arxiv.org/abs/2504.14117) |
| Knowledge Distillation Survey (Gou et al., 2021) | [arXiv:2006.05525](https://arxiv.org/abs/2006.05525) |
