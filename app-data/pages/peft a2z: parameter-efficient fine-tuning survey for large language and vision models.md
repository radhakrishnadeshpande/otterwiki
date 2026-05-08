# PEFT A2Z: Parameter-Efficient Fine-Tuning Survey for Large Language and Vision Models

> **arXiv:** [arXiv:2504.14117](https://arxiv.org/abs/2504.14117)
> **PDF:** [View PDF](https://arxiv.org/pdf/2504.14117)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2504.14117v1)
> **GitHub (Awesome List):** [Nusrat-Prottasha/PEFT-A2Z](https://github.com/Nusrat-Prottasha/PEFT-A2Z)
> **ResearchGate:** [Full Text](https://www.researchgate.net/publication/390990551_PEFT_A2Z_Parameter-Efficient_Fine-Tuning_Survey_for_Large_Language_and_Vision_Models)
> **Semantic Scholar:** [Paper Details](https://api.semanticscholar.org/arXiv:2504.14117)

---

## Authors

| Name | Affiliation |
|------|-------------|
| Nusrat Jahan Prottasha | University of Central Florida, USA |
| Upama Roy Chowdhury *(equal contribution)* | Khulna University of Engineering & Technology, Bangladesh |
| Shetu Mohanto *(equal contribution)* | Delineate Inc., USA |
| Tasfia Nuzhat *(equal contribution)* | Universiti Tenaga Nasional, Malaysia |
| Abdullah As Sami *(equal contribution)* | University of South Florida, USA |
| Md Shamol Ali *(equal contribution)* | Daffodil International University, Bangladesh |
| Md Shohanur Islam Sobuj | Anymate Me, Germany |
| Hafijur Raman | University of Central Florida, USA |
| Md Kowsher | University of Central Florida, USA |
| Ozlem Ozmen Garibay | University of Central Florida, USA |

**Submitted:** April 19, 2025
**Category:** Computation and Language (cs.CL)
**Evolution Timeline Covered:** 2019 → 2025

## Introduction

The rise of large-scale pretrained models — spanning LLMs such as GPT-4 and LLaMA, and Vision Language Models (VLMs) such as CLIP and LLaVA — has fundamentally shifted how AI systems are built and deployed. These models achieve strong performance across NLP, computer vision, and multimodal tasks, but adapting them to downstream use cases through full fine-tuning is computationally prohibitive for most practitioners. Full fine-tuning requires updating billions of parameters, consuming extensive GPU memory, task-specific data, and time — and introduces risks such as catastrophic forgetting and overfitting on small target datasets.

Parameter-Efficient Fine-Tuning (PEFT) has emerged as the practical solution: instead of updating all model weights, PEFT methods adapt large pretrained models by modifying only a small, targeted subset of parameters while keeping the base model frozen. This enables high-performance task adaptation with a fraction of the compute and memory cost of full fine-tuning. Despite its growing adoption, the PEFT landscape is fragmented — methods vary significantly in how they insert, select, or reparameterize parameters, and comparisons across language and vision modalities are rarely unified.

This survey provides an A-to-Z treatment of the field: from motivation and foundational concepts through a structured five-class taxonomy, cross-domain applications, and open challenges in scalability, interpretability, and theoretical grounding.

---

## Problem Statement

1. **Full fine-tuning is computationally prohibitive** — updating all parameters of billion-scale models requires GPU clusters unavailable to most research groups and practitioners.
2. **Catastrophic forgetting and overfitting** — standard fine-tuning on small target datasets risks degrading general capabilities of the pretrained model.
3. **Fragmented PEFT literature** — methods are scattered across additive, selective, reparameterization, hybrid, and unified approaches with no systematic cross-domain comparison.
4. **Architecture and task dependency** — PEFT method effectiveness varies significantly with the underlying model architecture and the nature of the downstream task, creating an unsolved method-selection problem.

---

## Contributions

1. **Comprehensive resource analysis** of the computational, memory, and storage demands of full fine-tuning of large-scale pretrained models (PLMs and LLMs).
2. **Five-class unified taxonomy** of PEFT methods: additive, selective, reparameterized, hybrid, and unified frameworks — with systematic comparison of mechanisms and trade-offs.
3. **Cross-domain coverage** spanning NLP, computer vision, multimodal tasks, and generative modeling.
4. **Open challenges and future directions** including federated learning, domain adaptation, theoretical grounding, interpretability, and robustness.

---

## Taxonomy of PEFT Methods

The survey organizes PEFT into five major categories:

| Category | Core Idea | Representative Methods |
|----------|-----------|----------------------|
| **Additive** | Insert new trainable modules or parameters into the frozen model | Adapters, Prefix Tuning, Prompt Tuning, LoRA |
| **Selective** | Select and fine-tune a small existing subset of model parameters | BitFit, Diff Pruning, Sparse Fine-Tuning |
| **Reparameterized** | Reparameterize weight updates as low-rank or structured decompositions | LoRA, IA³, KronA, AdaLoRA |
| **Hybrid** | Combine additive and selective or reparameterized strategies | MAM Adapter, UniPELT, LLaMA-Adapter |
| **Unified** | A single framework that subsumes or selects among multiple PEFT strategies automatically | S4, SparseAdapter, AutoPEFT |

---

## Domains Covered

- **NLP** — text classification, question answering, summarization, instruction following; evaluated on GLUE, SuperGLUE, SQuAD, and related QA benchmarks
- **Computer Vision** — image classification, object detection, segmentation; evaluated using pretrained Vision Transformers (ViT) and CLIP
- **Multimodal / VLMs** — cross-modal alignment and visual question answering with LLaVA and similar architectures
- **Generative Modeling** — text generation and image synthesis adaptation

---

## Key Findings

- **PEFT matches full fine-tuning** in many tasks while updating less than 1% of total parameters in the best cases.
- **Additive methods (especially LoRA)** have become the dominant practical choice due to their simplicity, modularity, and strong empirical performance.
- **Reparameterization approaches (LoRA family)** show the best balance of parameter efficiency and task performance across language tasks.
- **Prompt-based methods** are more sensitive to initialization and task format than adapter or LoRA variants.
- **Architecture compatibility** is a key constraint: not all PEFT methods transfer cleanly from encoder-only (BERT) to decoder-only (GPT) or encoder-decoder (T5) architectures.
- **Vision PEFT** is less mature than language PEFT; adapter-based approaches for ViTs are effective but hybrid strategies remain underexplored.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Currency** | The survey covers methods through April 2025; rapidly released post-survey innovations (e.g., newer LoRA variants, MoE-PEFT) may not be included. |
| **Architecture dependency** | PEFT method effectiveness is architecture-specific; the survey's comparisons may not generalize to all model families. |
| **Task design sensitivity** | PEFT performance is sensitive to prompt format, initialization, and task framing — factors not always controlled across compared papers. |
| **No new empirical results** | As a survey, it synthesizes prior results rather than providing new controlled experiments across a unified evaluation setting. |
| **Preprint status** | Not yet peer reviewed; classifications and conclusions reflect the authors' synthesis. |

---

## Citation

```bibtex
@article{prottasha2025peft,
  title   = {PEFT A2Z: Parameter-Efficient Fine-Tuning Survey for Large Language and Vision Models},
  author  = {Prottasha, Nusrat Jahan and Chowdhury, Upama Roy and Mohanto, Shetu and Nuzhat, Tasfia and Sami, Abdullah As and Ali, Md Shamol and Sobuj, Md Shohanur Islam and Raman, Hafijur and Kowsher, Md and Garibay, Ozlem Ozmen},
  journal = {arXiv preprint arXiv:2504.14117},
  year    = {2025},
  url     = {https://arxiv.org/abs/2504.14117}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2022) | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| The Power of Scale for Parameter-Efficient Prompt Tuning (Lester et al., 2021) | [arXiv:2104.08691](https://arxiv.org/abs/2104.08691) |
| Prefix-Tuning: Optimizing Continuous Prompts for Generation (Li & Liang, 2021) | [arXiv:2101.00190](https://arxiv.org/abs/2101.00190) |
| Parameter-Efficient Transfer Learning for NLP — Adapters (Houlsby et al., 2019) | [arXiv:1902.00751](https://arxiv.org/abs/1902.00751) |
| AdaLoRA: Adaptive Budget Allocation for PEFT (Zhang et al., 2023) | [arXiv:2303.10512](https://arxiv.org/abs/2303.10512) |
| QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023) | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| BitFit: Simple Parameter-efficient Fine-tuning for Transformers (Zaken et al., 2022) | [arXiv:2106.10199](https://arxiv.org/abs/2106.10199) |
| PEFT for Pre-trained Vision Models: A Survey (Xin et al., 2024) | [arXiv:2402.02242](https://arxiv.org/abs/2402.02242) |
| UniPELT: A Unified Framework for PEFT of Pre-trained Language Models (Mao et al., 2022) | [arXiv:2110.07577](https://arxiv.org/abs/2110.07577) |
| Scaling Down to Scale Up: A Guide to PEFT (Lialin et al., 2023) | [arXiv:2303.15647](https://arxiv.org/abs/2303.15647) |
