# MiniLLM: Knowledge Distillation of Large Language Models

> **arXiv:** [arXiv:2306.08543](https://arxiv.org/abs/2306.08543)
> **PDF:** [View PDF](https://arxiv.org/pdf/2306.08543)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2306.08543)
> **GitHub:** [microsoft/LMOps/minillm](https://github.com/microsoft/LMOps/tree/main/minillm)
> **HuggingFace Papers:** [HF Paper Page](https://huggingface.co/papers/2306.08543)
> **OpenReview:** [ICLR 2024 Submission](https://openreview.net/forum?id=5h0qf7IBZZ)
> **Semantic Scholar:** [Paper Details](https://api.semanticscholar.org/arXiv:2306.08543)


## Introduction

Knowledge distillation (KD) has long served as a standard technique for compressing neural networks — training a smaller student model to mimic a larger teacher model, thereby preserving predictive quality at reduced computational cost. For classification models, this typically means minimizing the forward Kullback-Leibler divergence (KLD) between the teacher and student output distributions. However, when applied to generative LLMs, forward KLD introduces a fundamental mismatch: teacher distributions over vocabulary are highly multimodal, and forward KLD forces the student to spread probability mass over all modes of the teacher — including low-probability "void" regions. This results in student models that systematically overestimate unlikely token sequences, producing diffuse, imprecise, and poorly calibrated generations.

MiniLLM addresses this by replacing forward KLD with **reverse KLD** as the distillation objective. Reverse KLD is mode-seeking rather than mean-seeking: the student concentrates its probability mass on the most probable teacher outputs rather than averaging over all of them. This alignment property is far better suited to generative language modeling, where generating one precise, high-quality response is more valuable than covering every plausible teacher output. To optimize this objective tractably, MiniLLM derives an **on-policy optimization algorithm** — sampling from the student's own distribution during training — which addresses the exposure bias problem inherent in standard sequence-level distillation methods.

---

## Problem Statement

1. **Forward KLD is misaligned with generation** — standard KD minimizes forward KLD, which forces students to overestimate low-probability regions of the teacher distribution, leading to imprecise, over-hedged generation.
2. **Prior KD methods do not target generative LLMs** — existing KD literature focuses on classification models or black-box API imitation (SFT on ChatGPT outputs); distillation of white-box open-source generative LLMs is underexplored.
3. **Exposure bias in sequence-level KD** — methods like SeqKD train students on teacher-generated sequences but do not expose the student to its own outputs during training, causing a distribution shift at inference time.
4. **Deployment constraint** — LLMs with billions of parameters cannot be deployed at the edge or in resource-constrained settings without effective compression.

---

## Contributions

1. **Identifies forward KLD as the root cause of student overestimation** in generative distillation and provides a formal justification for switching to reverse KLD.
2. **Derives an on-policy optimization algorithm** for minimizing reverse KLD in the generative setting, reducing exposure bias compared to sequence-level KD baselines.
3. **Demonstrates consistent gains** over SFT and SeqKD baselines in instruction-following quality, calibration, and long-text generation across multiple model families and scales.
4. **Shows positive scaling behavior** — student performance consistently improves as teacher model size increases, unlike some prior distillation methods where larger teachers do not always help.
5. **Releases code, data, and model checkpoints** across GPT-2 and OPT family model sizes (120M to 13B parameters).

---

## Method: Reverse KLD Distillation

### Why Not Forward KLD?

```
Forward KLD:  min KL(p_teacher || q_student)
  → Student must cover all modes of the teacher
  → Forces mass onto low-probability "void" regions
  → Results in over-diffuse, imprecise, miscalibrated outputs

Reverse KLD:  min KL(q_student || p_teacher)  ← MiniLLM
  → Student concentrates mass on the most probable teacher outputs
  → Mode-seeking: prefers one precise response over averaging across many
  → Produces more precise, better-calibrated, higher-quality generations
```

### On-Policy Optimization

Reverse KLD requires computing expectations under the student's own distribution. MiniLLM derives a policy-gradient-style on-policy algorithm:

```
Training loop:
  1. Sample full response sequence from student π_S (not from teacher or fixed dataset)
  2. Score the sampled sequence under both teacher p_T and student π_S
  3. Compute reverse KLD gradient using the sampled sequence
  4. Update student parameters

Key property:
  Student trains on its own outputs → eliminates train/inference distribution gap (exposure bias)
```

---

## Experimental Setup

| Property | Details |
|----------|---------|
| **Task** | Instruction following (open-ended generation) |
| **Training Data** | Dolly, Self-Instruct, Unnatural Instructions, Alpaca, Vicuna, OpenAssistant |
| **Evaluation** | GPT-4 feedback scoring (pairwise quality judgments) |
| **Model Families** | GPT-2 (125M → 1.5B) · OPT (1.3B → 13B) |
| **Teacher Models** | GPT-2-1.5B · GPT-J-6B · OPT-13B |
| **Student Models** | GPT-2-125M, 340M, 760M · GPT-Neo-2.7B · OPT-1.3B, 2.7B, 6.7B |
| **Baselines** | SFT (standard fine-tuning) · SeqKD (sequence-level KD) |

---

## Key Results

- **MiniLLM consistently outperforms SeqKD** on GPT-4 feedback scoring across all student–teacher pairs.
- **Lower exposure bias** — student models trained with on-policy sampling generalize better to novel inputs at inference time than SeqKD counterparts.
- **Better calibration** — MiniLLM students produce more confident correct outputs and less confident incorrect outputs than SeqKD.
- **Higher long-text generation performance** — gains are especially pronounced on responses requiring extended reasoning or multi-sentence outputs, where overestimating low-probability regions most severely degrades quality.
- **Positive teacher scaling** — unlike findings in some prior work (e.g., teaching assistant methods), MiniLLM student quality consistently improves as teacher model size increases from 340M to 1.5B (GPT-2 family).
- **Good generation diversity** — reverse KLD training does not cause mode collapse; students maintain healthy output diversity.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Distilled models may lack full reasoning capacity** | Despite improvements over SeqKD, student models remain bounded by their own parameter count; complex multi-hop reasoning may not fully transfer from the teacher. |
| **Requires white-box teacher access** | MiniLLM requires access to the teacher's full output distribution; it cannot be applied to black-box APIs (e.g., GPT-4 via API only). |
| **On-policy training cost** | Sampling from the student at each training step is more expensive than standard supervised training on fixed datasets. |
| **Evaluation reliance on GPT-4 feedback** | Quality judgments depend on GPT-4 as a judge, which may introduce model-specific biases. |
| **Experiments limited to English instruction following** | Cross-lingual distillation and task-specific fine-tuning settings are not evaluated. |

---

## Citation

```bibtex
@inproceedings{gu2024minillm,
  title     = {{MiniLLM}: Knowledge Distillation of Large Language Models},
  author    = {Yuxian Gu and Li Dong and Furu Wei and Minlie Huang},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=5h0qf7IBZZ}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| Distilling the Knowledge in a Neural Network (Hinton et al., 2015) | [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) |
| DistilBERT: A Distilled Version of BERT (Sanh et al., 2019) | [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) |
| Sequence-Level Knowledge Distillation (Kim & Rush, 2016) | [ACL Anthology](https://aclanthology.org/D16-1139/) |
| GKD: Generalized Knowledge Distillation for Auto-regressive Models (Agarwal et al., 2023) | [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) |
| f-Divergence Minimization for Sequence-Level KD (Wen et al., 2023) | [ACL 2023](https://aclanthology.org/2023.acl-long.605/) |
| Self-Instruct: Aligning LMs with Self-Generated Instructions (Wang et al., 2023) | [arXiv:2212.10560](https://arxiv.org/abs/2212.10560) |
| Alpaca: Instruction-Following LLaMA (Taori et al., 2023) | [GitHub](https://github.com/tatsu-lab/stanford_alpaca) |
| DRAG: Distilling RAG for SLMs from LLMs (Chen et al., 2025) | [ACL Anthology](https://aclanthology.org/2025.acl-long.358/) |
| Knowledge Distillation Survey (Gou et al., 2021) | [arXiv:2006.05525](https://arxiv.org/abs/2006.05525) |
