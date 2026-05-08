# Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability

> **arXiv:** [arXiv:2601.18778](https://arxiv.org/abs/2601.18778)
> **PDF:** [View PDF](https://arxiv.org/pdf/2601.18778)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2601.18778v1)
> **Blog Post:** [ssundaram21.github.io/soar](https://ssundaram21.github.io/soar/)
> **HuggingFace Papers:** [HF Paper Page](https://huggingface.co/papers/2601.18778)
> **ResearchGate:** [Full Text](https://www.researchgate.net/publication/400085337_Teaching_Models_to_Teach_Themselves_Reasoning_at_the_Edge_of_Learnability)
> **Semantic Scholar:** [Paper Details](https://api.semanticscholar.org/arXiv:2601.18778)

---

## Authors

| Name | Affiliation |
|------|-------------|
| Shobhita Sundaram|
| John Quan 
| Ariel Kwiatkowski 
| Kartik Ahuja 
| Yann Ollivier 

**Submitted:** January 26, 2026 (v1); Updated February 6, 2026 (v2)
**Category:** Machine Learning (cs.LG); Computation and Language (cs.CL)


## Introduction

Reinforcement learning from verifiable rewards (RLVR) has become a dominant paradigm for improving the reasoning capabilities of LLMs. Methods like GRPO and PPO fine-tune models by rewarding correct answers on math, code, and logic problems. However, this training signal completely collapses when the model cannot solve *any* instance of a hard problem set — a regime the authors call the **edge of learnability**. With 0% initial success rate, there are no positive reward signals to learn from, and standard RL simply stalls.

This paper addresses a fundamental question: can a pretrained LLM leverage its own latent knowledge to *construct a curriculum* for problems it cannot yet solve, without relying on expert-curated intermediate data? The answer is SOAR (**S**elf-**O**ptimization via **A**symmetric **R**L), a bilevel meta-RL framework in which a teacher copy of the model generates synthetic stepping-stone problems for a student copy, and is rewarded based directly on measured student improvement — not on proxy signals like problem difficulty or intrinsic curiosity. The key insight is that the teacher's policy is sharpened entirely through the student's progress on hard target problems, bootstrapping learning from a state of zero success.

---

## Problem Statement

1. **RLVR stalls at zero success rate** — standard RL fine-tuning for reasoning requires at least some correct solutions in training to generate reward signal; on the hardest problem subsets (0/128 success), it produces no learning.
2. **Expert-curated intermediate data is rarely available** — curriculum learning typically requires hand-designed stepping-stone datasets; this assumption breaks down for novel or highly specialized problem domains.
3. **Proxy rewards are fragile** — prior self-improvement approaches reward the teacher for generating "difficult" or "interesting" problems using intrinsic metrics; these proxies do not reliably correlate with student learning.
4. **Synthetic problems alone may not generalize** — it is unclear whether self-generated curricula teach transferable reasoning or merely memorize surface patterns of the training distribution.

---

## Contributions

1. **Introduces SOAR**, a bilevel meta-RL self-improvement framework that unlocks learning under sparse, binary rewards at zero initial success rate.
2. **Demonstrates that self-generated curricula can bootstrap learning** on the hardest subsets of mathematical benchmarks where standard RL produces no progress.
3. **Shows OOD generalization** — models trained with SOAR on one benchmark transfer to held-out datasets without any direct optimization for them.
4. **Establishes a new evaluation regime** — studying the fail@128 subset (problems no single sample of 128 attempts solved) as a principled frontier for testing the limits of RL fine-tuning.

---

## Framework: SOAR

SOAR implements a **bilevel meta-RL** loop with asymmetric roles for teacher and student:

```
Outer loop — Teacher optimization:
  Teacher model π_T generates synthetic stepping-stone problems P_synthetic
  Teacher is rewarded based on measured improvement of student on hard target problems P_hard
  Teacher never sees P_hard directly; it only observes student progress as its reward signal

Inner loop — Student training:
  Student model π_S trains on P_synthetic via standard RLVR (binary correctness reward)
  Student is periodically evaluated on P_hard to provide the teacher's reward signal

Key property:
  No intrinsic or proxy rewards — teacher reward is grounded in measured student progress
  No expert-curated intermediate data — curriculum emerges entirely from self-generation
```

### SOAR Configurations Evaluated

| Config | Description |
|--------|-------------|
| **PS (Problem Synthesis)** | Teacher generates synthetic problems in the same domain as P_hard |
| **PQ (Problem + Question)** | Teacher generates synthetic problems paired with reasoning-guided questions |
| **Hard-Only baseline** | Standard RL directly on P_hard (stalls at 0%) |
| **Intrinsic baseline** | Teacher rewarded by proxy difficulty signals (not student progress) |

---

## Experimental Setup

| Property | Details |
|----------|---------|
| **Primary Task** | Mathematical reasoning |
| **Benchmarks** | [MATH](https://arxiv.org/abs/2103.03874) (fail@128 subset), [HARP](https://arxiv.org/abs/2410.11743), [OlympiadBench](https://arxiv.org/abs/2402.14008) (OOD) |
| **Evaluation Regime** | fail@128 — problems with 0 correct solutions out of 128 sampled attempts |
| **Base Model** | Pretrained LLM (decoder-only; specific model disclosed in paper) |
| **Training Signal** | Binary correctness reward (RLVR) for student; student progress delta for teacher |

---

## Key Results

- **PS achieves +8.5% pass@32** on fail@128-MATH over the Hard-Only baseline that produces no learning.
- **PS achieves +3.6% pass@32** on fail@128-HARP over Hard-Only.
- **OOD transfer to OlympiadBench**: PQ-MATH and PQ-HARP achieve **+6% and +3%** respectively over Hard-Only on a fully held-out benchmark, despite no direct OOD optimization.
- **Student learning curves continue improving** after transitioning from synthetic curriculum to fail@128 direct training, indicating that synthetic stepping stones produce lasting shifts in the student's policy rather than temporary surface gains.
- **Intrinsic reward baselines underperform** SOAR across all settings, confirming that grounding teacher reward in measured student progress is critical.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Error propagation risk** | If the teacher generates incorrect or misleading synthetic problems, these errors can compound through student training; careful filtering of generated data is necessary. |
| **Teacher LLM inference cost** | Running a teacher copy of the model in the outer meta-RL loop adds significant training-time compute compared to standard RL fine-tuning. |
| **Domain specificity** | Experiments focus on mathematical reasoning; generalization of SOAR to code, science, or multi-step planning tasks is not yet established. |
| **Preprint status** | Not yet peer reviewed; empirical conclusions should be treated as preliminary. |
| **Cold-start dependency on latent knowledge** | SOAR relies on the teacher having sufficient latent knowledge to generate useful stepping stones; it may not generalize to domains entirely outside the pretraining distribution. |

---

## Citation

```bibtex
@misc{sundaram2026teachingmodels,
  title         = {Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability},
  author        = {Shobhita Sundaram and John Quan and Ariel Kwiatkowski and Kartik Ahuja and Yann Ollivier and Julia Kempe},
  year          = {2026},
  eprint        = {2601.18778},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2601.18778}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (DeepSeek, 2025) | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| GRPO: Group Relative Policy Optimization (Shao et al., 2024) | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| MATH: Measuring Mathematical Problem Solving (Hendrycks et al., 2021) | [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) |
| HARP: Hard Mathematical Reasoning Problems (Yue et al., 2024) | [arXiv:2410.11743](https://arxiv.org/abs/2410.11743) |
| OlympiadBench: Bilingual Olympiad-Level Benchmark (He et al., 2024) | [arXiv:2402.14008](https://arxiv.org/abs/2402.14008) |
| Self-Play Fine-Tuning (SPIN) (Chen et al., 2024) | [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) |
| STaR: Bootstrapping Reasoning with Reasoning (Zelikman et al., 2022) | [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) |
| Curriculum Learning (Bengio et al., 2009) | [ICML 2009](https://dl.acm.org/doi/10.1145/1553374.1553380) |
| DARC: Decoupled Asymmetric Reasoning Curriculum (2026) | [arXiv:2602.01999](https://arxiv.org/abs/2602.01999) |
