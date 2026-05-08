# Reinforcement Learning via Self-Distillation

> **arXiv:** [arXiv:2601.20802](https://arxiv.org/abs/2601.20802)
> **PDF:** [View PDF](https://arxiv.org/pdf/2601.20802)
> **Project Page:** [self-distillation.github.io/SDPO](https://self-distillation.github.io/SDPO)
> **GitHub:** [lasgroup/SDPO](https://github.com/lasgroup/SDPO)
> **HuggingFace Papers:** [HF Paper Page](https://huggingface.co/papers/2601.20802)
> **ResearchGate:** [Full Text](https://www.researchgate.net/publication/400178402_Reinforcement_Learning_via_Self-Distillation)
> **Semantic Scholar:** [Paper Details](https://api.semanticscholar.org/arXiv:2601.20802)

---


## Introduction

The dominant paradigm for post-training LLMs on reasoning tasks is Reinforcement Learning with Verifiable Rewards (RLVR) — training models with scalar binary feedback on correctness of answers to math problems or outcomes of code execution. Methods like GRPO and PPO have shown that this approach can meaningfully improve reasoning capability in LLMs. However, RLVR faces a fundamental credit-assignment bottleneck: a single scalar outcome reward (correct/incorrect) reveals nothing about *which specific tokens* led to the failure. The model must deduce the source of errors through massive exploration and high-variance gradient estimation — an inefficient and often unstable process, particularly on hard problems where success is rare.

The crucial observation motivating this paper is that most verifiable environments already emit **rich textual feedback** as a byproduct of verification — compiler error messages, stack traces, runtime exceptions, failed test case logs, or judge evaluations that precisely explain *why* an attempt failed. This information is routinely discarded in standard RLVR pipelines, which reduce the entire feedback signal to a single scalar. SDPO recovers and exploits this discarded signal by formalizing a new setting called **Reinforcement Learning with Rich Feedback (RLRF)** and introducing **Self-Distillation Policy Optimization (SDPO)** — an online RL algorithm that converts tokenized feedback into a dense, logit-level credit assignment signal using the model's own in-context reasoning ability, with no external teacher and no learned reward model.

---

## Problem Statement

1. **RLVR creates a severe credit-assignment bottleneck** — scalar binary rewards mask the underlying environment state, forcing high-variance gradient estimation and slow learning from failures.
2. **Rich textual feedback is routinely discarded** — compiler errors, runtime exceptions, and judge evaluations that explicitly localize failures are ignored in standard RLVR pipelines.
3. **Dense supervision requires expensive external teachers** — the obvious alternative (token-level distillation from a stronger teacher) requires access to a larger proprietary model, which is unavailable in many online learning settings.
4. **GRPO advantage collapse on sparse rewards** — on hard problems where all sampled responses are incorrect, GRPO advantages collapse to zero and training stalls entirely.

---

## Contributions

1. **Formalizes Reinforcement Learning with Rich Feedback (RLRF)** — a new problem setting that captures the structure of verifiable environments with textual feedback and distinguishes it from standard RLVR.
2. **Introduces SDPO (Self-Distillation Policy Optimization)** — an online RL algorithm that converts tokenized feedback into dense, logit-level credit assignment using the current model as its own self-teacher.
3. **No external teacher, no learned reward model** — dense supervision is derived entirely from the current policy conditioned on feedback, incurring no additional sampling overhead.
4. **Introduces test-time self-distillation** — SDPO can be applied at inference time on a single hard problem, iteratively refining the policy with environment feedback to solve problems that best-of-k sampling and multi-turn prompting fail to solve.
5. **Demonstrates positive model-size scaling** — SDPO's advantage over GRPO grows as model capacity increases (Qwen3 0.6B → 8B).

---

## Framework: SDPO

### RLVR vs. RLRF

```
RLVR (standard):
  Attempt y → Environment → Scalar reward r ∈ {0, 1}
  Credit assignment: entire sequence receives +1 or 0 — no token-level signal

RLRF (SDPO):
  Attempt y → Environment → Scalar reward r + Tokenized feedback f
  (e.g., f = "RuntimeError: index out of range at line 7")
  → Feedback f used to derive dense, token-level learning signal
```

### SDPO Training Step

```
1. Sample rollout y from current policy π_θ(y | x)
2. Obtain tokenized feedback f from environment
   (even if r = 0, f contains information about why y failed)
3. Compute self-teacher distribution:
   π_θ(· | x, f, y)  ← same model, feedback-conditioned context
   No new sampling required — re-compute log-probs of existing rollout y
4. Distill self-teacher into student policy:
   Minimize JS divergence between π_θ(· | x, f, y) and π_θ(· | x)
   (Jensen-Shannon chosen for on-policy stability over KL divergence)
5. Update policy parameters θ
```

### Key Design Properties

| Property | Details |
|----------|---------|
| **No sampling overhead** | Self-teacher re-evaluates the existing rollout — does not sample a new response |
| **No external teacher** | Dense signal derived entirely from current model conditioned on feedback |
| **Logit-level credit assignment** | Self-teacher can agree or disagree at specific tokens — far more informative than sequence-level reward |
| **JS divergence** | Symmetric Jensen-Shannon divergence used for distillation loss — more stable than forward/reverse KL for on-policy distillation |
| **Scalar-only fallback** | When environments provide only scalar rewards (no rich feedback), SDPO uses successful rollouts as implicit feedback — still outperforms GRPO |

---

## Experimental Setup

| Property | Details |
|----------|---------|
| **Base Models** | Qwen3-8B · OLMo3-7B-Instruct |
| **Model Scaling** | Qwen3 family: 0.6B, 1.7B, 4B, 8B |
| **Task Domains** | Mathematical reasoning · Competitive coding · Chemistry (undergraduate level) |
| **Benchmarks** | AIME · AMC · LiveCodeBench · Custom Chemistry QA |
| **Primary Metric** | avg@16 relative to wall-clock training time |
| **Baseline** | On-policy GRPO (both methods: one gradient step per generation batch) |
| **Hardware** | 4 × NVIDIA GH200 per run |

---

## Key Results

- **6× faster convergence than GRPO** on undergraduate Chemistry questions (OLMo3-7B-Instruct reaches the same accuracy GRPO achieves in 1 hour after only ~10 minutes).
- **Substantially higher final accuracy** than GRPO on both 1-hour and 5-hour wall-clock training budgets across all benchmarks.
- **11× shorter reasoning traces** than GRPO — GRPO-trained models frequently enter logical loops and produce verbose, repetitive chains; SDPO models reason more concisely and effectively.
- **Test-time self-distillation** solves hard LiveCodeBench problems (pass@64 < 0.03) with up to **~3× fewer attempts** than best-of-k sampling and multi-turn prompting, and discovers solutions the baselines fail to find entirely.
- **Positive model-size scaling** — SDPO's advantage over GRPO widens consistently across Qwen3 model sizes (0.6B → 8B), suggesting SDPO scales more effectively with model capacity.
- **Scalar-only SDPO still outperforms GRPO** — even without rich textual feedback, using successful rollouts as implicit demonstrations produces better credit assignment than scalar GRPO.

---

## Companion Paper

The same group released a closely related offline learning paper on the same day:

> Shenfeld et al. (2026) — [*Self-Distillation Enables Continual Learning*](https://arxiv.org/abs/2601.19897) [arXiv:2601.19897]
> Introduces **SDFT (Self-Distillation Fine-Tuning)** — the offline / demonstration-based analog of SDPO. Uses a demonstration-conditioned model as its own teacher to enable on-policy learning from expert demonstrations, reducing catastrophic forgetting in continual learning settings.

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Rich feedback availability** | The biggest gains require verifiable environments that emit structured textual feedback; domains without such feedback fall back to the scalar-only SDPO variant with reduced gains. |
| **Re-computation cost** | Computing self-teacher log-probabilities over the original rollout with a feedback-augmented context adds compute per training step relative to pure GRPO. |
| **Domain scope** | Primary experiments focus on math and code; generalization to open-ended generation, scientific reasoning, or agent tasks with unstructured feedback is not yet established. |
| **No external teacher signal** | While no-teacher is a design strength, it also means SDPO cannot bootstrap from a stronger teacher when one is available — combining SDPO with teacher distillation is left to future work. |
| **Preprint status** | Not yet peer reviewed; conclusions pending further validation and community scrutiny. |

---

## Citation

```bibtex
@article{hubotter2026reinforcement,
  title   = {Reinforcement Learning via Self-Distillation},
  author  = {Hübotter, Jonas and Lübeck, Frederike and Behric, Lejs and Baumann, Anton and Bagatella, Marco and Marta, Daniel and Hakimi, Ido and Shenfeld, Idan and Kleine Buening, Thomas and Guestrin, Carlos and Krause, Andreas},
  year    = {2026},
  journal = {arXiv preprint arXiv:2601.20802},
  url     = {https://arxiv.org/abs/2601.20802}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| GRPO: Group Relative Policy Optimization (Shao et al., 2024) | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| DeepSeek-R1: Incentivizing Reasoning via RL (DeepSeek-AI, 2025) | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| Self-Distillation Enables Continual Learning — SDFT (Shenfeld et al., 2026) | [arXiv:2601.19897](https://arxiv.org/abs/2601.19897) |
| SOAR: Self-Optimization via Asymmetric RL (Sundaram et al., 2026) | [arXiv:2601.18778](https://arxiv.org/abs/2601.18778) |
| Training LMs to Reason Efficiently (Arora & Zanette, 2025) | [arXiv:2502.04463](https://arxiv.org/abs/2502.04463) |
| MiniLLM: Knowledge Distillation via Reverse KLD (Gu et al., 2024) | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
| PPO: Proximal Policy Optimization (Schulman et al., 2017) | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| Hindsight Experience Replay (Andrychowicz et al., 2017) | [NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/453fadbd8a1a3af50a9df4df899537b5-Abstract.html) |
| Dense Reward for Free in RLHF (Chan et al., 2024) | [ICML 2024](https://arxiv.org/abs/2402.09702) |
| Better Estimation of KL Divergence between LMs (Amini et al., 2025) | [NeurIPS 2025](https://arxiv.org/abs/2312.14523) |
