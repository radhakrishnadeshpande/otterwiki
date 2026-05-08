# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

> **arXiv:** [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
> **PDF:** [View PDF](https://arxiv.org/pdf/2501.12948)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2501.12948v1)
> **Nature:** [Published Article (Sep 2025)](https://www.nature.com/articles/s41586-025-09422-z)
> **GitHub:** [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
> **HuggingFace:** [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
> **HuggingFace Papers:** [HF Paper Page](https://huggingface.co/papers/2501.12948)
> **Semantic Scholar:** [Paper Details](https://www.semanticscholar.org/paper/DeepSeek-R1:-Incentivizing-Reasoning-Capability-in-DeepSeek-AI-Guo/34471a2fa18ea22efad5287cf4aeb18542c98a9b)



## Introduction

The dominant paradigm for building instruction-following and reasoning LLMs has relied on a two-stage pipeline: first, large-scale supervised fine-tuning (SFT) on human-annotated demonstrations; then, reinforcement learning from human feedback (RLHF) to align model outputs with human preferences. This pipeline produces capable models but is inherently bottlenecked by the quality, scale, and cost of human annotation — particularly for complex reasoning tasks such as mathematics, competitive programming, and scientific problem-solving, where generating correct, well-reasoned demonstrations at scale is exceptionally expensive.

DeepSeek-R1 challenges this assumption directly. It demonstrates that **advanced reasoning capabilities can emerge from pure reinforcement learning applied to a base model**, without requiring human-labeled reasoning trajectories as a prerequisite. The key insight is the use of **Reinforcement Learning from Verifiable Rewards (RLVR)** — training on tasks where correctness can be verified programmatically (mathematical answer checking, code compilation and test case execution) — which eliminates the need for a neural reward model and avoids the reward hacking and instability issues that plague learned reward models at scale.

The paper introduces two models: **DeepSeek-R1-Zero**, trained via pure RL on a base model with no SFT warm-up, which exhibits emergent reasoning behaviors including self-reflection and the famous "aha moment" but suffers from poor readability and language mixing; and **DeepSeek-R1**, which addresses these issues through a multi-stage pipeline incorporating cold-start data and two RL stages, achieving performance **comparable to OpenAI o1-1217** on reasoning benchmarks while being fully open-source.

---

## Problem Statement

1. **SFT-dependent reasoning training is expensive and bottlenecked** — producing high-quality, human-annotated CoT demonstrations for hard reasoning tasks (competition math, advanced coding) is costly and does not scale easily.
2. **Neural reward models are fragile at scale** — process-based and outcome-based learned reward models are susceptible to reward hacking during large-scale RL, requiring expensive periodic retraining.
3. **Frontier reasoning capability was proprietary** — prior to DeepSeek-R1, models matching OpenAI o1's reasoning performance were closed-source; no open-source alternative existed.
4. **Reasoning emergence from RL was unvalidated at scale** — whether complex behaviors like self-reflection and dynamic strategy adaptation could emerge purely from RL without SFT warm-up remained an open empirical question.

---

## Contributions

1. **First open-source validation** that LLM reasoning capabilities can be fully incentivized through pure RL (RLVR), without any SFT as a prerequisite — demonstrated via DeepSeek-R1-Zero.
2. **Introduces GRPO (Group Relative Policy Optimization)** as the RL algorithm, a memory-efficient variant of PPO that eliminates the need for a separate critic model.
3. **Proposes a four-stage multi-stage training pipeline** for DeepSeek-R1 that resolves the readability and language mixing failures of R1-Zero while retaining and improving its reasoning capabilities.
4. **Achieves performance comparable to OpenAI o1-1217** on mathematics, coding, and STEM reasoning benchmarks with a fully open-source model.
5. **Open-sources six distilled models** (1.5B, 7B, 8B, 14B, 32B, 70B) derived from DeepSeek-R1 via knowledge distillation into Qwen and LLaMA backbones.
6. **Documents the "aha moment"** — a phenomenon where the model spontaneously learns to allocate more thinking time by re-evaluating its initial approach, emerging without explicit instruction.

---

## Models

| Model | Training Approach | Key Characteristics |
|-------|------------------|---------------------|
| **DeepSeek-R1-Zero** | Pure RLVR on DeepSeek-V3-Base, no SFT | Emergent reasoning; self-reflection; "aha moment"; suffers language mixing and poor readability |
| **DeepSeek-R1** | Multi-stage: cold-start SFT → RL → rejection sampling SFT → RL | Matches OpenAI o1-1217; human-readable CoT; language-consistent |
| **Distilled Models** | KD from DeepSeek-R1 into Qwen/LLaMA | 1.5B, 7B, 8B, 14B, 32B, 70B — strong reasoning at small scales |

---

## Training Pipeline: DeepSeek-R1

```
Stage 1 — Cold-Start SFT
  Train on a small, carefully curated CoT dataset
  Purpose: warm-start the model to produce readable, structured reasoning traces
  Avoids the readability and language mixing failures of pure R1-Zero

Stage 2 — Reasoning-Oriented RL (GRPO)
  Apply RLVR with verifiable rewards on math and code tasks
  Reward = Accuracy reward + Format reward
  No neural reward model — rule-based verification only

Stage 3 — Rejection Sampling SFT
  Use the RL-tuned model to generate new SFT data via rejection sampling
  Filters for high-quality outputs; adds non-reasoning data (writing, QA, etc.)
  Purpose: balance reasoning capability with general instruction following

Stage 4 — Final RL for Alignment
  Second RL stage incorporating a language consistency reward
  Aligns model with human preferences and reduces language mixing
  Produces the final DeepSeek-R1 model
```

---

## Reward Design (RLVR)

Two rule-based rewards — no learned neural reward model:

| Reward Component | Description |
|-----------------|-------------|
| **Accuracy Reward** | Verifies correctness: math answers checked against ground truth; code evaluated via test case execution |
| **Format Reward** | Enforces `<think>...</think>` tags for reasoning traces; penalizes internal chain-of-thought in final answer |

> The explicit rejection of neural reward models is a deliberate design choice — the paper cites susceptibility to reward hacking and prohibitive retraining costs as reasons for using rule-based verification exclusively.

---

## Emergent Behaviors in R1-Zero

A particularly notable aspect of R1-Zero training is the **spontaneous emergence of reasoning behaviors** not explicitly supervised:

- **Self-reflection and verification** — model learns to review its own intermediate steps and identify errors
- **Dynamic strategy adaptation** — model changes its reasoning approach mid-solution when initial strategies appear to fail
- **The "aha moment"** — at an intermediate training checkpoint, the model learns to re-evaluate its initial approach and allocate more compute to hard problems
- **Response length growth** — average response length increases naturally as RL training progresses, reflecting learned inference-time scaling

---

## Experimental Results

| Benchmark | DeepSeek-R1 | OpenAI o1-1217 | Notes |
|-----------|-------------|----------------|-------|
| AIME 2024 (pass@1) | 79.8% | 79.2% | Competitive math |
| MATH-500 | 97.3% | 96.4% | Mathematics |
| Codeforces Rating | 2029 | 2061 | Competitive coding |
| GPQA Diamond | 71.5% | 75.7% | Graduate-level STEM |
| LiveCodeBench | 65.9% | 63.4% | Live coding benchmark |
| SWE-bench Verified | 49.2% | 48.9% | Software engineering |

---

## Distilled Model Performance

Distillation from DeepSeek-R1 transfers strong reasoning into small models:

- **DeepSeek-R1-Distill-Qwen-7B** outperforms QwQ-32B on multiple benchmarks
- **DeepSeek-R1-Distill-Qwen-1.5B** achieves competitive math performance at minimal compute cost
- Distilled models are released under permissive licenses for research use

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Language mixing in R1-Zero** | Without language consistency rewards, R1-Zero mixes Chinese, English, and mathematical notation in its reasoning traces, reducing readability. |
| **Cold-start data dependency in R1** | DeepSeek-R1's four-stage pipeline requires a curated cold-start dataset; the fully pure RL path (R1-Zero) sacrifices readability for simplicity. |
| **Evaluation on verifiable tasks only** | RLVR requires tasks with programmatic correctness verification; extension to open-ended generation, creative writing, or subjective tasks is non-trivial. |
| **Software engineering underperformance vs o1** | DeepSeek-R1 slightly lags OpenAI o1 on GPQA Diamond (71.5% vs 75.7%), suggesting room for improvement on complex multi-discipline STEM reasoning. |
| **Distillation ceiling** | Distilled models are bounded by the teacher model's capabilities and may not replicate all emergent behaviors of the full R1 model. |

---

## Citation

```bibtex
@article{deepseekr1_2025,
  title     = {{DeepSeek-R1}: Incentivizing Reasoning Capability in {LLM}s via Reinforcement Learning},
  author    = {DeepSeek-AI and Daya Guo and Dejian Yang and Haowei Zhang and others},
  journal   = {Nature},
  volume    = {645},
  pages     = {633--638},
  year      = {2025},
  doi       = {10.1038/s41586-025-09422-z},
  url       = {https://www.nature.com/articles/s41586-025-09422-z}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| GRPO: Group Relative Policy Optimization (Shao et al., 2024) | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| Chain-of-Thought Prompting Elicits Reasoning (Wei et al., 2022) | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| PPO: Proximal Policy Optimization (Schulman et al., 2017) | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| InstructGPT / RLHF (Ouyang et al., 2022) | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| DeepSeek-V3 Technical Report (DeepSeek-AI, 2024) | [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) |
| Training LMs to Reason Efficiently — Arora & Zanette (2025) | [arXiv:2502.04463](https://arxiv.org/abs/2502.04463) |
| SOAR: Self-Optimization via Asymmetric RL (Sundaram et al., 2026) | [arXiv:2601.18778](https://arxiv.org/abs/2601.18778) |
| Self-Consistency for CoT Reasoning (Wang et al., 2022) | [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) |
| MATH Benchmark (Hendrycks et al., 2021) | [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) |
| MiniLLM: Knowledge Distillation of LLMs (Gu et al., 2024) | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
