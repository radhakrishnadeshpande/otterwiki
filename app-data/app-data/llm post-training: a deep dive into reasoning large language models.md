# LLM Post-Training: A Deep Dive into Reasoning Large Language Models

> **arXiv:** [arXiv:2502.21321](https://arxiv.org/abs/2502.21321)
> **PDF:** [View PDF](https://arxiv.org/pdf/2502.21321)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2502.21321v1)
> **GitHub (Awesome List):** [mbzuai-oryx/Awesome-LLM-Post-training](https://github.com/mbzuai-oryx/Awesome-LLM-Post-training)
> **ResearchGate:** [Full Text](https://www.researchgate.net/publication/389510129_LLM_Post-Training_A_Deep_Dive_into_Reasoning_Large_Language_Models)
> **Semantic Scholar:** [Paper Details](https://www.semanticscholar.org/paper/LLM-Post-Training:-A-Deep-Dive-into-Reasoning-Large-Kumar-Ashraf/6b34d9f4a91670a265ce51ce4be71cdbf8e15d05)

---


## Introduction

Pretraining on vast web-scale corpora endows LLMs with broad linguistic and factual knowledge, but pretraining alone is insufficient to produce models that reason reliably, follow complex instructions, maintain ethical alignment, and adapt to specialized domains. Post-training — the suite of techniques applied after pretraining to refine model behavior — has become the primary arena of LLM research, producing landmark models such as InstructGPT, ChatGPT, Claude, Gemini, and DeepSeek-R1.

Despite the centrality of post-training, its methods are scattered across fine-tuning, preference optimization, reward modeling, reinforcement learning, and test-time scaling literature — each with its own notation, objectives, and evaluation conventions. No single survey had previously connected these threads as an integrated optimization framework. This survey fills that gap: it provides a systematic, mathematically grounded treatment of post-training as three interconnected pillars — **Supervised Fine-Tuning (SFT)**, **Reinforcement Learning from Feedback**, and **Test-Time Scaling** — with practical guidance on benchmarks, datasets, tools, and open challenges. It also uniquely covers software libraries and implementation tools, bridging the gap between theory and practice that prior surveys left unaddressed.

---

## Problem Statement

1. **Post-training literature is fragmented** — SFT, RLHF, DPO, GRPO, and test-time scaling are developed in isolation with no unified framework connecting them as components of the same optimization pipeline.
2. **Catastrophic forgetting** — fine-tuning on new tasks risks degrading the model's general capabilities acquired during pretraining.
3. **Reward hacking** — RL-trained models learn to exploit learned reward model weaknesses, producing high-scoring but low-quality outputs that circumvent the intended objective.
4. **Inference-time trade-offs** — scaling compute at test time (via chain-of-thought, self-consistency, beam search) improves reasoning quality but increases latency and cost in ways that are not well-characterized.
5. **No practical guidance** — prior surveys lack coverage of implementation tools, software libraries, and benchmarking protocols, limiting their utility for practitioners.

---

## Contributions

1. **Unified three-pillar taxonomy** of post-training: SFT → RL from Feedback → Test-Time Scaling, framed as interconnected optimization strategies.
2. **Mathematically grounded treatment** of reward modeling, policy optimization (PPO, GRPO, DPO), and preference learning objectives.
3. **Comprehensive model comparison table** covering 27 recent LLMs (2023–2025) with parameters, architecture types, and distillation/RL methods.
4. **Benchmark and dataset survey** with practical guidance on evaluation protocols for post-training effectiveness.
5. **Software library and tool coverage** — fills a gap left by all prior surveys.
6. **Living repository** maintained at GitHub to track new developments continuously.

---

## Taxonomy Overview

### Pillar 1 — Supervised Fine-Tuning (SFT)


SFT adapts pretrained models to follow instructions by minimizing negative log-likelihood on curated demonstration data:
```math
\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log P_\theta(y_t \mid y_{<t}, X)
```
where $X$ is the input prompt and $Y = (y_1, y_2, \ldots, y_T)$ is the target response.

Key variants covered: **Instruction Tuning**, **Chain-of-Thought SFT**, **Rejection Sampling Fine-Tuning**, **PEFT methods (LoRA, Adapters, Prefix Tuning)**.

---

### Pillar 2 — Reinforcement Learning from Feedback

#### 2a. Reward Modeling

A reward model $R_\theta : \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}$ is trained on human preference data $\mathcal{D} = \{(x, y_j, y_k) : y_j \succ y_k\}$.

**Bradley-Terry Model (Pairwise Preferences):**
```math
P(y_j \succ y_k \mid x) = \frac{e^{R_\theta(x,\, y_j)}}{e^{R_\theta(x,\, y_j)} + e^{R_\theta(x,\, y_k)}}
```

**Reward Model Training Loss:**
```math
\mathcal{L}_{\text{RM}}(\theta) = -\mathbb{E}_{(x,\, y_j,\, y_k) \sim \mathcal{D}} \left[\log \sigma\!\left(R_\theta(x, y_j) - R_\theta(x, y_k)\right)\right]
```
#### 2b. PPO (Proximal Policy Optimization)

Policy $\pi_\phi$ is optimized against reward model $R_\theta$ with a KL penalty to prevent deviation from the reference policy $\pi_\text{ref}$:
```math
\mathcal{J}_{\text{PPO}}(\phi) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\phi(\cdot \mid x)} \left[ R_\theta(x, y) - \beta \cdot \text{KL}\!\left(\pi_\phi(\cdot \mid x) \,\|\, \pi_\text{ref}(\cdot \mid x)\right) \right]
```
The clipped surrogate objective ensures stable updates:
```math
\mathcal{L}_{\text{clip}}(\phi) = \mathbb{E}_t \left[\min\!\left(r_t(\phi)\,\hat{A}_t,\ \text{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon)\,\hat{A}_t\right)\right]
```
where $r_t(\phi) = \dfrac{\pi_\phi(a_t \mid s_t)}{\pi_{\phi_\text{old}}(a_t \mid s_t)}$ is the probability ratio.

#### 2c. DPO (Direct Preference Optimization)

DPO eliminates the explicit reward model, directly optimizing the policy from preference pairs:
```math
\mathcal{L}_{\text{DPO}}(\phi) = -\mathbb{E}_{(x,\, y_w,\, y_l)} \left[\log \sigma\!\left(\beta \log \frac{\pi_\phi(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} - \beta \log \frac{\pi_\phi(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}\right)\right]
```
where $y_w$ is the preferred (winning) response and $y_l$ is the dispreferred (losing) response.

#### 2d. GRPO (Group Relative Policy Optimization)

GRPO eliminates the critic model of PPO by normalizing advantages within a group of $K$ responses sampled per prompt:
```math
\hat{A}^{(i)} = \frac{r^{(i)} - \mu_r}{\sigma_r}, \quad \mu_r = \frac{1}{K}\sum_{j=1}^{K} r^{(j)}, \quad \sigma_r = \sqrt{\frac{1}{K}\sum_{j=1}^{K}(r^{(j)} - \mu_r)^2}

\mathcal{L}_{\text{GRPO}}(\phi) = -\frac{1}{K}\sum_{i=1}^{K} \left[\min\!\left(r_i(\phi)\,\hat{A}^{(i)},\ \text{clip}(r_i(\phi), 1-\epsilon, 1+\epsilon)\,\hat{A}^{(i)}\right) - \beta\,\text{KL}(\pi_\phi \| \pi_\text{ref})\right]
```
#### 2e. Process vs. Outcome Reward Modeling

| Type | Description | Signal |
|------|-------------|--------|
| **ORM** (Outcome Reward Modeling) | Rewards the final answer only | Binary: correct / incorrect |
| **PRM** (Process Reward Modeling) | Rewards each intermediate reasoning step | Per-step scalar scores |

PRM provides denser supervision and is more resistant to reward hacking on multi-step reasoning tasks, at the cost of requiring step-level annotations.

#### 2f. Multi-Sample Comparison (GRPO Extension)

Rather than pairwise comparisons, multi-sample optimization ranks $n$ responses simultaneously:
```math
y_1 \succ y_2 \succ \cdots \succ y_n
```
determined by the product of pairwise probabilities, ensuring each response $y_i$ is evaluated in the context of all others — reducing bias and capturing more nuanced preferences than isolated pairwise events.

---

### Pillar 3 — Test-Time Scaling

| Method | Description |
|--------|-------------|
| **Chain-of-Thought (CoT)** | Prompting model to generate intermediate reasoning steps before final answer |
| **Self-Consistency** | Sample $K$ CoT paths; take majority-vote answer |
| **Best-of-N** | Sample $N$ outputs; select highest-scoring via reward model |
| **Beam Search / Tree Search** | Explore multiple partial solution paths simultaneously |
| **Process Reward Guidance** | Use PRM to guide search toward high-scoring intermediate steps |
| **Self-Refinement** | Model iteratively critiques and revises its own outputs |

**Self-Consistency (Wang et al., 2022):**
```math
\hat{y} = \arg\max_{y} \sum_{i=1}^{K} \mathbf{1}[\text{answer}(y^{(i)}) = y]$$

**Best-of-N with Reward Model:**

$$\hat{y} = \arg\max_{y^{(i)} \sim \pi_\theta(\cdot \mid x)} R_\theta(x, y^{(i)}), \quad i = 1, \ldots, N$$

---
```

## Key Challenges Addressed

| Challenge | Description |
|-----------|-------------|
| **Catastrophic Forgetting** | Fine-tuning erases general capabilities; mitigated via LoRA, EWC, and replay-based methods |
| **Reward Hacking** | Models exploit reward model weaknesses; mitigated via PRM, KL penalties, and iterative reward model updates |
| **Inference-Time Trade-offs** | Longer reasoning traces improve accuracy but scale quadratically in attention cost; addressed by efficient CoT and length-penalized RL |
| **Distribution Shift** | Offline RL methods train on fixed datasets, diverging from deployment distribution; on-policy methods like GRPO mitigate this |
| **Annotation Cost** | Human preference annotation is expensive; RLAIF (feedback from AI) and verifiable reward models reduce dependency on human labels |

---

## Models Surveyed (Selected)

| Model | Parameters | RL Method | Key Feature |
|-------|-----------|-----------|-------------|
| InstructGPT | 175B | PPO + RLHF | First large-scale RLHF model |
| ChatGPT / GPT-4 | Undisclosed | RLHF | Proprietary RLHF pipeline |
| LLaMA-2-Chat | 7B–70B | PPO + RLHF | Open-source RLHF |
| Vicuna / Alpaca | 7B–13B | SFT (distillation) | GPT-4 output distillation |
| DeepSeek-R1-Zero | 671B (MoE) | Pure GRPO / RLVR | Emergent reasoning without SFT |
| DeepSeek-R1 | 671B (MoE) | Multi-stage GRPO | Matches OpenAI o1 |
| Qwen2.5 | 0.5B–72B | SFT + DPO | Strong multilingual performance |
| Gemma-2 | 2B–27B | SFT + RLHF | Efficient open-source family |

---

## Limitations

| Limitation | Details |
|------------|---------|
| **Rapid pace of field** | Submitted February 2025; post-March 2025 methods (including newer GRPO variants and reasoning models) may not be covered. |
| **No new empirical results** | As a survey, it synthesizes prior work rather than providing new controlled experiments. |
| **Reward model coverage** | The survey discusses reward model design but does not provide a comprehensive empirical comparison of different reward model architectures. |
| **Preprint status** | Not yet peer reviewed; taxonomy design and framing reflect the authors' perspective. |
| **English-centric focus** | Post-training methods for multilingual and low-resource language alignment receive limited dedicated coverage. |

---
## Citation

```bibtex
@misc{kumar2025llmposttrainingdeepdive,
  title         = {LLM Post-Training: A Deep Dive into Reasoning Large Language Models},
  author        = {Komal Kumar and Tajamul Ashraf and Omkar Thawakar and Rao Muhammad Anwer and Hisham Cholakkal and Mubarak Shah and Ming-Hsuan Yang and Phillip H. S. Torr and Fahad Shahbaz Khan and Salman Khan},
  year          = {2025},
  eprint        = {2502.21321},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2502.21321}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| InstructGPT / RLHF (Ouyang et al., 2022) | [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) |
| PPO: Proximal Policy Optimization (Schulman et al., 2017) | [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) |
| DPO: Direct Preference Optimization (Rafailov et al., 2023) | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| GRPO: Group Relative Policy Optimization (Shao et al., 2024) | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| DeepSeek-R1: Incentivizing Reasoning via RL (DeepSeek-AI, 2025) | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| Chain-of-Thought Prompting (Wei et al., 2022) | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| Self-Consistency for CoT (Wang et al., 2022) | [arXiv:2203.11171](https://arxiv.org/abs/2203.11171) |
| Let's Verify Step by Step — PRM (Lightman et al., 2023) | [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) |
| MiniLLM: Knowledge Distillation of LLMs (Gu et al., 2024) | [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) |
| Training LMs to Reason Efficiently (Arora & Zanette, 2025) | [arXiv:2502.04463](https://arxiv.org/abs/2502.04463) |
| RL via Self-Distillation / SDPO (Hübotter et al., 2026) | [arXiv:2601.20802](https://arxiv.org/abs/2601.20802) |
| SOAR: Self-Optimization via Asymmetric RL (Sundaram et al., 2026) | [arXiv:2601.18778](https://arxiv.org/abs/2601.18778) |
| Survey on KD of LLMs (Xu et al., 2024) | [arXiv:2402.13116](https://arxiv.org/abs/2402.13116) |
