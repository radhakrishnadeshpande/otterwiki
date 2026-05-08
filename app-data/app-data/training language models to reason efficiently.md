# Training Language Models to Reason Efficiently

> **arXiv:** [arXiv:2502.04463](https://arxiv.org/abs/2502.04463)
> **PDF:** [View PDF](https://arxiv.org/pdf/2502.04463)
> **HTML:** [HTML (experimental)](https://arxiv.org/html/2502.04463v3)
> **Project Page:** [zanette-labs.github.io/efficient-reasoning](https://zanette-labs.github.io/efficient-reasoning/)
> **GitHub:** [Zanette-Labs/efficient-reasoning](https://github.com/Zanette-Labs/efficient-reasoning)
> **OpenReview:** [NeurIPS 2025 Poster](https://openreview.net/forum?id=AiZxn84Wdo)
> **Semantic Scholar:** [Paper Details](https://www.semanticscholar.org/paper/Training-Language-Models-to-Reason-Efficiently-Arora-Zanette/03fa0740512a47758940415a6b3c1a635d9aca98)

---

## Introduction

The frontier of LLM reasoning is defined by models like OpenAI o1, Gemini Flash Thinking, and DeepSeek-R1 — systems trained via large-scale RL to produce long chain-of-thought (CoT) sequences before generating final answers. These extended reasoning traces unlock unprecedented problem-solving capability on mathematics, science, and code tasks. However, they come with a significant and growing deployment cost: each reasoning step generates additional tokens, and the attention mechanism's quadratic cost over sequence length means that long chain-of-thoughts translate directly into high inference latency and compute expenditure. As reasoning models scale, this cost becomes a practical barrier to economic deployment, user experience, and environmental sustainability.

The core insight of this paper is that long reasoning traces are not uniformly necessary — easy problems can be solved with shorter chains of thought, while hard problems genuinely benefit from extended deliberation. Standard RL training for reasoning models does not encode this distinction; models learn to generate long CoTs indiscriminately, regardless of task complexity. This paper addresses the problem directly: using RL to train reasoning models to **dynamically allocate inference-time compute based on task difficulty**, shortening chains of thought on easy problems while preserving reasoning depth on hard ones. The result is a family of models — derived from a single tunable hyperparameter α — that trace out the compute-performance tradeoff curve, allowing practitioners to select their preferred operating point without retraining.

---

## Problem Statement

1. **Long CoTs are expensive uniformly** — frontier reasoning models generate long reasoning traces for all problems regardless of difficulty, wasting compute on easy queries that require little reasoning.
2. **Inference cost scales quadratically** — the attention mechanism's quadratic cost over sequence length makes long CoTs disproportionately expensive at deployment scale.
3. **No dynamic compute allocation** — existing reasoning models lack a principled mechanism to adaptively allocate inference-time compute based on estimated task complexity.
4. **Overthinking is well-documented but underaddressed** — prior work identifies LLM overthinking (unnecessarily long CoTs) but existing mitigations either rely on offline heuristics or do not expose a tunable compute-performance tradeoff.

---

## Contributions

1. **Proposes a length-penalized online RL training objective** that incentivizes models to minimize CoT length while maintaining accuracy, built on top of RLOO (a variant of GRPO/REINFORCE).
2. **Derives a family of efficient reasoning models** controlled by a single hyperparameter α — the length penalty coefficient — enabling practitioners to navigate the compute-performance tradeoff without retraining for each operating point.
3. **Demonstrates graceful tradeoff navigation** — reducing CoT length significantly with minimal accuracy loss across math benchmarks.
4. **Provides the first online RL method** for this problem, distinguishing it from concurrent offline approaches (O1-Pruner, Kimi k1.5) that do not expose a tunable compute budget parameter.

---

## Method

MiniLLM addresses this by replacing forward KLD with **reverse KLD** as the distillation objective. This paper addresses the CoT length problem via a **length-penalized RL objective**:

```
Standard RLOO / GRPO reward:
  r(x, y) = correctness(y | x)       ← binary: 1 if answer correct, 0 otherwise

MiniLLM (length-penalized) reward:
  r(x, y) = correctness(y | x) − α · length(y)

Where:
  α  = length penalty coefficient (hyperparameter; controls compute-performance tradeoff)
  length(y) = number of tokens in the generated chain-of-thought + answer

Training: Online RL (student samples from its own distribution at each step)
  → Avoids exposure bias from offline methods
  → Model sees its own generation distribution throughout training
```

### Key Design Properties

| Property | Details |
|----------|---------|
| **Online RL** | Student samples from its own policy during training — no offline heuristics or fixed preference datasets |
| **Single hyperparameter α** | Controls the entire compute-performance tradeoff family; increasing α produces shorter but less accurate models |
| **GRPO normalization caveat** | Standard GRPO reward normalization is modified — naive normalization has unintended consequences when all responses to a prompt are correct (rewards cluster in [1−α, 1]), distorting the length penalty gradient |
| **No precise length targeting** | α controls overall generation cost but does not precisely target a specific token count; L1 (Aggarwal & Welleck, 2025) addresses exact length constraints as a concurrent follow-up |

---

## Experimental Setup

| Property | Details |
|----------|---------|
| **Base Models** | DeepSeek-R1-Distill-Qwen-1.5B · DeepSeek-R1-Distill-Qwen-7B |
| **Benchmarks** | MATH · AIME · GPQA · Other mathematical reasoning benchmarks |
| **Training Data** | Mathematical reasoning problems with verifiable binary correctness |
| **Baselines** | Standard RLOO (no length penalty) · O1-Pruner (offline RL) · Kimi k1.5 (length penalty with online RL but no tunable α) |

---

## Key Results

- **Substantial reduction in reasoning cost with minimal accuracy loss** — models trained with the length-penalized objective generate significantly shorter CoTs while approximately maintaining accuracy on math benchmarks.
- **Graceful tradeoff navigation** — the family of models derived by varying α smoothly traces out the compute-performance Pareto frontier, allowing precise operating point selection.
- **Online RL advantage** — the method outperforms offline RL baselines (O1-Pruner) on the accuracy-cost tradeoff, attributed to reduced exposure bias from on-policy training.
- **Tunable α uniquely enables a model family** — unlike Kimi k1.5's similar approach (which lacks a tunable budget parameter), varying α produces a complete Pareto set of models from a single training procedure.
- **Works on both 1.5B and 7B** — gains are consistent across model scales within the DeepSeek-R1-Distill family.

---

## Concurrent and Follow-up Work

This paper was part of a cluster of simultaneous and subsequent work on efficient reasoning:

| Paper | Approach | Link |
|-------|----------|------|
| **O1-Pruner** (Luo et al., 2025) | Offline RL with length-harmonizing objective | [arXiv:2501.12570](https://arxiv.org/abs/2501.12570) |
| **Kimi k1.5** (Team, 2025) | Online RL with length penalty (no tunable α) | [arXiv:2501.12599](https://arxiv.org/abs/2501.12599) |
| **L1** (Aggarwal & Welleck, 2025) | RL for exact token-budget constraints via prompts | [arXiv:2503.04697](https://arxiv.org/abs/2503.04697) |
| **TokenSkip** (Xia et al., 2025) | Semantic importance pruning + SFT | [arXiv:2502.12067](https://arxiv.org/abs/2502.12067) |
| **Chain of Draft** (Xu et al., 2025) | Shorter scratchpads via SFT | [arXiv:2502.18600](https://arxiv.org/abs/2502.18600) |

---

## Limitations

| Limitation | Details |
|------------|---------|
| **No precise length targeting** | α controls average CoT cost but not exact token count; latency-constrained applications requiring precise budget control must use follow-up methods like L1. |
| **Domain limited to math** | All experiments use mathematical reasoning benchmarks with verifiable binary rewards; generalization to code, science, or open-ended reasoning tasks is not evaluated. |
| **GRPO normalization sensitivity** | Naive use of standard GRPO reward normalization with the length penalty produces unintended gradient distortions; the paper's workaround adds implementation complexity. |
| **Requires RL training infrastructure** | Online RL fine-tuning is more infrastructure-intensive than SFT-based alternatives. |
| **Diminishing returns at extreme compression** | Very large α values cause excessive accuracy degradation; the graceful tradeoff holds only within a practical range of α. |

---

## Citation

```bibtex
@inproceedings{arora2025traininglanguagemodelsreason,
  title     = {Training Language Models to Reason Efficiently},
  author    = {Daman Arora and Andrea Zanette},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://openreview.net/forum?id=AiZxn84Wdo}
}
```

---

## Related Work & References

| Paper | Link |
|-------|------|
| DeepSeek-R1: Incentivizing Reasoning via RL (Guo et al., 2025) | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| GRPO: Group Relative Policy Optimization (Shao et al., 2024) | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| Chain-of-Thought Prompting (Wei et al., 2022) | [arXiv:2201.11903](https://arxiv.org/abs/2201.11903) |
| Stop Overthinking: Survey on Efficient Reasoning for LLMs (Sui et al., 2025) | [arXiv:2503.16419](https://arxiv.org/abs/2503.16419) |
| L1: Controlling Reasoning Length with RL (Aggarwal & Welleck, 2025) | [arXiv:2503.04697](https://arxiv.org/abs/2503.04697) |
| O1-Pruner: Length-Harmonizing Fine-Tuning (Luo et al., 2025) | [arXiv:2501.12570](https://arxiv.org/abs/2501.12570) |
| SOAR: Self-Optimization via Asymmetric RL (Sundaram et al., 2026) | [arXiv:2601.18778](https://arxiv.org/abs/2601.18778) |
| GPQA: Graduate-Level QA Benchmark (Rein et al., 2024) | [arXiv:2311.12022](https://arxiv.org/abs/2311.12022) |
| Efficient Reasoning Models: A Survey (Feng et al., 2025) | [arXiv:2504.10903](https://arxiv.org/abs/2504.10903) |

## Mathematics
Here are the some key mathematical formulas for **Training Language Models to Reason Efficiently**:

---

### 1. Standard RL Reward (RLOO / GRPO baseline)

$$r(x, y) = \mathbf{1}[\text{correct}(y \mid x)]$$

where $x$ is the input problem and $y$ is the generated response (CoT + answer).

---

### 2. Length-Penalized Reward (MoC Core Objective)

$$r_\alpha(x, y) = \mathbf{1}[\text{correct}(y \mid x)] - \alpha \cdot |y|$$

where:
- $\alpha \geq 0$ is the **length penalty coefficient** (hyperparameter)
- $|y|$ is the number of tokens in the generated chain-of-thought and answer
- $\alpha = 0$ recovers the standard RL baseline

---

### 3. RLOO Policy Gradient Objective

$$\mathcal{L}_\text{RLOO}(\theta) = -\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ \hat{A}(x, y) \cdot \log \pi_\theta(y \mid x) \right]$$

where the **leave-one-out advantage estimate** is:

$$\hat{A}(x, y^{(i)}) = r_\alpha(x, y^{(i)}) - \frac{1}{K-1} \sum_{j \neq i} r_\alpha(x, y^{(j)})$$

with $K$ responses sampled per prompt from the current policy $\pi_\theta$.

---

### 4. GRPO Normalization — the Caveat

Standard GRPO normalizes rewards within a group:

$$\hat{r}(x, y^{(i)}) = \frac{r_\alpha(x, y^{(i)}) - \mu_r}{\sigma_r + \epsilon}$$

**The problem:** when all $K$ responses are correct, rewards cluster in $[1 - \alpha \cdot |y|_\min,\ 1 - \alpha \cdot |y|_\max]$. After normalization, the length penalty gradient dominates in an unintended way. The paper modifies this normalization to avoid distorting the accuracy-length tradeoff signal.

---

### 5. Compute-Performance Tradeoff Family

By varying $\alpha$, the method produces a **Pareto frontier** of models:

$$\mathcal{F} = \left\{ \pi_\theta^{(\alpha)} : \alpha \in [0, \alpha_{\max}] \right\}$$

$$\text{Cost}(\alpha) = \mathbb{E}\left[ |y| \right] \downarrow \text{ as } \alpha \uparrow, \quad \text{Accuracy}(\alpha) \approx \text{const for small } \alpha$$

---

### 6. Expected Total Inference Cost

For a deployment serving $N$ queries:

$C_\text{total} = N \cdot \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi_\theta(\cdot \mid x)}\left[ |y| \right]$


Minimizing $r_\alpha$ during training directly minimizes $C_\text{total}$ at deployment, since $\mathbb{E}[|y|]$ decreases as $\alpha$ increases.
```math
