# L13.1 : Alignment of Language Models — Reward Maximization I (RLHF)

> In this lecture, I learned how RLHF aligns LLMs to human preferences.

https://youtu.be/f8ZhsaiCr1o?si=tjR-c6yIqkyesAKX

### RLHF Pipeline

1. **Supervised Fine-Tuning (SFT)**: Fine-tune on human-written demonstrations.
2. **Reward Model (RM) Training**: Train a model to predict human preference scores.
3. **RL Fine-Tuning**: Optimize the LM policy using PPO against the RM.

### Reward Model Training
Given a pair $(y_w, y_l)$ where $y_w$ is preferred over $y_l$:
`$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)} [\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]$`

### PPO (Proximal Policy Optimization) Objective
`$\mathcal{L}_{PPO}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$`

where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ is the **probability ratio**.

### KL Penalty (to prevent reward hacking)
`$\mathcal{L} = r_\phi(x, y) - \beta \cdot \text{KL}[\pi_\theta(\cdot|x) \| \pi_{ref}(\cdot|x)]$`

[[Next lecture notes-L13.2|L13.2 intro]]
