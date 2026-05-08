# L13.3 : Alignment of Language Models — Contrastive Learning (DPO)

> In this lecture, I learned about DPO, which eliminates the need for a separate reward model.

https://youtu.be/dCKPzxX71cU?si=l1HpBxbkjgCmu2RO

### Direct Preference Optimization (DPO)

DPO shows that RLHF implicitly optimizes:
`$\mathcal{L}_{DPO}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$`

- **No reward model needed** — directly optimizes policy from preference pairs.
- More **stable** and **computationally cheaper** than PPO.
- $\beta$ controls deviation from the reference policy.

### Comparison: RLHF vs DPO
| Property | RLHF (PPO) | DPO |
|----------|-----------|-----|
| Reward Model | Required | Not needed |
| Stability | Harder to train | More stable |
| Compute | Higher | Lower |
| Performance | Strong | Comparable |

[[Next lecture notes-L14.1|L14.1 intro]]
