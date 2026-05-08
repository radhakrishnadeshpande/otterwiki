# L13.2 : Alignment of Language Models — Reward Maximization II

> In this lecture, I deepened my understanding of RL-based alignment including reward hacking and alternatives.

https://youtu.be/aSico42v0DM?si=6QplhscXEMk3AuoC

### Reward Hacking
- The model learns to **game the reward model** rather than truly being helpful.
- The RM is imperfect → optimization pressure causes distribution shift.

### Constitutional AI (CAI)
- Uses **AI-generated critiques and revisions** instead of human feedback for every step.
- Model critiques its own output against a set of principles (the "constitution"), then revises.

### REINFORCE Algorithm (Policy Gradient)
`$\nabla_\theta \mathcal{L} = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) R(\tau)\right]$`

### Advantage Function
`$A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$`

- Using the advantage reduces variance in gradient estimates.

[[Next lecture notes-L13.3|L13.3 intro]]
