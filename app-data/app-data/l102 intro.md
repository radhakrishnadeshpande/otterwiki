# L10.2 : Mixture of Experts — Part II

> In this lecture, I explored advanced MoE designs and their use in state-of-the-art models.

https://youtu.be/ZrhfnJY9Gjs?si=7dqnKuPBmY21er0d

### Switch Transformer
- Uses **Top-1 routing** (each token goes to exactly 1 expert).
- Simplifies the routing and reduces communication overhead.

### Expert Choice Routing
- Instead of tokens choosing experts, **experts choose tokens**:
  - Each expert picks the top-$k$ tokens assigned to it.
  - Guarantees perfect load balance but causes some tokens to be dropped.

### GShard & Mixtral
- **GShard**: Distributed MoE with Top-2 routing across TPU pods.
- **Mixtral 8x7B**: 8 experts, top-2 routing per token; achieves quality of a 47B dense model at 13B active parameters.

### Key Insight
- MoE enables **conditional computation**: the same input activates different parameters, allowing specialization.
[[Next lecture notes-L11|L11 intro]]
