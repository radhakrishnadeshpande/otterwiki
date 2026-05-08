# L19 : Reasoning in LLMs

> In this lecture, I learned how LLMs approach complex reasoning tasks and the methods to improve reasoning.

https://youtu.be/B4pRRD2f8tk?si=dpvS7u12ct9SpPek

### Types of Reasoning
- **Mathematical reasoning**: Arithmetic, algebra, proofs.
- **Logical reasoning**: Deduction, abduction, induction.
- **Commonsense reasoning**: Real-world knowledge and intuition.
- **Multi-hop reasoning**: Chaining multiple inference steps.

### Chain-of-Thought (CoT) Prompting
`$\text{Prompt} = [(\text{question}_1, \text{reasoning}_1, \text{answer}_1), \ldots, (\text{question}_k, \text{reasoning}_k, \text{answer}_k), \text{question}]$`

### Self-Consistency
- Sample **multiple CoT reasoning paths** with temperature > 0.
- Take the **majority vote** on the final answer.
`$\hat{y} = \arg\max_y \sum_{i=1}^{m} \mathbb{1}[y_i = y]$`

### Tree of Thought (ToT)
- Explores **multiple reasoning branches** like a search tree.
- Uses **BFS/DFS** with a value function to guide exploration.

### Self-Reflection & Self-Correction
- The model critiques its own output and **iteratively refines** the answer.
- Example: **Reflexion** framework uses verbal reinforcement.

### Process Reward Models (PRM)
- Instead of rewarding only the final answer, **reward each reasoning step**:
`$R_{total} = \sum_{t=1}^{T} r_t \quad \text{where } r_t = \text{PRM}(s_t)$`



[[Next lecture notes-L20|L20 intro]]
