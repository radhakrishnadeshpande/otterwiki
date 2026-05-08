# L11 : Scaling Laws

> In this lecture, I learned how model performance scales predictably with compute, data, and parameters.

https://youtu.be/S7GOt85DWBs?si=ze3WGPIikVXE913S

### Chinchilla Scaling Laws (Hoffmann et al., 2022)

Optimal allocation of a compute budget $C$:
`$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$`

- **Chinchilla Rule of Thumb**: Train on ~**20 tokens per parameter**.
- GPT-3 (175B params) was significantly under-trained by this standard.

### Kaplan Scaling Laws (OpenAI, 2020)

`$L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D} + L_\infty$`

- Loss follows a **power law** in both model size $N$ and data size $D$.
- $\alpha_N \approx 0.076$, $\alpha_D \approx 0.095$ (empirically found).

### Compute-Optimal Scaling
`$C \approx 6ND \quad \text{(FLOPs for training a Transformer)}$`

### Key Takeaways
- **Bigger models** trained on **more data** consistently improve performance.
- There are **diminishing returns** but no observed ceiling yet.
- Scaling laws guide decisions on **how to spend a compute budget**.
[[Next lecture notes-L12.1|L12.1 intro]]
