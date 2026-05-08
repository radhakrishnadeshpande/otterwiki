# L17.2 : Multimodal Models — Part II

> In this lecture, I explored more advanced multimodal architectures and capabilities.

https://youtu.be/IuNheUd_fZ4?si=5d9Th5HZ7LYqptPK

### Vision Transformer (ViT)
- Splits image into **patches**; each patch is a token.
`$x_{patch} = \text{Flatten}(\text{Image}[i:i+p, j:j+p])$`
- Applies Transformer on patch embeddings.
- **No convolution** needed — pure attention-based.

### GPT-4V & Gemini
- Natively multimodal — trained jointly on text, images, audio, and video.
- Can answer complex visual reasoning questions.

### Image Generation (Diffusion Models — overview)
**Forward process (add noise):**
`$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$`

**Reverse process (denoise):**
`$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$`

- **DALL·E, Stable Diffusion** generate images from text using diffusion.

---


[[Next lecture notes-L18.1|L18.1 intro]]
