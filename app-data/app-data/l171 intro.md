# L17.1 : Multimodal Models — Part I

> In this lecture, I learned how LLMs are extended to process images alongside text.

https://youtu.be/IXp3P1MPB0Q?si=ezMX9rjMM5UVQGgI

### Vision-Language Models
- Process **both images and text** jointly.
- Architectures typically connect a **vision encoder** to a **language decoder**.

### CLIP (Contrastive Language-Image Pre-training)
`$\mathcal{L}_{CLIP} = -\frac{1}{N}\sum_{i=1}^{N}\left[\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)} + \log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ji}/\tau)}\right]$`

- Contrastive loss aligns image and text embeddings.
- $s_{ij} = f_I(x_i)^T f_T(y_j)$ = dot product of image and text embeddings.
- $\tau$ = learned temperature parameter.

### Flamingo
- Freezes a pre-trained vision encoder and LLM.
- Adds **cross-attention layers** between them.
- Uses **interleaved image-text** training data.
- Achieves strong few-shot multimodal performance.

### LLaVA
- Projects image features from CLIP ViT into the language space using a **linear projection**.
- Fine-tuned on visual instruction data.

[[Next lecture notes-L17.2|L17.2 intro]]
