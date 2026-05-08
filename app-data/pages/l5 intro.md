# L5.1 : Neural Language Models — RNNs

> In this lecture, I learned how Recurrent Neural Networks model sequential data like text.

https://youtu.be/KZPh8F-rymY?si=FL4Xc0VdzC3nbxk5

- **Feedforward NNs** cannot handle variable-length sequences → **RNNs** solve this.
- RNNs have a **hidden state** $h_t$ that summarizes past information.

### RNN Equations

`$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$`
`$y_t = \text{softmax}(W_y h_t + b_y)$`

### Training: Backpropagation Through Time (BPTT)
- The gradient is computed by **unrolling** the RNN through time.
- **Vanishing Gradient Problem**: Gradients shrink exponentially over many time steps → model forgets long-range dependencies.
- **Exploding Gradient Problem**: Gradients grow uncontrollably → fixed with **gradient clipping**.

**Gradient Clipping:**
`$\text{if } \|\nabla\| > \theta, \quad \nabla \leftarrow \frac{\theta}{\|\nabla\|} \nabla$`

### RNN Language Model Loss
`$L = -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_1, \ldots, w_{t-1})$`

[[Next lecture notes-L5.2|L5.2 intro]]
