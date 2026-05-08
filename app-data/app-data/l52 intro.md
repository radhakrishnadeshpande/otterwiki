# L5.2 : Neural Language Models — LSTMs and GRUs

> In this lecture, I learned how LSTMs and GRUs solve the vanishing gradient problem.

https://youtu.be/dn4_S_mDXSQ?si=UypSSCIv8SkDnnUI

### LSTM (Long Short-Term Memory)

LSTMs use **gating mechanisms** to control information flow:

**Forget Gate:** `$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$`

**Input Gate:** `$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$`

**Candidate Cell:** `$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$`

**Cell State Update:** `$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$`

**Output Gate:**` $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$`

**Hidden State:** `$h_t = o_t \odot \tanh(C_t)$`

### GRU (Gated Recurrent Unit)

Simpler than LSTM with only two gates:

**Reset Gate:** `$r_t = \sigma(W_r [h_{t-1}, x_t])$`

**Update Gate:** `$z_t = \sigma(W_z [h_{t-1}, x_t])$`

**New Memory:** `$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])$`

**Hidden State:** `$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$`

- GRU is **computationally cheaper** than LSTM with comparable performance.
[[Next lecture notes-L5.3|L5.3 intro]]
