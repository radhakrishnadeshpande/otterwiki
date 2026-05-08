# L4.2 : Word Representation — GloVe

> In this lecture, I learned about GloVe, a global co-occurrence based word embedding method.

https://youtu.be/qI0O8ksKImw?si=He2h0wj_xoIJX0gp

- **GloVe (Global Vectors for Word Representation)** combines the benefits of matrix factorization and local context window methods.
- Uses a **word co-occurrence matrix** $X$ where $X_{ij}$ = number of times word $j$ appears in the context of word $i$.

### GloVe Objective
`$J = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$`

Where:
- $f(X_{ij})$ is a **weighting function** that reduces the influence of very frequent co-occurrences.
- $w_i$ and $\tilde{w}_j$ are word and context vectors.

`$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$`

- GloVe captures **global statistical information** unlike Word2Vec which uses only local context.
- Both GloVe and Word2Vec produce similar quality embeddings in practice.


[[Next lecture notes-L5|L5 intro]]
