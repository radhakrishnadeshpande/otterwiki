# L3.1: Introduction to Language Models — Statistical Language Modeling

> In this lecture, I learned how language can be modeled probabilistically using statistical methods.

Lecture: https://youtu.be/GIX79DnzwOk?si=9mehA8kVwA1Hg2IP

---

## What is a Language Model?

A **Language Model (LM)** assigns a probability to a sequence of words.

Goal:

`$P(w_1, w_2, \ldots, w_n)$`

This represents the probability of an entire sentence occurring.

---

## Chain Rule of Probability

The probability of a sentence can be decomposed using the chain rule:

`
$P(w_1, \dots, w_n)
=
\prod_{i=1}^{n} P(w_i \mid w_1, \dots, w_{i-1})$
`

This means each word depends on all previous words.

---

## Bigram Approximation (Markov Assumption)

To simplify computation, we assume each word depends only on the previous word:

`$
P(w_i \mid w_1, \ldots, w_{i-1})
\approx
P(w_i \mid w_{i-1})
$`

This is called a **Bigram Model**.

---

## N-gram Probability Estimation (MLE)

The probability of a word using an N-gram model is estimated as:

`$
P(w_i \mid w_{i-n+1}, \ldots, w_{i-1})
=
\frac{
C(w_{i-n+1}, \ldots, w_i)
}{
C(w_{i-n+1}, \ldots, w_{i-1})
}
$`

Where:

- $( C(\cdot))$  denotes the count of occurrences in the training corpus.
- This is called **Maximum Likelihood Estimation (MLE)**.

---

## Common N-gram Models

- **Unigram** → depends on 0 previous words
- **Bigram** → depends on 1 previous word
- **Trigram** → depends on 2 previous words

---

## Sparsity Problem

Many N-grams never appear in training data.

Result:

- Their counts become zero
- Probability becomes zero
- The model fails on unseen sequences

This is known as the **sparsity problem**.

---

## Log Probabilities

Very small probabilities can cause numerical underflow.

Instead of multiplying probabilities:

`$
P = p_1 \times p_2 \times \cdots \times p_n
$`

we use log probabilities:

`$
\log P
=
\sum_i \log p_i
$`

Advantages:

- Numerically stable
- Faster computation
- Easier optimization

---

## Key Takeaways

- Language modeling is fundamentally a probability estimation task.
- Chain rule decomposes sentence probability into conditional probabilities.
- N-gram models approximate long dependencies with shorter contexts.
- MLE estimates probabilities using corpus counts.
- Sparsity is a major limitation of statistical language models.
- Log probabilities help avoid numerical instability.

---

[[Next lecture notes-L3.2|L3.2 intro]]
