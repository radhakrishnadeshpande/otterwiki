# L12.2 : Instruction Tuning

> In this lecture, I learned how pre-trained LLMs are fine-tuned to follow natural language instructions.

https://youtu.be/-rBYX6PMXXQ?si=9_ltCnMwt1l1U_e0

- **Instruction Tuning** fine-tunes a pre-trained LM on (instruction, output) pairs.
- Makes models **better at following diverse user instructions** without needing few-shot examples.

### Training Objective
`$\mathcal{L}_{IT} = -\sum_{t} \log P(y_t \mid x, y_{<t})$`
where $x$ is the instruction and $y$ is the expected response.

### Key Datasets & Models
- **FLAN** (Fine-tuned Language Net): 60+ NLP tasks reformatted as instructions.
- **InstructGPT**: SFT + RLHF pipeline.
- **Alpaca**: LLaMA fine-tuned on 52K instruction pairs (self-instruct).
- **FLAN-T5, Dolly, Vicuna**: Popular open-source instruction-tuned models.

### Self-Instruct
- Use an LLM to **generate its own instruction data** → fine-tune on it → iterate.
- Bootstraps instruction data without manual labeling.

### Benefits of Instruction Tuning
- Better **zero-shot generalization** across tasks.
- Improved **task format understanding**.
- More **useful and aligned** outputs for end users.
- 
[[Next lecture notes-L13.1|L13.1 intro]]
