# L18.1 : LLMs and Tools — Tool Augmentation

> In this lecture, I learned how LLMs can call external tools to extend their capabilities.

https://youtu.be/eUfaCUkZrVQ?si=l0l2--JI6bxOagpA

### Why Tools?
- LLMs lack: real-time knowledge, mathematical precision, ability to take actions.
- **Tool augmentation** allows LLMs to call calculators, search engines, APIs, databases, etc.

### Toolformer (Schick et al.)
- LLM learns to **insert API calls** in its own text and use the result.
- Self-supervised training: model learns when and how to call tools.

### ReAct (Reasoning + Acting)
- Interleaves **reasoning traces** with **tool-use actions**:
  1. *Thought*: "I need to find the population of India."
  2. *Action*: `Search[India population 2024]`
  3. *Observation*: "India population is 1.4 billion."
  4. *Thought*: "Now I can answer."
  5. *Answer*: "1.4 billion."

### Tool Calling in Practice (OpenAI Function Calling)
- Define tools as JSON schemas.
- LLM outputs a structured **function call** with arguments.
- Application executes the function and returns result to LLM.

[[Next lecture notes-L18.2|L18.2 intro]]
