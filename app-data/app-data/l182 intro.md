# L18.2 : LLMs and Tools — Function Calling

> In this lecture, I learned the technical details of how LLMs interface with structured function calls.

https://youtu.be/SMx_UmpiNmA?si=xBiE9NTHavDTDGFg

### Function Calling Mechanism
- The model is provided with a **schema** of available functions.
- At each step, the model can output either:
  - A text response, OR
  - A structured function call: `{"name": "get_weather", "arguments": {"city": "Delhi"}}`

### Parallel Function Calling
- Model can issue **multiple tool calls simultaneously** when they are independent.

### Tool Selection & Planning
- With many tools, models must select the right tool — a **tool retrieval** step may be needed.
- **Chain-of-thought planning** before tool use improves accuracy.

### HotPotQA & Multi-hop Reasoning
- Some questions require **multiple tool calls** (multi-hop).
- The model must chain information across calls.

### Reliability Challenges
- Tool calls may **fail or return unexpected outputs**.
- Models must handle errors gracefully and **retry or fall back**.

[[Next lecture notes-L18.3|L18.3 intro]]
