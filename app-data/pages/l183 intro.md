# L18.3 : LLMs and Tools — Agentic Workflow

> In this lecture, I learned how LLMs operate as autonomous agents in multi-step workflows.

https://youtu.be/D0KawZMdRIU?si=IcApA8X3wYmtkh65

### LLM Agents
- An **agent** is an LLM that can: perceive observations, reason, take actions, and iterate.
- Operates in a **loop**: Observe → Think → Act → Observe → …

### Agent Frameworks
- **AutoGPT, BabyAGI**: Early autonomous agents.
- **LangChain, LlamaIndex**: Frameworks for building LLM pipelines.

### Multi-Agent Systems
- **Orchestrator** agent decomposes tasks.
- **Specialist** agents execute sub-tasks.
- Agents communicate via shared memory or message passing.

### Memory Types in Agents
| Memory | Description |
|--------|-------------|
| In-context | Information in the current prompt window |
| External (Episodic) | Retrieved from a vector store |
| Working | Scratchpad for intermediate reasoning |
| Parametric | Knowledge encoded in model weights |

### Challenges
- **Hallucinated actions**: Agent calls non-existent functions.
- **Infinite loops**: No progress, keeps repeating.
- **Context overflow**: Long task histories exceed context window.

[[Next lecture notes-L19|L19 intro]]
