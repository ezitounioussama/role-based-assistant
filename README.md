# Role-Based Assistant with Chat Memory

An assistant that holds a persona (system message) and remembers previous turns (conversation
buffer). Two roles, local model, no API key.

> **[DEMO.md](DEMO.md)** — the demo dialogues, the Step 5 reflection, and the few-shot
> contamination bug this build uncovered.

```bash
ollama serve && ollama pull llama3.2:3b

python3 run_demo.py              # scripted demos + reset experiment
python3 assistant.py             # talk to it (travel role)
python3 assistant.py --role grammar
python3 tests.py                 # 17 tests, no model needed
```

Interactive commands: `reset` clears memory, `memory` prints the buffer, `quit` exits.

| File | Contents |
|---|---|
| `assistant.py` | Roles, `ConversationBuffer`, the model call |
| `run_demo.py` | Both demos, the reset experiment, the measured bug |
| `tests.py` | 17 tests on the buffer and the roles |
| **[`DEMO.md`](DEMO.md)** | **The deliverable dialogues + reflection** |
| `docs/output.txt` | Raw log |

## The five steps

**1. Role** — one system message per persona: *budget travel assistant* and *strict grammar
teacher*. Both end with the same two rules, keep it short and do not invent facts, because a 3B
model drifts into essays and invents plausible prices otherwise.

**2. Memory** — `ConversationBuffer` stores the turns and replays them on every call. The model is
**stateless**; it remembers nothing between requests. `max_turns` caps the replay, because the
request otherwise grows until it silently overflows the context window.

**3. Few-shot (optional)** — one worked exchange per role, folded into the system message and
labelled as an example. See the bug below for why placement and content both matter.

**4. Testing** — turns 3–5 of each demo refer back to earlier details *without repeating them*
("within my budget", "how many days would that need"), which is the only way to tell real memory
from a lucky guess.

**5. Reflection** — in [DEMO.md](DEMO.md): the role held except for two honest slips, all three
remembered details came back correctly, and a reset wipes the conversation while leaving the role
intact.

## The bug worth reading

The travel example originally said *"I have 300 euros"*. Asked later to recall a budget the user
had given as **800**, the assistant answered **300** — measured wrong **4/4 times** with examples
injected as chat turns, **3/4** with them in the system message, **0/4** once the figure was
removed. An example turn is indistinguishable from real history to the model, and examples survive
a memory reset, so the wrong figure outlived the conversation.

---

Author: **Oussama Ezitouni**
