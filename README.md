# Role-Based Assistant with Chat Memory

An assistant that holds a persona (a system message) and remembers previous turns (a conversation
buffer). Two roles — budget travel assistant and strict grammar teacher — running on a local
model, no API key.

Building it turned up a failure worth knowing about, and then re-running it turned up something
better. The travel role's few-shot example mentioned *"I have 300 euros"*. Asked later to recall a
budget the user had actually given as **800**, `llama3.2:3b` answered **300** — wrong 4/4 times
with the example injected as chat turns, 3/4 with it in the system message, 0/4 once the figure was
removed. To that model an example turn was indistinguishable from real history, and examples
survive a memory reset, so the wrong number outlived the conversation that should have corrected
it.

On `qwen3:8b` the leak never appeared — 0/4 in every variant, including the worst one. Same
prompts, same code. So it is a capability failure rather than a structural one, which is worth
knowing but changes nothing about the fix: a bug that depends on which model you loaded is not a
bug you have fixed, and taking the number out of an example costs nothing.

```bash
ollama serve && ollama pull qwen3:8b

python3 run_demo.py              # scripted demos + reset experiment
python3 assistant.py             # talk to it (travel role)
python3 assistant.py --role grammar
python3 tests.py                 # 17 tests, no model needed
```

Interactive commands: `reset` clears memory, `memory` prints the buffer, `quit` exits.

## Also in this repo

- **[DEMO.md](DEMO.md)** — the demo dialogues, the Step 5 reflection, the five steps in detail,
  and the contamination bug written up with its numbers
- [`docs/output.txt`](docs/output.txt) — raw log

---

Author: **Oussama Ezitouni**
