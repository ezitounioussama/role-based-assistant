# Role-Based Assistant with Chat Memory

An assistant that holds a persona (a system message) and remembers previous turns (a conversation
buffer). Two roles — budget travel assistant and strict grammar teacher — running on a local
model, no API key.

Building it turned up a failure worth knowing about. The travel role's few-shot example mentioned
*"I have 300 euros"*. Later, asked to recall a budget the user had actually given as **800**, the
assistant answered **300** — wrong 4/4 times with the example injected as chat turns, 3/4 with it
in the system message, 0/4 once the figure was removed. To the model an example turn is
indistinguishable from real history, and examples survive a memory reset, so the wrong number
outlived the conversation that should have corrected it.

```bash
ollama serve && ollama pull llama3.2:3b

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
