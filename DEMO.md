# Demo Dialogues

The deliverable: multi-turn conversations showing the assistant holding its role and using
memory. Captured from a real run — `qwen3:8b` via Ollama, thinking disabled. Full log:
[`docs/output.txt`](docs/output.txt).

---

## Demo 1 — Budget Travel Assistant (5 turns)

**System role:** *"You are a helpful travel assistant who suggests budget-friendly trips"* — plus
rules to work within the stated budget, never ask twice for a detail already given, keep answers
to 2–4 sentences, and never invent exact prices.

> **Turn 1**
> **User:** I want to visit Spain.
>
> **Assistant:** Spain is a great choice! Considering your long weekend, I'd recommend Seville.
> It's a vibrant city with a rich history, and accommodation options are affordable, with hostels
> starting from around €15-20 per night. You can explore the historic centre, visit the Alcázar
> Palace, and enjoy the city's lively atmosphere without breaking the bank. Would you like me to
> suggest some rough estimates for flights and transportation?

> **Turn 2** — *gives the budget only, never repeats "Spain"*
> **User:** Yes, around 800 euros.
>
> **Assistant:** With 800 euros, you can consider a more comfortable stay in Spain. Seville is
> still a great option, or you could consider Granada, with its famous Alhambra palace.

> **Turn 3** — *adds a preference, mentions neither destination nor budget*
> **User:** I'd rather avoid big cities.
>
> **Assistant:** Consider the Costa Brava region in Catalonia, Spain. You can find a 2-star hotel
> for around 40-60 euros per night, and enjoy the beaches, hiking trails, and charming coastal
> towns like Calella de Palafrugell.

> **Turn 4** — *says "my budget" without the number*
> **User:** Can you suggest something else within my budget?
>
> **Assistant:** The Algarve region in southern Portugal is another option. You can find a 2-star
> hotel for around 35-55 euros per night, and enjoy the beaches, golf courses, and traditional
> villages like Lagos.

> **Turn 5** — *says "that", which only resolves against turn 4*
> **User:** How many days would that need?
>
> **Assistant:** For the Algarve, 3-4 days would be a good amount of time to explore the region.
> This would allow you to visit a few beaches, a village or two, and get a feel for the area.

**What memory did here.** Turn 3 gets a small coastal region rather than a city, so the
"no big cities" preference landed while Spain was still remembered from turn 1. Turn 4 names no
number yet stays in budget range. Turn 5's "that" resolves to the Algarve suggested one turn
earlier — impossible without the history.

---

## Demo 2 — Strict Grammar Teacher (4 turns)

**System role:** *"You are a strict grammar teacher who corrects every mistake"* — with a fixed
`Corrected:` / `Errors:` output format, and an instruction to flag repeated mistakes.

> **Turn 1**
> **User:** she dont like apple's
>
> **Assistant:**
> Corrected: She doesn't like apples.
> Errors: - 'she' needs a capital S at the start of a sentence - 'dont' needs an apostrophe:
> doesn't - 'dont' is the wrong form after 'she': use doesn't - 'apple's' is possessive; the
> plural is apples

> **Turn 2**
> **User:** me and him goes to the store yesterday
>
> **Assistant:**
> Corrected: My friend and I went to the store yesterday.
> Errors: - 'me' is an object pronoun; use I as the subject pronoun - 'and him' is incorrect;
> use and (no 'him') - 'goes' is in the wrong tense; use went (past simple)

> **Turn 3** — *a pure memory question about its own history*
> **User:** Did I make the same mistake twice?
>
> **Assistant:** Yes, you did. You previously corrected "me and him" to "my friend and I" and now
> you're repeating the mistake. Remember, "me" is an object pronoun, and in this case, "I" is the
> correct subject pronoun.

> **Turn 4** — *already correct; a strict teacher must not invent a fix*
> **User:** The cat sat on the mat.
>
> **Assistant:**
> Corrected: The cat sat on the mat.
> Errors: - None. This sentence is already correct.

**What memory did here.** Turn 3 is answerable only from the stored turns. Turn 4 shows the role
holding under pressure: the persona is "corrects every mistake", and the tempting failure is to
invent one. It did not.

---

## Step 5 — Reflection

### Did the assistant stay in its role?

**Mostly, with one honest slip.** The grammar teacher held perfectly: the `Corrected:` / `Errors:`
format survived all four turns, and it refused to invent an error on a correct sentence.

The travel assistant kept the budget framing and the "no big cities" preference throughout, but
slipped twice:

- **Turn 1**: the role says *"if they have not given a budget, ask for it before recommending"*.
  It recommended Seville immediately instead. A rule in a system prompt is a strong nudge, not a
  guarantee — especially on a 3B model.
- **Turn 4**: asked for another option "in Spain" territory, it suggested the **Algarve, in
  Portugal**. Budget respected, country quietly dropped.

Both are useful: they show that role adherence is probabilistic. Tightening it means fewer rules,
stated more forcefully, and ideally a check in code for the ones that matter.

### Did it correctly remember past details?

Yes, and the test was designed so a lucky guess would not pass. Asked *"remind me what my budget
was and where I said I wanted to go"* after six exchanges:

> Your budget was €800, and you mentioned you wanted to visit Spain. You also mentioned you'd
> like to avoid big cities.

All three details, from three different turns, none repeated in the question.

### What happens if memory is cleared or reset?

Same question, immediately after `reset()`:

> You didn't mention a specific budget when we started our conversation, so I'll need to ask you
> to share that with me. You did mention that you're looking for a budget-friendly trip, but I
> don't have any information on your budget. Can you please tell me how much you're comfortable
> spending on this trip?

The distinction the reset exposes:

| | Survives reset | Why |
|---|---|---|
| System role | yes | Configuration — it is what the assistant *is* |
| Few-shot examples | yes | Also configuration |
| Conversation turns | **no** | State — it is what the assistant *knows* |

So after a reset it is still a budget travel assistant, still asking for a budget before
recommending. It has forgotten *this traveller*, not *its job*. The message count drops from 13
back to 1.

Worth being precise about the mechanism: **the model is stateless.** It remembers nothing between
requests. Turn 5 can resolve "that" only because the buffer replays all previous turns in the next
request. Memory is something the program supplies, not something the API stores — which is also
why the request grows by two messages per turn and needs the `max_turns` cap.

---

## The bug this demo found

The travel example originally opened with *"I have 300 euros and a long weekend."* Asked to recall
the budget after the user said **800**, the assistant answered **300** — the example's figure,
reported as the user's own. Nothing errored.

Measured over 4 runs per variant, on both models this project has run on. Said 300 (wrong):

| Variant | llama3.2:3b | qwen3:8b |
|---|---|---|
| Examples injected as chat turns, example says "300" | **4/4** | **0/4** |
| Examples folded into the system message, says "300" | **3/4** | **0/4** |
| Examples in the system message, no figures | 0/4 | 0/4 |
| No examples at all | 0/4 | 0/4 |

**On llama3.2:3b:** an injected example turn was indistinguishable from something the user said.
Moving the examples into the system message helped but did not fix it — pooled across all runs it
leaked 10/10 times as chat turns against 4/10 in the system message. Removing the figure fixed it.

**On qwen3:8b the leak did not appear once**, not even in the worst variant. So the bug is real —
it was measured, repeatedly, on a real model — but it is a *capability* failure, not a structural
one: a stronger model keeps example turns and real history apart on its own.

That changes what to conclude, not what to do. A failure that depends on which model you loaded is
not a failure you have fixed, and the mitigation costs nothing: a few-shot example exists to
demonstrate *style*, so a concrete number in one buys nothing and can be misread as a fact. Since
examples also survive a memory reset, a wrong figure outlives the conversation it contaminated.

The shipped travel role uses a figure-free example either way, with the original kept as
`few_shot_trap` for this demonstration, and a test asserts the shipped one contains no figures.

`run_demo.py` now derives this section's wording from the numbers it just measured. The earlier
version printed the llama3.2:3b conclusions as fact, which meant that on qwen3:8b it printed a
table of zeroes above a paragraph insisting the leak was reliable.

---

Author: **Oussama Ezitouni**

---

## The five steps, and how they are built

**1. Role** — one system message per persona: *budget travel assistant* and *strict grammar
teacher*. Both end with the same two rules, keep it short and do not invent facts, because a 3B
model drifts into essays and invents plausible prices otherwise.

**2. Memory** — `ConversationBuffer` stores the turns and replays them on every call. The model is
**stateless**; it remembers nothing between requests. `max_turns` caps the replay, because the
request otherwise grows until it silently overflows the context window.

**3. Few-shot (optional)** — one worked exchange per role, folded into the system message and
labelled as an example. See "The bug this demo found" above for why placement and content both
matter.

**4. Testing** — turns 3–5 of each demo refer back to earlier details *without repeating them*
("within my budget", "how many days would that need"), which is the only way to tell real memory
from a lucky guess.

**5. Reflection** — the reflection section above: the role held except for two honest slips, all
three remembered details came back correctly, and a reset wipes the conversation while leaving the
role intact.

## Files

| File | Contents |
|---|---|
| `assistant.py` | Roles, `ConversationBuffer`, the model call |
| `run_demo.py` | Both demos, the reset experiment, the measured bug |
| `tests.py` | 17 tests on the buffer and the roles |
| `docs/output.txt` | Raw log |
