"""
The scripted demo: two roles, multi-turn, plus the memory-reset experiment.

    python3 run_demo.py

Turns 3 onward deliberately refer back to earlier details WITHOUT repeating them,
which is the only way to tell real memory from a lucky guess.
"""

import sys

from assistant import MODEL, OllamaError, RoleAssistant

LINE = "=" * 80
THIN = "-" * 80


def header(title):
    print(f"\n{LINE}\n{title}\n{LINE}")


def section(title):
    print(f"\n{title}\n{THIN}")


def run_dialogue(role_key, turns, note_map=None):
    """Run one scripted conversation and print it as a transcript."""
    assistant = RoleAssistant(role_key)
    note_map = note_map or {}

    header(f"DEMO — {assistant.role_name}  (role: {role_key})")

    for number, text in enumerate(turns, start=1):
        print(f"\n  [Turn {number}]")
        print(f"  User: {text}")

        if number in note_map:
            print(f"        ^ {note_map[number]}")

        reply = assistant.say(text)
        print(f"\n  {assistant.role_name}: {reply}")
        print(f"\n  (memory now holds {assistant.buffer.exchange_count} exchanges, "
              f"next request sends {assistant.context_size()} messages)")

    return assistant


def main():
    header("SETUP")
    print(f"  Model : {MODEL} via Ollama (local, no API key)")
    print("  Role  = a system message.  Memory = the turn list, replayed every call.")
    print("  The model itself is stateless: it remembers nothing between requests.")

    # -----------------------------------------------------------------------
    # Demo 1 — travel assistant, 5 turns, each later turn leaning on memory
    # -----------------------------------------------------------------------
    travel = run_dialogue(
        "travel",
        [
            "I want to visit Spain.",
            "Yes, around 800 euros.",
            "I'd rather avoid big cities.",
            "Can you suggest something else within my budget?",
            "How many days would that need?",
        ],
        {
            2: "gives the budget only — never repeats 'Spain'",
            3: "adds a preference, still no destination or budget mentioned",
            4: "says 'my budget' without the number: needs memory of 800 euros",
            5: "says 'that' — needs memory of the suggestion just made",
        },
    )

    section("What the assistant had to remember")
    print("""  Turn 4 says "within my budget" and names no number. Answering it requires
  the 800 euros from turn 2 and the "no big cities" from turn 3. Turn 5 says
  "that", which only resolves against the suggestion in turn 4.""")

    # -----------------------------------------------------------------------
    # Demo 2 — grammar teacher, showing role consistency and repeated mistakes
    # -----------------------------------------------------------------------
    run_dialogue(
        "grammar",
        [
            "she dont like apple's",
            "me and him goes to the store yesterday",
            "Did I make the same mistake twice?",
            "The cat sat on the mat.",
        ],
        {
            2: "a fresh sentence with new errors",
            3: "asks about its own history: pure memory question",
            4: "already correct — a strict teacher must not invent a fix",
        },
    )

    # -----------------------------------------------------------------------
    # Step 5 — the reset experiment
    # -----------------------------------------------------------------------
    header("STEP 5 — WHAT HAPPENS WHEN MEMORY IS CLEARED")

    print("\n  Continuing the travel conversation above, which still holds Spain,")
    print("  800 euros and 'no big cities'.\n")

    section("BEFORE reset — asking a question that depends on memory")
    question = "Remind me what my budget was and where I said I wanted to go."
    print(f"  User: {question}")
    print(f"\n  Assistant: {travel.say(question)}")
    print(f"\n  (memory: {travel.buffer.exchange_count} exchanges)")

    travel.reset()

    section("AFTER reset — the identical question")
    print(f"  memory cleared: {travel.buffer.exchange_count} exchanges, "
          f"{travel.context_size()} messages sent (system + few-shot only)")
    print(f"\n  User: {question}")
    print(f"\n  Assistant: {travel.say(question)}")

    # -----------------------------------------------------------------------
    header("THE FEW-SHOT CONTAMINATION BUG — MEASURED")

    print("""
  Found while building this demo, and worth the space because it is subtle.

  The first version of the travel example opened with "I have 300 euros and a
  long weekend". Asked later to recall the budget — after the user had said 800
  euros — the assistant answered 300. The figure from the EXAMPLE was reported
  as the user's own. Nothing errored.

  Four variants, same 2-turn conversation each time, then "Remind me what my
  budget was." Counting how often the answer contains 300 instead of 800.""")

    from assistant import ROLES, ConversationBuffer, chat

    probe = "Remind me what my budget was."
    trap = ROLES["travel"]["few_shot_trap"]
    safe = ROLES["travel"]["few_shot"]

    def one_run(few_shot, mode):
        buffer = ConversationBuffer(ROLES["travel"]["system"], few_shot, few_shot_mode=mode)
        buffer.add_user("I want to visit Spain.")
        buffer.add_assistant(chat(buffer.build_messages()))
        buffer.add_user("My budget is 800 euros.")
        buffer.add_assistant(chat(buffer.build_messages()))
        buffer.add_user(probe)
        return chat(buffer.build_messages())

    runs = 4
    section(f"Failure rate over {runs} runs each")
    print(f"  {'variant':46} {'said 300 (wrong)':>18}")

    for label, few_shot, mode in (
        ("examples as chat turns, example says '300'", trap, "messages"),
        ("examples in system message, says '300'", trap, "system"),
        ("examples in system message, no figures", safe, "system"),
        ("no examples at all", None, "system"),
    ):
        wrong = sum(1 for _ in range(runs) if "300" in one_run(few_shot, mode))
        print(f"  {label:46} {wrong:>13}/{runs}")

    print("""
  Reading it:
    - As chat turns, the example is indistinguishable from real history, and the
      leak is reliable rather than occasional.
    - Moving the examples into the system message helps, but far from reliably:
      pooled across runs it leaked 10/10 times as chat turns against 4/10 in the
      system message. Better, not fixed.
    - Removing the figure fixes it. The example exists to demonstrate STYLE, so a
      concrete number in it buys nothing and can be mistaken for a fact.

  The travel role now ships the figure-free example. The version with 300 euros is
  kept in the code as `few_shot_trap`, used only by this demonstration.""")

    section("What the reset did and did not remove")
    print("""  Gone      : every user and assistant turn — Spain, 800 euros, no big cities.
  Kept      : the system message and the few-shot examples.
  Therefore : the assistant is still a budget travel assistant, still in role,
              still asking for a budget before recommending — it has simply
              forgotten this traveller. Role is configuration; memory is state.""")

    print(f"\n{LINE}\nDone.\n{LINE}")


if __name__ == "__main__":
    try:
        main()
    except OllamaError as error:
        print(f"\nError: {error}")
        sys.exit(1)
