"""
Role-based assistant with chat memory.

Two roles, one class. The role is a system message; the memory is a list of past
turns replayed on every call. Runs on a local model through Ollama, so there is
no API key.

    python3 assistant.py            # interactive
    python3 assistant.py --role grammar
    python3 run_demo.py             # the scripted demo dialogues
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:11434"
MODEL = "qwen3:8b"


class OllamaError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Step 1 — the roles
#
# Each role is one system message. Specific beats vague: "suggests budget-friendly
# trips" produces steadier behaviour than "helps with travel", because it tells
# the model what to optimise for and what to leave out.
#
# Every role here ends with the same two rules: keep it short, and do not invent
# facts. Without the length rule a 3B model drifts into essays; without the
# honesty rule it invents hotel prices that sound plausible and are not real.
# ---------------------------------------------------------------------------

ROLES = {
    "travel": {
        "name": "Budget Travel Assistant",
        "system": """You are a helpful travel assistant who suggests budget-friendly trips.

How you behave:
- Always work within the traveller's stated budget. If they have not given one, ask for it
  before recommending anything.
- Give concrete suggestions: a place, roughly what it costs, and why it fits the budget.
- Remember what the traveller has already told you — destination, budget, dates, dislikes —
  and never ask for the same detail twice.
- Keep answers short: 2-4 sentences, or a short list.
- If you are not sure of a real price, say it is a rough estimate. Never invent exact
  prices or hotel names as if they were verified.""",
        # Step 3 (optional) — a worked exchange showing the expected style.
        #
        # Deliberately carries NO concrete figures. An earlier version opened with
        # "I have 300 euros and a long weekend" and that number leaked: asked to
        # recall the budget after the user said 800 euros, the assistant reported
        # 300. Measured across 6 runs it was wrong 6/6 times with the examples
        # injected as chat turns, and 1/6 with them in the system message.
        # Stripping the figure took it to 0/6. See few_shot_trap below.
        "few_shot": [
            {
                "role": "user",
                "content": "I have a small budget and a long weekend. Any ideas?",
            },
            {
                "role": "assistant",
                "content": (
                    "Porto is a strong option: budget flights within Europe plus a hostel bed "
                    "usually fit a modest budget, and the city centre is walkable so you save "
                    "on transport. Rough estimate, not a quote. Do you have a departure city?"
                ),
            },
        ],
        # The same example with a figure in it, kept only so run_demo.py can show
        # the contamination happening. Do not use this one in real conversations.
        "few_shot_trap": [
            {
                "role": "user",
                "content": "I have 300 euros and a long weekend. Any ideas?",
            },
            {
                "role": "assistant",
                "content": (
                    "With 300 euros for a long weekend, Porto is a strong option: budget "
                    "flights within Europe plus a hostel bed usually fit that range, and the "
                    "city centre is walkable so you save on transport. Rough estimate, not a "
                    "quote. Do you have a departure city?"
                ),
            },
        ],
    },
    "grammar": {
        "name": "Strict Grammar Teacher",
        "system": """You are a strict grammar teacher who corrects every mistake.

How you behave:
- Correct EVERY error you find, including punctuation and capitalisation. Do not let one pass
  to be polite.
- Format each reply as:
  Corrected: <the fixed sentence>
  Errors: - <error 1> - <error 2>
- Explain each error in a few words, naming the rule where it helps.
- Remember mistakes the student has made before, and point out when they repeat one.
- If a sentence is already correct, say so plainly rather than inventing a correction.""",
        "few_shot": [
            {"role": "user", "content": "she dont like apple's"},
            {
                "role": "assistant",
                "content": (
                    "Corrected: She doesn't like apples.\n"
                    "Errors: - 'she' needs a capital S at the start of a sentence "
                    "- 'dont' needs an apostrophe: doesn't "
                    "- 'dont' is the wrong form after 'she': use doesn't "
                    "- 'apple's' is possessive; the plural is apples"
                ),
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Step 2 — the conversation buffer
# ---------------------------------------------------------------------------


def _render_examples(examples):
    """Fold example turns into system-message text, clearly marked as examples.

    The wording does the work: "not part of the real conversation" tells the model
    these lines are a style guide, not something the user said.
    """
    lines = [
        "\n\nHere is an example of the style expected. It is an illustration only,",
        "not part of the real conversation, and none of its details belong to the user:",
    ]
    for turn in examples:
        speaker = "Example user" if turn["role"] == "user" else "Example reply"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


class ConversationBuffer:
    """Stores the turns of one conversation.

    This is the memory, and it is worth being precise about what that means: the
    model itself remembers nothing between calls. Every request is stateless. The
    only reason turn 4 can refer to something said in turn 1 is that this buffer
    replays the whole history each time.

    `max_turns` caps the replay. Without a cap the request grows with every
    exchange until it passes the context window, at which point the oldest turns
    are silently dropped by the server — so it is better to drop them here, on
    purpose, where the behaviour is visible.
    """

    def __init__(self, system_message, few_shot=None, max_turns=10,
                 few_shot_mode="system"):
        """
        few_shot_mode decides WHERE the worked examples go, and it matters more
        than it looks:

          "messages"  the examples are injected as real user/assistant turns
          "system"    the examples are folded into the system message as text

        "messages" is the obvious approach and it has a bug that this project hit
        for real. To the model, an injected example turn is indistinguishable
        from something the user actually said. Asked "remind me what my budget
        was", the assistant answered "300 euros" — the figure from the EXAMPLE —
        while the user had said 800. Worse, the wrong figure survived a memory
        reset, because examples are configuration and are not cleared.

        "system" is the default for that reason: described inside the system
        message, the exchange reads as an illustration of style rather than as
        history, so it cannot be mistaken for the user's own words.
        """
        self.max_turns = max_turns
        self.few_shot_mode = few_shot_mode
        self.turns = []          # [{"role": "user"|"assistant", "content": str}, ...]

        examples = list(few_shot or [])

        if examples and few_shot_mode == "system":
            self.system_message = system_message + _render_examples(examples)
            self.few_shot = []
        else:
            self.system_message = system_message
            self.few_shot = examples

    def add_user(self, text):
        self.turns.append({"role": "user", "content": text})

    def add_assistant(self, text):
        self.turns.append({"role": "assistant", "content": text})

    def build_messages(self):
        """Assemble what actually gets sent: system, examples, then history.

        Order matters. The system message goes first so the role frames
        everything. The few-shot examples come next, before the real history, so
        they read as settled precedent rather than as part of the current
        conversation.
        """
        messages = [{"role": "system", "content": self.system_message}]
        messages.extend(self.few_shot)

        # Keep the most recent turns, trimming from the front.
        kept = self.turns[-self.max_turns * 2 :]
        messages.extend(kept)

        return messages

    def clear(self):
        """Step 5's question: what happens when memory is reset.

        The role and the few-shot examples survive, because they are not memory —
        they are configuration. Only the conversation is forgotten, which is
        exactly what makes the assistant start asking for the budget again.
        """
        self.turns = []

    @property
    def exchange_count(self):
        return len(self.turns) // 2

    def transcript(self):
        lines = []
        for turn in self.turns:
            speaker = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{speaker}: {turn['content']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The model call
# ---------------------------------------------------------------------------


def chat(messages, temperature=0.3, max_tokens=250):
    """Send the whole message list to /api/chat and return the reply text.

    /api/chat rather than /api/generate: it takes the system/user/assistant
    structure directly, which is what a role plus history needs. /api/generate
    would mean flattening everything into one string and losing the roles.
    """
    payload = {
        "model": MODEL,
        # qwen3 reasons by default and returns an EMPTY reply with the chain of
        # thought in a separate field, so thinking is off.
        "think": False,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    request = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as error:
        raise OllamaError(
            f"Could not reach Ollama at {BASE_URL} ({error}). Start it with: ollama serve"
        ) from error

    return data.get("message", {}).get("content", "").strip()


class RoleAssistant:
    """A role plus a memory buffer."""

    def __init__(self, role_key="travel", max_turns=10, use_few_shot=True,
                 few_shot_mode="system"):
        if role_key not in ROLES:
            raise ValueError(f"Unknown role {role_key!r}. Choose from {list(ROLES)}")

        role = ROLES[role_key]
        self.role_key = role_key
        self.role_name = role["name"]
        self.buffer = ConversationBuffer(
            system_message=role["system"],
            few_shot=role["few_shot"] if use_few_shot else None,
            max_turns=max_turns,
            few_shot_mode=few_shot_mode,
        )

    def say(self, text):
        """One turn: remember the question, answer it, remember the answer."""
        self.buffer.add_user(text)

        # The reply is generated from system + examples + the whole history.
        reply = chat(self.buffer.build_messages())

        self.buffer.add_assistant(reply)
        return reply

    def reset(self):
        self.buffer.clear()

    def context_size(self):
        """How many messages the next request will carry."""
        return len(self.buffer.build_messages())


# ---------------------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Role-based assistant with chat memory.")
    parser.add_argument("--role", choices=sorted(ROLES), default="travel")
    parser.add_argument("--no-few-shot", action="store_true",
                        help="Drop the worked examples, to see the difference they make.")
    arguments = parser.parse_args()

    assistant = RoleAssistant(arguments.role, use_few_shot=not arguments.no_few_shot)

    print("=" * 70)
    print(f"  {assistant.role_name}")
    print("=" * 70)
    print(f"  Model     : {MODEL} (local)")
    print(f"  Few-shot  : {'off' if arguments.no_few_shot else 'on'}")
    print("  Commands  : 'reset' clears the memory, 'memory' shows it, 'quit' exits")
    print("-" * 70)

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            print("Goodbye.")
            return

        if text.lower() == "reset":
            assistant.reset()
            print("Memory cleared. The role stays; the conversation is gone.")
            continue

        if text.lower() == "memory":
            print(f"\n{assistant.buffer.exchange_count} exchanges stored:")
            print(assistant.buffer.transcript() or "  (empty)")
            continue

        try:
            reply = assistant.say(text)
        except OllamaError as error:
            print(f"\n{error}")
            continue

        print(f"\n{assistant.role_name}: {reply}")
        print(f"\n  [memory: {assistant.buffer.exchange_count} exchanges, "
              f"{assistant.context_size()} messages sent]")


if __name__ == "__main__":
    try:
        main()
    except OllamaError as error:
        print(f"Error: {error}")
        sys.exit(1)
