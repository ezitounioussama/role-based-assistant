"""
Tests for the memory buffer. No model and no Ollama needed — the buffer is pure
data structure, so its behaviour can be pinned down exactly.

    python3 tests.py
"""

import unittest

from assistant import ROLES, ConversationBuffer, RoleAssistant


class TestBufferBasics(unittest.TestCase):
    def setUp(self):
        self.buffer = ConversationBuffer("You are a test role.")

    def test_system_message_is_always_first(self):
        self.buffer.add_user("hello")
        messages = self.buffer.build_messages()

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a test role.")

    def test_turns_are_kept_in_order(self):
        self.buffer.add_user("first")
        self.buffer.add_assistant("reply one")
        self.buffer.add_user("second")

        roles = [m["role"] for m in self.buffer.build_messages()]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_exchange_count(self):
        self.assertEqual(self.buffer.exchange_count, 0)
        self.buffer.add_user("q")
        self.buffer.add_assistant("a")
        self.assertEqual(self.buffer.exchange_count, 1)

    def test_history_grows_by_two_per_turn(self):
        sizes = []
        for number in range(3):
            self.buffer.add_user(f"question {number}")
            sizes.append(len(self.buffer.build_messages()))
            self.buffer.add_assistant(f"answer {number}")

        # system + 1, system + 3, system + 5
        self.assertEqual(sizes, [2, 4, 6])


class TestMemoryReset(unittest.TestCase):
    """Step 5: what a reset removes, and what it does not."""

    def setUp(self):
        self.buffer = ConversationBuffer(
            "You are a travel assistant.",
            few_shot=[{"role": "user", "content": "example"},
                      {"role": "assistant", "content": "example reply"}],
            few_shot_mode="messages",
        )
        self.buffer.add_user("I want to visit Spain.")
        self.buffer.add_assistant("Where from?")

    def test_clear_removes_the_conversation(self):
        self.buffer.clear()

        self.assertEqual(self.buffer.turns, [])
        self.assertEqual(self.buffer.exchange_count, 0)
        self.assertNotIn("Spain", str(self.buffer.build_messages()))

    def test_clear_keeps_the_role(self):
        """Role is configuration, not memory."""
        self.buffer.clear()
        messages = self.buffer.build_messages()

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("travel assistant", messages[0]["content"])

    def test_clear_keeps_the_few_shot_examples(self):
        """Which is exactly why a leaked example figure survives a reset."""
        self.buffer.clear()
        self.assertIn("example reply", str(self.buffer.build_messages()))


class TestTrimming(unittest.TestCase):
    def test_old_turns_are_dropped_at_the_limit(self):
        buffer = ConversationBuffer("role", max_turns=3)

        for number in range(6):
            buffer.add_user(f"question {number}")
            buffer.add_assistant(f"answer {number}")

        messages = buffer.build_messages()
        text = str(messages)

        # system + 3 exchanges = 1 + 6
        self.assertEqual(len(messages), 7)
        self.assertNotIn("question 0", text)   # oldest dropped
        self.assertIn("question 5", text)      # newest kept

    def test_trimming_never_drops_the_system_message(self):
        buffer = ConversationBuffer("role", max_turns=1)
        for number in range(5):
            buffer.add_user("q")
            buffer.add_assistant("a")

        self.assertEqual(buffer.build_messages()[0]["role"], "system")


class TestFewShotPlacement(unittest.TestCase):
    """The contamination fix, at the structural level."""

    EXAMPLES = [{"role": "user", "content": "example question"},
                {"role": "assistant", "content": "example answer"}]

    def test_messages_mode_injects_them_as_turns(self):
        buffer = ConversationBuffer("role", self.EXAMPLES, few_shot_mode="messages")
        roles = [m["role"] for m in buffer.build_messages()]

        self.assertEqual(roles, ["system", "user", "assistant"])

    def test_system_mode_folds_them_into_one_message(self):
        buffer = ConversationBuffer("role", self.EXAMPLES, few_shot_mode="system")
        messages = buffer.build_messages()

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("example answer", messages[0]["content"])

    def test_system_mode_marks_them_as_examples(self):
        """The label is what stops them reading as the user's own words."""
        buffer = ConversationBuffer("role", self.EXAMPLES, few_shot_mode="system")
        content = buffer.build_messages()[0]["content"]

        self.assertIn("not part of the real conversation", content)


class TestRoles(unittest.TestCase):
    def test_both_roles_exist_and_are_specific(self):
        for key in ("travel", "grammar"):
            self.assertIn(key, ROLES)
            self.assertGreater(len(ROLES[key]["system"]), 200)

    def test_the_shipped_travel_example_contains_no_figures(self):
        """The contamination fix, asserted so it cannot regress."""
        import re

        text = str(ROLES["travel"]["few_shot"])
        self.assertIsNone(re.search(r"\d+\s*euro", text, re.IGNORECASE), text)
        self.assertNotIn("300", text)

    def test_the_trap_example_is_kept_separate(self):
        self.assertIn("300", str(ROLES["travel"]["few_shot_trap"]))

    def test_unknown_role_is_rejected(self):
        with self.assertRaises(ValueError):
            RoleAssistant("astronaut")

    def test_roles_forbid_inventing_facts(self):
        """Both roles must forbid invention, in the wording each one needs.

        The travel role says "never invent exact prices"; the grammar role says
        "rather than inventing a correction". Asserting one exact phrase would
        fail on the other, so the check is for the concept.
        """
        for key in ("travel", "grammar"):
            self.assertIn("invent", ROLES[key]["system"].lower(), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
