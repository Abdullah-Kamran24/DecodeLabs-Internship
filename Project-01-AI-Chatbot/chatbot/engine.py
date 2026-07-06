"""
engine.py
=========
Phase 2 of the IPO model: PROCESS (Intent Matching & State).

The ResponseEngine resolves a sanitized string into a reply using a
FOUR-stage cascade — this is the "Hybrid Architecture" from the project
blueprint, made concrete. Each stage only runs if every earlier stage
failed to produce an answer:

    Stage 1 - O(1) EXACT MATCH
        A direct dictionary lookup (PATTERN_TO_INTENT.get(text)).
        This is the fast path and mirrors the spec's
        `responses.get(user_input, default)` pattern exactly.

    Stage 2 - O(n) KEYWORD CONTAINMENT SCAN
        If the user typed something that *contains* a known pattern as a
        whole word (e.g. "hello there friend" contains "hello"), we still
        recognize the intent. This trades a small amount of speed for a
        much more natural conversational feel.

    Stage 3 - AI FALLBACK (opt-in, "PASS TO LLM" in the diagram)
        If no rule matched at all, and the user has connected a live AI
        backend (see ai_fallback.py — Google Gemini, free, or Anthropic
        Claude, paid), the raw question is sent to that model for a
        genuine, open-ended answer.

    Stage 4 - DEFAULT FALLBACK
        If AI mode is off, or the AI call itself fails for any reason
        (no internet, bad key, timeout), return a random default response
        instead of crashing or staying silent. The bot should never be
        "stumped" with an unhandled error.
"""

import random
import re
from typing import Optional

from .knowledge_base import KNOWLEDGE_BASE, PATTERN_TO_INTENT, FALLBACK_RESPONSES
from .ai_fallback import ask_ai


class ResponseEngine:
    """Deterministic rule-based response resolver."""

    def __init__(self):
        self._kb = KNOWLEDGE_BASE
        self._exact_lookup = PATTERN_TO_INTENT
        # Pre-compile one regex per pattern up front (built once, reused
        # on every message) instead of recompiling on every user turn.
        self._compiled_patterns = [
            (intent_key, pattern, re.compile(rf"\b{re.escape(pattern)}\b"))
            for intent_key, data in self._kb.items()
            for pattern in data["patterns"]
        ]

    def get_response(self, clean_input: str, raw_input: Optional[str] = None) -> str:
        """
        Resolve a sanitized input string into a chatbot reply.

        Args:
            clean_input: Output of sanitizer.sanitize_input(). Used for
                rule matching (Stages 1 & 2).
            raw_input: The original, un-sanitized text the user typed.
                Used for Stage 3 (AI fallback) so the model sees natural
                punctuation/capitalization instead of stripped text.
                Falls back to clean_input if not provided.

        Returns:
            A response string. Never raises, never returns None/empty.
        """
        if not clean_input:
            return random.choice(FALLBACK_RESPONSES)

        intent_key = self._exact_match(clean_input)

        if intent_key is None:
            intent_key = self._fuzzy_match(clean_input)

        if intent_key is not None:
            return self._resolve(intent_key)

        # Stage 3: no rule matched at all -> Hybrid Architecture kicks in.
        # ask_ai() returns None instantly if AI mode isn't enabled, so this
        # costs nothing when the feature is off.
        question = raw_input if raw_input else clean_input
        ai_answer = ask_ai(question)
        if ai_answer:
            return f"[AI mode] {ai_answer}"

        # Stage 4: final, guaranteed-safe fallback.
        return random.choice(FALLBACK_RESPONSES)

    # ------------------------------------------------------------------
    # Internal stages
    # ------------------------------------------------------------------
    def _exact_match(self, text: str) -> Optional[str]:
        """Stage 1: O(1) average-time hash map lookup."""
        return self._exact_lookup.get(text)

    def _fuzzy_match(self, text: str) -> Optional[str]:
        """Stage 2: O(n) whole-word containment scan."""
        for intent_key, _pattern, compiled in self._compiled_patterns:
            if compiled.search(text):
                return intent_key
        return None

    def _resolve(self, intent_key: str) -> str:
        """Turn a matched intent key into an actual reply string."""
        intent_data = self._kb[intent_key]
        handler = intent_data.get("handler")
        if handler is not None:
            return handler()
        return random.choice(intent_data["responses"])
