"""
core.py
=======
Phase 3 of the IPO model: OUTPUT (Response Generation & Feedback Loop),
plus "The Heartbeat" — the infinite loop that keeps the bot alive until
a kill command is received.

This module wires together sanitizer -> engine -> terminal I/O -> logging
into one runnable Chatbot class.
"""

import logging
import random

from .config import BOT_NAME, EXIT_COMMANDS, FAREWELL_RESPONSES, LOG_FILE, BANNER_WIDTH
from .sanitizer import sanitize_input
from .engine import ResponseEngine
from .setup_wizard import maybe_run_setup_wizard


class Chatbot:
    """
    The orchestrator. Owns the conversation loop, the logger, and a single
    ResponseEngine instance (built once, reused for every turn).
    """

    def __init__(self):
        self.engine = ResponseEngine()
        self._configure_logging()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _configure_logging(self) -> None:
        """Every conversation turn is appended to logs/chatbot_log.txt.
        This is the 'Feedback Loop' referenced in the IPO blueprint —
        a durable record of inputs and outputs for later review/debugging."""
        logger = logging.getLogger("chatbot")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            logger.addHandler(handler)

        self._logger = logger

    def _log_turn(self, speaker: str, text: str) -> None:
        self._logger.info("%s: %s", speaker, text)

    def _print_banner(self) -> None:
        print("=" * BANNER_WIDTH)
        print(f"  {BOT_NAME} — Rule-Based AI Chatbot (Project 1)".center(BANNER_WIDTH))
        print("  Type 'help' for what I can do, or 'exit' to quit.".center(BANNER_WIDTH))
        print("=" * BANNER_WIDTH)

    # ------------------------------------------------------------------
    # The main loop ("The Heartbeat: The Infinite Loop")
    # ------------------------------------------------------------------
    def run(self) -> None:
        """
        Starts the conversation. The organism stays alive (while True)
        until the user issues a kill command, the input stream closes
        (EOF), or the process is interrupted (Ctrl+C) — all three exits
        are handled gracefully, with no stack traces shown to the user.
        """
        self._print_banner()
        maybe_run_setup_wizard()

        try:
            while True:  # <-- THE HEARTBEAT
                try:
                    raw_text = input("You: ")
                except EOFError:
                    # Input stream closed (e.g. piped input ran out).
                    print(f"\n{BOT_NAME}: Input stream closed. Goodbye!")
                    break

                clean_text = sanitize_input(raw_text)
                self._log_turn("You", raw_text)

                if not clean_text:
                    # Empty / whitespace-only message: nudge, don't crash.
                    print(f"{BOT_NAME}: I didn't catch that — try typing something!")
                    continue

                if clean_text in EXIT_COMMANDS:
                    farewell = random.choice(FAREWELL_RESPONSES)
                    print(f"{BOT_NAME}: {farewell}")
                    self._log_turn(BOT_NAME, farewell)
                    break  # <-- KILL COMMAND: clean break out of the loop

                response = self.engine.get_response(clean_text, raw_text)
                print(f"{BOT_NAME}: {response}")
                self._log_turn(BOT_NAME, response)

        except KeyboardInterrupt:
            # Ctrl+C pressed: exit on a new line instead of a raw traceback.
            print(f"\n{BOT_NAME}: Interrupted. Shutting down gracefully. Goodbye!")
