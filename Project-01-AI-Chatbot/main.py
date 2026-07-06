#!/usr/bin/env python3
"""
main.py
=======
Entry point for the Rule-Based AI Chatbot (Project 1).

Usage:
    python main.py

This file intentionally contains almost no logic — it just imports the
Chatbot class and starts it. Keeping the entry point thin makes the
project easy to test (the tests/ folder imports the chatbot package
directly, without ever running this script) and easy to reuse (another
script, or a future GUI/web wrapper, could import Chatbot the same way).
"""

from chatbot import Chatbot


def main() -> None:
    bot = Chatbot()
    bot.run()


if __name__ == "__main__":
    main()
