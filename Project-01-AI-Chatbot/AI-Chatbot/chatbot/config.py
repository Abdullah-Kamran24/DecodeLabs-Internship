"""
config.py
=========
Centralized, static configuration for the chatbot.

Keeping these values in one place (instead of scattered magic strings
throughout the codebase) is a basic but important professional habit:
if you ever want to rename the bot, add an exit word, or relocate the
log file, you change exactly one line, in exactly one place.
"""

import os

# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------
BOT_NAME = "Nova"
VERSION = "1.0.0"

# ----------------------------------------------------------------------
# Exit Strategy (Project Spec: "Clean break command")
# ----------------------------------------------------------------------
# A set gives O(1) average-time membership checks ("word in EXIT_COMMANDS"),
# the same efficiency principle used by the knowledge base dictionary.
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "good bye", "see you", "stop", "q"}

FAREWELL_RESPONSES = [
    "Goodbye! Have a great day.",
    "See you later, take care!",
    "Bye! Thanks for chatting with me.",
    "Catch you next time!",
]

# ----------------------------------------------------------------------
# Paths / Logging
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "chatbot_log.txt")

# ----------------------------------------------------------------------
# Misc display settings
# ----------------------------------------------------------------------
BANNER_WIDTH = 64
