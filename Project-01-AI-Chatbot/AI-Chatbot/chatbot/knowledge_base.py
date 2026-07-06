"""
knowledge_base.py
==================
The chatbot's "brain": a dictionary (hash map) mapping intents to the
patterns that trigger them and the responses they produce.

Why a dictionary and not an if/elif ladder?
--------------------------------------------
An if/elif chain checks conditions one by one -> O(n) time, and it gets
slower and harder to maintain as you add more rules (a classic anti-pattern).
A dictionary lookup is O(1) on average, regardless of how many intents you
add. This file is intentionally structured as pure data (no control flow)
so that *adding a new skill to the bot never requires touching the engine
logic* — you only ever edit the KNOWLEDGE_BASE dict below.

Why this file is intentionally SMALL
--------------------------------------
An earlier version of this bot tried to hard-code general-knowledge facts
(capitals, physics constants, etc.) directly into this dictionary. That
approach is fundamentally broken: the world has millions of facts, and
no static dictionary will ever cover all of them — ask it "what's the
capital of Pakistan" and watch it fail, simply because nobody happened
to type that exact fact in by hand.

So this file now only hard-codes things that genuinely SHOULD be fixed,
deterministic, on-brand responses: greetings, identity, small talk, and
utility commands (time/date/help/AI-status). Every other question —
real knowledge, open-ended questions, anything — is handed straight to
the real Claude API by engine.py's hybrid fallback (see ai_fallback.py
and setup_wizard.py). That's the "connect to the backend, that's it"
design: a thin rule layer for personality, and a live model for
everything else.

Structure of each intent
-------------------------
{
    "category":  a label used to group topics in the 'help' output,
    "patterns":  [list of trigger words/phrases],
    "responses": [list of possible replies]   # one is chosen at random
    # OR, instead of "responses":
    "handler":   a zero-argument function returning a string
                 (used for dynamic content like the current time/date,
                 or content that depends on runtime config like AI mode)
}
"""

import random
from datetime import datetime

from . import ai_fallback


# ----------------------------------------------------------------------
# Dynamic response handlers (used by intents whose answer changes at runtime)
# ----------------------------------------------------------------------
def _tell_time() -> str:
    return f"The current time is {datetime.now().strftime('%I:%M %p')}."


def _tell_date() -> str:
    return f"Today's date is {datetime.now().strftime('%B %d, %Y')}."


def _ai_status() -> str:
    provider = ai_fallback.get_active_provider()
    if provider == "gemini":
        return (
            "I'm connected to Google Gemini (free tier) right now. Anything outside my small "
            "set of built-in rules gets answered live, in real time, through the API."
        )
    if provider == "claude":
        return (
            "I'm connected to Anthropic Claude right now. Anything outside my small set of "
            "built-in rules gets answered live, in real time, through the API."
        )
    return (
        "I'm not connected to a live AI backend yet, so unmatched questions get a plain "
        "fallback reply instead of a real answer. Restart me and say 'y' at the connection "
        "prompt to enable it (a free option is available) — see the README for details."
    )


def _help() -> str:
    """Builds the help text directly from KNOWLEDGE_BASE, grouped by
    category, so it can never go out of sync with the bot's actual
    capabilities."""
    redundant_prefixes = ("meta_",)

    grouped = {}
    for key, data in KNOWLEDGE_BASE.items():
        if key == "help":
            continue
        category = data.get("category", "general")
        label = key
        for prefix in redundant_prefixes:
            if label.startswith(prefix):
                label = label[len(prefix):]
                break
        grouped.setdefault(category, []).append(label.replace("_", " "))

    lines = ["Here's what I can help with directly:"]
    label_order = ["chat", "about", "utility"]
    for category in label_order:
        if category not in grouped:
            continue
        topics = ", ".join(sorted(grouped[category]))
        lines.append(f"  - {category.title()}: {topics}")

    lines.append("Say 'exit', 'quit', or 'bye' anytime to end our chat.")
    lines.append("Ask me literally anything else, and:")
    lines.append(_ai_status())
    return "\n".join(lines)


# ----------------------------------------------------------------------
# THE KNOWLEDGE BASE  (Project Spec: "Dictionary with 5+ intents")
# ----------------------------------------------------------------------
# Deliberately small — see the module docstring for why. This covers
# personality/identity/utility only. Real knowledge goes through the
# live Claude connection (ai_fallback.py), not a hand-typed fact list.
KNOWLEDGE_BASE = {

    # ---------------------------- CHAT -------------------------------
    "greeting": {
        "category": "chat",
        "patterns": [
            "hi", "hii", "hello", "hey", "yo", "sup",
            "good morning", "good afternoon", "good evening",
        ],
        "responses": [
            "Hello there! How can I help you today?",
            "Hi! Great to see you.",
            "Hey! What can I do for you?",
            "Greetings, human. I'm listening.",
        ],
    },
    "wellbeing": {
        "category": "chat",
        "patterns": [
            "how are you", "how're you", "how are you doing",
            "whats up", "what's up", "how is it going",
        ],
        "responses": [
            "I'm just lines of code, but I'm running smoothly! How about you?",
            "Doing great, thanks for asking! What about you?",
            "All systems nominal. How can I help?",
        ],
    },
    "thanks": {
        "category": "chat",
        "patterns": ["thank you", "thanks", "thx", "thank u", "appreciate it"],
        "responses": ["You're welcome!", "Anytime!", "Happy to help.", "No problem at all!"],
    },
    "compliment": {
        "category": "chat",
        "patterns": ["good bot", "nice bot", "you are smart", "you're smart", "youre cool"],
        "responses": [
            "Thank you! I do my best.",
            "That means a lot, even to a bunch of if-statements.",
        ],
    },
    "joke": {
        "category": "chat",
        "patterns": ["joke", "tell me a joke", "make me laugh", "something funny"],
        "responses": [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "I told my computer I needed a break, and it said no problem - it would go to sleep.",
            "Why was the function sad after a party? It didn't get a callback.",
            "Why do Python programmers wear glasses? Because they can't C.",
        ],
    },

    # -------------------------- ABOUT / META --------------------------
    "identity": {
        "category": "about",
        "patterns": ["your name", "who are you", "what are you", "what is your name"],
        "responses": [
            "I'm Nova. I handle greetings and small talk myself, and connect to Claude "
            "for everything else — so I can actually answer real questions, not just guess.",
        ],
    },
    "creator": {
        "category": "about",
        "patterns": ["who made you", "who created you", "who built you", "your creator", "your developer"],
        "responses": [
            "I was built as a small rule-based front end with a real Claude connection in the backend.",
        ],
    },
    "meta_are_you_claude": {
        "category": "about",
        "patterns": ["are you claude", "are you chatgpt", "are you gpt", "are you gemini", "are you a real ai"],
        "responses": [
            "I'm Nova — a thin rule-based layer that handles greetings and small talk myself, and forwards "
            "everything else straight to Claude's API in the backend, so the actual answers come from Claude.",
        ],
    },
    "meta_sentience": {
        "category": "about",
        "patterns": ["are you alive", "are you conscious", "are you sentient", "do you have feelings"],
        "responses": [
            "No — I don't have feelings or consciousness. The small-talk you see is hard-coded; anything "
            "deeper is answered by a real model (Claude) when connected, not by me 'thinking'.",
        ],
    },
    "meta_purpose": {
        "category": "about",
        "patterns": ["what is your purpose", "why were you made", "why do you exist"],
        "responses": [
            "I exist to show a clean hybrid pattern: a fast, deterministic layer for identity and small talk, "
            "and a live connection to a real AI model for everything that requires actual knowledge.",
        ],
    },

    # ---------------------------- UTILITY -------------------------------
    "time": {
        "category": "utility",
        "patterns": ["time", "what time is it", "current time", "tell me the time"],
        "handler": _tell_time,
    },
    "date": {
        "category": "utility",
        "patterns": ["date", "today's date", "what day is it", "what is the date"],
        "handler": _tell_date,
    },
    "ai_status": {
        "category": "utility",
        "patterns": ["ai mode", "is ai mode on", "ai status", "are you using ai", "are you connected"],
        "handler": _ai_status,
    },
    "help": {
        "category": "utility",
        "patterns": ["help", "what can you do", "commands", "options", "menu"],
        "handler": _help,
    },
}

# ----------------------------------------------------------------------
# Fallback (Project Spec: "Default response for unknowns")
# ----------------------------------------------------------------------
# This is ONLY reached when no rule matched AND ask_ai() returned None,
# which means the user has NOT connected a live AI backend at all.
# When connected but the API call fails, ask_ai() now returns a
# descriptive error string which the engine shows directly — so the
# user sees the actual problem instead of a misleading "not connected."
FALLBACK_RESPONSES = [
    "I don't have a live AI connection set up yet, so I can't answer that. "
    "Restart me and say 'y' at the setup prompt — a free option (Google Gemini) "
    "is available and takes under a minute to connect.",
    "That's outside my built-in rules and I'm not connected to a live AI backend. "
    "Restart me and say 'y' to connect for free via Google Gemini.",
]

# ----------------------------------------------------------------------
# Flattened exact-match lookup table: pattern -> intent_key
# ----------------------------------------------------------------------
# This is the literal "responses.get(user_input, default)" pattern from the
# spec, generalized to point at an intent instead of a raw string, so the
# engine still gets an O(1) average-time exact lookup before it ever falls
# back to slower keyword scanning or the live Claude connection.
PATTERN_TO_INTENT = {
    pattern: intent_key
    for intent_key, intent_data in KNOWLEDGE_BASE.items()
    for pattern in intent_data["patterns"]
}
