"""
chatbot
=======
A hybrid chatbot: a thin, deterministic rule layer (greetings, identity,
small talk, utility commands) backed by a live connection to a real AI
model for everything else — built around the IPO (Input -> Process ->
Output) model. Supports Google Gemini (free, default) and Anthropic
Claude (paid, optional) as the live backend.

Modules
-------
config          : Static configuration (bot name, exit words, file paths).
sanitizer       : Phase 1 - input cleaning & normalization.
knowledge_base  : The chatbot's small, deterministic rule set (personality,
                  identity, utility commands) - intentionally NOT an
                  encyclopedia; real questions go to the live AI backend.
ai_fallback     : The live backend connection - sends anything the rules
                  don't cover to Gemini (free) or Claude (paid), whichever
                  is connected.
setup_wizard    : One-time "connect a live AI backend" prompt, with a
                  choice of provider; saves the chosen key locally so the
                  connection persists across runs.
engine          : Phase 2 - intent matching + hybrid fallback (the Process stage).
core            : Phase 3 - the Chatbot class that runs the main infinite loop (Output + Heartbeat).
"""

from .core import Chatbot

__all__ = ["Chatbot"]
__version__ = "2.1.0"
