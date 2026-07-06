"""
sanitizer.py
============
Phase 1 of the IPO model: INPUT (Raw Feed) -> Sanitization & Normalization.

Raw human input is messy: "  HELLO!! "  vs "hello" vs "Hello?" should all
collapse to the same internal representation before any logic runs on it.
This module is the single funnel every raw string must pass through.
"""

import re

# Matches anything that is NOT a word character, whitespace, or apostrophe.
# Apostrophes are kept so contractions like "what's" / "how're" survive cleaning.
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s']")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def sanitize_input(raw_input: str) -> str:
    """
    Normalize a raw user string into a clean, comparable string.

    Steps:
        1. Guard against None (e.g. a programmatic caller passing no text).
        2. Lowercase the text  -> case-insensitive matching.
        3. Strip leading/trailing whitespace.
        4. Remove punctuation (keeping apostrophes for contractions).
        5. Collapse repeated internal whitespace to a single space.

    Args:
        raw_input: The exact string typed by the user.

    Returns:
        A cleaned, lowercase string safe for dictionary lookups and
        keyword matching. Returns "" for None or empty/whitespace-only input.

    Examples:
        >>> sanitize_input("  HELLO!!  ")
        'hello'
        >>> sanitize_input("What's   up???")
        "what's up"
    """
    if raw_input is None:
        return ""

    text = raw_input.lower().strip()
    text = _PUNCTUATION_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
