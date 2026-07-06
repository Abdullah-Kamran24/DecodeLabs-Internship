"""
ai_fallback.py
===============
The live backend connection — the "PASS TO LLM" step in the Hybrid
Architecture.  Supports two providers:

  - "gemini"  (DEFAULT, FREE)
        Google Gemini via Google AI Studio.
        Free tier, no credit card.
        Keys start with  AIza...
        Get one at: https://aistudio.google.com/apikey

  - "claude"  (OPTIONAL, PAID)
        Anthropic Claude API.
        Pay-as-you-go, requires billing.
        Keys start with  sk-ant-...
        Get one at: https://console.anthropic.com

Key rules:
  * This module NEVER crashes the chatbot.
  * If no key is set -> ask_ai() returns None (truly not connected).
  * If a key IS set but the API call fails -> ask_ai() returns a
    human-readable error string so the user can see what went wrong
    (wrong key, no internet, quota exceeded, etc.) instead of a
    misleading "not connected" message.
"""

import json
import os
import re
import urllib.error
import urllib.request

# -----------------------------------------------------------------------
# Provider endpoints / defaults
# -----------------------------------------------------------------------
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/{model}:generateContent?key={key}"
)
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6"

DEFAULT_TIMEOUT   = 15.0
MAX_TOKENS        = 350

SYSTEM_PROMPT = (
    "You are Nova, the AI assistant voice of a rule-based chatbot. "
    "A user question did not match any hard-coded rule, so it was "
    "forwarded to you. Answer accurately and concisely in 1-5 sentences. "
    "If you are unsure, say so honestly."
)

# -----------------------------------------------------------------------
# Key sanitization (strips Windows-paste control chars + duplication)
# -----------------------------------------------------------------------
def _sanitize_key(raw: str) -> str:
    cleaned = re.sub(r"[^\x20-\x7E]", "", raw).strip()
    n = len(cleaned)
    if n > 0 and n % 2 == 0:
        half = n // 2
        if cleaned[:half] == cleaned[half:]:
            cleaned = cleaned[:half]
    return cleaned.strip()

# -----------------------------------------------------------------------
# Provider detection
# -----------------------------------------------------------------------
def get_active_provider():
    """Returns 'gemini', 'claude', or None."""
    explicit   = os.environ.get("AI_PROVIDER", "").strip().lower()
    has_gemini = bool(_sanitize_key(os.environ.get("GEMINI_API_KEY", "")))
    has_claude = bool(_sanitize_key(os.environ.get("ANTHROPIC_API_KEY", "")))

    if explicit == "gemini" and has_gemini:
        return "gemini"
    if explicit == "claude" and has_claude:
        return "claude"
    if has_gemini:
        return "gemini"
    if has_claude:
        return "claude"
    return None

def is_ai_mode_enabled() -> bool:
    return get_active_provider() is not None

# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------
def ask_ai(question: str, timeout: float = DEFAULT_TIMEOUT):
    """
    Returns:
        None   -> no key set (truly not connected).
        str    -> the model's answer  OR  a clear error message
                  (so the user always knows what happened).
    """
    if not question:
        return None

    provider = get_active_provider()
    if provider == "gemini":
        return _ask_gemini(question, timeout)
    if provider == "claude":
        return _ask_claude(question, timeout)
    return None   # no key at all

# -----------------------------------------------------------------------
# Gemini
# -----------------------------------------------------------------------
def _ask_gemini(question: str, timeout: float):
    api_key = _sanitize_key(os.environ.get("GEMINI_API_KEY", ""))
    if not api_key:
        return None

    model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    url   = GEMINI_URL_TEMPLATE.format(model=model, key=api_key)

    payload = json.dumps({
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents":           [{"parts": [{"text": question}]}],
        "generationConfig":   {"maxOutputTokens": MAX_TOKENS},
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"content-type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = _read_error_body(e)
        return (
            f"⚠ Gemini API error {e.code}. "
            + _gemini_error_hint(e.code, detail)
        )
    except urllib.error.URLError as e:
        return f"⚠ Could not reach Gemini — check your internet connection. ({e.reason})"
    except OSError as e:
        return f"⚠ Network error: {e}"
    except (ValueError, KeyError):
        return "⚠ Gemini returned an unexpected response. Please try again."

    try:
        parts = body["candidates"][0]["content"]["parts"]
        text  = " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        return text or "⚠ Gemini returned an empty answer. Please try rephrasing."
    except (KeyError, IndexError, TypeError):
        return "⚠ Gemini returned an unexpected format. Please try again."

def _gemini_error_hint(code: int, detail: str) -> str:
    if code == 400:
        return "Your key may be malformed. Gemini keys start with AIza — get one free at https://aistudio.google.com/apikey"
    if code == 403:
        return "Your key was rejected (403 Forbidden). Make sure you copied it correctly from https://aistudio.google.com/apikey"
    if code == 429:
        return "Rate limit hit (429). You've sent too many requests — wait a few seconds and try again."
    return f"Details: {detail}"

# -----------------------------------------------------------------------
# Claude
# -----------------------------------------------------------------------
def _ask_claude(question: str, timeout: float):
    api_key = _sanitize_key(os.environ.get("ANTHROPIC_API_KEY", ""))
    if not api_key:
        return None

    # Warn early if the key format looks wrong.
    if not api_key.startswith("sk-ant-"):
        return (
            f"⚠ That doesn't look like a valid Anthropic API key. "
            f"(Received: {api_key[:8]}...) "
            f"Anthropic keys always start with 'sk-ant-'. "
            f"Get a real key at https://console.anthropic.com  —  or use "
            f"Google Gemini (option 1) which is completely free."
        )

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL

    payload = json.dumps({
        "model":      model,
        "max_tokens": MAX_TOKENS,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user", "content": question}],
    }).encode("utf-8")

    req = urllib.request.Request(
        ANTHROPIC_URL, data=payload, method="POST",
        headers={
            "x-api-key":          api_key,
            "anthropic-version":  ANTHROPIC_VERSION,
            "content-type":       "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = _read_error_body(e)
        return (
            f"⚠ Anthropic API error {e.code}. "
            + _claude_error_hint(e.code, detail)
        )
    except urllib.error.URLError as e:
        return f"⚠ Could not reach Anthropic — check your internet connection. ({e.reason})"
    except OSError as e:
        return f"⚠ Network error: {e}"
    except ValueError:
        return "⚠ Anthropic returned an unexpected response. Please try again."

    try:
        texts = [b.get("text", "") for b in body.get("content", [])
                 if isinstance(b, dict) and b.get("type") == "text"]
        text  = " ".join(t for t in texts if t).strip()
        return text or "⚠ Claude returned an empty answer. Please try rephrasing."
    except (KeyError, TypeError):
        return "⚠ Anthropic returned an unexpected format. Please try again."

def _claude_error_hint(code: int, detail: str) -> str:
    if code == 401:
        return (
            "Your key was rejected (401 Unauthorized). "
            "Make sure it starts with 'sk-ant-' and was copied correctly "
            "from https://console.anthropic.com  —  or use Google Gemini "
            "(option 1) which is completely free."
        )
    if code == 403:
        return "Access denied (403). Your account may not have API access enabled."
    if code == 429:
        return "Rate limit or quota exceeded (429). Check your plan at https://console.anthropic.com"
    if code == 529:
        return "Anthropic servers are overloaded right now (529). Try again in a moment."
    return f"Details: {detail}"

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _read_error_body(e: urllib.error.HTTPError) -> str:
    try:
        return e.read().decode("utf-8", errors="ignore")[:300]
    except Exception:
        return ""
