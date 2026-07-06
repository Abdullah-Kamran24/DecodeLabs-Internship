"""
setup_wizard.py
================
One-time interactive flow to connect Nova to a live AI backend.

Why input() instead of getpass.getpass()
------------------------------------------
getpass is unreliable on Windows CMD/PowerShell: it hides input
completely (so users cannot tell if anything was pasted) and often
silently drops pasted text entirely, returning an empty string.
input() works correctly on every platform. We compensate for the
lack of masking by showing only a short preview after entry so the
user can confirm the key was received without it lingering on screen.

Key format guide
-----------------
  Google Gemini  (free)  : starts with  AIza...
  Anthropic Claude (paid): starts with  sk-ant-...
"""

import json
import os
import re

from .config import BASE_DIR

SECRETS_PATH       = os.path.join(BASE_DIR, ".secrets.json")
GEMINI_SIGNUP_URL  = "https://aistudio.google.com/apikey"
CLAUDE_SIGNUP_URL  = "https://console.anthropic.com"


# ------------------------------------------------------------------
# Key helpers
# ------------------------------------------------------------------
def _clean_key(raw: str) -> str:
    """
    Strip control characters and fix Windows-paste duplication.
    1. Remove every non-printable character (keep only 0x20-0x7E).
    2. If the string is doubled (first half == second half), keep half.
    3. Strip whitespace.
    """
    cleaned = re.sub(r"[^\x20-\x7E]", "", raw).strip()
    n = len(cleaned)
    if n > 0 and n % 2 == 0:
        half = n // 2
        if cleaned[:half] == cleaned[half:]:
            cleaned = cleaned[:half]
    return cleaned.strip()


def _mask(key: str) -> str:
    """Return first-4 ... last-4 preview of an API key."""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "..." + key[-4:]


def _validate_key_format(provider: str, key: str):
    """
    Return a warning string if the key format looks wrong, else None.
      Gemini keys  start with  AIza
      Claude keys  start with  sk-ant-
    """
    if provider == "gemini" and not key.startswith("AIza"):
        return (
            f"  ⚠  Gemini keys normally start with 'AIza' "
            f"but yours starts with '{key[:8]}...'.\n"
            f"     Get a valid free key at: {GEMINI_SIGNUP_URL}"
        )
    if provider == "claude" and not key.startswith("sk-ant-"):
        return (
            f"  ⚠  Claude keys always start with 'sk-ant-' "
            f"but yours starts with '{key[:8]}...'.\n"
            f"     This key will likely be rejected.\n"
            f"     Get a real key at: {CLAUDE_SIGNUP_URL}\n"
            f"     Or use Google Gemini (option 1) — it is completely free."
        )
    return None


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------
def _load_saved_key() -> None:
    """Load provider + key from .secrets.json into os.environ."""
    if os.environ.get("GEMINI_API_KEY", "").strip() or \
       os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return  # real env vars always win

    try:
        with open(SECRETS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return

    provider = str(data.get("AI_PROVIDER", "")).strip()
    if provider:
        os.environ["AI_PROVIDER"] = provider

    for field in ("GEMINI_API_KEY", "GEMINI_MODEL",
                  "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        raw = str(data.get(field, ""))
        val = _clean_key(raw) if "KEY" in field else raw.strip()
        if val:
            os.environ[field] = val


def _save_key(provider: str) -> bool:
    """Write provider + current env keys to .secrets.json."""
    try:
        with open(SECRETS_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "AI_PROVIDER":      provider,
                "GEMINI_API_KEY":    os.environ.get("GEMINI_API_KEY", ""),
                "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
            }, f)
        return True
    except OSError:
        return False


# ------------------------------------------------------------------
# Interactive prompts
# ------------------------------------------------------------------
def _prompt_provider() -> str:
    """Ask which provider to use; default = Gemini (free)."""
    print()
    print("  Which AI backend would you like to use?")
    print(f"  [1] Google Gemini  — FREE, no credit card  "
          f"(get key: {GEMINI_SIGNUP_URL})")
    print(f"  [2] Anthropic Claude — paid API             "
          f"(get key: {CLAUDE_SIGNUP_URL})")
    print()
    try:
        choice = input(
            "  Enter 1 or 2 (press Enter for default = 1 free Gemini): "
        ).strip()
    except EOFError:
        choice = ""
    return "claude" if choice == "2" else "gemini"


def _prompt_key(provider: str) -> str:
    """
    Prompt for the API key using plain input() — reliable on all
    Windows/Mac/Linux terminals. Shows a masked preview + format
    validation so users know exactly what was captured and whether
    it looks correct before anything is saved.
    """
    if provider == "gemini":
        label = "Gemini"
        hint  = "Gemini keys start with: AIza..."
    else:
        label = "Anthropic Claude"
        hint  = "Claude keys start with: sk-ant-..."

    print()
    print(f"  Paste your {label} API key and press Enter.")
    print(f"  ({hint})")
    print()

    while True:
        try:
            raw = input("  API Key: ").strip()
        except EOFError:
            return ""

        api_key = _clean_key(raw)

        if not api_key:
            print()
            print("  [!] Nothing was captured.")
            print("      In Windows CMD          : right-click to paste")
            print("      In PowerShell/Terminal  : Ctrl+Shift+V or right-click")
            print()
            try:
                retry = input("  Try again? [Y/n]: ").strip().lower()
            except EOFError:
                return ""
            if retry in ("n", "no"):
                return ""
            continue

        # Show masked preview so user can confirm the key arrived.
        print()
        print(f"  Key received: {_mask(api_key)}  ({len(api_key)} characters)")

        # Validate format before asking for confirmation.
        warning = _validate_key_format(provider, api_key)
        if warning:
            print()
            print(warning)
            print()
            try:
                proceed = input(
                    "  Save anyway and try? [y/N]: "
                ).strip().lower()
            except EOFError:
                proceed = "n"
            if proceed not in ("y", "yes"):
                print()
                print("  Okay — let's try again. Paste a different key.")
                continue
        else:
            print()
            try:
                confirm = input("  Does that look right? [Y/n]: ").strip().lower()
            except EOFError:
                confirm = "y"
            if confirm in ("n", "no"):
                print()
                print("  Okay — let's try again.")
                continue

        return api_key


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def maybe_run_setup_wizard() -> None:
    """
    Called once right before the main chat loop starts.
    Silent if already connected; otherwise runs the one-time setup.
    """
    _load_saved_key()

    if (os.environ.get("GEMINI_API_KEY", "").strip() or
            os.environ.get("ANTHROPIC_API_KEY", "").strip()):
        return  # already connected

    print()
    print("=" * 64)
    print("  Nova — Live AI Connection Setup")
    print("=" * 64)
    print()
    print("  Nova can connect to a live AI backend for real-time answers")
    print("  to anything outside her built-in rules.")
    print("  A completely FREE option is available (Google Gemini).")
    print("  You only need to do this ONCE — it saves automatically.")
    print()

    try:
        choice = input("  Connect now? [Y/n]: ").strip().lower()
    except EOFError:
        return

    if choice in ("n", "no"):
        print()
        print("  Running rule-based only. Restart anytime to connect.")
        print()
        return

    provider = _prompt_provider()
    api_key  = _prompt_key(provider)

    if not api_key:
        print()
        print("  No key saved — running rule-based only for now.")
        print("  Restart anytime to try again.")
        print()
        return

    env_var = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
    os.environ["AI_PROVIDER"] = provider
    os.environ[env_var]       = api_key

    label = "Google Gemini (free)" if provider == "gemini" else "Anthropic Claude (paid)"

    if _save_key(provider):
        print()
        print(f"  Connected to {label}!")
        print("  Saved to .secrets.json — you won't be asked again on this machine.")
        print()
    else:
        print()
        print(f"  Connected to {label} for this session.")
        print("  (Could not save to disk — you'll be asked again next time.)")
        print()
