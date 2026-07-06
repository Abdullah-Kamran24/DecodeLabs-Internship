# 🤖 Nova — Hybrid AI Chatbot (Rule-Based Core + Free Live AI Backend)

A small, deterministic rule layer for personality and small talk, backed by
a **real, live connection to a free AI model** for everything else. Instead
of trying to hand-type the world's facts into a dictionary (which breaks
the moment someone asks something nobody anticipated), Nova handles what
*should* be fixed and predictable herself, and forwards everything else to
a real AI backend in real time — connected once, for free, in under a
minute.

**The default backend is Google Gemini — genuinely free, no credit card,
no trial period, no expiration.** Anthropic Claude is also supported as
an optional paid alternative, if you'd rather use that instead.

This started as a pure rule-based chatbot demonstrating the **IPO pipeline**
(Input → Process → Output) and a dictionary-driven "brain" instead of a
fragile if/elif ladder. It now extends that foundation with the **Hybrid
Architecture** pattern: a fast, deterministic layer up front, and a real
AI model wired in behind it for anything the rules don't cover.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation & Setup](#installation--setup)
6. [How to Run](#how-to-run)
7. [Connecting to a Free AI Backend (One-Time Setup)](#connecting-to-a-free-ai-backend-one-time-setup)
8. [Example Conversation](#example-conversation)
9. [Available Commands / Intents](#available-commands--intents)
10. [Running the Tests](#running-the-tests)
11. [How to Extend the Bot](#how-to-extend-the-bot)
12. [Troubleshooting](#troubleshooting)
13. [Design Notes](#design-notes)
14. [License](#license)

---

## Features

- ✅ **Continuous conversation loop** — the bot stays alive until you tell it to stop.
- ✅ **Input sanitization** — case-insensitive, punctuation-stripped, whitespace-normalized.
- ✅ **A small, deliberately lean rule layer** — greetings, identity, small talk, and utility commands, resolved instantly with an O(1) dictionary lookup. No giant hand-typed fact list to maintain or outgrow.
- ✅ **Free live AI backend by default** — anything outside the rule layer (real knowledge, open-ended questions, "what's the capital of X") is answered live by **Google Gemini**, which has a genuine free tier: no credit card, no expiration.
- ✅ **Anthropic Claude also supported** — pick it instead at setup if you'd rather use a paid API.
- ✅ **One-time connection setup** — no manual environment-variable juggling. Run the bot, say "yes" once, pick a provider, paste your key, and you're connected from then on.
- ✅ **Transparent answers** — AI-generated replies are tagged `[AI mode]` so you always know whether an answer came from a hard-coded rule or a live model call.
- ✅ **Guaranteed-safe fallback** — if you're not connected, or the live call fails for any reason (no internet, bad key, rate limit), the bot says so plainly instead of crashing or guessing.
- ✅ **Clean exit strategy** — multiple exit words (`exit`, `quit`, `bye`, `stop`...), plus graceful handling of `Ctrl+C` and closed input streams.
- ✅ **Conversation logging** — every turn is timestamped and saved to `logs/chatbot_log.txt`.
- ✅ **Fully unit tested** — 39 automated tests, covering both providers, the setup wizard, and mocked success/failure paths for each. No real API key or network access needed to run them.
- ✅ **Zero required dependencies** — runs anywhere Python 3.8+ runs. Both AI backends use only `urllib` from the standard library — no `pip install` needed even for that.

---

## Architecture

```
 INPUT                       PROCESS                                  OUTPUT
(Raw Feed)              (The Logic Skeleton)                      (Feedback Loop)
   │                              │                                      │
   ▼                              ▼                                      ▼
sanitizer.py    ──────►    engine.py + knowledge_base.py  ────►      core.py
(lower, strip,         Stage 1: O(1) exact match                 (print reply,
 remove punctuation)    Stage 2: O(n) fuzzy match                 write to log)
                        Stage 3: live AI call (ai_fallback.py)
                        Stage 4: safe default fallback
```

**The Hybrid flow, end to end:**
```
USER QUESTION
     │
     ▼
RULE MATCH? ──YES──► INSTANT RESPONSE (greeting, identity, utility — zero network)
     │
     NO
     ▼
CONNECTED?  ──YES──► live call to Gemini (free) or Claude (paid)
     │                          │
     NO                    call fails
     │                          │
     ▼                          ▼
        SAFE FALLBACK MESSAGE (never crashes, never guesses)
```

**Why the rule layer is small on purpose.** An earlier version of this
project tried to hard-code general-knowledge facts (capitals, constants,
historical dates) directly into the dictionary. That approach is
fundamentally broken — ask it something nobody happened to type in by
hand (like "what's the capital of Pakistan") and it simply fails, no
matter how many facts you add, because the real world doesn't fit in a
static dictionary. So this version draws the line differently: the rule
layer only owns things that genuinely *should* be fixed and predictable
(your name, your identity, greetings, utility commands). Real knowledge
is **always** answered live, by a real model, through the backend
connection — not memorized in advance.

**Why Gemini is the default.** Anthropic's Claude API is pay-as-you-go
with no permanent free tier. Google's Gemini API, accessed through Google
AI Studio, genuinely is free — no credit card, no expiration, just a
Google account. So Nova defaults new connections to Gemini and only uses
Claude if you specifically choose it.

**Why a dictionary instead of if/elif, for the rule layer itself.**
An if/elif ladder checks rules one-by-one — performance degrades linearly
(`O(n)`) as you add features, and becomes a fragile anti-pattern to
maintain. A Python dictionary gives **O(1)** average-time lookups no
matter how many rules you add:

```python
intent_key = PATTERN_TO_INTENT.get(clean_input)   # O(1) — the fast path
```

**The Heartbeat (infinite loop + kill command)**
```python
while True:                      # the organism stays alive...
    user_input = get_input()
    if user_input in EXIT_COMMANDS:
        break                     # ...until the kill command arrives
    process(user_input)
```

---

## Project Structure

```
rule_based_chatbot/
│
├── main.py                  # Entry point — run this file to start chatting
├── requirements.txt         # No external deps; documents that fact
├── run.bat                  # One-click launcher for Windows
├── run.sh                   # One-click launcher for macOS/Linux
├── .gitignore                # Excludes logs, caches, and your saved API key
├── README.md                # You are here
│
├── chatbot/                 # The application package
│   ├── __init__.py          # Exposes the Chatbot class
│   ├── config.py            # Bot name, exit words, log file path
│   ├── sanitizer.py         # Phase 1: input cleaning & normalization
│   ├── knowledge_base.py    # The small rule layer — personality + utility only
│   ├── ai_fallback.py       # The live backend connection (Gemini free / Claude paid)
│   ├── setup_wizard.py      # One-time "connect" prompt with provider choice, saved locally
│   ├── engine.py            # Phase 2: 4-stage hybrid matching/fallback logic
│   └── core.py               # Phase 3: the Chatbot class + main loop
│
├── tests/                   # Automated unit tests
│   ├── __init__.py
│   └── test_chatbot.py
│
├── logs/                    # Auto-created at runtime
│   └── chatbot_log.txt      # Timestamped conversation history
│
└── .secrets.json             # Created after you connect — your local provider + key (gitignored, never commit)
```

---

## Requirements

- **Python 3.8 or newer** (uses only the standard library: `re`, `random`,
  `logging`, `datetime`, `os`, `json`, `getpass`, `urllib`, `unittest`).
- No `pip install` needed, ever — even the live AI backends use only the
  standard library.
- An internet connection and a free
  [Google AI Studio API key](https://aistudio.google.com/apikey)
  **only if** you choose to connect (recommended — it's free). Without
  one, the bot still runs fine in rule-based-only mode.

Check your Python version:

```bash
python --version
```

> 💡 On some systems (especially macOS/Linux) the command is `python3`
> instead of `python`. If `python --version` shows Python 2.x, use
> `python3` throughout this guide instead.

---

## Installation & Setup

1. **Download / unzip the project** into any folder, e.g. `C:\Projects\rule_based_chatbot` or `~/Projects/rule_based_chatbot`.
2. Open a terminal (Command Prompt, PowerShell, or macOS/Linux terminal).
3. Navigate into the project folder:

   ```bash
   cd path/to/rule_based_chatbot
   ```

That's it — there is no `pip install` step.

---

## How to Run

### Windows (Command Prompt / PowerShell)

```cmd
cd path\to\rule_based_chatbot
python main.py
```

Or simply **double-click `run.bat`** in File Explorer.

### macOS / Linux (Terminal)

```bash
cd path/to/rule_based_chatbot
python3 main.py
```

Or, after making it executable once:

```bash
chmod +x run.sh
./run.sh
```

### Stopping the bot

Type any of: `exit`, `quit`, `bye`, `goodbye`, `stop`, `q` — or press `Ctrl+C`.

---

## Connecting to a Free AI Backend (One-Time Setup)

The first time you run the bot, it will ask:

```
----------------------------------------------------------------
Nova can connect to a live AI backend for real-time answers to
anything outside her built-in rules (general knowledge,
open-ended questions, etc). A free option is available.
----------------------------------------------------------------
Connect to a live AI backend now? [y/N]:
```

**Say `y`**, then choose a provider:

```
Which AI backend would you like to connect?
  [1] Google Gemini  — FREE, no credit card required  (get a key: https://aistudio.google.com/apikey)
  [2] Anthropic Claude — paid API, requires billing    (get a key: https://console.anthropic.com)
Choose [1/2] (default 1 - free):
```

Press Enter or type `1` for the **free** option (recommended).

### Getting a free Gemini key (takes about a minute)

1. Go to **[aistudio.google.com/apikey](https://aistudio.google.com/apikey)**.
2. Sign in with any Google account.
3. Click **"Create API key"**.
4. Copy the key it gives you (starts with `AIza...`).
5. Paste it into the bot's prompt when asked (input is hidden, like a password field).

No credit card, no trial period, no billing setup. Just a Google account.

That's it — the key is saved locally to `.secrets.json` in the project
folder, and you will **never be asked again on this machine**. From then
on, every question outside Nova's small rule set is answered live, for
free.

If you say `n` (or just press Enter at the first prompt), Nova keeps
running in rule-based-only mode — unmatched questions get a plain, honest
fallback message instead of a real answer. You can connect anytime later
by simply restarting the bot.

### Checking connection status

```
You: ai status
Nova: I'm connected to Google Gemini (free tier) right now. Anything outside
      my small set of built-in rules gets answered live, in real time,
      through the API.
```

### Advanced: connecting via environment variable instead

If you'd rather not use the interactive prompt (for example, when running
in a script or CI), you can set the relevant variables yourself before
launching — the wizard detects them and skips the prompt entirely:

```bash
# macOS/Linux — free Gemini option
export AI_PROVIDER="gemini"
export GEMINI_API_KEY="your-key-here"
python3 main.py

# macOS/Linux — paid Claude option, if you'd rather use that
export AI_PROVIDER="claude"
export ANTHROPIC_API_KEY="your-key-here"
python3 main.py

# Windows PowerShell — free Gemini option
$env:AI_PROVIDER = "gemini"
$env:GEMINI_API_KEY = "your-key-here"
python main.py
```

You can also pick a specific model for either provider:

```bash
export GEMINI_MODEL="gemini-2.5-flash"          # default for Gemini
export ANTHROPIC_MODEL="claude-sonnet-4-6"      # default for Claude
```

### What happens if you're not connected

Nothing breaks. `ask_ai()` in `chatbot/ai_fallback.py` simply returns
`None` immediately, and the bot falls back to a plain, honest message
instead of guessing or crashing.

> 🔒 **Security note:** `.secrets.json` is excluded via `.gitignore` and
> should never be committed to source control or shared. Anyone with your
> key can use your account's API quota.

---

## Example Conversation

**Without connecting (rule-based only):**
```
================================================================
             Nova — Rule-Based AI Chatbot (Project 1)
        Type 'help' for what I can do, or 'exit' to quit.
================================================================
----------------------------------------------------------------
Nova can connect to a live AI backend for real-time answers to
anything outside her built-in rules (general knowledge,
open-ended questions, etc). A free option is available.
----------------------------------------------------------------
Connect to a live AI backend now? [y/N]: n
Okay — running rule-based only for now. You can connect anytime; just restart the bot.

You: hello
Nova: Hi! Great to see you.
You: what is the capital of pakistan?
Nova: That's outside my small rule set, and I don't have a live connection to answer it right now.
You: bye
Nova: Goodbye! Have a great day.
```

**After connecting once (free Gemini option):**
```
Connect to a live AI backend now? [y/N]: y

Which AI backend would you like to connect?
  [1] Google Gemini  — FREE, no credit card required  (get a key: https://aistudio.google.com/apikey)
  [2] Anthropic Claude — paid API, requires billing    (get a key: https://console.anthropic.com)
Choose [1/2] (default 1 - free): 1
Paste your Gemini API key (input hidden, get one free at https://aistudio.google.com/apikey):
Connected to Gemini (free)! Saved locally to .secrets.json so you won't be asked again on this machine.

You: hello
Nova: Hi! Great to see you.
You: what is the capital of pakistan?
Nova: [AI mode] The capital of Pakistan is Islamabad.
You: who painted the mona lisa
Nova: [AI mode] The Mona Lisa was painted by Leonardo da Vinci in the early 1500s.
You: bye
Nova: Goodbye! Have a great day.
```

---

## Available Commands / Intents

Nova's rule layer covers personality and utility only (type `help` in the
bot to see this generated live):

| Category | Examples |
|---|---|
| **Chat** | `hi`, `hello`, `how are you`, `thanks`, `tell me a joke`, `good bot` |
| **About** | `who are you`, `who made you`, `are you claude`, `are you conscious`, `what is your purpose` |
| **Utility** | `what time is it`, `what's the date`, `ai status`, `help` |
| **Exit** | `exit`, `quit`, `bye`, `goodbye`, `stop`, `q` |

**Everything else** — real knowledge, open-ended questions, anything you'd
normally ask an AI assistant — is answered live once you're
[connected](#connecting-to-a-free-ai-backend-one-time-setup). If you're
not connected, you'll get an honest fallback message instead of a crash
or a wrong guess.

---

## Running the Tests

The project ships with 39 automated unit tests covering the sanitizer, the
rule layer, the hybrid response engine, both AI providers (Gemini and
Claude), the setup wizard, and the exit-command set. No real API key or
network access is needed — every provider call is mocked.

```bash
# From the project root:
python -m unittest discover tests -v
```

Expected output ends with:

```
----------------------------------------------------------------------
Ran 39 tests in 0.0XXs

OK
```

---

## How to Extend the Bot

The whole point of a dictionary-based design is that **adding a new rule
never requires touching the matching logic** — you only edit data in
`chatbot/knowledge_base.py`. Remember: only add things here that should
genuinely be fixed and predictable. Real knowledge belongs to the live
AI connection, not a hand-typed list.

### 1. Add a brand-new intent (static response)

```python
"sports": {
    "category": "chat",   # groups it under the right heading in 'help'
    "patterns": ["football", "cricket", "favorite sport"],
    "responses": [
        "I'm more of a chess-engine kind of bot, but I respect the hustle!",
    ],
},
```

### 2. Add an intent with a dynamic (computed) response

```python
def _flip_coin() -> str:
    import random
    return random.choice(["Heads!", "Tails!"])

# inside KNOWLEDGE_BASE:
"coin_flip": {
    "category": "chat",
    "patterns": ["flip a coin", "heads or tails"],
    "handler": _flip_coin,
},
```

### 3. Add a new exit word

Edit `chatbot/config.py`:

```python
EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", "see you", "stop", "q", "later"}
```

### 4. Give the bot a different personality

Change `BOT_NAME` in `chatbot/config.py`, and rewrite the response strings
in `knowledge_base.py` to match a new voice (formal, sarcastic, cheerful,
pirate-speak — anything!).

### 5. Customize the system prompt the AI backend sees

Edit `SYSTEM_PROMPT` in `chatbot/ai_fallback.py` to change Nova's tone or
constraints for live-answered questions (e.g. "always answer in two
sentences or fewer", or "always mention a relevant follow-up question").
This prompt is shared by both providers.

### 6. Add a third provider

`ai_fallback.py` is structured so each provider is a self-contained
`_ask_<provider>()` function, dispatched from `ask_ai()` based on
`get_active_provider()`. Adding a new one (e.g. a local model via Ollama)
means writing one more function in that pattern and adding it to the
dispatch — no other file needs to change.

---

## Troubleshooting

| Problem | Likely Cause | Fix |
|---|---|---|
| `'python' is not recognized as an internal or external command` | Python isn't installed or isn't on PATH | Install Python from python.org and check "Add Python to PATH" during setup, or use `python3` instead |
| `ModuleNotFoundError: No module named 'chatbot'` | Running `main.py` from the wrong directory | `cd` into the project root folder (the one containing `main.py`) before running |
| Nothing happens when you press Enter on an empty line | Expected behavior — empty input is ignored | Type an actual message; the bot will nudge you to try again |
| Asked to connect every time, even after saying yes once | `.secrets.json` couldn't be written (e.g. read-only folder) | Run the project from a folder you have write access to, like your home directory or Desktop |
| `ai status` says "not connected" even though you set up once | You're running from a different copy/folder of the project | `.secrets.json` lives next to `main.py` — make sure you're launching the same folder you connected from |
| Connected but you still get a fallback message | The live call failed (bad/expired key, no internet, rate limit hit) | For Gemini, check your key at aistudio.google.com; for Claude, check console.anthropic.com and your billing. The bot never crashes on this, it just falls back gracefully |
| Gemini gives a rate-limit-style fallback after many questions in a row | The free tier has a requests-per-minute cap | Wait a few seconds between questions, or switch to Claude (paid) if you need higher throughput |
| `UnicodeEncodeError` printing emoji on old Windows cmd | Some legacy Windows terminals default to a non-UTF-8 code page | Run `chcp 65001` once in cmd before launching, or use Windows Terminal/PowerShell |

---

## Design Notes

- **Separation of concerns**: sanitization, the rule layer, the live AI
  connection, the setup wizard, and the conversation loop each live in
  their own module.
- **Data/logic separation**: `knowledge_base.py` is pure data; `engine.py`
  is pure logic. You can add new rules without ever opening `engine.py`.
- **Provider abstraction**: `engine.py` and `knowledge_base.py` never
  know or care whether Gemini or Claude is answering — they just call
  `ask_ai()` and `get_active_provider()`. Swapping or adding providers
  only touches `ai_fallback.py` and `setup_wizard.py`.
- **Fail-open onboarding, fail-safe runtime**: connecting is one easy
  prompt, but every downstream call (`ask_ai()`) is wrapped so a missing
  key, dead network, or bad response degrades to a plain message instead
  of a crash.
- **Defensive programming**: `None`/empty input never crashes the bot;
  `Ctrl+C` and EOF are caught and exit gracefully.
- **Observability**: every turn is logged with a timestamp, giving you a
  durable feedback loop for debugging or reviewing past conversations.
- **Testability**: business logic (`sanitizer`, `engine`, `ai_fallback`,
  `setup_wizard`) is fully decoupled from terminal I/O, so all of it is
  unit tested without a real network call or a real interactive terminal.

---

## License

This project is free to use, modify, and learn from for educational
purposes.

---

**Remember:** the goal isn't just to run this once — open
`chatbot/knowledge_base.py` and add a personality touch of your own, or
open `chatbot/ai_fallback.py` and tune how the AI backend responds.
That's how mastery happens.
