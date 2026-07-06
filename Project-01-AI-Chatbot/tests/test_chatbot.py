"""
test_chatbot.py — full test suite (44 tests).

Run with:   python -m unittest discover tests -v
No real API key or network needed — every provider call is mocked.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chatbot.sanitizer    import sanitize_input
from chatbot.engine       import ResponseEngine
from chatbot.config       import EXIT_COMMANDS
from chatbot.knowledge_base import KNOWLEDGE_BASE
from chatbot              import ai_fallback
from chatbot              import setup_wizard


# ------------------------------------------------------------------
# Helpers — build mock HTTP responses shaped like each provider
# ------------------------------------------------------------------
def _gemini_ok(text):
    body = json.dumps(
        {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    ).encode()
    m = mock.MagicMock()
    m.read.return_value = body
    m.__enter__.return_value = m
    return m

def _claude_ok(text):
    body = json.dumps(
        {"content": [{"type": "text", "text": text}]}
    ).encode()
    m = mock.MagicMock()
    m.read.return_value = body
    m.__enter__.return_value = m
    return m

def _http_error(code):
    e = urllib_error_HTTPError(code)
    return e

import urllib.error as _ue

class _FakeHTTPError(_ue.HTTPError):
    def __init__(self, code):
        super().__init__("http://x", code, "err", {}, None)
    def read(self):
        return b'{"error":"bad key"}'


# ------------------------------------------------------------------
# Sanitizer
# ------------------------------------------------------------------
class TestSanitizer(unittest.TestCase):
    def test_lowercases_and_strips(self):
        self.assertEqual(sanitize_input("  HELLO  "), "hello")

    def test_removes_punctuation(self):
        self.assertEqual(sanitize_input("Hello!!!"), "hello")

    def test_keeps_apostrophes(self):
        self.assertEqual(sanitize_input("What's up?"), "what's up")

    def test_collapses_whitespace(self):
        self.assertEqual(sanitize_input("hi    there"), "hi there")

    def test_empty_string(self):
        self.assertEqual(sanitize_input(""), "")

    def test_none_input(self):
        self.assertEqual(sanitize_input(None), "")


# ------------------------------------------------------------------
# Knowledge base structure
# ------------------------------------------------------------------
class TestKnowledgeBase(unittest.TestCase):
    def test_has_at_least_five_intents(self):
        self.assertGreaterEqual(len(KNOWLEDGE_BASE), 5)

    def test_small_not_encyclopedia(self):
        self.assertLess(len(KNOWLEDGE_BASE), 25)

    def test_every_intent_has_patterns(self):
        for k, v in KNOWLEDGE_BASE.items():
            self.assertIn("patterns", v, f"{k} missing patterns")
            self.assertGreater(len(v["patterns"]), 0)

    def test_every_intent_has_responses_or_handler(self):
        for k, v in KNOWLEDGE_BASE.items():
            self.assertTrue("responses" in v or "handler" in v, f"{k} has neither")

    def test_every_intent_has_category(self):
        for k, v in KNOWLEDGE_BASE.items():
            self.assertIn("category", v, f"{k} missing category")


# ------------------------------------------------------------------
# Response engine
# ------------------------------------------------------------------
class TestResponseEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ResponseEngine()

    def test_exact_match(self):
        r = self.engine.get_response("hi")
        self.assertIsInstance(r, str)
        self.assertTrue(len(r) > 0)

    def test_fuzzy_match(self):
        r = self.engine.get_response("hello there my friend")
        self.assertIsInstance(r, str)
        self.assertTrue(len(r) > 0)

    def test_empty_input_no_crash(self):
        r = self.engine.get_response("")
        self.assertIsInstance(r, str)

    def test_time_handler(self):
        r = self.engine.get_response("what time is it")
        self.assertIn("time", r.lower())

    def test_no_connection_falls_back(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            r = self.engine.get_response("what is the capital of pakistan")
        self.assertIsInstance(r, str)
        self.assertTrue(len(r) > 0)

    def test_gemini_success_shows_answer(self):
        mock_resp = _gemini_ok("Islamabad is the capital of Pakistan.")
        env = {"AI_PROVIDER": "gemini", "GEMINI_API_KEY": "AIzaFakeKey1234"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", return_value=mock_resp):
                r = self.engine.get_response(
                    "what is the capital of pakistan",
                    "What is the capital of Pakistan?"
                )
        self.assertIn("Islamabad", r)
        self.assertIn("[AI mode]", r)

    def test_invalid_claude_key_shows_error_not_silent_fallback(self):
        """If the key is wrong format, the user sees WHY, not a vague message."""
        env = {"AI_PROVIDER": "claude", "ANTHROPIC_API_KEY": "AQ.BadKey123"}
        with mock.patch.dict(os.environ, env, clear=True):
            r = self.engine.get_response("where is paris", "where is paris")
        # Should contain the key-format warning, not a silent "not connected"
        self.assertIn("[AI mode]", r)
        self.assertIn("sk-ant-", r)   # the hint about correct format


# ------------------------------------------------------------------
# AI fallback — provider selection
# ------------------------------------------------------------------
class TestAIFallbackProviderSelection(unittest.TestCase):
    def test_no_key_returns_none(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ai_fallback.ask_ai("anything"))
            self.assertFalse(ai_fallback.is_ai_mode_enabled())

    def test_gemini_key_selects_gemini(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFake"}, clear=True):
            self.assertEqual(ai_fallback.get_active_provider(), "gemini")

    def test_claude_key_selects_claude(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-fake"}, clear=True):
            self.assertEqual(ai_fallback.get_active_provider(), "claude")

    def test_explicit_provider_breaks_tie(self):
        env = {"GEMINI_API_KEY": "AIzaG", "ANTHROPIC_API_KEY": "sk-ant-C",
               "AI_PROVIDER": "claude"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(ai_fallback.get_active_provider(), "claude")

    def test_gemini_preferred_when_both_present(self):
        env = {"GEMINI_API_KEY": "AIzaG", "ANTHROPIC_API_KEY": "sk-ant-C"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(ai_fallback.get_active_provider(), "gemini")


# ------------------------------------------------------------------
# Gemini backend
# ------------------------------------------------------------------
class TestGeminiBackend(unittest.TestCase):
    def test_success(self):
        env = {"GEMINI_API_KEY": "AIzaTestKey"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", return_value=_gemini_ok("Tokyo.")):
                self.assertEqual(ai_fallback.ask_ai("capital of japan"), "Tokyo.")

    def test_network_error_returns_string(self):
        env = {"GEMINI_API_KEY": "AIzaTestKey"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
                r = ai_fallback.ask_ai("anything")
        # Must be a non-empty string (error message), NOT None
        self.assertIsNotNone(r)
        self.assertIsInstance(r, str)
        self.assertTrue(len(r) > 0)

    def test_http_error_returns_string(self):
        env = {"GEMINI_API_KEY": "AIzaTestKey"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", side_effect=_FakeHTTPError(403)):
                r = ai_fallback.ask_ai("anything")
        self.assertIsNotNone(r)
        self.assertIn("403", r)

    def test_bad_json_returns_string(self):
        m = mock.MagicMock()
        m.read.return_value = b"not json"
        m.__enter__.return_value = m
        env = {"GEMINI_API_KEY": "AIzaTestKey"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", return_value=m):
                r = ai_fallback.ask_ai("anything")
        self.assertIsNotNone(r)


# ------------------------------------------------------------------
# Claude backend
# ------------------------------------------------------------------
class TestClaudeBackend(unittest.TestCase):
    def test_success(self):
        env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", return_value=_claude_ok("Tokyo.")):
                self.assertEqual(ai_fallback.ask_ai("capital of japan"), "Tokyo.")

    def test_invalid_key_format_returns_clear_error(self):
        """Key not starting with sk-ant- returns a helpful error, not None."""
        env = {"ANTHROPIC_API_KEY": "AQ.BadFormatKey"}
        with mock.patch.dict(os.environ, env, clear=True):
            r = ai_fallback.ask_ai("anything")
        self.assertIsNotNone(r)
        self.assertIn("sk-ant-", r)   # the format hint

    def test_401_returns_clear_error(self):
        env = {"ANTHROPIC_API_KEY": "sk-ant-wrongkey"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", side_effect=_FakeHTTPError(401)):
                r = ai_fallback.ask_ai("anything")
        self.assertIsNotNone(r)
        self.assertIn("401", r)

    def test_network_error_returns_string(self):
        env = {"ANTHROPIC_API_KEY": "sk-ant-test"}
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
                r = ai_fallback.ask_ai("anything")
        self.assertIsNotNone(r)
        self.assertIsInstance(r, str)
        self.assertTrue(len(r) > 0)


# ------------------------------------------------------------------
# Setup wizard
# ------------------------------------------------------------------
class TestSetupWizard(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._patched = os.path.join(self._tmpdir.name, ".secrets.json")
        self._pp = mock.patch.object(setup_wizard, "SECRETS_PATH", self._patched)
        self._pp.start()

    def tearDown(self):
        self._pp.stop()
        self._tmpdir.cleanup()

    # _clean_key
    def test_clean_key_removes_ctrl_v_and_deduplicates(self):
        real = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
        broken = "\x16" + real + "\x16" + real
        self.assertEqual(setup_wizard._clean_key(broken), real)

    def test_clean_key_normal_unchanged(self):
        k = "AIzaSyTestKey12345"
        self.assertEqual(setup_wizard._clean_key(k), k)

    # _mask
    def test_mask_shows_start_and_end(self):
        k = "AIzaSyABCDEFGHIJKLMNOP"
        m = setup_wizard._mask(k)
        self.assertTrue(m.startswith("AIza"))
        self.assertIn("...", m)

    # _validate_key_format
    def test_validate_gemini_good(self):
        self.assertIsNone(setup_wizard._validate_key_format("gemini", "AIzaABC"))

    def test_validate_gemini_bad(self):
        w = setup_wizard._validate_key_format("gemini", "AQ.wrongkey")
        self.assertIsNotNone(w)
        self.assertIn("AIza", w)

    def test_validate_claude_good(self):
        self.assertIsNone(setup_wizard._validate_key_format("claude", "sk-ant-abc"))

    def test_validate_claude_bad(self):
        w = setup_wizard._validate_key_format("claude", "AQ.wrongkey")
        self.assertIsNotNone(w)
        self.assertIn("sk-ant-", w)

    # persistence
    def test_save_and_load_round_trip(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaKey123",
                                           "AI_PROVIDER": "gemini"}, clear=True):
            self.assertTrue(setup_wizard._save_key("gemini"))
        with mock.patch.dict(os.environ, {}, clear=True):
            setup_wizard._load_saved_key()
            self.assertEqual(os.environ.get("GEMINI_API_KEY"), "AIzaKey123")

    def test_real_env_var_not_overridden(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFromFile",
                                           "AI_PROVIDER": "gemini"}, clear=True):
            setup_wizard._save_key("gemini")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaFromEnv"}, clear=True):
            setup_wizard._load_saved_key()
            self.assertEqual(os.environ.get("GEMINI_API_KEY"), "AIzaFromEnv")

    # wizard flow
    def test_silent_when_already_connected(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSet"}, clear=True):
            with mock.patch("builtins.input") as mi:
                setup_wizard.maybe_run_setup_wizard()
                mi.assert_not_called()

    def test_decline_does_not_save(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.input", return_value="n"):
                setup_wizard.maybe_run_setup_wizard()
        self.assertFalse(os.path.exists(self._patched))

    def test_gemini_connect_saves_correctly(self):
        # Connect now=y, provider=1(gemini), key, looks-right=y
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.input",
                            side_effect=["y", "1", "AIzaRealKey999", "y"]):
                setup_wizard.maybe_run_setup_wizard()
            self.assertEqual(os.environ.get("GEMINI_API_KEY"), "AIzaRealKey999")
            self.assertEqual(os.environ.get("AI_PROVIDER"), "gemini")
        self.assertTrue(os.path.exists(self._patched))

    def test_claude_connect_saves_correctly(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.input",
                            side_effect=["y", "2", "sk-ant-realkey", "y"]):
                setup_wizard.maybe_run_setup_wizard()
            self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), "sk-ant-realkey")
            self.assertEqual(os.environ.get("AI_PROVIDER"), "claude")

    def test_bad_format_key_shows_warning_and_can_retry(self):
        # Warn on bad format, user says save-anyway, key is accepted
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.input",
                            side_effect=["y", "2", "AQ.BadKey", "y"]):
                # "y" at "Save anyway?" prompt
                setup_wizard.maybe_run_setup_wizard()
            # Key saved despite warning (user chose to proceed)
            self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), "AQ.BadKey")

    def test_blank_provider_defaults_to_gemini(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with mock.patch("builtins.input",
                            side_effect=["y", "", "AIzaDefaultKey", "y"]):
                setup_wizard.maybe_run_setup_wizard()
            self.assertEqual(os.environ.get("AI_PROVIDER"), "gemini")


# ------------------------------------------------------------------
# Exit commands
# ------------------------------------------------------------------
class TestExitCommands(unittest.TestCase):
    def test_common_exit_words(self):
        for w in ["exit", "quit", "bye"]:
            self.assertIn(w, EXIT_COMMANDS)


if __name__ == "__main__":
    unittest.main()
