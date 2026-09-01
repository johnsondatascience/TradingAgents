"""Tests for TRADINGAGENTS_* env-var overlay onto DEFAULT_CONFIG."""

from __future__ import annotations

import importlib
import os

import pytest

import tradingagents.default_config as default_config_module


@pytest.fixture(autouse=True)
def _restore_default_config():
    """Put DEFAULT_CONFIG back after each test in this module.

    Every test here reloads default_config to re-evaluate DEFAULT_CONFIG, and
    a reload is permanent in a way the env changes are not: monkeypatch
    restores the environment at teardown, but nothing puts the module back, so
    the last test's env-derived values stay in DEFAULT_CONFIG for the rest of
    the session. Any later fixture doing set_config(deepcopy(DEFAULT_CONFIG))
    then inherits them.

    That is not hypothetical. Without this, running
    test_gexter_candidate_keys_coerce_from_env before
    tests/test_market_structure_state.py leaves gexter_candidates False, and
    the market-structure node is asserted to request candidates it no longer
    asks for -- a failure in a different file, about a key that test never
    mentions. The full suite happened to order around it; a -k selection did
    not.

    The env is cleared explicitly rather than leaning on monkeypatch having
    already unwound, so the reload sees a clean environment whichever order
    the two finalizers run in.
    """
    saved = {k: os.environ.get(k) for k in default_config_module._ENV_OVERRIDES}
    yield
    for key in default_config_module._ENV_OVERRIDES:
        os.environ.pop(key, None)
    try:
        importlib.reload(default_config_module)
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def test_no_env_uses_built_in_defaults(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gpt-5.6"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gpt-5.6-luna"
    assert dc.DEFAULT_CONFIG["backend_url"] is None
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is False
    assert dc.DEFAULT_CONFIG["gexter_timeout"] == 120


def test_string_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="google",
        TRADINGAGENTS_DEEP_THINK_LLM="gemini-3-pro-preview",
        TRADINGAGENTS_QUICK_THINK_LLM="gemini-3-flash-preview",
        TRADINGAGENTS_LLM_BACKEND_URL="https://example.invalid/v1",
        TRADINGAGENTS_OUTPUT_LANGUAGE="Chinese",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "google"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gemini-3-pro-preview"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gemini-3-flash-preview"
    assert dc.DEFAULT_CONFIG["backend_url"] == "https://example.invalid/v1"
    assert dc.DEFAULT_CONFIG["output_language"] == "Chinese"


def test_int_coercion(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="3",
        TRADINGAGENTS_MAX_RISK_ROUNDS="2",
    )
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 3
    assert isinstance(dc.DEFAULT_CONFIG["max_debate_rounds"], int)
    assert dc.DEFAULT_CONFIG["max_risk_discuss_rounds"] == 2
    assert isinstance(dc.DEFAULT_CONFIG["max_risk_discuss_rounds"], int)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ],
)
def test_bool_coercion(monkeypatch, raw, expected):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_CHECKPOINT_ENABLED=raw)
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is expected


def test_reasoning_thinking_overrides(monkeypatch):
    """The provider reasoning/thinking knobs are env-configurable (non-interactive runs)."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_OPENAI_REASONING_EFFORT="high",
        TRADINGAGENTS_GOOGLE_THINKING_LEVEL="minimal",
        TRADINGAGENTS_ANTHROPIC_EFFORT="low",
    )
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] == "high"
    assert dc.DEFAULT_CONFIG["google_thinking_level"] == "minimal"
    assert dc.DEFAULT_CONFIG["anthropic_effort"] == "low"


def test_reasoning_effort_defaults_to_none(monkeypatch):
    """Unset reasoning/thinking knobs stay None so each provider uses its own default."""
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] is None
    assert dc.DEFAULT_CONFIG["google_thinking_level"] is None
    assert dc.DEFAULT_CONFIG["anthropic_effort"] is None


def test_empty_env_value_is_passthrough(monkeypatch):
    """Empty TRADINGAGENTS_* values must not clobber the built-in default."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="",
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1


def test_gexter_timeout_is_coerced_like_every_other_int(monkeypatch):
    """Fork-local key, same mechanism: a plain string in .env still works."""
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_GEXTER_TIMEOUT="45")
    assert dc.DEFAULT_CONFIG["gexter_timeout"] == 45
    assert isinstance(dc.DEFAULT_CONFIG["gexter_timeout"], int)


def test_invalid_gexter_timeout_names_the_env_var(monkeypatch):
    """A bare int() would raise "invalid literal for int()" without naming the
    variable, and would fail the import of tradingagents.default_config itself.
    """
    monkeypatch.setenv("TRADINGAGENTS_GEXTER_TIMEOUT", "abc")
    with pytest.raises(ValueError, match="TRADINGAGENTS_GEXTER_TIMEOUT"):
        importlib.reload(default_config_module)
    monkeypatch.delenv("TRADINGAGENTS_GEXTER_TIMEOUT", raising=False)
    importlib.reload(default_config_module)


def test_invalid_int_raises(monkeypatch):
    """Garbage int values should surface a ValueError at import, not silently misconfigure."""
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "not-a-number")
    with pytest.raises(ValueError, match="TRADINGAGENTS_MAX_DEBATE_ROUNDS"):
        importlib.reload(default_config_module)
    # Restore module state for subsequent tests in this process
    monkeypatch.delenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", raising=False)
    importlib.reload(default_config_module)


@pytest.mark.parametrize("bad", ["treu", "flase", "maybe", "2", "enabled"])
def test_invalid_bool_raises(monkeypatch, bad):
    """A misspelled boolean must fail loudly (like ints) instead of silently False."""
    monkeypatch.setenv("TRADINGAGENTS_CHECKPOINT_ENABLED", bad)
    with pytest.raises(ValueError, match="TRADINGAGENTS_CHECKPOINT_ENABLED"):
        importlib.reload(default_config_module)
    monkeypatch.delenv("TRADINGAGENTS_CHECKPOINT_ENABLED", raising=False)
    importlib.reload(default_config_module)


def test_unknown_env_var_is_ignored(monkeypatch):
    """Env vars outside _ENV_OVERRIDES must not bleed into DEFAULT_CONFIG."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_NONEXISTENT_KEY="oops",
    )
    assert "nonexistent_key" not in dc.DEFAULT_CONFIG


def test_gexter_candidate_keys_have_defaults():
    from tradingagents.default_config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["gexter_candidates"] is True
    assert DEFAULT_CONFIG["gexter_dte_max"] == 2
    assert DEFAULT_CONFIG["gexter_cutoff"] is None
    assert DEFAULT_CONFIG["gexter_max_quote_age_seconds"] == 1800


def test_gexter_candidate_keys_coerce_from_env(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_GEXTER_CANDIDATES", "false")
    monkeypatch.setenv("TRADINGAGENTS_GEXTER_DTE_MAX", "1")
    monkeypatch.setenv("TRADINGAGENTS_GEXTER_MAX_QUOTE_AGE", "600")
    monkeypatch.setenv("TRADINGAGENTS_GEXTER_CUTOFF", "11:00")
    import importlib
    from tradingagents import default_config
    importlib.reload(default_config)
    cfg = default_config.DEFAULT_CONFIG
    assert cfg["gexter_candidates"] is False
    assert cfg["gexter_dte_max"] == 1
    assert cfg["gexter_max_quote_age_seconds"] == 600
    assert cfg["gexter_cutoff"] == "11:00"
