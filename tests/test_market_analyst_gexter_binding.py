import copy

import pytest

from tradingagents import default_config
from tradingagents.agents.analysts import market_analyst
from tradingagents.dataflows.config import set_config


class _RecordingLLM:
    """Captures the tools bound to it; never calls a model."""

    def __init__(self):
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self


def _bound_tool_names(monkeypatch, configured):
    monkeypatch.setattr(market_analyst, "gexter_configured", lambda: configured)
    llm = _RecordingLLM()
    node = market_analyst.create_market_analyst(llm)
    state = {
        "trade_date": "2026-08-30",
        "company_of_interest": "^GSPC",
        "asset_type": "stock",
        "instrument_context": "Analyzing ^GSPC.",
        "messages": [],
    }
    with pytest.raises(Exception):
        # The recording LLM has no invoke(); we only need bind_tools to have run.
        node(state)
    return [tool.name for tool in llm.bound]


@pytest.fixture(autouse=True)
def _base_config():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def test_tool_is_bound_when_gexter_is_configured(monkeypatch):
    assert "get_market_structure" in _bound_tool_names(monkeypatch, configured=True)


def test_tool_is_absent_when_gexter_is_not_configured(monkeypatch):
    names = _bound_tool_names(monkeypatch, configured=False)
    assert "get_market_structure" not in names
    assert names == ["get_stock_data", "get_indicators", "get_verified_market_snapshot"]


def test_system_message_mentions_market_structure_when_configured():
    assert "get_market_structure" in market_analyst._build_system_message(True)


def test_system_message_omits_market_structure_when_not_configured():
    assert "get_market_structure" not in market_analyst._build_system_message(False)


def test_configured_system_message_differs_from_unconfigured_only_by_paragraph():
    configured = market_analyst._build_system_message(True)
    unconfigured = market_analyst._build_system_message(False)

    # The two messages must differ by exactly one contiguous insertion (the
    # GEXter paragraph) and nothing else. Locate that insertion via the
    # common prefix/suffix of the two strings, then confirm that slicing it
    # out of the configured message reconstitutes the unconfigured message
    # exactly - proving the conditional branch injects the paragraph and
    # nothing more.
    shortest = min(len(configured), len(unconfigured))

    prefix_len = 0
    while prefix_len < shortest and configured[prefix_len] == unconfigured[prefix_len]:
        prefix_len += 1

    suffix_len = 0
    while (
        suffix_len < shortest - prefix_len
        and configured[-1 - suffix_len] == unconfigured[-1 - suffix_len]
    ):
        suffix_len += 1

    inserted = configured[prefix_len : len(configured) - suffix_len]
    assert "get_market_structure" in inserted
    assert configured[:prefix_len] + configured[len(configured) - suffix_len :] == unconfigured
