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
