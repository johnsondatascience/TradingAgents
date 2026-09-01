import copy

import pytest
from langchain_core.runnables import RunnableLambda

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


class _StubResult:
    """Stands in for an AIMessage: a tool call keeps the graph looping."""

    def __init__(self, tool_calls):
        self.tool_calls = tool_calls
        self.content = "" if tool_calls else "done"


class _ScriptedLLM:
    """Returns each scripted result in turn, recording nothing else.

    bind_tools has to hand back a Runnable: the node composes it into a chain
    with the prompt, so a bare object fails before the node body is reached.
    """

    def __init__(self, *results):
        self._results = list(results)

    def bind_tools(self, tools):
        return RunnableLambda(lambda _: self._results.pop(0))


_DOCUMENT = {"schema_version": 1, "symbols": {"SPX": {"regime": "compression"}}}


def _counting_fetch(monkeypatch, *returns):
    """Patch the node's document fetch and count how often it is reached."""
    calls = {"n": 0}

    def fake_fetch(symbol, **kwargs):
        calls["n"] += 1
        value = returns[min(calls["n"] - 1, len(returns) - 1)]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(market_analyst, "fetch_document", fake_fetch)
    monkeypatch.setattr(market_analyst, "gexter_configured", lambda: True)
    return calls


def _state(**extra):
    state = {
        "trade_date": "2026-08-30",
        "company_of_interest": "^GSPC",
        "asset_type": "stock",
        "instrument_context": "Analyzing ^GSPC.",
        "messages": [],
    }
    state.update(extra)
    return state


def test_the_document_is_fetched_once_per_run_not_once_per_tool_round_trip(
        monkeypatch):
    """The tools node edges back to the analyst, so the node body re-runs.

    Each re-entry was spawning GEXter's CLI again -- three or four subprocess
    launches for one report, all of them redundant, because the document is a
    property of the run rather than of the tool call that preceded it.
    """
    calls = _counting_fetch(monkeypatch, _DOCUMENT)
    node = market_analyst.create_market_analyst(
        _ScriptedLLM(_StubResult([{"name": "get_stock_data", "args": {}, "id": "1"}]),
                     _StubResult([])))
    state = _state()
    first = node(state)
    # What the graph does after the tools node: same state, one more message.
    second = node(_state(market_structure=first["market_structure"]))
    assert calls["n"] == 1
    assert second["market_structure"] == _DOCUMENT


def test_a_later_fetch_failure_never_clobbers_a_document_already_in_state(
        monkeypatch):
    """market_structure has plain-overwrite semantics.

    A timeout on the second spawn returned None, and None replaced a perfectly
    good document -- so the Trader and the Portfolio Manager saw no candidates
    on a run that had produced them.
    """
    _counting_fetch(monkeypatch, _DOCUMENT, OSError("timed out"))
    node = market_analyst.create_market_analyst(
        _ScriptedLLM(_StubResult([{"name": "get_stock_data", "args": {}, "id": "1"}]),
                     _StubResult([])))
    first = node(_state())
    assert first["market_structure"] == _DOCUMENT
    second = node(_state(market_structure=first["market_structure"]))
    assert second["market_structure"] == _DOCUMENT


def test_the_node_records_the_run_date_for_the_llm_facing_tool(monkeypatch):
    """The tool cannot read state, so the node has to leave the date where
    the tool will look. Without this the model's own call to
    get_market_structure is the one unbounded path in an otherwise
    point-in-time run.
    """
    from tradingagents.dataflows.config import get_config

    _counting_fetch(monkeypatch, _DOCUMENT)
    market_analyst.fetch_market_structure(_state(trade_date="2026-06-02"))
    assert get_config()["gexter_trade_date"] == "2026-06-02"
