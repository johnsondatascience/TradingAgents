"""The market analyst binds tools to its LLM; the market ToolNode executes them.

A tool bound to the LLM but missing from the ToolNode is answered by langgraph
with "... is not a valid tool", so the model reports the tool unavailable and
the feature silently never runs. This module guards that wiring three ways:

1. ``get_verified_market_snapshot`` is registered (a real past bug).
2. *Every* tool the analyst binds is registered — including the fork-local
   ``get_market_structure``, which the analyst binds only when GEXter is
   configured. A single-name assertion did not generalize; parity does.
3. The market ToolNode can actually execute a ``get_market_structure`` call end
   to end, degrading to a ``DATA_UNAVAILABLE`` sentinel when GEXter cannot
   answer. This is the graph path a configured run takes.
"""
import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from tradingagents.agents.analysts import market_analyst
from tradingagents.dataflows import interface
from tradingagents.dataflows.gexter import GexterUnavailableError
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _market_tool_node():
    # _create_tool_nodes does not use self -> call unbound (avoids building LLMs).
    return TradingAgentsGraph._create_tool_nodes(None)["market"]


def _registered_market_tools():
    return set(_market_tool_node().tools_by_name)


def _compiled_market_tool_graph():
    """The market ToolNode wired as a graph node, the way setup.py wires it.

    A ToolNode invoked bare has no langgraph Runtime in its config and raises
    before reaching any tool; compiling a one-node graph exercises the real
    execution path instead of hand-rolling langgraph internals.
    """
    graph = StateGraph(MessagesState)
    graph.add_node("tools", _market_tool_node())
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    return graph.compile()


class _RecordingLLM:
    """Captures the tools bound to it; never calls a model."""

    def __init__(self):
        self.bound = None

    def bind_tools(self, tools):
        self.bound = tools
        return self


def _analyst_bound_tools(monkeypatch, gexter_available):
    """The tool objects create_market_analyst binds, for a given GEXter state."""
    monkeypatch.setattr(market_analyst, "gexter_configured", lambda: gexter_available)
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
    return llm.bound


@pytest.mark.unit
def test_market_toolnode_can_execute_verified_snapshot():
    market_tools = _registered_market_tools()
    assert "get_verified_market_snapshot" in market_tools, (
        "get_verified_market_snapshot is bound to the market analyst but not "
        "registered in the market ToolNode, so the model's call fails."
    )
    # the other core market tools must remain too
    assert {"get_stock_data", "get_indicators"} <= market_tools


@pytest.mark.unit
def test_every_tool_the_market_analyst_binds_is_registered(monkeypatch):
    """Parity, not a name list: whatever the analyst binds must be executable."""
    bound = {tool.name for tool in _analyst_bound_tools(monkeypatch, True)}
    # Guard the guard: with GEXter "configured" the conditional branch must have
    # been taken, or this test would pass vacuously against the upstream three.
    assert "get_market_structure" in bound
    registered = _registered_market_tools()
    assert bound <= registered, (
        f"bound to the market analyst but absent from the market ToolNode: "
        f"{sorted(bound - registered)}. langgraph answers such a call with "
        f"'is not a valid tool' and the tool never runs."
    )


@pytest.mark.unit
def test_market_structure_is_registered_even_when_gexter_is_unconfigured(monkeypatch):
    """Registration is unconditional; only the analyst's binding is conditional.

    A ToolNode may hold a tool the model was never told about, so an upstream
    user without GEXter is unaffected — the tool is simply unreachable.
    """
    bound = {tool.name for tool in _analyst_bound_tools(monkeypatch, False)}
    assert "get_market_structure" not in bound
    assert "get_market_structure" in _registered_market_tools()


@pytest.mark.unit
def test_market_toolnode_executes_market_structure_and_degrades(monkeypatch):
    """The graph path, with GEXter configured but unable to answer.

    Patches the vendor entry rather than spawning a subprocess, so this needs no
    GEXter checkout and no Postgres. What it proves is that the tool call is
    routable at all: before the ToolNode registration it came back as
    "get_market_structure is not a valid tool".
    """
    def unavailable(*args, **kwargs):
        raise GexterUnavailableError("GEXter reported: connection refused")

    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_market_structure"], "gexter", unavailable
    )
    call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_market_structure",
                "args": {"ticker": "^GSPC"},
                "id": "call-market-structure-1",
                "type": "tool_call",
            }
        ],
    )
    result = _compiled_market_tool_graph().invoke({"messages": [call]})

    message = result["messages"][-1]
    assert message.type == "tool"
    assert message.name == "get_market_structure"
    assert message.content.startswith("DATA_UNAVAILABLE:"), message.content
    assert "not a valid tool" not in message.content
