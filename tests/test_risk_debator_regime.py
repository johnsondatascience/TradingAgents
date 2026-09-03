"""GEXter regime context for the three risk debators.

The debators have no tool access, so the regime and its position-size
multiplier can only reach them through the prompt. The block is advisory:
it is evidence the debate weighs, not a constraint it obeys.

The "renders nothing" cases carry the most weight — they are what keeps
every equity run, every non-S&P ticker, and every un-configured install
byte-identical to upstream.
"""

from types import SimpleNamespace

import pytest

from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.schemas import render_risk_context


def _view(regime="compression", bias="sell_premium", multiplier=1.0):
    return {
        "regime": regime,
        "strength": "moderate",
        "confidence": 0.72,
        "trade_bias": bias,
        "risk_adjustment": multiplier,
        "interpretation": "dealers long gamma, mean reversion favoured",
    }


def _document(stale=None, nowcast=None, divergence=False, status="ok"):
    entry = {"status": status, "regime_divergence": divergence}
    if status == "ok":
        entry["stale"] = stale if stale is not None else _view()
        entry["nowcast"] = nowcast
    return {"schema_version": 1, "trading_day": "2026-09-02",
            "symbols": {"SPX": entry}}


def test_no_document_renders_nothing():
    assert render_risk_context(None) == ""


def test_empty_document_renders_nothing():
    assert render_risk_context({}) == ""


def test_document_without_symbols_renders_nothing():
    assert render_risk_context({"schema_version": 1}) == ""


def test_symbol_without_usable_data_renders_nothing():
    assert render_risk_context(_document(status="no_data")) == ""


def test_regime_and_multiplier_reach_the_debators():
    out = render_risk_context(_document(stale=_view(multiplier=0.5)))
    assert "COMPRESSION" in out
    assert "0.5" in out
    assert "sell_premium" in out
    assert "dealers long gamma" in out


def test_context_is_framed_as_advisory_not_binding():
    out = render_risk_context(_document()).lower()
    # The debate weighs this; it does not obey it.
    assert "advisory" in out
    assert "not a directional forecast" in out


def test_context_names_the_symbol_it_measured():
    assert "SPX" in render_risk_context(_document())


def test_nowcast_view_wins_over_stale():
    out = render_risk_context(_document(
        stale=_view(regime="compression"),
        nowcast=_view(regime="expansion")))
    assert "EXPANSION" in out
    assert "COMPRESSION" not in out


def test_falls_back_to_stale_when_no_model_produced_a_nowcast():
    out = render_risk_context(_document(
        stale=_view(regime="compression"), nowcast=None))
    assert "COMPRESSION" in out


def test_divergence_is_called_out_when_the_regime_has_shifted():
    out = render_risk_context(_document(divergence=True))
    assert "less settled" in out


def test_no_divergence_note_when_the_regime_is_stable():
    assert "less settled" not in render_risk_context(_document(divergence=False))


def test_missing_fields_degrade_instead_of_raising():
    bare = {"regime": None, "strength": None, "confidence": None,
            "trade_bias": None, "risk_adjustment": None, "interpretation": None}
    out = render_risk_context(_document(stale=bare))
    assert "UNKNOWN" in out
    assert "multiplier" not in out      # nothing to advise, so nothing claimed


def test_a_zero_multiplier_is_still_reported():
    # 0.0 is falsy but meaningful: it is GEXter saying "do not size into this".
    out = render_risk_context(_document(stale=_view(multiplier=0.0)))
    assert "multiplier: 0.0" in out


class _CapturingLLM:
    """Stands in for the chat model and keeps the prompt it was handed."""

    def __init__(self):
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return SimpleNamespace(content="argument text")


def _debate_state(market_structure=None):
    return {
        "company_of_interest": "^GSPC",
        "market_report": "m",
        "sentiment_report": "s",
        "news_report": "n",
        "fundamentals_report": "f",
        "trader_investment_plan": "plan",
        "market_structure": market_structure,
        "risk_debate_state": {"history": "", "count": 0},
    }


FACTORIES = [create_aggressive_debator,
             create_conservative_debator,
             create_neutral_debator]


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.__name__)
def test_every_debator_receives_the_regime_context(factory):
    llm = _CapturingLLM()
    factory(llm)(_debate_state(_document()))
    assert "GEXTER MARKET-STRUCTURE CONTEXT" in llm.prompt
    assert "COMPRESSION" in llm.prompt


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.__name__)
def test_debator_prompt_is_untouched_without_gexter(factory):
    llm = _CapturingLLM()
    factory(llm)(_debate_state(None))
    assert "GEXTER" not in llm.prompt


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.__name__)
def test_debator_still_returns_its_debate_state(factory):
    # The suffix must not disturb the node's contract with the graph.
    out = factory(_CapturingLLM())(_debate_state(_document()))
    assert out["risk_debate_state"]["count"] == 1
