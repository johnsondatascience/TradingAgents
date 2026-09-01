"""AgentState plumbing for the GEXter market-structure document.

The node fetches before the LLM runs, so the model cannot choose the date and
cannot skip the fetch, and the Trader later reads the same object rather than a
paraphrase of it.
"""
import pytest

from tradingagents.agents.analysts.market_analyst import fetch_market_structure
from tradingagents.agents.utils.agent_states import AgentState

_MODULE = "tradingagents.agents.analysts.market_analyst"


def test_agent_state_declares_market_structure():
    assert "market_structure" in AgentState.__annotations__


def _state(ticker="^GSPC", trade_date="2026-08-31"):
    return {"company_of_interest": ticker, "trade_date": trade_date, "messages": []}


def test_no_fetch_when_gexter_is_unconfigured(monkeypatch):
    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: False)
    assert fetch_market_structure(_state()) is None


def test_no_fetch_for_a_ticker_outside_the_sp_complex(monkeypatch):
    called = []
    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: True)
    monkeypatch.setattr(f"{_MODULE}.fetch_document", lambda **kw: called.append(kw))
    assert fetch_market_structure(_state(ticker="NVDA")) is None
    assert called == []          # no subprocess for an equity run


def test_fetch_passes_the_states_trade_date(monkeypatch):
    seen = {}
    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: True)
    monkeypatch.setattr(
        f"{_MODULE}.fetch_document",
        lambda **kw: seen.update(kw) or {"symbols": {"SPX": {"status": "ok"}}})
    fetch_market_structure(_state(trade_date="2026-08-27"))
    assert seen["trade_date"] == "2026-08-27"
    assert seen["symbol"] == "SPX"
    assert seen["candidates"] is True
    assert seen["dte_max"] == 2


def test_a_vendor_failure_yields_none_rather_than_aborting(monkeypatch):
    from tradingagents.dataflows.gexter import GexterUnavailableError

    def boom(**kw):
        raise GexterUnavailableError("postgres down")

    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: True)
    monkeypatch.setattr(f"{_MODULE}.fetch_document", boom)
    # Market structure is optional context. A GEXter outage degrades the run;
    # it must never abort the graph.
    assert fetch_market_structure(_state()) is None


def test_the_freshness_gate_is_applied_to_the_fetched_entry(monkeypatch):
    stale = {
        "schema_version": 1,
        "symbols": {"SPX": {
            "status": "ok", "spot_context": {"session_em_points": 47.1},
            "candidates": [{"candidate_id": "x",
                            "quoted_asof": "2020-01-01T13:45:00-05:00"}],
            "candidates_suppressed_reason": None}},
    }
    monkeypatch.setattr(f"{_MODULE}.gexter_configured", lambda: True)
    monkeypatch.setattr(f"{_MODULE}.fetch_document", lambda **kw: stale)
    import datetime as _dt
    from zoneinfo import ZoneInfo
    # The market date, not the local one: this machine is not on ET, and the
    # gate decides "today" in market time by design.
    today_et = _dt.datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    doc = fetch_market_structure(
        {"company_of_interest": "^GSPC",
         "trade_date": today_et, "messages": []})
    entry = doc["symbols"]["SPX"]
    assert entry["candidates"] == []
    assert entry["spot_context"] is not None      # levels survive the gate


# --- Trader selects by id; the renderer resolves it --------------------------

from tradingagents.agents.schemas import (  # noqa: E402
    CandidateResponse, TraderProposal, find_candidate, render_trader_proposal,
)

_DOC = {"schema_version": 1, "symbols": {"SPX": {"candidates": [{
    "candidate_id": "SPX_20260831_iron_condor_6380_6450",
    "structure": "iron_condor", "expiry": "2026-08-31",
    "legs": [{"option_type": "put", "strike": 6380, "qty": -1},
             {"option_type": "put", "strike": 6355, "qty": 1},
             {"option_type": "call", "strike": 6450, "qty": -1},
             {"option_type": "call", "strike": 6475, "qty": 1}],
    "net_premium": 2.50, "premium_kind": "credit", "max_loss": 22.50,
    "quoted_asof": "2026-08-31T13:45:00-04:00",
}]}}}


def test_find_candidate_resolves_a_known_id():
    assert find_candidate(_DOC, "SPX_20260831_iron_condor_6380_6450")["net_premium"] == 2.50


def test_find_candidate_returns_none_for_an_unknown_id():
    assert find_candidate(_DOC, "SPX_20260831_iron_condor_1_2") is None
    assert find_candidate(None, "anything") is None
    assert find_candidate(_DOC, None) is None


def test_trader_proposal_defaults_leave_the_equity_shape_untouched():
    proposal = TraderProposal(action="Buy", reasoning="because")
    assert proposal.candidate_response is None
    assert proposal.candidate_id is None
    rendered = render_trader_proposal(proposal)
    assert "Options Structure" not in rendered
    assert rendered.endswith("FINAL TRANSACTION PROPOSAL: **BUY**")


def test_render_uses_the_documents_numbers_not_the_models():
    proposal = TraderProposal(
        action="Buy", reasoning="walls are in play",
        candidate_response=CandidateResponse.ACCEPT,
        candidate_id="SPX_20260831_iron_condor_6380_6450",
        candidate_reasoning="short call sits on the wall", contracts=2)
    rendered = render_trader_proposal(proposal, document=_DOC)
    assert "short 6380P" in rendered and "long 6475C" in rendered
    assert "2.5" in rendered and "22.5" in rendered
    assert "Contracts**: 2" in rendered
    assert rendered.endswith("FINAL TRANSACTION PROPOSAL: **BUY**")


def test_render_declares_a_decline_without_inventing_a_structure():
    proposal = TraderProposal(
        action="Hold", reasoning="conviction too low",
        candidate_response=CandidateResponse.DECLINE,
        candidate_id="SPX_20260831_iron_condor_6380_6450",
        candidate_reasoning="transition regime, 0.25 size")
    rendered = render_trader_proposal(proposal, document=_DOC)
    assert "Declined" in rendered
    assert "6380P" not in rendered


def test_an_unknown_id_renders_as_a_decline():
    # The model invented an id. Printing a structure it described would be
    # fabrication, so the only safe rendering is a decline.
    proposal = TraderProposal(
        action="Buy", reasoning="x",
        candidate_response=CandidateResponse.ACCEPT,
        candidate_id="SPX_hallucinated_9999")
    rendered = render_trader_proposal(proposal, document=_DOC)
    assert "Declined" in rendered or "unavailable" in rendered
    assert "6380" not in rendered


def test_candidate_response_has_no_modify_member():
    # A modify path would be a numeric output surface under another name.
    assert {m.value for m in CandidateResponse} == {"accept", "decline"}


# --- Portfolio Manager verdict, appended not interleaved ---------------------

from tradingagents.agents.schemas import (  # noqa: E402
    PortfolioDecision, render_pm_decision,
)

_CID = "SPX_20260831_iron_condor_6380_6450"


def _decision(**kw):
    base = dict(rating="Hold", executive_summary="summary text",
                investment_thesis="thesis text")
    base.update(kw)
    return PortfolioDecision(**base)


def test_pm_render_without_a_verdict_is_byte_identical_to_today():
    assert render_pm_decision(_decision()) == (
        "**Rating**: Hold\n\n**Executive Summary**: summary text"
        "\n\n**Investment Thesis**: thesis text")


def test_pm_render_appends_the_structure_after_the_parsed_headers():
    # The memory log, CLI display and report writers parse these three headers.
    rendered = render_pm_decision(
        _decision(structure_verdict="approve", contracts=2),
        document=_DOC, candidate_id=_CID)
    assert rendered.index("**Rating**") < rendered.index("**Executive Summary**")
    assert rendered.index("**Executive Summary**") < rendered.index("**Investment Thesis**")
    assert rendered.index("**Investment Thesis**") < rendered.index("**Options Structure**")
    assert "Contracts**: 2" in rendered


def test_pm_reject_prints_no_structure():
    rendered = render_pm_decision(
        _decision(structure_verdict="reject", structure_note="too wide"),
        document=_DOC, candidate_id=_CID)
    assert "Rejected" in rendered and "too wide" in rendered
    assert "6380" not in rendered


def test_pm_resize_changes_only_the_contract_count():
    rendered = render_pm_decision(
        _decision(structure_verdict="resize", contracts=1),
        document=_DOC, candidate_id=_CID)
    assert "Contracts**: 1" in rendered
    assert "2.5" in rendered            # premium is still the document's


def test_pm_unknown_id_approves_nothing():
    rendered = render_pm_decision(
        _decision(structure_verdict="approve"),
        document=_DOC, candidate_id="SPX_hallucinated_1")
    assert "unavailable" in rendered
    assert "6380" not in rendered


def test_pm_defaults_leave_the_equity_shape_untouched():
    assert _decision().structure_verdict is None
    assert "Options Structure" not in render_pm_decision(_decision())
