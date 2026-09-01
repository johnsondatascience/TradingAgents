"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import (
    CandidateResponse,
    TraderProposal,
    find_candidate,
    render_trader_proposal,
)
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def _candidate_prompt(document) -> str:
    """Present GEXter's priced candidates for an accept/decline verdict.

    Strikes and premiums are listed for the model to reason about but never
    to restate: it answers with an id, and the renderer resolves that id
    back to these same numbers. Declining is stated as acceptable because
    candidates are emitted in every regime, including ones whose own bias
    is no_trade.
    """
    candidates = []
    for entry in ((document or {}).get("symbols") or {}).values():
        candidates.extend((entry or {}).get("candidates") or [])
    if not candidates:
        return ""
    lines = [
        "",
        "",
        "GEXter has priced the following 0-2DTE structures. Respond to "
        "exactly one by setting candidate_response and copying its "
        "candidate_id verbatim. Do not restate or recompute its strikes "
        "or premiums. Declining is a legitimate answer - prefer it when "
        "conviction is low or the size multiplier is small.",
    ]
    for candidate in candidates:
        legs = ", ".join(
            f"{'short' if leg.get('qty', 0) < 0 else 'long'} "
            f"{leg.get('strike')}{'P' if leg.get('option_type') == 'put' else 'C'}"
            for leg in candidate.get("legs") or [])
        lines.append(
            f"- {candidate.get('candidate_id')}: "
            f"{candidate.get('structure')} ({legs}), net "
            f"{candidate.get('net_premium')} pts "
            f"{candidate.get('premium_kind')}, max loss "
            f"{candidate.get('max_loss')} pts, conviction "
            f"{candidate.get('conviction')}, size multiplier "
            f"{candidate.get('size_multiplier')}")
    return chr(10).join(lines)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]
        market_structure = state.get("market_structure")
        # The render callback is the only place the parsed proposal is
        # visible, so it is captured there rather than re-invoking the model.
        proposal_ref = {}
        # The research plan digests the debate but loses exact price structure;
        # give the Trader the technical market report so entry/stop levels are
        # grounded in real ATR / support-resistance / current price (#1167). The
        # report is empty when the user did not select the market analyst, so
        # only offer it (and the grounding instruction) when it has content.
        market_report = (state["market_report"] or "").strip()

        if market_report:
            grounding = (
                "Ground concrete price levels (entry, stop-loss, position sizing) in the technical "
                "market report's price structure -- current price, support/resistance, ATR, and "
                "volatility -- and use the research plan for direction and strategy. "
            )
            report_section = f"Technical Market Report:\n{market_report}\n\n"
        else:
            grounding = ""
            report_section = ""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    + grounding
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research team's investment plan for {company_name}. "
                    f"{instrument_context}\n\n"
                    f"{report_section}"
                    f"Proposed Investment Plan:\n{investment_plan}\n\n"
                    f"Make an informed, strategic trading decision."
                    # Appended after upstream's prose, never woven into it:
                    # the candidate block is a suffix the fork owns, so an
                    # upstream rewording of the plan text cannot silently drop it.
                    + _candidate_prompt(market_structure)
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            # The document is the authority on every number. The model supplies
            # an id and a verdict; this renderer supplies the arithmetic.
            lambda proposal: (
                proposal_ref.update(proposal=proposal)
                or render_trader_proposal(proposal, document=market_structure)),
            "Trader",
        )

        # Only an accepted, resolvable id travels on. A declined or invented
        # one must not reach the Portfolio Manager as something to approve.
        chosen = None
        proposal = proposal_ref.get("proposal")
        if (isinstance(proposal, TraderProposal)
                and proposal.candidate_response is CandidateResponse.ACCEPT
                and find_candidate(market_structure, proposal.candidate_id)):
            chosen = proposal.candidate_id

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "selected_candidate_id": chosen,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
