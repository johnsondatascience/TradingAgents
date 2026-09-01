"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
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

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Based on your analysis, provide a specific recommendation to buy, sell, or hold. "
                    "Anchor your reasoning in the analysts' reports and the research plan. "
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on a comprehensive analysis by a team of analysts, here is an investment "
                    f"plan tailored for {company_name}. {instrument_context} This plan incorporates "
                    f"insights from current technical market trends, macroeconomic indicators, and "
                    f"social media sentiment. Use this plan as a foundation for evaluating your next "
                    f"trading decision.\n\nProposed Investment Plan: {investment_plan}\n\n"
                    f"Leverage these insights to make an informed and strategic decision."
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
            lambda proposal: render_trader_proposal(
                proposal, document=market_structure),
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
