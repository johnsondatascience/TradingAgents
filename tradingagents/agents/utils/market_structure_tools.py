from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_market_structure(
    ticker: Annotated[
        str,
        "The ticker under analysis, e.g. '^GSPC' or 'NVDA'. Determines whether "
        "the directional bias applies to your instrument or is index context.",
    ],
    symbols: Annotated[
        str | None,
        "A SINGLE GEXter symbol (SPX, XSP or ES) overriding the one derived "
        "from the ticker. Not a comma-separated list: only one symbol is "
        "rendered. Omit to derive it from the ticker.",
    ] = None,
    top_strikes: Annotated[
        int | None, "How many strikes by |GEX| to include; omit for a default of 10"
    ] = None,
) -> str:
    """
    Retrieve index options positioning and the gamma-exposure regime for the
    S&P complex: the regime label and confidence, net GEX, the gamma flip
    strike, call and put walls, a directional bias and a position-size
    multiplier. Describes how dealer hedging is likely to shape index price
    behavior today — mean reversion when dealers are long gamma, trend
    acceleration when short.

    This is INDEX-level data. When the ticker under analysis is the S&P complex
    (^GSPC, SPY, XSP, ES=F) the bias applies to it directly; for any other
    ticker it is market-regime context only. Uses the configured
    market_structure vendor.

    The report covers GEXter's most recently collected trading day, which is
    not necessarily the date under analysis: this tool takes no date. The
    rendered header states the day it covers; if that differs from the date
    you are analyzing, treat the positioning as background only and do not
    apply it to that date.

    Args:
        ticker (str): The instrument under analysis
        symbols (str): Optional override, a single GEXter symbol (not a list)
        top_strikes (int): Max strikes to include; omit for a default of 10

    Returns:
        str: A formatted markdown report of index options positioning
    """
    return route_to_vendor("get_market_structure", ticker, symbols, top_strikes)
