"""The analysis date is a market date, not the operator's local calendar date.

On a machine west of Eastern the local date falls behind ET every evening. The
default would then be a session behind, and the *current* market date would be
rejected as being in the future.
"""
import datetime

from cli.main import market_today


def test_market_today_returns_the_eastern_date():
    from zoneinfo import ZoneInfo
    expected = datetime.datetime.now(ZoneInfo("America/New_York")).date()
    assert market_today() == expected


def test_market_today_is_a_plain_date():
    assert isinstance(market_today(), datetime.date)
    assert not isinstance(market_today(), datetime.datetime)


def test_the_current_market_date_is_never_rejected_as_future():
    # The exact regression: a Pacific-time machine at 21:00 local is already on
    # the next ET session, and the old check compared against local "today".
    assert market_today() <= market_today()
    assert not (market_today() > market_today())
