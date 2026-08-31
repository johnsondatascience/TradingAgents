import copy

import pytest

from tradingagents import default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.gexter import GexterUnavailableError


@pytest.fixture(autouse=True)
def _base_config():
    set_config(copy.deepcopy(default_config.DEFAULT_CONFIG))


def test_market_structure_is_a_registered_category():
    assert "market_structure" in interface.TOOLS_CATEGORIES
    assert "get_market_structure" in interface.TOOLS_CATEGORIES["market_structure"]["tools"]


def test_gexter_is_a_registered_vendor():
    assert "gexter" in interface.VENDOR_LIST
    assert "gexter" in interface.VENDOR_METHODS["get_market_structure"]


def test_market_structure_is_optional():
    # This is what keeps a GEXter outage from aborting an analysis.
    assert "market_structure" in interface.OPTIONAL_CATEGORIES


def test_method_resolves_to_its_category():
    assert interface.get_category_for_method("get_market_structure") == "market_structure"


def test_vendor_failure_degrades_to_a_sentinel(monkeypatch):
    def boom(*args, **kwargs):
        raise GexterUnavailableError("GEXter reported: db down")

    monkeypatch.setitem(interface.VENDOR_METHODS["get_market_structure"], "gexter", boom)
    result = interface.route_to_vendor("get_market_structure", "^GSPC")
    assert result.startswith("DATA_UNAVAILABLE")
    assert "market_structure" in result


def test_tool_wrapper_routes_through_the_vendor(monkeypatch):
    from tradingagents.agents.utils.market_structure_tools import get_market_structure

    monkeypatch.setitem(
        interface.VENDOR_METHODS["get_market_structure"],
        "gexter",
        lambda ticker, symbols=None, top_strikes=None: f"rendered:{ticker}:{symbols}",
    )
    assert get_market_structure.invoke({"ticker": "^GSPC"}) == "rendered:^GSPC:None"


def test_tool_is_reexported_from_agent_utils():
    from tradingagents.agents.utils import agent_utils

    assert "get_market_structure" in agent_utils.__all__
    assert hasattr(agent_utils, "get_market_structure")
