# GEXter Market-Structure Vendor

**Date:** 2026-08-30
**Branch:** feat/gexter-market-structure
**Status:** Design approved, pending written-spec review

## Problem

TradingAgents has no notion of index-level options positioning. Its analysts
see price, indicators, fundamentals, news, macro series, and prediction-market
probabilities — all of which describe *what has happened* or *what is expected*,
none of which describe *how dealer hedging will shape price behavior today*.

GEXter (`../gexter`, a separate repo) computes exactly that for the S&P complex:
a gamma-exposure regime classification with a flip strike, call/put walls, a
directional bias, and a position-size multiplier. Until now it had no
machine-readable output, so nothing could consume it.

That gap is closed. GEXter ships a versioned JSON contract behind
`scripts/oi_model/nowcast_signals.py --json` (schema_version 1), which emits
parseable JSON on every path including total failure, with meaningful exit
codes.

This project makes that contract available to the market analyst as a bound
tool, through TradingAgents' existing vendor-routing layer.

## Scope

**In scope**
- New vendor module `tradingagents/dataflows/gexter.py` — runs the GEXter CLI as
  a subprocess, parses its JSON, formats markdown for an LLM reader.
- New `market_structure` category registered in `interface.py`, with `gexter` as
  its only vendor, added to `OPTIONAL_CATEGORIES`.
- New tool wrapper `tradingagents/agents/utils/market_structure_tools.py`.
- Configuration keys for the GEXter interpreter, repo path, and timeout.
- Conditional binding of the tool into the market analyst.
- Tests, with no subprocess execution, no GEXter checkout, and no Postgres.

**Out of scope (deferred)**
- Propagating the regime into `AgentState` for the researchers and risk
  debators. This is the natural next project — `risk_adjustment` is a
  position-size multiplier and the risk debators are the audience for it — but
  they have no tool access, so it requires graph plumbing that this project
  deliberately does not touch.
- Historical GEX for backtest-mode dates. GEXter only holds days its collector
  ran, and `nowcast_signals.py` reports the latest collected day; reconciling
  that with an arbitrary `curr_date` is its own project.
- Any HTTP service, scheduled artifact, or direct database read. See
  Alternatives considered.
- Any change to GEXter. This project consumes the phase-1 contract as shipped.

## Non-negotiable context

- **No imports from GEXter.** TradingAgents must not import GEXter code. GEXter
  depends on `psycopg2`, `lightgbm`, `polars`, and `ml4t-*` betas; hard-importing
  it would multiply TradingAgents' install weight for one optional feature. The
  process boundary is the whole point.
- **This repo is a fork.** `upstream` is `TauricResearch/TradingAgents`. Every
  change here must be inert for a user without GEXter, and the footprint on
  upstream-shared files must stay minimal so future merges are cheap.
- **A GEXter failure must never abort a run.** GEXter needs Postgres, a live
  collector, and a model artifact; all three are frequently unavailable. The
  category is registered in `OPTIONAL_CATEGORIES` so the router degrades to a
  sentinel string instead of raising.
- **The two systems use different symbol namespaces.** TradingAgents resolves
  the S&P to the Yahoo symbol `^GSPC` (`symbol_utils.py:64` maps `SPX` → `^GSPC`).
  GEXter collects under the Tradier-side names `SPX`, `XSP`, and `ES`. A run
  analyzing the index arrives as `^GSPC` and must be translated.
- **The bias is index-scoped, and whether that matters depends on the ticker.**
  `trade_bias` and `risk_adjustment` describe the S&P complex. Presenting them
  unqualified to a run analyzing an unrelated small-cap is actively misleading.
  Presenting them *disclaimed* to a run analyzing `^GSPC` is equally wrong in the
  opposite direction — it suppresses the signal exactly where it applies.

## Alternatives considered

**HTTP service in GEXter** — the original assessment's recommendation: a thin
read-only FastAPI service that TradingAgents calls over the network. Rejected
for now because it requires building and supervising a new service in GEXter
before any TradingAgents work can begin, and the subprocess path reaches the
same data with zero new GEXter work. Worth revisiting if the two ever run on
different machines, or if per-call latency becomes a problem.

**Scheduled artifact file** — GEXter writes the document to a shared path on a
cron; TradingAgents reads the file. Rejected because it makes staleness a
first-class concern for a marginal gain, and GEXter's phase-1 spec already
deferred an artifact-writing mode to a later project.

**Direct database read** — TradingAgents queries GEXter's Postgres itself.
Rejected: it would duplicate GEX math that already exists in GEXter and
re-introduce the `psycopg2` dependency the process boundary avoids.

**Subprocess (chosen)** — the original assessment rejected "subprocess + stdout
parsing" as brittle against print formatting that was actively changing. That
objection was specific to scraping *human-formatted text*. It no longer applies:
the phase-1 contract is versioned, is valid JSON on every path including total
failure, and signals failure by exit code. The cost is a same-machine
constraint and paying GEXter's startup (Postgres connect plus model load) per
call.

## Component 1 — `tradingagents/dataflows/gexter.py`

The vendor. One public function, matching the `route_to_vendor` calling
convention:

```python
def get_gexter_market_structure(ticker, symbols=None, top_strikes=None) -> str:
    """Index options-positioning context from GEXter, formatted for an LLM."""
```

**Behavior**

1. Read `gexter_repo`, `gexter_python`, and `gexter_timeout` from config. If
   either path is unset or does not exist, raise `VendorNotConfiguredError`.
2. Resolve the GEXter symbol from `ticker` via the symbol map below, unless
   `symbols` is given explicitly (which overrides).
3. Run the CLI:
   `[gexter_python, "scripts/oi_model/nowcast_signals.py", "--json",
     "--symbols", <symbol>]`, with `cwd=gexter_repo`, `capture_output=True`,
   `text=True`, and `timeout=gexter_timeout`. When `top_strikes` is given,
   `--top-strikes <n>` is appended; when it is `None` the flag is omitted and
   GEXter's own default of 10 applies.
4. Parse stdout as JSON. Every failure below raises a `VendorError` subclass,
   which the router converts to a sentinel because the category is optional.
5. Format the document as markdown and return it.

The module also exports a predicate used by the analyst binding:

```python
def gexter_configured() -> bool:
    """True when both GEXter paths are configured and exist on disk."""
```

It performs the same check as step 1 and is the single source of truth for
"is GEXter available here" — the vendor raises on it, the analyst branches on
it, and neither restates the condition.

**Symbol map** (TradingAgents ticker → GEXter symbol). Lives here, not in
`symbol_utils.py`, which is upstream-shared:

| Run ticker | GEXter symbol | S&P complex? |
|---|---|---|
| `^GSPC`, `SPX`, `SPX500`, `US500` | `SPX` | yes |
| `SPY` | `SPX` | yes, as a proxy |
| `XSP` | `XSP` | yes |
| `ES=F`, `ES` | `ES` | yes |
| anything else | `SPX` | no |

Matching is case-insensitive on the stripped ticker. An unrecognized ticker
still gets SPX positioning — it is legitimate market context — but takes the
disclaiming branch of the header.

`SPY` is a deliberate approximation: GEXter collects SPX and XSP chains, not
SPY's, and SPY has its own distinct gamma profile. It maps to `SPX` and is
treated as S&P complex, but the rendered output states that the measurement is
on SPX chains. Silently substituting would overstate the result.

## Component 2 — Output format

The header is two-branch, keyed on whether the run's ticker is S&P complex.

**S&P-complex ticker** (`^GSPC`, `SPY`, `XSP`, `ES=F`):

```markdown
## S&P 500 Index Options Positioning — SPX, 2026-08-30

This describes options positioning in the index you are analyzing. The
directional bias and risk multiplier below apply to this instrument.

**Regime:** compression (moderate, confidence 72%)
**Net GEX:** +1.23Bn  ·  flip 5400  ·  call wall 5450  ·  put wall 5350
**Bias:** sell_premium  ·  risk multiplier 1.0
**Structures:** iron_condor

Dealers are long gamma; expect mean reversion toward the flip strike.

*Nowcast and stale regimes diverge (compression → transition): real-time open
interest has shifted the regime.*
```

**Any other ticker** (illustrated for `NVDA`):

```markdown
## S&P 500 Index Options Positioning — SPX, 2026-08-30

INDEX-level market regime context. The directional bias and risk multiplier
below describe the S&P complex, NOT a recommendation for NVDA.

...same body...
```

For `SPY`, the header additionally notes that GEX is measured on SPX chains.

**Body rules**
- `spot`, `levels`, and `top_strikes` render only when present; a `null` level
  is omitted rather than printed as "None".
- The divergence line renders only when `regime_divergence` is `true`.
- When `model_available` is `false`, the stale view is rendered and the output
  states that the real-time nowcast is unavailable, so the LLM knows it is
  reading end-of-day positioning rather than live.
- A symbol whose entry is `{"status": "no_data"}` renders as an explicit
  unavailability line carrying the `reason`, not an error.
- `near_spot_concentration` and `zero_dte_proportion` are percentages, not
  fractions; render them with a `%` suffix.

## Component 3 — Registration in `interface.py`

Four additions, each a few lines:

```python
TOOLS_CATEGORIES["market_structure"] = {
    "description": "Index options positioning and gamma-exposure regime",
    "tools": ["get_market_structure"],
}

VENDOR_LIST += ["gexter"]

OPTIONAL_CATEGORIES = {"macro_data", "prediction_markets", "market_structure"}

VENDOR_METHODS["get_market_structure"] = {
    "gexter": get_gexter_market_structure,
}
```

Registering in `OPTIONAL_CATEGORIES` is what makes a GEXter failure degrade:
`route_to_vendor` returns a `DATA_UNAVAILABLE:` sentinel instead of raising
(`interface.py:253-259`).

## Component 4 — Tool wrapper

`tradingagents/agents/utils/market_structure_tools.py`, mirroring
`prediction_markets_tools.py`:

```python
@tool
def get_market_structure(ticker, symbols=None, top_strikes=None) -> str:
    """Index options positioning and gamma-exposure regime for the S&P complex."""
    return route_to_vendor("get_market_structure", ticker, symbols, top_strikes)
```

`ticker` is required and carries the instrument under analysis — the same
ticker-first convention as `get_stock_data`. The market analyst's system
message already holds the instrument context, so the model has the value to
pass. The tool is re-exported from `agent_utils` alongside the others.

## Component 5 — Configuration

```python
"gexter_repo": os.getenv("TRADINGAGENTS_GEXTER_REPO"),
"gexter_python": os.getenv("TRADINGAGENTS_GEXTER_PYTHON"),
"gexter_timeout": int(os.getenv("TRADINGAGENTS_GEXTER_TIMEOUT", "120")),
```

Both paths default to `None`, so the feature is off unless explicitly
configured. `data_vendors["market_structure"]` is set to `"gexter"` — harmless
when unconfigured, because the tool is not bound in that case.

The 120-second default reflects that GEXter pays a Postgres connect plus a
model load per invocation.

## Component 6 — Conditional binding in the market analyst

```python
tools = [get_stock_data, get_indicators, get_verified_market_snapshot]
if gexter_configured():
    tools.append(get_market_structure)
```

Binding only when configured is what keeps this fork's addition invisible to an
upstream user. The soft-fail sentinel already prevents a crash; conditional
binding additionally prevents every upstream user's market analyst from burning
a tool call per run on a tool that can never succeed.

The system message gains a short paragraph, included only when the tool is
bound, telling the analyst what gamma positioning means and when to consult it.

## Data flow

```
market_analyst (tool bound only if configured)
  -> get_market_structure(ticker="^GSPC")
  -> route_to_vendor("get_market_structure", ...)
  -> get_gexter_market_structure(...)
       ticker -> GEXter symbol ("^GSPC" -> "SPX", S&P complex = true)
       subprocess: {gexter_python} scripts/oi_model/nowcast_signals.py
                   --json --symbols SPX          [cwd = gexter_repo]
       stdout -> json.loads -> schema_version check -> markdown
  -> str to the LLM
       (or "DATA_UNAVAILABLE: optional market_structure ..." on any failure)
```

## Error handling

Every failure below raises from the vendor and is converted by the router into
a `DATA_UNAVAILABLE` sentinel. No failure aborts a run.

| Condition | Vendor behavior |
|---|---|
| `gexter_repo` / `gexter_python` unset or missing on disk | `VendorNotConfiguredError` |
| CLI exits 1 with `{"error": ...}` (Postgres down) | `VendorError` carrying the document's `error` string |
| CLI exits non-zero with unparseable stdout | `VendorError` with a truncated stdout/stderr excerpt |
| `subprocess.TimeoutExpired` | Killed; `VendorError` naming the timeout — a hung Postgres cannot hang an analysis |
| stdout is not JSON | `VendorError` with a truncated excerpt |
| `schema_version` is absent or != 1 | `VendorError` naming the observed version |
| Requested symbol has `{"status": "no_data"}` | **Not an error.** Rendered as an explicit unavailability line |
| `model_available` is `false` | **Not an error.** Stale view rendered, with a note |

The `schema_version` check is deliberate and strict. GEXter's contract states
that additive changes keep the version at 1 while a removal or retype bumps it.
Refusing an unrecognized version turns a future breaking change into a loud,
diagnosable failure rather than a silently misparsed document.

## Testing

No subprocess execution, no GEXter checkout, no Postgres, no network.
`subprocess.run` is monkeypatched to return canned documents captured from real
phase-1 output.

**`tests/test_gexter_vendor.py`** (new) — the vendor:
- Happy path: a full document renders regime, levels, and bias.
- S&P-complex ticker (`^GSPC`) takes the applies-to-your-instrument header.
- Non-complex ticker (`NVDA`) takes the disclaiming header and names the ticker.
- `SPY` takes the complex header and states that GEX is measured on SPX chains.
- Symbol mapping: `^GSPC`/`SPX`/`SPX500` → `SPX`; `ES=F` → `ES`; `XSP` → `XSP`;
  an unknown ticker → `SPX` with the disclaiming branch.
- An explicit `symbols` argument overrides the mapping.
- `regime_divergence: true` renders the divergence line; `false` omits it.
- `model_available: false` renders the stale view and says the nowcast is
  unavailable; no fabricated nowcast.
- `{"status": "no_data"}` renders an unavailability line carrying the reason.
- Null levels are omitted rather than rendered as "None".
- Error document plus exit 1 raises `VendorError` carrying the error text.
- Timeout, non-JSON stdout, and a wrong `schema_version` each raise.
- Unset config raises `VendorNotConfiguredError`.
- The subprocess is invoked with the expected argv, `cwd`, and timeout.

**`tests/test_market_structure_routing.py`** (new) — the routing seam:
- `get_market_structure` resolves to the `gexter` vendor.
- A vendor failure returns the `DATA_UNAVAILABLE` sentinel rather than raising,
  proving `market_structure` is genuinely optional.

**`tests/test_market_analyst_gexter_binding.py`** (new) — the binding:
- With config set, the market analyst binds `get_market_structure`.
- Without it, the analyst's tool list is byte-for-byte what it is today.

## Success criteria

1. With GEXter configured and Postgres up, a market-analyst run can call
   `get_market_structure` and receive rendered positioning.
2. With GEXter configured and Postgres **down**, the run completes and the
   analyst sees a `DATA_UNAVAILABLE` sentinel — no traceback, no abort.
3. With GEXter unconfigured, the market analyst's bound tool list is identical
   to today's and no GEXter subprocess is ever spawned. (The vendor module is
   still imported at `interface.py` load time to populate `VENDOR_METHODS`; it
   imports only the standard library plus this repo's own config and errors, so
   importing it costs nothing and reaches nothing.)
4. A run analyzing `^GSPC` sees the applies-to-your-instrument header; a run
   analyzing `NVDA` sees the disclaiming header naming NVDA.
5. A `schema_version` other than 1 produces a loud, named failure.
6. The full existing test suite passes unchanged.
