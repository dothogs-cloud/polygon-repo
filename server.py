import os
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer

load_dotenv()

POLYGON_API_KEY = os.environ["POLYGON_API_KEY"]
POLYGON_BASE = "https://api.polygon.io"

mcp = MCPServer("polygon-options-flow")


async def _get(client: httpx.AsyncClient, path: str, params: dict[str, Any]) -> dict:
    params = {**params, "apiKey": POLYGON_API_KEY}
    resp = await client.get(f"{POLYGON_BASE}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


async def _fetch_option_snapshots(symbol: str, max_pages: int = 3) -> list[dict]:
    """Pull the options chain snapshot for an underlying, following pagination up to max_pages."""
    results: list[dict] = []
    async with httpx.AsyncClient() as client:
        data = await _get(
            client,
            f"/v3/snapshot/options/{symbol.upper()}",
            {"limit": 250},
        )
        results.extend(data.get("results", []))
        next_url = data.get("next_url")
        pages = 1
        while next_url and pages < max_pages:
            resp = await client.get(next_url, params={"apiKey": POLYGON_API_KEY}, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            next_url = data.get("next_url")
            pages += 1
    return results


@mcp.tool()
async def get_unusual_options_activity(
    symbol: str,
    min_volume_oi_ratio: float = 2.0,
    min_day_volume: int = 100,
    top_n: int = 10,
) -> dict:
    """Screen an underlying's options chain for unusual activity today.

    Flags contracts where today's volume is high relative to existing open
    interest (a common "unusual options activity" heuristic — volume that
    already exceeds a multiple of existing open interest suggests new
    positioning rather than existing holders trading around a position).

    Args:
        symbol: Underlying ticker, e.g. "NVDA".
        min_volume_oi_ratio: Minimum (day volume / open interest) to flag a contract.
        min_day_volume: Minimum absolute day volume, to filter out illiquid noise.
        top_n: Max number of contracts to return, sorted by estimated premium traded.
    """
    snapshots = await _fetch_option_snapshots(symbol)

    flagged = []
    for s in snapshots:
        details = s.get("details", {})
        day = s.get("day", {})
        volume = day.get("volume") or 0
        open_interest = s.get("open_interest") or 0
        if volume < min_day_volume:
            continue
        ratio = volume / max(open_interest, 1)
        if ratio < min_volume_oi_ratio:
            continue
        last_price = (day.get("close") or day.get("vwap") or 0)
        est_premium = last_price * volume * 100
        flagged.append(
            {
                "ticker": s.get("ticker") or details.get("ticker"),
                "contract_type": details.get("contract_type"),
                "strike_price": details.get("strike_price"),
                "expiration_date": details.get("expiration_date"),
                "day_volume": volume,
                "open_interest": open_interest,
                "volume_oi_ratio": round(ratio, 2),
                "last_price": last_price,
                "implied_volatility": s.get("implied_volatility"),
                "est_premium_traded_usd": round(est_premium, 2),
            }
        )

    flagged.sort(key=lambda c: c["est_premium_traded_usd"], reverse=True)
    return {
        "symbol": symbol.upper(),
        "contracts_scanned": len(snapshots),
        "flagged_count": len(flagged),
        "flagged_contracts": flagged[:top_n],
    }


@mcp.tool()
async def get_options_flow_summary(symbol: str) -> dict:
    """Summarize today's call vs. put volume for an underlying's options chain.

    A low put/call volume ratio has historically been associated with modestly
    bullish next-day/next-week returns in academic research (and vice versa for
    a high ratio) — this is a raw same-day snapshot, not a backtested signal.

    Args:
        symbol: Underlying ticker, e.g. "NVDA".
    """
    snapshots = await _fetch_option_snapshots(symbol)

    call_volume = 0
    put_volume = 0
    call_premium = 0.0
    put_premium = 0.0
    for s in snapshots:
        details = s.get("details", {})
        day = s.get("day", {})
        volume = day.get("volume") or 0
        last_price = day.get("close") or day.get("vwap") or 0
        premium = last_price * volume * 100
        if details.get("contract_type") == "call":
            call_volume += volume
            call_premium += premium
        elif details.get("contract_type") == "put":
            put_volume += volume
            put_premium += premium

    put_call_ratio = (put_volume / call_volume) if call_volume else None
    return {
        "symbol": symbol.upper(),
        "contracts_scanned": len(snapshots),
        "call_volume": call_volume,
        "put_volume": put_volume,
        "put_call_volume_ratio": round(put_call_ratio, 3) if put_call_ratio is not None else None,
        "call_premium_traded_usd": round(call_premium, 2),
        "put_premium_traded_usd": round(put_premium, 2),
    }


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True,
    )
