from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any
from urllib.request import Request, urlopen


COPY_TRADER_LIST_URL_ENV = "SIRTRADE_COPY_TRADER_LIST_URL"
COPY_TRADER_POSITIONS_URL_ENV = "SIRTRADE_COPY_TRADER_POSITIONS_URL_TEMPLATE"
COPY_TRADER_HEADERS_ENV = "SIRTRADE_COPY_TRADER_HEADERS_JSON"


@dataclass
class LeadTraderProfile:
    trader_id: str
    nickname: str
    roi: float
    pnl_usd: float
    win_rate: float
    max_drawdown: float
    followers: float
    score: float


@dataclass
class LeadTraderPosition:
    trader_id: str
    symbol: str
    side: str
    leverage: float
    notional_usd: float
    entry_price: float


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_side(value: Any) -> str:
    side = str(value).strip().upper()
    if side in {"BUY", "LONG", "1", "+1"}:
        return "LONG"
    if side in {"SELL", "SHORT", "-1"}:
        return "SHORT"
    return side


def _extract_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    for key in ("data", "rows", "list", "items", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _extract_items(value)
            if nested:
                return nested
    return []


def _load_headers() -> dict[str, str]:
    raw = os.getenv(COPY_TRADER_HEADERS_ENV, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _http_get_json_url(url: str, headers: dict[str, str] | None = None) -> list | dict:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _profile_from_payload(item: dict[str, Any]) -> LeadTraderProfile | None:
    trader_id = str(
        item.get("traderId")
        or item.get("portfolioId")
        or item.get("leadPortfolioId")
        or item.get("id")
        or ""
    ).strip()
    if not trader_id:
        return None

    nickname = str(item.get("nickname") or item.get("name") or item.get("traderName") or trader_id).strip()
    roi = _coerce_float(item.get("roi") or item.get("roi30d") or item.get("pnlRate") or item.get("yieldRate"))
    pnl_usd = _coerce_float(item.get("pnl") or item.get("pnlUsd") or item.get("profit") or item.get("copyProfit"))
    win_rate = _coerce_float(item.get("winRate") or item.get("win_ratio") or item.get("successRate"))
    max_drawdown = abs(_coerce_float(item.get("maxDrawdown") or item.get("drawdown") or item.get("mdd")))
    followers = _coerce_float(item.get("followers") or item.get("copiers") or item.get("aum") or item.get("followersCount"))

    score = (0.45 * roi) + (0.25 * pnl_usd / 1000.0) + (0.15 * win_rate) + (0.05 * followers / 1000.0) - (0.20 * max_drawdown)
    return LeadTraderProfile(
        trader_id=trader_id,
        nickname=nickname,
        roi=roi,
        pnl_usd=pnl_usd,
        win_rate=win_rate,
        max_drawdown=max_drawdown,
        followers=followers,
        score=score,
    )


def _position_from_payload(item: dict[str, Any], trader_id: str) -> LeadTraderPosition | None:
    symbol = str(item.get("symbol") or item.get("pair") or item.get("ticker") or "").upper().strip()
    if not symbol:
        return None

    side = _normalize_side(item.get("side") or item.get("direction") or item.get("positionSide"))
    if side not in {"LONG", "SHORT"}:
        return None

    leverage = max(0.0, _coerce_float(item.get("leverage") or item.get("marginMultiple") or 1.0, default=1.0))
    notional_usd = abs(_coerce_float(item.get("notional") or item.get("notionalUsd") or item.get("amountUsd") or item.get("value")))
    entry_price = max(0.0, _coerce_float(item.get("entryPrice") or item.get("avgPrice") or item.get("price")))
    return LeadTraderPosition(
        trader_id=trader_id,
        symbol=symbol,
        side=side,
        leverage=leverage,
        notional_usd=notional_usd,
        entry_price=entry_price,
    )


def fetch_copy_trader_leaderboard() -> list[LeadTraderProfile]:
    url = os.getenv(COPY_TRADER_LIST_URL_ENV, "").strip()
    if not url:
        return []

    payload = _http_get_json_url(url, headers=_load_headers())
    profiles: list[LeadTraderProfile] = []
    for item in _extract_items(payload):
        profile = _profile_from_payload(item)
        if profile is not None:
            profiles.append(profile)
    return profiles


def select_best_lead_trader(profiles: list[LeadTraderProfile]) -> LeadTraderProfile | None:
    if not profiles:
        return None
    ranked = sorted(
        profiles,
        key=lambda profile: (
            float(profile.score),
            float(profile.roi),
            float(profile.pnl_usd),
            float(profile.win_rate),
            -float(profile.max_drawdown),
        ),
        reverse=True,
    )
    return ranked[0]


def fetch_copy_trader_positions(trader_id: str) -> list[LeadTraderPosition]:
    template = os.getenv(COPY_TRADER_POSITIONS_URL_ENV, "").strip()
    if not template or not trader_id:
        return []

    url = template.format(trader_id=trader_id)
    payload = _http_get_json_url(url, headers=_load_headers())
    positions: list[LeadTraderPosition] = []
    for item in _extract_items(payload):
        position = _position_from_payload(item, trader_id=trader_id)
        if position is not None:
            positions.append(position)
    return positions


def load_top_copy_trader_snapshot(
    allow_shorts: bool,
    allow_leverage: bool,
) -> dict[str, Any] | None:
    leader = select_best_lead_trader(fetch_copy_trader_leaderboard())
    if leader is None:
        return None

    raw_positions = fetch_copy_trader_positions(leader.trader_id)
    filtered_positions: list[LeadTraderPosition] = []
    seen: set[tuple[str, str]] = set()
    for position in sorted(raw_positions, key=lambda item: float(item.notional_usd), reverse=True):
        if not allow_leverage and float(position.leverage) > 1.0:
            continue
        if not allow_shorts and position.side == "SHORT":
            continue
        dedupe_key = (position.symbol, position.side)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        filtered_positions.append(position)

    return {
        "leader": asdict(leader),
        "positions": [asdict(position) for position in filtered_positions],
    }