"""Paper order execution via Alpaca (paper=True only).

Never touches live keys. All orders are submitted to the paper endpoint.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, QueryOrderStatus
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, ClosePositionRequest
from alpaca.common.exceptions import APIError

from data.errors import DataSourceError, TransientDataError
from monitoring.logger import get_logger

log = get_logger("orders")


@dataclass(frozen=True)
class OrderResult:
    success: bool
    order_id: str | None = None
    symbol: str = ""
    side: str = ""
    qty: float = 0.0
    status: str = ""
    filled_avg_price: float | None = None
    message: str = ""


class PaperOrderExecutor:
    """Thin wrapper around Alpaca TradingClient (forced paper=True)."""

    def __init__(self, api_key: str, api_secret: str) -> None:
        if not api_key or not api_secret:
            raise DataSourceError("Alpaca paper keys required for order execution")
        # Hard-coded paper=True — live trading is not possible through this class
        self._client = TradingClient(api_key, api_secret, paper=True)
        self._lock = asyncio.Lock()

    async def _run(self, func, *args, **kwargs):
        async with self._lock:
            try:
                return await asyncio.to_thread(func, *args, **kwargs)
            except APIError as exc:
                status = getattr(exc, "status_code", None)
                if status in {408, 429, 500, 502, 503, 504}:
                    raise TransientDataError(f"Alpaca order HTTP {status}") from None
                raise DataSourceError(f"Alpaca order error (HTTP {status}): {exc}") from None

    async def get_account(self) -> Any:
        return await self._run(self._client.get_account)

    async def get_positions(self) -> list[Any]:
        return await self._run(self._client.get_all_positions)

    async def submit_market_buy(self, symbol: str, qty: float) -> OrderResult:
        if qty <= 0:
            return OrderResult(False, message="qty must be positive")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        try:
            order = await self._run(self._client.submit_order, req)
            log.info("PAPER BUY submitted: %s qty=%.6f id=%s", symbol, qty, order.id)
            return OrderResult(
                success=True,
                order_id=str(order.id),
                symbol=symbol,
                side="buy",
                qty=float(order.qty),
                status=str(order.status),
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )
        except Exception as exc:
            log.error("PAPER BUY failed %s: %s", symbol, exc)
            return OrderResult(False, symbol=symbol, side="buy", qty=qty, message=str(exc)[:200])

    async def submit_market_sell(self, symbol: str, qty: float) -> OrderResult:
        if qty <= 0:
            return OrderResult(False, message="qty must be positive")
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        try:
            order = await self._run(self._client.submit_order, req)
            log.info("PAPER SELL submitted: %s qty=%.6f id=%s", symbol, qty, order.id)
            return OrderResult(
                success=True,
                order_id=str(order.id),
                symbol=symbol,
                side="sell",
                qty=float(order.qty),
                status=str(order.status),
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            )
        except Exception as exc:
            log.error("PAPER SELL failed %s: %s", symbol, exc)
            return OrderResult(False, symbol=symbol, side="sell", qty=qty, message=str(exc)[:200])

    async def close_position(self, symbol: str) -> OrderResult:
        try:
            order = await self._run(self._client.close_position, symbol)
            log.info("PAPER CLOSE submitted: %s id=%s", symbol, getattr(order, "id", None))
            return OrderResult(success=True, order_id=str(getattr(order, "id", "")),
                               symbol=symbol, side="sell", status="submitted")
        except Exception as exc:
            log.error("PAPER CLOSE failed %s: %s", symbol, exc)
            return OrderResult(False, symbol=symbol, message=str(exc)[:200])

    async def cancel_all_orders(self) -> int:
        try:
            await self._run(self._client.cancel_orders)
            log.warning("Cancelled all open paper orders")
            return 1
        except Exception as exc:
            log.error("cancel_all failed: %s", exc)
            return 0
