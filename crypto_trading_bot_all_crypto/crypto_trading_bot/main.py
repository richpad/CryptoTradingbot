"""CLI entry point (Stage 1: data only; no trading of any kind).

  python main.py backfill [--days N]     download historical candles
  python main.py collect [--once]        live snapshots + incremental candles
  python main.py gaps [--repair]         detect (and optionally repair) missing candles
  python main.py status                  show what is stored (no network needed)
  python main.py signals                 Strategy A decision on the latest closed candle (no orders)

Stage 1b: the default data source is Alpaca (paper keys; data only, no orders are ever placed).
Set EXCHANGE_ID=kraken (or another ccxt exchange) to use the previous source.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import signal
import sys
from datetime import datetime, timezone
from typing import Any

import ccxt

from config import Config, ConfigError, load_config, timeframe_ms
from data.errors import DataSourceError
from data.historical_data import HistoricalDataFetcher, find_gaps, now_ms
from data.market_data import MarketDataCollector
from database.database_manager import DatabaseManager
from monitoring.logger import get_logger, setup_logging

log = get_logger("main")


def fmt_ts(ms: int | None) -> str:
    if ms is None:
        return "-"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


async def open_exchange(config: Config) -> Any:
    if config.exchange_id == "alpaca":
        if not (config.alpaca_api_key and config.alpaca_api_secret):
            raise ConfigError("Set ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET in .env "
                              "(paper keys only; never paste them into chat or commit them).")
        from data.alpaca_source import AlpacaDataAdapter     # lazy: ccxt-only installs still work
        adapter = AlpacaDataAdapter(config.alpaca_api_key, config.alpaca_api_secret)
        await adapter.load_markets()
        return adapter

    import ccxt.async_support as ccxt_async
    if config.exchange_id not in ccxt_async.exchanges:
        raise ConfigError(f"Unknown exchange {config.exchange_id!r}")
    exchange = getattr(ccxt_async, config.exchange_id)(
        {"enableRateLimit": True, "timeout": config.request_timeout_ms})   # public data: no credentials
    try:
        await exchange.load_markets()
        if not exchange.has.get("fetchOHLCV"):
            raise ConfigError(f"{config.exchange_id} does not support OHLCV")
    except BaseException:
        await exchange.close()
        raise
    return exchange


def restrict_to_supported(config: Config, exchange: Any) -> Config:
    supported = sorted(exchange.symbols or [])
    if config.use_all_crypto:
        if not supported:
            raise ConfigError("Exchange returned no tradable symbols")
        log.info("USE_ALL_CRYPTO=true → trading %d pairs: %s", len(supported),
                 ", ".join(supported[:12]) + (" ..." if len(supported) > 12 else ""))
        return dataclasses.replace(config, pairs=tuple(supported))

    supported_set = set(supported)
    valid = tuple(p for p in config.pairs if p in supported_set)
    for p in config.pairs:
        if p not in supported_set:
            log.warning("%s is not listed on %s; skipping", p, config.exchange_id)
    if not valid:
        raise ConfigError("None of the configured pairs are available on this exchange")
    return dataclasses.replace(config, pairs=valid)


async def cmd_backfill(config: Config, db: DatabaseManager, exchange: Any, days: int) -> int:
    fetcher = HistoricalDataFetcher(exchange, db, config.exchange_id,
                                    daily_grace_ms=config.daily_close_grace_minutes * 60_000)
    start, sem = now_ms() - days * 86_400_000, asyncio.Semaphore(config.max_concurrency)

    async def one(symbol: str, tf: str):
        async with sem:
            return await fetcher.backfill(symbol, tf, start)

    results = await asyncio.gather(*(one(s, tf) for s in config.pairs for tf in config.timeframes),
                                   return_exceptions=True)
    failures = 0
    for r in results:
        if isinstance(r, BaseException):
            failures += 1
            log.error("backfill failed: %s: %s", type(r).__name__, r)
        else:
            note = f" offset={r.utc_offset_ms / 3_600_000:+.1f}h" if r.utc_offset_ms else ""
            note += f" dst_like={r.dst_like_spacing}" if r.dst_like_spacing else ""
            print(f"{r.symbol:<10} {r.timeframe:<4} inserted={r.inserted:<6} pages={r.pages:<3} "
                  f"dropped={r.dropped_invalid:<3} earliest={fmt_ts(r.first_ts_ms)}{note}")
    return 1 if failures else 0


async def cmd_collect(config: Config, db: DatabaseManager, exchange: Any, once: bool) -> int:
    collector = MarketDataCollector(exchange, db, config)
    if once:
        report = await collector.run_once()
        print(report)
        return 1 if report.errors else 0
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:   # Windows
            pass
    log.info("collecting %s on %s every %.0fs (Ctrl+C to stop)",
             ", ".join(config.pairs), config.exchange_id, config.poll_interval_seconds)
    await collector.run_forever(stop)
    return 0


async def cmd_gaps(config: Config, db: DatabaseManager, exchange: Any | None, days: int, repair: bool) -> int:
    start, now, remaining = now_ms() - days * 86_400_000, now_ms(), 0
    fetcher = (HistoricalDataFetcher(exchange, db, config.exchange_id,
                                     daily_grace_ms=config.daily_close_grace_minutes * 60_000)
               if exchange else None)
    for symbol in config.pairs:
        for tf in config.timeframes:
            tf_ms = timeframe_ms(tf)
            if repair and fetcher:
                await fetcher.repair_gaps(symbol, tf, start)
            stamps = db.get_timestamps(config.exchange_id, symbol, tf, start)
            gaps = find_gaps(stamps, tf_ms)
            stale = max(0, ((now // tf_ms) * tf_ms - tf_ms - stamps[-1]) // tf_ms) if stamps else None
            remaining += sum(g.missing for g in gaps)
            print(f"{symbol:<10} {tf:<4} candles={len(stamps):<6} gaps={len(gaps):<3} "
                  f"missing={sum(g.missing for g in gaps):<5} stale_by={'-' if stale is None else stale} candles")
    return 1 if remaining else 0


def cmd_status(config: Config, db: DatabaseManager) -> int:
    print(f"active source: {config.exchange_id}\n")
    rows = db.coverage(None)
    if not rows:
        print("No candles stored yet. Run: python main.py backfill")
    for r in rows:
        print(f"{r['exchange']:<8} {r['symbol']:<10} {r['timeframe']:<4} candles={r['count']:<7} "
              f"{fmt_ts(r['first_ts_ms'])} -> {fmt_ts(r['last_ts_ms'])}")
    errors = db.get_errors(5)
    print(f"\nrecent errors: {len(errors)}")
    for e in errors:
        print(f"  {fmt_ts(e.ts_ms)} {e.component}: {e.message[:100]}")
    return 0


def cmd_signals(config: Config, db: DatabaseManager) -> int:
    """Evaluate Strategy A on the latest CLOSED candle of every stored series. Prints only; places nothing."""
    from indicators.scoring import score_frame
    from indicators.technical_indicators import IndicatorParams, compute_indicators
    from strategies.trend_following import TrendFollowing

    strat, need = TrendFollowing(), IndicatorParams().warmup_rows
    print(f"Strategy A (trend following) on '{config.exchange_id}', latest closed candle per series. "
          f"Decisions only: no orders are placed.\n")
    for symbol in config.pairs:
        for tf in config.timeframes:
            df = db.get_candles(config.exchange_id, symbol, tf, limit=2000)
            if len(df) < need:
                print(f"{symbol:<10} {tf:<4} need {need} candles, have {len(df)} -> run: python main.py backfill --days 250")
                continue
            ind = compute_indicators(df)
            score = score_frame(ind)["score"]
            sig = strat.evaluate(ind, symbol, tf, score)
            entries = int(strat.signals(ind)["entry"].sum())
            rsi = sig.indicators.get("rsi_14", float("nan"))
            print(f"{symbol:<10} {tf:<4} {fmt_ts(sig.candle_open_ts_ms)}  {sig.action.value:<4} close={sig.price:<11.4f} "
                  f"rsi={rsi:5.1f} score={(sig.score or 0.0):+5.2f} entries_in_window={entries:<3} {sig.reasons[0]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crypto trading bot - Stage 1+2 (data + paper trading)")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backfill"); b.add_argument("--days", type=int)
    c = sub.add_parser("collect"); c.add_argument("--once", action="store_true")
    g = sub.add_parser("gaps"); g.add_argument("--days", type=int); g.add_argument("--repair", action="store_true")
    sub.add_parser("status")
    sub.add_parser("signals")
    paper = sub.add_parser("paper", help="Run paper trading engine (requires PAPER_TRADING_ENABLED=true)")
    paper.add_argument("--once", action="store_true", help="Run a single cycle then exit")
    paper.add_argument("--interval", type=float, default=60.0, help="Seconds between cycles (default 60)")
    return p


async def cmd_paper(config: Config, db: DatabaseManager, once: bool, interval: float) -> int:
    """Stage 2: paper trading execution."""
    if not config.paper_trading_enabled:
        raise ConfigError(
            "PAPER_TRADING_ENABLED is not set to true in .env. "
            "Paper trading is opt-in for safety."
        )
    if not (config.alpaca_api_key and config.alpaca_api_secret):
        raise ConfigError("ALPACA_PAPER_API_KEY and ALPACA_PAPER_API_SECRET are required for paper trading")

    # Load markets so USE_ALL_CRYPTO can expand the pair list
    exchange = await open_exchange(config)
    try:
        config = restrict_to_supported(config, exchange)
    finally:
        await exchange.close()

    from execution.engine import PaperTradingEngine
    from execution.orders import PaperOrderExecutor
    from execution.risk import RiskParams

    executor = PaperOrderExecutor(config.alpaca_api_key, config.alpaca_api_secret)
    engine = PaperTradingEngine(config, db, executor, RiskParams())  # aggressive defaults already set

    if once:
        await engine.run_once()
        return 0

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, engine.stop)
        except NotImplementedError:
            pass
    await engine.run_forever(interval_seconds=interval)
    return 0


async def run(args: argparse.Namespace, config: Config, db: DatabaseManager) -> int:
    if args.command == "status":
        return cmd_status(config, db)
    if args.command == "signals":
        return cmd_signals(config, db)
    if args.command == "paper":
        return await cmd_paper(config, db, args.once, args.interval)

    exchange = None
    try:
        if args.command != "gaps" or args.repair:
            exchange = await open_exchange(config)
            config = restrict_to_supported(config, exchange)
        if args.command == "backfill":
            return await cmd_backfill(config, db, exchange, args.days or config.history_days)
        if args.command == "collect":
            return await cmd_collect(config, db, exchange, args.once)
        return await cmd_gaps(config, db, exchange, args.days or config.history_days, args.repair)
    finally:
        if exchange is not None:
            await exchange.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    setup_logging(config.log_level, config.log_dir)
    mode = "PAPER TRADING ENABLED" if config.paper_trading_enabled else "data only (no orders)"
    log.info("data source: %s | mode: %s", config.exchange_id, mode)
    db = DatabaseManager(config.database_url)
    db.init_schema()
    try:
        return asyncio.run(run(args, config, db))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    except (ccxt.BaseError, DataSourceError) as exc:   # network down, rejected request, bad credentials...
        log.error("Exchange error: %s: %s", type(exc).__name__, str(exc)[:300])
        print(f"Data source error ({type(exc).__name__}): could not complete the request. "
              f"See logs/bot.log.", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        return 130
    finally:
        db.dispose()


if __name__ == "__main__":
    sys.exit(main())
