"""Dashboard: market data + Strategy A + live Paper Portfolio / P&L.

Run from the project folder:   streamlit run dashboard/app.py
Keep `python main.py collect` (and optionally `python main.py paper`) running in other terminals.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from config import ConfigError, load_config, timeframe_ms
from database.database_manager import DatabaseManager
from indicators.scoring import score_frame
from indicators.technical_indicators import IndicatorParams, compute_indicators
from strategies.trend_following import TrendFollowing

st.set_page_config(page_title="Trading bot – Paper Dashboard", layout="wide")


@st.cache_resource
def get_db(url: str) -> DatabaseManager:
    db = DatabaseManager(url)
    db.init_schema()
    return db


@st.cache_data(ttl=10)
def load_coverage(url: str) -> list[dict]:
    return get_db(url).coverage(None)


@st.cache_data(ttl=10)
def load_candles(url: str, exchange: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    return get_db(url).get_candles(exchange, symbol, timeframe, limit=limit)


@st.cache_data(ttl=10)
def load_snapshots(url: str, exchange: str, symbol: str) -> pd.DataFrame:
    return get_db(url).snapshots_df(exchange, symbol, 500)


@st.cache_data(ttl=10)
def load_errors(url: str) -> pd.DataFrame:
    rows = get_db(url).get_errors(20)
    return pd.DataFrame(
        [{"time (UTC)": pd.to_datetime(r.ts_ms, unit="ms", utc=True), "component": r.component,
          "message": r.message} for r in rows])


def fetch_paper_portfolio(api_key: str, api_secret: str):
    """Live snapshot from Alpaca Paper. Returns None on any failure."""
    if not api_key or not api_secret:
        return None
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(api_key, api_secret, paper=True)
        account = client.get_account()
        positions = client.get_all_positions()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(getattr(account, "buying_power", account.cash) or account.cash),
            "last_equity": float(getattr(account, "last_equity", account.equity) or account.equity),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry": float(p.avg_entry_price),
                    "current": float(getattr(p, "current_price", p.avg_entry_price) or p.avg_entry_price),
                    "market_value": float(getattr(p, "market_value", 0) or 0),
                    "unrealized_pl": float(getattr(p, "unrealized_pl", 0) or 0),
                    "unrealized_plpc": float(getattr(p, "unrealized_plpc", 0) or 0) * 100,
                    "side": str(getattr(p, "side", "long")),
                }
                for p in positions if float(p.qty) != 0
            ],
        }
    except Exception as exc:
        return {"error": str(exc)[:300]}


EMA_COLORS = {"ema_9": "#f59e0b", "ema_21": "#3b82f6", "ema_50": "#8b5cf6", "ema_200": "#64748b"}


def candle_chart(df: pd.DataFrame, title: str, sigs: pd.DataFrame, emas: bool, bands: bool,
                 show_rsi: bool, markers: bool) -> go.Figure:
    rows = 3 if show_rsi else 2
    heights = [0.62, 0.17, 0.21] if show_rsi else [0.78, 0.22]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights, vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name="price"), row=1, col=1)
    if emas:
        for col, color in EMA_COLORS.items():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col.upper().replace("_", " "),
                                     line=dict(width=1.2, color=color)), row=1, col=1)
    if bands:
        for col in ("bb_upper", "bb_lower"):
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=col,
                                     line=dict(width=1, dash="dot", color="#94a3b8")), row=1, col=1)
    if markers:
        entry, leave = sigs["entry"], sigs["exit_cross"]
        fig.add_trace(go.Scatter(x=df.index[entry], y=df["low"][entry] * 0.998, mode="markers", name="Strategy A entry",
                                 marker=dict(symbol="triangle-up", size=13, color="#16a34a")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index[leave], y=df["high"][leave] * 1.002, mode="markers", name="EMA9<21 exit",
                                 marker=dict(symbol="triangle-down", size=11, color="#dc2626")), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="volume", marker_opacity=0.5), row=2, col=1)
    fig.update_yaxes(title_text="volume", row=2, col=1)
    if show_rsi:
        fig.add_trace(go.Scatter(x=df.index, y=df["rsi_14"], mode="lines", name="RSI 14",
                                 line=dict(width=1.2, color="#0ea5e9")), row=3, col=1)
        for level in (30, 70):
            fig.add_hline(y=level, line_dash="dot", line_color="#94a3b8", row=3, col=1)
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=3, col=1)
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, showlegend=emas or bands or markers,
                      margin=dict(l=10, r=10, t=50, b=10), height=680 if show_rsi else 560,
                      legend=dict(orientation="h", y=1.02, x=0))
    return fig


def render_portfolio(config) -> None:
    st.subheader("Paper Portfolio & P&L")
    if not config.paper_trading_enabled:
        st.info("Set `PAPER_TRADING_ENABLED=true` in `.env` and run `python main.py paper` to enable live portfolio view.")
        return

    data = fetch_paper_portfolio(config.alpaca_api_key, config.alpaca_api_secret)
    if data is None:
        st.warning("Alpaca paper keys missing – cannot load portfolio.")
        return
    if "error" in data:
        st.error(f"Could not reach Alpaca Paper: {data['error']}")
        return

    equity = data["equity"]
    cash = data["cash"]
    bp = data["buying_power"]
    last_eq = data["last_equity"]
    day_pnl = equity - last_eq
    day_pnl_pct = (day_pnl / last_eq * 100) if last_eq else 0.0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity", f"${equity:,.2f}")
    c2.metric("Cash", f"${cash:,.2f}")
    c3.metric("Buying Power", f"${bp:,.2f}")
    c4.metric("Day P&L", f"${day_pnl:+,.2f}", f"{day_pnl_pct:+.2f}%")
    c5.metric("Open Positions", str(len(data["positions"])))

    if not data["positions"]:
        st.caption("No open positions.")
        return

    rows = []
    total_unrealized = 0.0
    for p in data["positions"]:
        total_unrealized += p["unrealized_pl"]
        rows.append({
            "Symbol": p["symbol"],
            "Qty": f"{p['qty']:.6f}",
            "Avg Entry": f"{p['avg_entry']:,.4f}",
            "Current": f"{p['current']:,.4f}",
            "Market Value": f"${p['market_value']:,.2f}",
            "Unrealized P&L": f"${p['unrealized_pl']:+,.2f}",
            "Unrealized %": f"{p['unrealized_plpc']:+.2f}%",
            "Side": p["side"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"Total unrealized P&L: **${total_unrealized:+,.2f}**")


def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()
    url = config.database_url

    st.title("Trading Bot – Paper Dashboard")
    st.caption(f"Source: **{config.exchange_id}**  |  database: `{url}`  |  "
               f"paper trading: **{'ON' if config.paper_trading_enabled else 'OFF'}**  |  all times UTC")

    # ----- Portfolio section (live from Alpaca Paper) -----
    render_portfolio(config)
    st.divider()

    coverage = load_coverage(url)
    if not coverage:
        st.info("No candles stored yet. In a terminal run `python main.py backfill`, then reload this page.")
        st.stop()

    cov = pd.DataFrame(coverage)
    cov["first"] = pd.to_datetime(cov["first_ts_ms"], unit="ms", utc=True)
    cov["last"] = pd.to_datetime(cov["last_ts_ms"], unit="ms", utc=True)

    with st.sidebar:
        st.header("Series")
        exchanges = sorted(cov["exchange"].unique())
        default_ex = exchanges.index(config.exchange_id) if config.exchange_id in exchanges else 0
        exchange = st.selectbox("Source", exchanges, index=default_ex)
        sub = cov[cov["exchange"] == exchange]
        symbol = st.selectbox("Pair", sorted(sub["symbol"].unique()))
        tf_order = ["5m", "15m", "1h", "4h", "1d"]
        tfs = [t for t in tf_order if t in set(sub[sub["symbol"] == symbol]["timeframe"])]
        timeframe = st.selectbox("Timeframe", tfs, index=min(2, len(tfs) - 1))
        n_candles = st.slider("Candles shown", 50, 1000, 300, step=50)
        st.header("Overlays")
        show_emas = st.checkbox("EMAs (9 / 21 / 50 / 200)", value=True)
        show_bands = st.checkbox("Bollinger bands", value=False)
        show_rsi = st.checkbox("RSI panel", value=True)
        show_markers = st.checkbox("Strategy A markers", value=True)
        if st.button("Refresh now"):
            st.cache_data.clear()
            st.rerun()

    warmup = IndicatorParams().warmup_rows
    full = load_candles(url, exchange, symbol, timeframe, n_candles + warmup + 100)
    ind = compute_indicators(full)
    strat = TrendFollowing()
    sig_all = strat.signals(ind)
    df, sigs = ind.iloc[-n_candles:], sig_all.iloc[-n_candles:]
    st.subheader(f"{symbol} - {timeframe} ({exchange})")
    if df.empty:
        st.warning("No candles for this selection.")
    else:
        last, prev = df["close"].iloc[-1], df["close"].iloc[-2] if len(df) > 1 else df["close"].iloc[-1]
        age_min = (time.time() * 1000 - (df.index[-1].value // 1_000_000)) / 60_000
        width_min = timeframe_ms(timeframe) / 60_000
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Last close", f"{last:,.4f}", f"{(last / prev - 1) * 100:+.2f}% vs prev candle")
        c2.metric("Candles stored (series)", f"{int(sub[(sub['symbol'] == symbol) & (sub['timeframe'] == timeframe)]['count'].iloc[0]):,}")
        c3.metric("Newest candle opened", df.index[-1].strftime("%Y-%m-%d %H:%M"))
        stale = age_min > 2.5 * width_min
        c4.metric("Data freshness", "STALE" if stale else "OK",
                  f"newest opened {age_min / 60:.1f}h ago", delta_color="inverse" if stale else "off")
        decision = strat.evaluate(ind, symbol, timeframe, score_frame(ind)["score"])
        c5.metric("Strategy A (latest closed candle)", decision.action.value,
                  f"score {decision.score:+.2f}" if decision.score is not None else None, delta_color="off")
        if len(full) < warmup:
            st.warning(f"Indicators need {warmup} candles (EMA200); this series has {len(full)}. "
                       f"Run `python main.py backfill --days 250`. Lines and signals appear once there is enough history.")
        else:
            st.caption("Strategy A on this candle: " + "; ".join(decision.reasons))
        st.plotly_chart(candle_chart(df, f"{symbol} {timeframe}", sigs, show_emas, show_bands, show_rsi, show_markers),
                        use_container_width=True)
        with st.expander("Raw candles (latest first)"):
            st.dataframe(df.iloc[::-1])

    st.subheader("Live market snapshots (spread and depth)")
    snaps = load_snapshots(url, exchange, symbol)
    if snaps.empty:
        st.caption("No snapshots stored for this source/pair. Run `python main.py collect` to record them.")
    else:
        s = snaps.iloc[-1]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Bid", f"{s['bid']:,.4f}")
        m2.metric("Ask", f"{s['ask']:,.4f}")
        m3.metric("Spread", f"{s['spread_bps']:.2f} bps")
        m4.metric("Book imbalance", f"{s['imbalance']:+.2f}", help="+1 all bids, -1 all asks (top levels)")
        m5.metric("Snapshots stored (last 500)", f"{len(snaps):,}")
        st.line_chart(snaps["spread_bps"], height=200)

    st.subheader("Coverage - everything stored")
    st.dataframe(cov[["exchange", "symbol", "timeframe", "count", "first", "last"]], hide_index=True)

    st.subheader("Recent errors")
    errors = load_errors(url)
    if errors.empty:
        st.caption("No errors logged.")
    else:
        st.dataframe(errors, hide_index=True)


main()
