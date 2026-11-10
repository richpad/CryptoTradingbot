# Crypto Trading Bot - Stage 1 + Stage 2 (Paper Trading)

**Stage 1**: Data collection, indicators, Strategy A signals (no orders).  
**Stage 2**: Paper trading execution on Alpaca Paper with aggressive risk parameters.

Live trading is still deliberately blocked.

## Setup

```bash
python3.12 -m venv .venv
# activate it
pip install -r requirements.txt
```

Create `.env`:

```env
EXCHANGE_ID=alpaca
ALPACA_PAPER_API_KEY=your_paper_key
ALPACA_PAPER_API_SECRET=your_paper_secret
PAPER_TRADING_ENABLED=true          # required to enable paper orders
```

## Commands

```bash
python main.py backfill --days 250   # download history (needed for indicators)
python main.py collect               # keep candles + snapshots fresh
python main.py signals               # show Strategy A decisions (no orders)
python main.py status

# Stage 2 – paper trading
python main.py paper --once          # one cycle (recommended first)
python main.py paper                 # continuous paper trading loop
```

## Aggressive Risk Parameters (Stage 2)

- Risk per trade: **1.5%** of equity
- Max open positions: **6**
- Max portfolio heat: **6%**
- Daily loss limit: **4%** → stops new entries
- 4 consecutive losses → 2-hour pause
- Mandatory stop-loss on every entry
- ATR-based position sizing

## Dashboard

```bash
streamlit run dashboard/app.py
```

Still read-only for now (charts + signals). Full P&L / positions view comes later.

**Important**: This is paper trading only. Nothing here has been proven profitable. Always monitor the first sessions closely.
