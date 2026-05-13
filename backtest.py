#!/usr/bin/env python3
"""
Multi-Symbol Backtest: UT Bot + HMA/EMA Strategy with Staged Exits
===================================================================

Strategy:
  - Indicators: UT Bot (ATR 10, K=2.5), HMA(100), EMA(200) via pandas-ta
  - Entry:  UT Bot Buy ENV (≥2 days)  AND  Close > HMA(100)  AND  Close > EMA(200)
            AND  ADX(14) > 20  AND  Volume > 20d SMA
            → execute at NEXT day's open price
            Position size = 5% of current equity (compounding, max 20 concurrent)
            Cooldown: 3 days after stop-out before re-entering same symbol
            Max 2 entries per buy-environment episode (same symbol)
  - Exit:   TP at 5×ATR from entry (adaptive per symbol)
            Breakeven stop: once profit ≥ 2.5×ATR, stop moves to entry price
  - Stop-loss: Close < HMA(100)  OR  Close < EMA(200)  OR  UT Bot Sell env

Usage:
    uv run backtest.py
"""

import os
import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from jinja2 import Template
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
DATA_DIR = Path("data")
OUTPUT_FILE = "backtest_report.html"
INITIAL_CAPITAL = 1_000_000.0
POSITION_PCT = 0.05          # 5 % per position → max 20 concurrent

ATR_PERIOD = 10
UT_K = 2.5
HMA_LENGTH = 100
EMA_LENGTH = 200

# ATR-adaptive profit-taking & breakeven stop
# TP target  = entry_price × (1 + TP_ATR_MULT × ATR / entry_price)
# BE trigger = entry_price × (1 + BE_ATR_MULT × ATR / entry_price)
TP_ATR_MULT = 4.0                   # take profit at 5× ATR from entry
BE_ATR_MULT = 2.0                   # activate breakeven stop at 2.5× ATR (half of TP)
BE_STOP_PCT = 0.0                   # breakeven stop at entry price (0 % profit)

# Entry filters
COOLDOWN_DAYS = 3                   # days to wait after stop-out before re-entering same symbol
BUY_ENV_MIN_DAYS = 2                # buy environment must persist ≥ N days before entry
MAX_ENTRIES_PER_ENV = 3             # max entries per buy-environment episode (same symbol)

REPORTS_DIR = Path("reports")       # per-symbol reports

# ETFs to exclude (only keep individual stocks for trading)
ETF_SYMBOLS = {
    "DBA", "DIA", "EEM", "EWY", "FNDA", "FNDC", "FNDX",
    "GLD", "HYG", "IBIT", "IGV", "IWM", "KWEB",
    "PRF", "PRFZ", "QQQ", "SCHG", "SCHI", "SLV",
    "SMH", "SOXL", "SOXX", "SPY", "SQQQ", "TLT",
    "TQQQ", "TSLL",
}

# Blacklisted Symbols
BLACKLIST_SYMBOLS = {
    "BMNR", "BAC", "CLSK", "ERNA", "GME", "MRVL", "QBTS", "QUBT", "POET", "SGHC", "MARA", "NOVT", "NVTS", "SMBS"
}

# ── File-ownership helper (when running under sudo) ──────────────────────────
_SUDO_UID = int(os.environ.get("SUDO_UID", -1))
_SUDO_GID = int(os.environ.get("SUDO_GID", -1))

def _fix_owner(path: Path) -> None:
    """chown *path* (and its parent dir) back to the real user when running via sudo."""
    if _SUDO_UID < 0:
        return
    try:
        os.chown(path, _SUDO_UID, _SUDO_GID)
        # Also fix the parent directory if we created it
        parent = path.parent
        if parent != Path(".") and parent.exists():
            os.chown(parent, _SUDO_UID, _SUDO_GID)
    except OSError:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# UT Bot Indicator
# ═══════════════════════════════════════════════════════════════════════════════
def calc_ut_bot(close: pd.Series, high: pd.Series, low: pd.Series,
                atr_period: int = ATR_PERIOD, key_value: float = UT_K):
    """
    UT Bot trailing-stop indicator (ATR-based).

    Returns
    -------
    trailing_stop : pd.Series
    buy_env       : pd.Series[bool]   close > trailing stop
    sell_env      : pd.Series[bool]   close < trailing stop
    buy_signal    : pd.Series[bool]   cross-over: close crosses above trailing stop
    """
    atr = ta.atr(high, low, close, length=atr_period)
    n_loss = key_value * atr

    n = len(close)
    ts = np.zeros(n)
    c_arr = close.values
    nl_arr = n_loss.values

    for i in range(1, n):
        if np.isnan(nl_arr[i]):
            ts[i] = ts[i - 1]
            continue
        prev = ts[i - 1]
        c, pc, nl = c_arr[i], c_arr[i - 1], nl_arr[i]

        if c > prev and pc > prev:
            ts[i] = max(prev, c - nl)
        elif c < prev and pc < prev:
            ts[i] = min(prev, c + nl)
        elif c > prev:
            ts[i] = c - nl
        else:
            ts[i] = c + nl

    trailing_stop = pd.Series(ts, index=close.index)
    buy_env = close > trailing_stop
    sell_env = close < trailing_stop
    # Buy signal: close crosses above the trailing stop (crossover)
    buy_signal = buy_env & (~buy_env.shift(1, fill_value=False))
    return trailing_stop, buy_env, sell_env, buy_signal, atr


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading & Indicator Computation
# ═══════════════════════════════════════════════════════════════════════════════
def load_all_data(exclude_etfs: bool = True) -> dict[str, pd.DataFrame]:
    """Load every *_25Y_daily.csv from DATA_DIR and compute indicators."""
    all_data: dict[str, pd.DataFrame] = {}
    csv_files = sorted(DATA_DIR.glob("*_25Y_daily.csv"))
    print(f"[data] Found {len(csv_files)} CSV files in {DATA_DIR}/")
    skipped_symbols: list[str] = []

    for f in csv_files:
        symbol = f.stem.replace("_25Y_daily", "")
        if exclude_etfs and symbol in ETF_SYMBOLS:
            skipped_symbols.append(symbol)
            continue
        if symbol in BLACKLIST_SYMBOLS:
            skipped_symbols.append(symbol)
            continue
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if len(df) < EMA_LENGTH + 60:
            continue

        # Indicators
        df["hma"] = ta.hma(df["close"], length=HMA_LENGTH)
        df["ema200"] = ta.ema(df["close"], length=EMA_LENGTH)
        ts, buy_env, sell_env, buy_signal, atr_series = calc_ut_bot(
            df["close"], df["high"], df["low"])
        df["ut_stop"] = ts
        df["ut_buy"] = buy_env
        df["ut_sell"] = sell_env
        df["ut_buy_signal"] = buy_signal
        df["atr"] = atr_series

        # ADX(14)
        adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
        df["adx"] = adx_df[f"ADX_14"]

        df = df.dropna(subset=["hma", "ema200", "adx"])
        if len(df) < 50:
            continue

        # Buy environment duration: consecutive days in buy env
        buy_streak = df["ut_buy"].astype(int)
        buy_streak = buy_streak.groupby(
            (~df["ut_buy"]).cumsum()
        ).cumsum()
        df["buy_env_days"] = buy_streak

        # Buy episode ID: increments each time a new buy-env period starts
        new_episode = df["ut_buy"] & (~df["ut_buy"].shift(1, fill_value=False))
        df["buy_episode_id"] = new_episode.cumsum()

        # Volume filter: today's volume > 20-day SMA of volume
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["vol_above_avg"] = df["volume"] > df["vol_sma20"]

        # Pre-computed signals
        # Entry: UT Bot buy env ≥ N days + close > HMA + close > EMA
        #        + ADX > 20 + volume above 20d avg
        # (actual buy happens at NEXT day's open)
        df["entry_ok"] = (
            df["ut_buy"]
            & (df["buy_env_days"] >= BUY_ENV_MIN_DAYS)
            & (df["close"] > df["hma"])
            & (df["close"] > df["ema200"])
            & (df["adx"] > 20)
            & df["vol_above_avg"]
        )
        df["stop_hit"] = (
            (df["close"] < df["hma"])
            | (df["close"] < df["ema200"])
            | df["ut_sell"]
        )
        all_data[symbol] = df

    print(f"[data] Excluded {len(skipped_symbols)} symbols: {', '.join(skipped_symbols)}")
    print(f"[data] {len(all_data)} stocks ready after indicator warm-up")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# Position Tracker
# ═══════════════════════════════════════════════════════════════════════════════
class Position:
    __slots__ = (
        "symbol", "entry_price", "entry_date",
        "initial_shares", "remaining_shares",
        "breakeven_active",
        "tp_pct", "be_trigger_pct",
    )

    def __init__(self, symbol, entry_price, entry_date, shares, entry_atr: float = 0.0):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.initial_shares = shares
        self.remaining_shares = shares
        self.breakeven_active = False
        # ATR-adaptive thresholds (computed once at entry)
        atr_pct = entry_atr / entry_price if entry_price > 0 else 0.05
        self.tp_pct = TP_ATR_MULT * atr_pct          # e.g. 5 × 3% = 15%
        self.be_trigger_pct = BE_ATR_MULT * atr_pct   # e.g. 2.5 × 3% = 7.5%


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Engine
# ═══════════════════════════════════════════════════════════════════════════════
def run_backtest(all_data: dict[str, pd.DataFrame]):
    """
    Run the event-driven portfolio simulation.

    Entry signals fire at day-T close; execution happens at day-(T+1) open.
    Exits (stop-loss & profit-taking) are evaluated and executed at close.

    Returns
    -------
    equity_df : DataFrame  (date index → equity, cash, invested, n_positions)
    trade_df  : DataFrame  (trade log)
    """
    all_dates = sorted(set().union(*(df.index for df in all_data.values())))
    print(f"[sim]  Period {all_dates[0].date()} → {all_dates[-1].date()}  "
          f"({len(all_dates)} bars, {len(all_data)} symbols)")

    cash = INITIAL_CAPITAL
    positions: dict[str, Position] = {}

    equity_records: list[dict] = []
    trade_log: list[dict] = []

    # Pending entries: signals that fired yesterday, to be filled at today's open
    pending_entries: set[str] = set()
    # Cooldown tracker: sym → earliest date allowed to re-enter
    cooldown_until: dict[str, pd.Timestamp] = {}
    # Per buy-environment episode entry counter: (sym, episode_id) → count
    episode_entries: dict[tuple[str, int], int] = {}

    for date in all_dates:
        closed_today: set[str] = set()

        # ── Phase 0: Execute pending entries at today's OPEN ──────────────
        # Compounding: compute current equity at today's open and size each
        # position as POSITION_PCT of that equity.
        if pending_entries:
            pos_value_open = sum(
                pos.remaining_shares * (
                    all_data[s].loc[date, "open"]
                    if date in all_data[s].index
                    else pos.entry_price
                )
                for s, pos in positions.items()
            )
            current_equity = cash + pos_value_open
            alloc_amount = current_equity * POSITION_PCT

        for sym in sorted(pending_entries):
            if sym in positions:
                continue
            df = all_data[sym]
            if date not in df.index:
                continue
            open_price = df.loc[date, "open"]
            if open_price <= 0 or np.isnan(open_price):
                continue

            budget = min(alloc_amount, cash)
            if budget < 100:
                continue
            shares = int(budget / open_price)
            if shares <= 0:
                continue

            cost = shares * open_price
            cash -= cost
            entry_atr = df.loc[date, "atr"] if not np.isnan(df.loc[date, "atr"]) else 0.0
            positions[sym] = Position(sym, open_price, date, shares, entry_atr)
            trade_log.append(dict(
                symbol=sym, date=date, side="BUY",
                shares=shares, price=open_price,
                reason="ENTRY", pnl_pct=0.0,
                entry_price=open_price, entry_date=date,
            ))
            # Increment episode entry counter
            ep_id = int(df.loc[date, "buy_episode_id"])
            key = (sym, ep_id)
            episode_entries[key] = episode_entries.get(key, 0) + 1
        # Clear all pending (one-shot signals, execute or discard)
        pending_entries.clear()

        # ── Phase 1: Exits at today's CLOSE ───────────────────────────────
        to_remove: list[str] = []
        for sym, pos in positions.items():
            df = all_data[sym]
            if date not in df.index:
                continue
            row = df.loc[date]
            price = row["close"]
            pnl_pct = (price - pos.entry_price) / pos.entry_price

            # Activate breakeven stop once profit ≥ ATR-based trigger
            if not pos.breakeven_active and pnl_pct >= pos.be_trigger_pct:
                pos.breakeven_active = True

            # ─ Breakeven stop: once activated, if price drops back to entry → exit
            if pos.breakeven_active and pnl_pct <= BE_STOP_PCT:
                sell_n = pos.remaining_shares
                if sell_n > 0:
                    cash += sell_n * price
                    trade_log.append(dict(
                        symbol=sym, date=date, side="SELL",
                        shares=sell_n, price=price,
                        reason="BE_STOP", pnl_pct=pnl_pct,
                        entry_price=pos.entry_price, entry_date=pos.entry_date,
                    ))
                    pos.remaining_shares = 0
                to_remove.append(sym)
                closed_today.add(sym)
                cooldown_until[sym] = date + pd.Timedelta(days=COOLDOWN_DAYS)
                continue

            # ─ Normal stop-loss
            if row["stop_hit"]:
                sell_n = pos.remaining_shares
                if sell_n > 0:
                    cash += sell_n * price
                    trade_log.append(dict(
                        symbol=sym, date=date, side="SELL",
                        shares=sell_n, price=price,
                        reason="STOP", pnl_pct=pnl_pct,
                        entry_price=pos.entry_price, entry_date=pos.entry_date,
                    ))
                    pos.remaining_shares = 0
                to_remove.append(sym)
                closed_today.add(sym)
                cooldown_until[sym] = date + pd.Timedelta(days=COOLDOWN_DAYS)
                continue

            # ─ ATR-adaptive take profit → sell all
            if pnl_pct >= pos.tp_pct:
                sell_n = pos.remaining_shares
                if sell_n > 0:
                    cash += sell_n * price
                    trade_log.append(dict(
                        symbol=sym, date=date, side="SELL",
                        shares=sell_n, price=price,
                        reason=f"TP_{pos.tp_pct*100:.0f}%", pnl_pct=pnl_pct,
                        entry_price=pos.entry_price, entry_date=pos.entry_date,
                    ))
                    pos.remaining_shares = 0
                to_remove.append(sym)
                closed_today.add(sym)

        for sym in to_remove:
            positions.pop(sym, None)

        # ── Phase 2: Generate entry SIGNALS at close (fill tomorrow) ──────
        for sym in sorted(all_data.keys()):
            if sym in positions or sym in closed_today:
                continue
            # Cooldown check
            if sym in cooldown_until and date < cooldown_until[sym]:
                continue
            df = all_data[sym]
            if date not in df.index:
                continue
            if df.loc[date, "entry_ok"]:
                # Episode entry limit
                ep_id = int(df.loc[date, "buy_episode_id"])
                if episode_entries.get((sym, ep_id), 0) >= MAX_ENTRIES_PER_ENV:
                    continue
                pending_entries.add(sym)

        # ── Phase 3: Record equity ───────────────────────────────────────
        pos_value = sum(
            pos.remaining_shares * (
                all_data[sym].loc[date, "close"]
                if date in all_data[sym].index
                else pos.entry_price
            )
            for sym, pos in positions.items()
        )
        equity_records.append(dict(
            date=date, equity=cash + pos_value,
            cash=cash, invested=pos_value,
            n_positions=len(positions),
        ))

    equity_df = pd.DataFrame(equity_records).set_index("date")
    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
        columns=["symbol", "date", "side", "shares", "price",
                 "reason", "pnl_pct", "entry_price", "entry_date"]
    )
    print(f"[sim]  Done. Final equity ${equity_df['equity'].iloc[-1]:,.0f}  "
          f"| {len(trade_df)} order records")
    return equity_df, trade_df


# ═══════════════════════════════════════════════════════════════════════════════
# Round-Trip Trade Builder
# ═══════════════════════════════════════════════════════════════════════════════
def build_round_trips(trade_df: pd.DataFrame) -> list[dict]:
    """Match BUY→SELL* sequences into round-trip trade records."""
    if trade_df.empty:
        return []

    trips: list[dict] = []
    for symbol in trade_df["symbol"].unique():
        sym_t = (
            trade_df[trade_df["symbol"] == symbol]
            .sort_values("date")
            .reset_index(drop=True)
        )
        i = 0
        while i < len(sym_t):
            if sym_t.iloc[i]["side"] != "BUY":
                i += 1
                continue
            entry = sym_t.iloc[i]
            entry_cost = entry["shares"] * entry["price"]
            proceeds = 0.0
            last_sell = None
            j = i + 1
            while j < len(sym_t) and sym_t.iloc[j]["side"] == "SELL":
                s = sym_t.iloc[j]
                proceeds += s["shares"] * s["price"]
                last_sell = s
                j += 1
            if last_sell is not None:
                pnl = proceeds - entry_cost
                trips.append(dict(
                    symbol=symbol,
                    entry_date=entry["date"],
                    exit_date=last_sell["date"],
                    entry_price=entry["price"],
                    exit_reason=last_sell["reason"],
                    initial_shares=int(entry["shares"]),
                    entry_cost=entry_cost,
                    proceeds=proceeds,
                    pnl=pnl,
                    pnl_pct=pnl / entry_cost if entry_cost else 0,
                    hold_days=(last_sell["date"] - entry["date"]).days,
                ))
            i = j
    return trips


# ═══════════════════════════════════════════════════════════════════════════════
# SPY Buy-and-Hold Benchmark
# ═══════════════════════════════════════════════════════════════════════════════
def load_spy_benchmark(backtest_start, backtest_end) -> pd.DataFrame | None:
    """Load SPY data and compute buy-and-hold equity over the backtest period."""
    spy_path = DATA_DIR / "SPY_25Y_daily.csv"
    if not spy_path.exists():
        print("[bench] SPY_25Y_daily.csv not found — skipping benchmark")
        return None

    df = pd.read_csv(spy_path, parse_dates=["date"], index_col="date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df[(df.index >= backtest_start) & (df.index <= backtest_end)]
    if df.empty:
        return None

    initial_price = df["close"].iloc[0]
    shares = INITIAL_CAPITAL / initial_price
    df["spy_equity"] = shares * df["close"]
    print(f"[bench] SPY B&H loaded: {df.index[0].date()} → {df.index[-1].date()}, "
          f"final ${df['spy_equity'].iloc[-1]:,.0f}")
    return df


def _spy_stats(spy_df: pd.DataFrame) -> dict:
    """Compute basic stats for the SPY buy-and-hold benchmark."""
    eq = spy_df["spy_equity"]
    rets = eq.pct_change().dropna()
    total_days = (eq.index[-1] - eq.index[0]).days
    total_years = max(total_days / 365.25, 1e-6)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / total_years) - 1
    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min()
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    return dict(
        spy_total_return=total_ret, spy_cagr=cagr,
        spy_max_dd=max_dd, spy_ann_vol=ann_vol, spy_sharpe=sharpe,
        spy_final_equity=eq.iloc[-1],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Performance Analytics (vectorbt + manual)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_overall_stats(equity_df: pd.DataFrame, round_trips: list[dict],
                          spy_df: pd.DataFrame | None = None) -> dict:
    """Compute portfolio-level performance metrics."""
    equity = equity_df["equity"]
    rets = equity.pct_change().dropna()

    total_days = (equity.index[-1] - equity.index[0]).days
    total_years = max(total_days / 365.25, 1e-6)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / total_years) - 1

    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    ann_vol = rets.std() * np.sqrt(252)
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    down = rets[rets < 0]
    sortino = (rets.mean() / down.std() * np.sqrt(252)
               if len(down) > 0 and down.std() > 0 else 0)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # vectorbt returns analytics
    try:
        vbt_rets = rets.vbt.returns(freq="1D")
        vbt_sharpe = vbt_rets.sharpe_ratio()
        vbt_sortino = vbt_rets.sortino_ratio()
        vbt_max_dd = vbt_rets.max_drawdown()
    except Exception:
        vbt_sharpe = vbt_sortino = vbt_max_dd = None

    n_rt = len(round_trips)
    if n_rt > 0:
        wins = [t for t in round_trips if t["pnl"] > 0]
        losses = [t for t in round_trips if t["pnl"] <= 0]
        win_rate = len(wins) / n_rt
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_hold = np.mean([t["hold_days"] for t in round_trips])
        best_trade = max(round_trips, key=lambda t: t["pnl_pct"])
        worst_trade = min(round_trips, key=lambda t: t["pnl_pct"])
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_hold = 0
        best_trade = worst_trade = None

    result = dict(
        start_date=equity.index[0].strftime("%Y-%m-%d"),
        end_date=equity.index[-1].strftime("%Y-%m-%d"),
        total_years=total_years,
        initial_capital=INITIAL_CAPITAL,
        final_equity=equity.iloc[-1],
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_dd,
        annual_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        vbt_sharpe=vbt_sharpe,
        vbt_sortino=vbt_sortino,
        vbt_max_dd=vbt_max_dd,
        n_round_trips=n_rt,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=profit_factor,
        avg_hold_days=avg_hold,
        best_trade=best_trade,
        worst_trade=worst_trade,
    )

    # SPY benchmark comparison
    if spy_df is not None:
        spy = _spy_stats(spy_df)
        result.update(spy)
        result["alpha"] = cagr - spy["spy_cagr"]
    else:
        result.update(dict(
            spy_total_return=None, spy_cagr=None, spy_max_dd=None,
            spy_ann_vol=None, spy_sharpe=None, spy_final_equity=None, alpha=None,
        ))
    return result


def compute_per_symbol_stats(round_trips: list[dict]) -> pd.DataFrame:
    if not round_trips:
        return pd.DataFrame()
    rt = pd.DataFrame(round_trips)
    ps = rt.groupby("symbol").agg(
        n_trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl_pct=("pnl_pct", "mean"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        avg_hold_days=("hold_days", "mean"),
        max_win_pct=("pnl_pct", "max"),
        max_loss_pct=("pnl_pct", "min"),
        total_entry_cost=("entry_cost", "sum"),
    ).reset_index()
    ps["return_pct"] = ps["total_pnl"] / ps["total_entry_cost"]
    return ps.sort_values("total_pnl", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Symbol Full-Position Backtest
# ═══════════════════════════════════════════════════════════════════════════════
def run_single_backtest(df: pd.DataFrame, symbol: str,
                        initial_capital: float = INITIAL_CAPITAL):
    """
    Backtest a single symbol with 100 % allocation (full position).

    Returns  (equity_df, trade_df, round_trips)
    """
    dates = df.index
    cash = initial_capital
    pos: Position | None = None
    pending = False        # signal fired at close, buy at next open
    cooldown_end: pd.Timestamp | None = None  # cooldown after stop-out
    episode_entries: dict[int, int] = {}       # episode_id → entry count

    equity_records: list[dict] = []
    trade_log: list[dict] = []

    for i, date in enumerate(dates):
        # ── Phase 0: Execute pending entry at today's open ────────────────
        if pending and pos is None:
            op = df.loc[date, "open"]
            if op > 0 and not np.isnan(op):
                shares = int(cash / op)
                if shares > 0:
                    cost = shares * op
                    cash -= cost
                    entry_atr = df.loc[date, "atr"] if not np.isnan(df.loc[date, "atr"]) else 0.0
                    pos = Position(symbol, op, date, shares, entry_atr)
                    trade_log.append(dict(
                        symbol=symbol, date=date, side="BUY",
                        shares=shares, price=op,
                        reason="ENTRY", pnl_pct=0.0,
                        entry_price=op, entry_date=date,
                    ))
                    ep_id = int(df.loc[date, "buy_episode_id"])
                    episode_entries[ep_id] = episode_entries.get(ep_id, 0) + 1
            pending = False

        # ── Phase 1: Exits at close ───────────────────────────────────────
        close_price = df.loc[date, "close"]
        if pos is not None and pos.remaining_shares > 0:
            pnl_pct = (close_price - pos.entry_price) / pos.entry_price
            closed = False

            # Activate breakeven stop (ATR-adaptive trigger)
            if not pos.breakeven_active and pnl_pct >= pos.be_trigger_pct:
                pos.breakeven_active = True

            # Breakeven stop
            if pos.breakeven_active and pnl_pct <= BE_STOP_PCT:
                sell_n = pos.remaining_shares
                cash += sell_n * close_price
                trade_log.append(dict(
                    symbol=symbol, date=date, side="SELL",
                    shares=sell_n, price=close_price,
                    reason="BE_STOP", pnl_pct=pnl_pct,
                    entry_price=pos.entry_price, entry_date=pos.entry_date,
                ))
                pos.remaining_shares = 0
                closed = True
                cooldown_end = date + pd.Timedelta(days=COOLDOWN_DAYS)

            # Stop-loss
            elif df.loc[date, "stop_hit"]:
                sell_n = pos.remaining_shares
                cash += sell_n * close_price
                trade_log.append(dict(
                    symbol=symbol, date=date, side="SELL",
                    shares=sell_n, price=close_price,
                    reason="STOP", pnl_pct=pnl_pct,
                    entry_price=pos.entry_price, entry_date=pos.entry_date,
                ))
                pos.remaining_shares = 0
                closed = True
                cooldown_end = date + pd.Timedelta(days=COOLDOWN_DAYS)

            # ATR-adaptive take profit → sell all
            elif pnl_pct >= pos.tp_pct:
                sell_n = pos.remaining_shares
                cash += sell_n * close_price
                trade_log.append(dict(
                    symbol=symbol, date=date, side="SELL",
                    shares=sell_n, price=close_price,
                    reason=f"TP_{pos.tp_pct*100:.0f}%", pnl_pct=pnl_pct,
                    entry_price=pos.entry_price, entry_date=pos.entry_date,
                ))
                pos.remaining_shares = 0
                closed = True

            if closed:
                pos = None

        # ── Phase 2: Generate entry signal at close ───────────────────────
        pending = False
        if pos is None and df.loc[date, "entry_ok"]:
            # Respect cooldown
            if cooldown_end is not None and date < cooldown_end:
                pass
            else:
                # Episode entry limit
                ep_id = int(df.loc[date, "buy_episode_id"])
                if episode_entries.get(ep_id, 0) < MAX_ENTRIES_PER_ENV:
                    pending = True

        # ── Phase 3: Record equity ────────────────────────────────────────
        held_value = (pos.remaining_shares * close_price) if pos else 0.0
        equity_records.append(dict(date=date, equity=cash + held_value,
                                   cash=cash, invested=held_value))

    equity_df = pd.DataFrame(equity_records).set_index("date")
    trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(
        columns=["symbol", "date", "side", "shares", "price",
                 "reason", "pnl_pct", "entry_price", "entry_date"])
    rt = build_round_trips(trade_df)
    return equity_df, trade_df, rt


# ═══════════════════════════════════════════════════════════════════════════════
# Plotly Chart Generators
# ═══════════════════════════════════════════════════════════════════════════════
_PLOTLY_CFG = dict(full_html=False, include_plotlyjs=False)
_COLORS = dict(up="#26a69a", down="#ef5350", blue="#2196f3",
               gray="#9e9e9e", orange="#ff9800")


def _chart_equity(equity_df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> str:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
        subplot_titles=("Portfolio Equity Curve", "Active Positions"),
    )
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["equity"],
        name="Strategy", line=dict(color=_COLORS["blue"], width=1.5),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
    ), row=1, col=1)
    if spy_df is not None:
        fig.add_trace(go.Scatter(
            x=spy_df.index, y=spy_df["spy_equity"],
            name="SPY B&H", line=dict(color=_COLORS["orange"], width=1.5, dash="dash"),
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["cash"],
        name="Cash", line=dict(color=_COLORS["gray"], width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=equity_df.index, y=equity_df["n_positions"],
        name="# Positions", marker_color=_COLORS["orange"], opacity=0.6,
    ), row=2, col=1)
    fig.update_layout(
        height=560, margin=dict(l=50, r=30, t=40, b=30),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    return pio.to_html(fig, **_PLOTLY_CFG)


def _chart_drawdown(equity_df: pd.DataFrame) -> str:
    equity = equity_df["equity"]
    dd = (equity - equity.cummax()) / equity.cummax()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values * 100,
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
        line=dict(color=_COLORS["down"], width=1),
        name="Drawdown %",
    ))
    fig.update_layout(
        title="Portfolio Drawdown",
        height=320, margin=dict(l=50, r=30, t=40, b=30),
        yaxis_title="Drawdown %", hovermode="x unified",
    )
    return pio.to_html(fig, **_PLOTLY_CFG)


def _chart_monthly_returns(equity_df: pd.DataFrame) -> str:
    equity = equity_df["equity"]
    monthly = equity.resample("ME").last().pct_change().dropna()
    if monthly.empty:
        return "<p>No monthly data available.</p>"
    mdf = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "ret": monthly.values * 100,
    })
    pivot = mdf.pivot_table(index="year", columns="month", values="ret", aggfunc="sum")
    pivot = pivot.reindex(columns=range(1, 13))
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=month_labels,
        y=pivot.index.astype(str),
        colorscale=[[0, _COLORS["down"]], [0.5, "white"], [1, _COLORS["up"]]],
        zmid=0,
        text=np.where(np.isnan(pivot.values), "",
                      np.vectorize(lambda v: f"{v:.1f}%")(pivot.values)),
        texttemplate="%{text}",
        hovertemplate="Year %{y}, %{x}: %{z:.2f}%<extra></extra>",
    ))
    # Annual column
    annual = mdf.groupby("year")["ret"].sum()
    fig.add_trace(go.Heatmap(
        z=annual.values.reshape(-1, 1),
        x=["Annual"],
        y=annual.index.astype(str),
        colorscale=[[0, _COLORS["down"]], [0.5, "white"], [1, _COLORS["up"]]],
        zmid=0,
        text=np.vectorize(lambda v: f"{v:.1f}%")(annual.values.reshape(-1, 1)),
        texttemplate="%{text}",
        showscale=False,
        hovertemplate="Year %{y}: %{z:.2f}%<extra></extra>",
        xaxis="x2", yaxis="y",
    ))
    fig.update_layout(
        title="Monthly Returns (%)",
        height=max(350, len(pivot) * 26 + 100),
        margin=dict(l=50, r=30, t=40, b=30),
        xaxis=dict(domain=[0, 0.85], side="top"),
        xaxis2=dict(domain=[0.87, 1.0], side="top"),
        yaxis=dict(autorange="reversed"),
    )
    return pio.to_html(fig, **_PLOTLY_CFG)


def _chart_symbol_pnl(per_sym: pd.DataFrame) -> str:
    if per_sym.empty:
        return "<p>No per-symbol data.</p>"
    df = per_sym.sort_values("total_pnl")
    colors = [_COLORS["up"] if v > 0 else _COLORS["down"] for v in df["total_pnl"]]
    fig = go.Figure(go.Bar(
        x=df["total_pnl"], y=df["symbol"],
        orientation="h", marker_color=colors,
        text=[f"${v:,.0f}" for v in df["total_pnl"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Total P&L by Symbol",
        height=max(400, len(df) * 22 + 80),
        margin=dict(l=70, r=80, t=40, b=30),
        xaxis_title="P&L (USD)", yaxis=dict(dtick=1),
    )
    return pio.to_html(fig, **_PLOTLY_CFG)


def _chart_trade_distribution(round_trips: list[dict]) -> str:
    if not round_trips:
        return "<p>No trades.</p>"
    pnl_pcts = [t["pnl_pct"] * 100 for t in round_trips]
    colors = [_COLORS["up"] if v > 0 else _COLORS["down"] for v in pnl_pcts]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("P&L Distribution (%)", "Holding Period (Days)"))
    fig.add_trace(go.Histogram(
        x=pnl_pcts, nbinsx=50, marker_color=_COLORS["blue"], opacity=0.7, name="P&L %",
    ), row=1, col=1)
    hold_days = [t["hold_days"] for t in round_trips]
    fig.add_trace(go.Histogram(
        x=hold_days, nbinsx=40, marker_color=_COLORS["orange"], opacity=0.7, name="Days",
    ), row=1, col=2)
    fig.update_layout(height=350, margin=dict(l=50, r=30, t=40, b=30), showlegend=False)
    return pio.to_html(fig, **_PLOTLY_CFG)


def _chart_yearly_returns(equity_df: pd.DataFrame,
                          spy_df: pd.DataFrame | None = None) -> str:
    equity = equity_df["equity"]
    yearly = equity.resample("YE").last().pct_change().dropna()
    if yearly.empty:
        return ""
    years = yearly.index.year.astype(str)

    fig = go.Figure()
    # Strategy bars
    colors = [_COLORS["up"] if v > 0 else _COLORS["down"] for v in yearly.values]
    fig.add_trace(go.Bar(
        name="Strategy", x=years, y=yearly.values * 100,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in yearly.values * 100],
        textposition="outside",
    ))
    # SPY bars (side-by-side)
    if spy_df is not None:
        spy_yearly = spy_df["spy_equity"].resample("YE").last().pct_change().dropna()
        spy_years = spy_yearly.index.year.astype(str)
        spy_colors = ["rgba(33,150,243,0.6)" if v > 0 else "rgba(33,150,243,0.35)"
                      for v in spy_yearly.values]
        fig.add_trace(go.Bar(
            name="SPY B&H", x=spy_years, y=spy_yearly.values * 100,
            marker_color=spy_colors,
            text=[f"{v:.1f}%" for v in spy_yearly.values * 100],
            textposition="outside",
        ))
        fig.update_layout(barmode="group")

    fig.update_layout(
        title="Annual Returns — Strategy vs SPY Buy & Hold",
        height=400, margin=dict(l=50, r=30, t=40, b=30),
        yaxis_title="Return %",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
    )
    return pio.to_html(fig, **_PLOTLY_CFG)


def _chart_exit_reasons(round_trips: list[dict]) -> str:
    if not round_trips:
        return ""
    # Consolidate ATR-adaptive TP_xx% variants into a single "TP" category
    raw = [t["exit_reason"] for t in round_trips]
    consolidated = [r if not r.startswith("TP_") else "TP" for r in raw]
    reasons = pd.Series(consolidated).value_counts()
    colors = {"STOP": "#ef5350", "TP": "#26a69a", "BE_STOP": "#ff9800"}
    marker_colors = [colors.get(r, "#9e9e9e") for r in reasons.index]
    fig = go.Figure(go.Pie(
        labels=reasons.index, values=reasons.values,
        hole=0.4, textinfo="label+percent+value",
        marker_colors=marker_colors,
    ))
    fig.update_layout(title="Exit Reason Breakdown", height=350,
                      margin=dict(l=30, r=30, t=40, b=30))
    return pio.to_html(fig, **_PLOTLY_CFG)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Symbol OHLC Chart (American Bar Style)
# ═══════════════════════════════════════════════════════════════════════════════
def _chart_single_symbol(df: pd.DataFrame, symbol: str,
                         trade_df: pd.DataFrame) -> str:
    """
    Full OHLC chart with UT Bot trailing stop, HMA, EMA, and entry/exit markers.
    Uses go.Ohlc (American bar style) instead of go.Candlestick.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.80, 0.20],
        vertical_spacing=0.03,
        subplot_titles=(f"{symbol} — OHLC with Indicators", "Volume"),
    )

    # ── Row 1: OHLC bars (American style) ─────────────────────────────────
    fig.add_trace(go.Ohlc(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # UT Bot trailing stop
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ut_stop"], name="UT Bot Stop",
        line=dict(color="#ff9800", width=1, dash="dot"), opacity=0.85,
    ), row=1, col=1)

    # HMA 100
    fig.add_trace(go.Scatter(
        x=df.index, y=df["hma"], name="HMA 100",
        line=dict(color="#2196f3", width=1.3),
    ), row=1, col=1)

    # EMA 200
    fig.add_trace(go.Scatter(
        x=df.index, y=df["ema200"], name="EMA 200",
        line=dict(color="#9c27b0", width=1.3),
    ), row=1, col=1)

    # Entry / Exit markers
    if not trade_df.empty:
        buys = trade_df[trade_df["side"] == "BUY"]
        sells = trade_df[trade_df["side"] == "SELL"]
        if not buys.empty:
            fig.add_trace(go.Scatter(
                x=buys["date"], y=buys["price"], mode="markers",
                name="Buy",
                marker=dict(symbol="triangle-up", size=10,
                            color="#26a69a", line=dict(width=1, color="white")),
            ), row=1, col=1)
        if not sells.empty:
            # color by reason
            sell_colors = []
            for r in sells["reason"]:
                if "STOP" in r:
                    sell_colors.append("#ef5350")
                elif "TP3" in r:
                    sell_colors.append("#4caf50")
                elif "TP2" in r:
                    sell_colors.append("#8bc34a")
                else:
                    sell_colors.append("#cddc39")
            fig.add_trace(go.Scatter(
                x=sells["date"], y=sells["price"], mode="markers",
                name="Sell",
                marker=dict(symbol="triangle-down", size=9,
                            color=sell_colors,
                            line=dict(width=1, color="white")),
                text=sells["reason"], hovertemplate="%{x}<br>$%{y:.2f}<br>%{text}",
            ), row=1, col=1)

    # ── Row 2: Volume ─────────────────────────────────────────────────────
    vol_colors = np.where(df["close"] >= df["open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color=vol_colors, opacity=0.5, showlegend=False,
    ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        height=700,
        margin=dict(l=50, r=30, t=40, b=30),
        legend=dict(orientation="h", y=1.015, x=0.5, xanchor="center"),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=True,
        xaxis2_rangeslider_thickness=0.05,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    return pio.to_html(fig, **_PLOTLY_CFG)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Symbol HTML Report
# ═══════════════════════════════════════════════════════════════════════════════
_SINGLE_TPL = Template(r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ symbol }} — Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg:#f8f9fa;--card:#fff;--border:#e0e0e0;--text:#212121;
          --muted:#757575;--up:#26a69a;--down:#ef5350;--blue:#2196f3;--r:8px; }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:var(--bg);color:var(--text);max-width:1440px;margin:0 auto;
       padding:24px;line-height:1.5}
  h1{font-size:1.8rem;margin-bottom:4px}
  h2{font-size:1.3rem;margin:28px 0 14px;padding-bottom:6px;
     border-bottom:2px solid var(--blue)}
  .sub{color:var(--muted);margin-bottom:20px}
  .metrics{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));
           gap:10px;margin-bottom:20px}
  .card{background:var(--card);border:1px solid var(--border);
        border-radius:var(--r);padding:14px}
  .card .label{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .card .value{font-size:1.4rem;font-weight:700;margin-top:3px}
  .card .value.pos{color:var(--up)} .card .value.neg{color:var(--down)}
  .chart-box{background:var(--card);border:1px solid var(--border);
             border-radius:var(--r);padding:10px;margin-bottom:18px;overflow-x:auto}
  table{width:100%;border-collapse:collapse;font-size:.82rem}
  th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--border)}
  th{background:#f1f3f5;font-weight:600;position:sticky;top:0}
  td:first-child,th:first-child{text-align:left}
  tr:hover td{background:#f5f5f5}
  .tbl-wrap{max-height:500px;overflow-y:auto;border:1px solid var(--border);border-radius:var(--r)}
  a.back{display:inline-block;margin-bottom:16px;color:var(--blue);text-decoration:none;font-weight:600}
  a.back:hover{text-decoration:underline}
  footer{margin-top:32px;padding-top:12px;border-top:1px solid var(--border);
         color:var(--muted);font-size:.78rem;text-align:center}
</style>
</head>
<body>
<a class="back" href="../backtest_report.html">&larr; Back to Portfolio Report</a>
<h1>{{ symbol }}</h1>
<p class="sub">{{ start }} &rarr; {{ end }} ({{ n_bars }} bars, {{ "%.1f"|format(years) }} years)
 &mdash; Full position backtest &mdash; Initial ${{ "{:,.0f}".format(capital) }}</p>

<h2>Performance</h2>
<div class="metrics">
  <div class="card"><div class="label">Total Return</div>
    <div class="value {{ 'pos' if total_ret>=0 else 'neg' }}">{{ "%.2f"|format(total_ret*100) }}%</div></div>
  <div class="card"><div class="label">CAGR</div>
    <div class="value {{ 'pos' if cagr>=0 else 'neg' }}">{{ "%.2f"|format(cagr*100) }}%</div></div>
  <div class="card"><div class="label">Max Drawdown</div>
    <div class="value neg">{{ "%.2f"|format(max_dd*100) }}%</div></div>
  <div class="card"><div class="label">Sharpe</div>
    <div class="value">{{ "%.2f"|format(sharpe) }}</div></div>
  <div class="card"><div class="label">Sortino</div>
    <div class="value">{{ "%.2f"|format(sortino) }}</div></div>
  <div class="card"><div class="label">Final Equity</div>
    <div class="value">${{ "{:,.0f}".format(final_eq) }}</div></div>
  <div class="card"><div class="label">Trades</div>
    <div class="value">{{ n_trades }}</div></div>
  <div class="card"><div class="label">Win Rate</div>
    <div class="value">{{ "%.0f"|format(win_rate*100) }}%</div></div>
  <div class="card"><div class="label">Profit Factor</div>
    <div class="value">{{ "%.2f"|format(pf) if pf < 1e6 else "∞" }}</div></div>
  <div class="card"><div class="label">Avg Hold Days</div>
    <div class="value">{{ "%.1f"|format(avg_hold) }}</div></div>
</div>

<h2>OHLC Chart with Indicators &amp; Signals</h2>
<div class="chart-box">{{ ohlc_chart }}</div>

<h2>Equity Curve &amp; Drawdown</h2>
<div class="chart-box">{{ eq_dd_chart }}</div>

<h2>Trade Log ({{ n_trades }} round-trips)</h2>
<div class="tbl-wrap">
<table>
<thead><tr>
  <th>Entry</th><th>Exit</th><th>Hold</th><th>Entry $</th>
  <th>Exit Reason</th><th>P&amp;L %</th><th>P&amp;L $</th>
</tr></thead>
<tbody>
{% for t in trades %}
<tr>
  <td>{{ t.entry_date.strftime('%Y-%m-%d') }}</td>
  <td>{{ t.exit_date.strftime('%Y-%m-%d') }}</td>
  <td>{{ t.hold_days }}d</td>
  <td>${{ "%.2f"|format(t.entry_price) }}</td>
  <td>{{ t.exit_reason }}</td>
  <td style="color:{{ '#26a69a' if t.pnl_pct>=0 else '#ef5350' }}">{{ "%.2f"|format(t.pnl_pct*100) }}%</td>
  <td style="color:{{ '#26a69a' if t.pnl>=0 else '#ef5350' }}">${{ "{:,.0f}".format(t.pnl) }}</td>
</tr>
{% endfor %}
</tbody></table></div>

<footer>Generated by backtest.py &mdash; {{ symbol }} full-position backtest</footer>
</body></html>
""")


def _chart_eq_dd_single(equity_df: pd.DataFrame) -> str:
    """Small equity + drawdown combo chart for per-symbol report."""
    eq = equity_df["equity"]
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.04,
                        subplot_titles=("Equity", "Drawdown %"))
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq, name="Equity",
        line=dict(color="#2196f3", width=1.3),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=dd.index, y=dd, name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239,83,80,0.12)",
        line=dict(color="#ef5350", width=1),
    ), row=2, col=1)
    fig.update_layout(height=400, margin=dict(l=50, r=30, t=35, b=30),
                      hovermode="x unified", showlegend=False)
    return pio.to_html(fig, **_PLOTLY_CFG)


def generate_single_report(symbol: str, df: pd.DataFrame,
                           equity_df: pd.DataFrame, trade_df: pd.DataFrame,
                           round_trips: list[dict]):
    """Write an HTML report for one symbol into REPORTS_DIR/."""
    eq = equity_df["equity"]
    rets = eq.pct_change().dropna()
    total_days = (eq.index[-1] - eq.index[0]).days
    years = max(total_days / 365.25, 1e-6)
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    cagr_val = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1
    peak = eq.cummax()
    max_dd = ((eq - peak) / peak).min()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    down = rets[rets < 0]
    sortino = (rets.mean() / down.std() * np.sqrt(252)
               if len(down) > 0 and down.std() > 0 else 0)

    n_rt = len(round_trips)
    if n_rt > 0:
        wins = [t for t in round_trips if t["pnl"] > 0]
        losses = [t for t in round_trips if t["pnl"] <= 0]
        win_rate = len(wins) / n_rt
        gp = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf_val = gp / gl if gl > 0 else float("inf")
        avg_hold = np.mean([t["hold_days"] for t in round_trips])
    else:
        win_rate = pf_val = avg_hold = 0

    ohlc_chart = _chart_single_symbol(df, symbol, trade_df)
    eq_dd_chart = _chart_eq_dd_single(equity_df)

    html = _SINGLE_TPL.render(
        symbol=symbol,
        start=df.index[0].strftime("%Y-%m-%d"),
        end=df.index[-1].strftime("%Y-%m-%d"),
        n_bars=len(df), years=years, capital=INITIAL_CAPITAL,
        total_ret=total_ret, cagr=cagr_val, max_dd=max_dd,
        sharpe=sharpe, sortino=sortino, final_eq=eq.iloc[-1],
        n_trades=n_rt, win_rate=win_rate, pf=pf_val, avg_hold=avg_hold,
        ohlc_chart=ohlc_chart, eq_dd_chart=eq_dd_chart,
        trades=round_trips,
    )
    out = REPORTS_DIR / f"{symbol}_report.html"
    out.write_text(html, encoding="utf-8")
    _fix_owner(out)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# vectorbt Portfolio Integration
# ═══════════════════════════════════════════════════════════════════════════════
def create_vbt_portfolio(all_data, equity_df, trade_df):
    """Build a vectorbt Portfolio from pre-computed orders for analytics."""
    try:
        symbols = sorted(all_data.keys())
        dates = equity_df.index

        # Close price matrix (forward-fill gaps)
        close_dict = {}
        for sym in symbols:
            s = all_data[sym]["close"].reindex(dates)
            s = s.ffill().bfill()
            close_dict[sym] = s
        close_df = pd.DataFrame(close_dict)

        # Order size matrix
        size_df = pd.DataFrame(0.0, index=dates, columns=symbols)
        for _, t in trade_df.iterrows():
            d, sym = t["date"], t["symbol"]
            if d in size_df.index and sym in size_df.columns:
                delta = t["shares"] if t["side"] == "BUY" else -t["shares"]
                size_df.loc[d, sym] += delta

        pf = vbt.Portfolio.from_orders(
            close=close_df,
            size=size_df,
            size_type="amount",
            init_cash=INITIAL_CAPITAL,
            cash_sharing=True,
            group_by=True,
            freq="1D",
        )
        print(f"[vbt]  Portfolio created successfully")
        return pf
    except Exception as e:
        print(f"[vbt]  Warning: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Report Generator
# ═══════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = Template(r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Backtest Report — UT Bot + HMA/EMA</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --bg: #f8f9fa; --card: #ffffff; --border: #e0e0e0;
          --text: #212121; --muted: #757575; --up: #26a69a; --down: #ef5350;
          --blue: #2196f3; --radius: 8px; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
         sans-serif; background: var(--bg); color: var(--text);
         max-width: 1440px; margin: 0 auto; padding: 24px; line-height: 1.5; }
  h1 { font-size: 1.8rem; margin-bottom: 4px; }
  h2 { font-size: 1.3rem; margin: 32px 0 16px; padding-bottom: 8px;
       border-bottom: 2px solid var(--blue); }
  .subtitle { color: var(--muted); margin-bottom: 24px; }
  .metrics { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
             gap: 12px; margin-bottom: 24px; }
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: var(--radius); padding: 16px; }
  .card .label { font-size: 0.78rem; color: var(--muted); text-transform: uppercase;
                 letter-spacing: 0.5px; }
  .card .value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; }
  .card .value.pos { color: var(--up); }
  .card .value.neg { color: var(--down); }
  .chart-box { background: var(--card); border: 1px solid var(--border);
               border-radius: var(--radius); padding: 12px; margin-bottom: 20px;
               overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th, td { padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--border); }
  th { background: #f1f3f5; font-weight: 600; position: sticky; top: 0; }
  td:first-child, th:first-child { text-align: left; }
  tr:hover td { background: #f5f5f5; }
  .tbl-wrap { max-height: 600px; overflow-y: auto; border: 1px solid var(--border);
              border-radius: var(--radius); }
  .strategy-box { background: var(--card); border: 1px solid var(--border);
                  border-radius: var(--radius); padding: 20px; margin-bottom: 20px; }
  .strategy-box ul { margin-left: 20px; }
  .strategy-box li { margin-bottom: 4px; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
  .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
         font-size: 0.75rem; font-weight: 600; }
  .tag.win { background: #e8f5e9; color: #2e7d32; }
  .tag.loss { background: #ffebee; color: #c62828; }
  footer { margin-top: 40px; padding-top: 16px; border-top: 1px solid var(--border);
           color: var(--muted); font-size: 0.8rem; text-align: center; }
</style>
</head>
<body>

<h1>Backtest Report: UT Bot + HMA / EMA Strategy</h1>
<p class="subtitle">Generated {{ gen_time }} &mdash; vectorbt {{ vbt_version }}</p>

<!-- Strategy Description -->
<div class="strategy-box">
<h3>Strategy Rules</h3>
<ul>
  <li><b>Indicators:</b> UT Bot (ATR {{ atr_period }}, K={{ ut_k }}), HMA({{ hma_len }}), EMA({{ ema_len }})</li>
  <li><b>Entry:</b> UT Bot <em>Buy environment (&ge;2 days)</em> AND Close &gt; HMA({{ hma_len }}) AND Close &gt; EMA({{ ema_len }}) AND ADX(14) &gt; 20 AND Volume &gt; 20d SMA &rarr; buy at <b>next day's open</b> &mdash; allocate {{ pos_pct }}% of <b>current equity</b> per position (compounding, max 20 concurrent). <b>Cooldown:</b> 3 days after stop-out. <b>Max 2 entries</b> per buy-environment episode.</li>
  <li><b>Take profit:</b> ATR-adaptive &mdash; sell all at <b>5&times;ATR</b> from entry (varies per symbol &amp; trade). <b>Breakeven stop:</b> once profit &ge; <b>2.5&times;ATR</b>, stop-loss moves to entry price (保本止损).</li>
  <li><b>Stop-loss:</b> Close &lt; HMA({{ hma_len }}) OR Close &lt; EMA({{ ema_len }}) OR UT Bot enters <em>Sell</em> environment</li>
</ul>
<p style="margin-top:8px;color:var(--muted);">Data: {{ n_symbols }} symbols, {{ start_date }} &rarr; {{ end_date }} ({{ total_years|round(1) }} years). Initial capital ${{ "{:,.0f}".format(initial_capital) }}.</p>
</div>

<!-- Overall Metrics -->
<h2>Overall Performance</h2>
<div class="metrics">
  <div class="card"><div class="label">Total Return</div>
    <div class="value {{ 'pos' if total_return >= 0 else 'neg' }}">{{ "%.2f"|format(total_return*100) }}%</div></div>
  <div class="card"><div class="label">CAGR</div>
    <div class="value {{ 'pos' if cagr >= 0 else 'neg' }}">{{ "%.2f"|format(cagr*100) }}%</div></div>
  <div class="card"><div class="label">Max Drawdown</div>
    <div class="value neg">{{ "%.2f"|format(max_drawdown*100) }}%</div></div>
  <div class="card"><div class="label">Sharpe Ratio</div>
    <div class="value">{{ "%.2f"|format(sharpe_ratio) }}</div></div>
  <div class="card"><div class="label">Sortino Ratio</div>
    <div class="value">{{ "%.2f"|format(sortino_ratio) }}</div></div>
  <div class="card"><div class="label">Calmar Ratio</div>
    <div class="value">{{ "%.2f"|format(calmar_ratio) }}</div></div>
  <div class="card"><div class="label">Annual Volatility</div>
    <div class="value">{{ "%.2f"|format(annual_volatility*100) }}%</div></div>
  <div class="card"><div class="label">Final Equity</div>
    <div class="value">${{ "{:,.0f}".format(final_equity) }}</div></div>
  <div class="card"><div class="label">Round-Trip Trades</div>
    <div class="value">{{ n_round_trips }}</div></div>
  <div class="card"><div class="label">Win Rate</div>
    <div class="value">{{ "%.1f"|format(win_rate*100) }}%</div></div>
  <div class="card"><div class="label">Profit Factor</div>
    <div class="value">{{ "%.2f"|format(profit_factor) if profit_factor < 1000 else "∞" }}</div></div>
  <div class="card"><div class="label">Avg Hold Days</div>
    <div class="value">{{ "%.1f"|format(avg_hold_days) }}</div></div>
  <div class="card"><div class="label">Avg Win</div>
    <div class="value pos">{{ "%.2f"|format(avg_win_pct*100) }}%</div></div>
  <div class="card"><div class="label">Avg Loss</div>
    <div class="value neg">{{ "%.2f"|format(avg_loss_pct*100) }}%</div></div>
</div>

{% if vbt_sharpe is not none %}
<div class="metrics">
  <div class="card"><div class="label">VBT Sharpe</div>
    <div class="value">{{ "%.2f"|format(vbt_sharpe) }}</div></div>
  <div class="card"><div class="label">VBT Sortino</div>
    <div class="value">{{ "%.2f"|format(vbt_sortino) }}</div></div>
  <div class="card"><div class="label">VBT Max DD</div>
    <div class="value neg">{{ "%.2f"|format(vbt_max_dd*100) }}%</div></div>
</div>
{% endif %}

{% if spy_total_return is not none %}
<h2>Benchmark Comparison — SPY Buy &amp; Hold</h2>
<div class="metrics">
  <div class="card"><div class="label">SPY Total Return</div>
    <div class="value {{ 'pos' if spy_total_return >= 0 else 'neg' }}">{{ "%.2f"|format(spy_total_return*100) }}%</div></div>
  <div class="card"><div class="label">SPY CAGR</div>
    <div class="value {{ 'pos' if spy_cagr >= 0 else 'neg' }}">{{ "%.2f"|format(spy_cagr*100) }}%</div></div>
  <div class="card"><div class="label">SPY Max Drawdown</div>
    <div class="value neg">{{ "%.2f"|format(spy_max_dd*100) }}%</div></div>
  <div class="card"><div class="label">SPY Sharpe</div>
    <div class="value">{{ "%.2f"|format(spy_sharpe) }}</div></div>
  <div class="card"><div class="label">SPY Ann. Volatility</div>
    <div class="value">{{ "%.2f"|format(spy_ann_vol*100) }}%</div></div>
  <div class="card"><div class="label">SPY Final Equity</div>
    <div class="value">${{ "{:,.0f}".format(spy_final_equity) }}</div></div>
  <div class="card" style="border-left: 3px solid var(--blue);">
    <div class="label">Alpha (CAGR - SPY)</div>
    <div class="value {{ 'pos' if alpha >= 0 else 'neg' }}">{{ "%.2f"|format(alpha*100) }}%</div></div>
</div>
{% endif %}

<!-- Equity & Drawdown -->
<h2>Equity Curve &amp; Drawdown</h2>
<div class="chart-box">{{ equity_chart }}</div>
<div class="chart-box">{{ drawdown_chart }}</div>

<!-- Annual & Monthly Returns -->
<h2>Returns Breakdown</h2>
<div class="chart-box">{{ yearly_chart }}</div>
<div class="chart-box">{{ monthly_chart }}</div>

<!-- Exit Reason & Trade Distribution -->
<h2>Trade Analysis</h2>
<div class="two-col">
  <div class="chart-box">{{ exit_chart }}</div>
  <div class="chart-box">{{ trade_dist_chart }}</div>
</div>

<!-- Per-Symbol Performance -->
<h2>Per-Symbol Performance</h2>
<div class="chart-box">{{ symbol_pnl_chart }}</div>

<div class="tbl-wrap">
<table>
<thead>
<tr>
  <th>Symbol</th><th># Trades</th><th>Total P&amp;L</th><th>Return %</th>
  <th>Win Rate</th><th>Avg P&amp;L %</th><th>Max Win %</th><th>Max Loss %</th>
  <th>Avg Hold</th>
</tr>
</thead>
<tbody>
{% for row in per_symbol_rows %}
<tr>
  <td><b><a href="reports/{{ row.symbol }}_report.html" style="color:var(--blue);text-decoration:none">{{ row.symbol }}</a></b></td>
  <td>{{ row.n_trades }}</td>
  <td class="{{ 'pos' if row.total_pnl >= 0 else 'neg' }}"
      style="color: {{ '#26a69a' if row.total_pnl >= 0 else '#ef5350' }}">
      ${{ "{:,.0f}".format(row.total_pnl) }}</td>
  <td>{{ "%.2f"|format(row.return_pct*100) }}%</td>
  <td>{{ "%.0f"|format(row.win_rate*100) }}%</td>
  <td>{{ "%.2f"|format(row.avg_pnl_pct*100) }}%</td>
  <td style="color:#26a69a">{{ "%.2f"|format(row.max_win_pct*100) }}%</td>
  <td style="color:#ef5350">{{ "%.2f"|format(row.max_loss_pct*100) }}%</td>
  <td>{{ "%.0f"|format(row.avg_hold_days) }}d</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

{% if best_trade %}
<h2>Notable Trades</h2>
<div class="two-col">
  <div class="card">
    <div class="label">Best Trade</div>
    <div class="value pos">{{ best_trade.symbol }} +{{ "%.1f"|format(best_trade.pnl_pct*100) }}%</div>
    <p style="color:var(--muted);font-size:0.85rem;margin-top:4px;">
      {{ best_trade.entry_date.strftime('%Y-%m-%d') }} → {{ best_trade.exit_date.strftime('%Y-%m-%d') }}
      ({{ best_trade.hold_days }}d) | P&amp;L ${{ "{:,.0f}".format(best_trade.pnl) }}</p>
  </div>
  <div class="card">
    <div class="label">Worst Trade</div>
    <div class="value neg">{{ worst_trade.symbol }} {{ "%.1f"|format(worst_trade.pnl_pct*100) }}%</div>
    <p style="color:var(--muted);font-size:0.85rem;margin-top:4px;">
      {{ worst_trade.entry_date.strftime('%Y-%m-%d') }} → {{ worst_trade.exit_date.strftime('%Y-%m-%d') }}
      ({{ worst_trade.hold_days }}d) | P&amp;L ${{ "{:,.0f}".format(worst_trade.pnl) }}</p>
  </div>
</div>
{% endif %}

<footer>
  Generated by backtest.py &mdash; vectorbt {{ vbt_version }} &bull; pandas-ta &bull; plotly
</footer>
</body>
</html>
""")


def generate_report(equity_df, trade_df, round_trips, stats, per_sym, all_data,
                    spy_df=None):
    """Render the full HTML report and write to disk."""
    # Generate all charts
    equity_chart = _chart_equity(equity_df, spy_df=spy_df)
    drawdown_chart = _chart_drawdown(equity_df)
    monthly_chart = _chart_monthly_returns(equity_df)
    yearly_chart = _chart_yearly_returns(equity_df, spy_df=spy_df)
    symbol_pnl_chart = _chart_symbol_pnl(per_sym)
    trade_dist_chart = _chart_trade_distribution(round_trips)
    exit_chart = _chart_exit_reasons(round_trips)

    # Prepare per-symbol table rows
    per_symbol_rows = per_sym.to_dict("records") if not per_sym.empty else []

    # Merge stats with chart/template vars (stats keys take lower priority)
    tpl_vars = dict(
        gen_time=datetime.now().strftime("%Y-%m-%d %H:%M"),
        vbt_version=vbt.__version__,
        atr_period=ATR_PERIOD,
        ut_k=UT_K,
        hma_len=HMA_LENGTH,
        ema_len=EMA_LENGTH,
        pos_pct=int(POSITION_PCT * 100),
        n_symbols=len(all_data),
        equity_chart=equity_chart,
        drawdown_chart=drawdown_chart,
        monthly_chart=monthly_chart,
        yearly_chart=yearly_chart,
        symbol_pnl_chart=symbol_pnl_chart,
        trade_dist_chart=trade_dist_chart,
        exit_chart=exit_chart,
        per_symbol_rows=per_symbol_rows,
    )
    tpl_vars.update(stats)  # stats includes initial_capital, start_date, etc.
    html = HTML_TEMPLATE.render(**tpl_vars)
    out_path = Path(OUTPUT_FILE)
    out_path.write_text(html, encoding="utf-8")
    _fix_owner(out_path)
    print(f"[report] Written to {OUTPUT_FILE} ({len(html)/1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  UT Bot + HMA/EMA Multi-Symbol Backtest")
    print("=" * 60)

    # 1. Load data & indicators (stocks only, ETFs excluded)
    all_data = load_all_data(exclude_etfs=True)

    # ══════════════════════════════════════════════════════════
    # A) Per-Symbol Full-Position Backtests
    # ══════════════════════════════════════════════════════════
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _fix_owner(REPORTS_DIR)
    print(f"\n{'─'*60}")
    print(f"  Per-symbol full-position backtests → {REPORTS_DIR}/")
    print(f"{'─'*60}")
    sym_summary: list[dict] = []
    for sym in sorted(all_data.keys()):
        df = all_data[sym]
        eq, tdf, rts = run_single_backtest(df, sym)
        out = generate_single_report(sym, df, eq, tdf, rts)

        # Quick summary for console
        total_ret = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1
        sym_summary.append(dict(symbol=sym, ret=total_ret, trades=len(rts)))
        print(f"  {sym:6s}  ret={total_ret*100:+7.1f}%  trades={len(rts):4d}  → {out.name}")

    print(f"[per-sym] {len(sym_summary)} symbol reports written to {REPORTS_DIR}/")

    # ══════════════════════════════════════════════════════════
    # B) Portfolio Backtest (5 % allocation per symbol)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"  Portfolio backtest (5% allocation, stocks only)")
    print(f"{'─'*60}")

    # 2. Run portfolio simulation
    equity_df, trade_df = run_backtest(all_data)

    # 3. Build round-trip trades
    round_trips = build_round_trips(trade_df)
    print(f"[stat] {len(round_trips)} round-trip trades built")

    # 4. Load SPY benchmark
    spy_df = load_spy_benchmark(equity_df.index[0], equity_df.index[-1])

    # 5. Compute stats (with SPY comparison)
    stats = compute_overall_stats(equity_df, round_trips, spy_df=spy_df)
    per_sym = compute_per_symbol_stats(round_trips)

    # 6. Create vectorbt portfolio (for cross-validation)
    pf = create_vbt_portfolio(all_data, equity_df, trade_df)
    if pf is not None:
        try:
            print(f"[vbt]  VBT Total Return: {pf.total_return():.4f}")
            print(f"[vbt]  VBT Sharpe:       {pf.sharpe_ratio():.4f}")
            print(f"[vbt]  VBT Max DD:       {pf.max_drawdown():.4f}")
        except Exception as e:
            print(f"[vbt]  Stats extraction warning: {e}")

    # 7. Generate portfolio HTML report
    generate_report(equity_df, trade_df, round_trips, stats, per_sym, all_data,
                    spy_df=spy_df)

    print("=" * 60)
    print(f"  DONE — Total Return: {stats['total_return']*100:.2f}%  |  "
          f"CAGR: {stats['cagr']*100:.2f}%  |  Max DD: {stats['max_drawdown']*100:.2f}%")
    print(f"  Portfolio report:  {OUTPUT_FILE}")
    print(f"  Per-symbol reports: {REPORTS_DIR}/ ({len(sym_summary)} files)")
    print("=" * 60)


if __name__ == "__main__":
    main()
