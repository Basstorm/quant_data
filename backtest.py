#!/usr/bin/env python3
"""
Multi-Symbol Backtest: UT Bot + HMA/EMA Strategy with Staged Exits
===================================================================

Strategy:
  - Indicators: UT Bot (ATR 10, K=2.5), HMA(100), EMA(200) via pandas-ta
  - Entry:  UT Bot Buy env  AND  Close > HMA(100)  AND  Close > EMA(200)
            Position size = 5% of initial capital (max 20 concurrent positions)
  - Exit (staged profit-taking):
            +5%  → sell 30% of initial shares
            +10% → sell 50% of initial shares
            +20% → sell all remaining
  - Stop-loss: Close < HMA(100)  OR  Close < EMA(200)  OR  UT Bot Sell env

Usage:
    uv run backtest.py
"""

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

# Staged profit-taking
TP1_PCT = 0.05;  TP1_EXIT = 0.30   # +5 % → sell 30 %
TP2_PCT = 0.10;  TP2_EXIT = 0.50   # +10 % → sell 50 %
TP3_PCT = 0.20                      # +20 % → sell all remaining


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
    return trailing_stop, buy_env, sell_env


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading & Indicator Computation
# ═══════════════════════════════════════════════════════════════════════════════
def load_all_data() -> dict[str, pd.DataFrame]:
    """Load every *_25Y_daily.csv from DATA_DIR and compute indicators."""
    all_data: dict[str, pd.DataFrame] = {}
    csv_files = sorted(DATA_DIR.glob("*_25Y_daily.csv"))
    print(f"[data] Found {len(csv_files)} CSV files in {DATA_DIR}/")

    for f in csv_files:
        symbol = f.stem.replace("_25Y_daily", "")
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if len(df) < EMA_LENGTH + 60:
            continue

        # Indicators
        df["hma"] = ta.hma(df["close"], length=HMA_LENGTH)
        df["ema200"] = ta.ema(df["close"], length=EMA_LENGTH)
        ts, buy_env, sell_env = calc_ut_bot(df["close"], df["high"], df["low"])
        df["ut_stop"] = ts
        df["ut_buy"] = buy_env
        df["ut_sell"] = sell_env

        df = df.dropna(subset=["hma", "ema200"])
        if len(df) < 50:
            continue

        # Pre-computed signals
        df["entry_ok"] = (
            df["ut_buy"]
            & (df["close"] > df["hma"])
            & (df["close"] > df["ema200"])
        )
        df["stop_hit"] = (
            (df["close"] < df["hma"])
            | (df["close"] < df["ema200"])
            | df["ut_sell"]
        )
        all_data[symbol] = df

    print(f"[data] {len(all_data)} symbols ready after indicator warm-up")
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
# Position Tracker
# ═══════════════════════════════════════════════════════════════════════════════
class Position:
    __slots__ = (
        "symbol", "entry_price", "entry_date",
        "initial_shares", "remaining_shares",
        "took_tp1", "took_tp2",
    )

    def __init__(self, symbol, entry_price, entry_date, shares):
        self.symbol = symbol
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.initial_shares = shares
        self.remaining_shares = shares
        self.took_tp1 = False
        self.took_tp2 = False


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Engine
# ═══════════════════════════════════════════════════════════════════════════════
def run_backtest(all_data: dict[str, pd.DataFrame]):
    """
    Run the event-driven portfolio simulation.

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
    alloc_amount = INITIAL_CAPITAL * POSITION_PCT

    equity_records: list[dict] = []
    trade_log: list[dict] = []

    for date in all_dates:
        closed_today: set[str] = set()

        # ── Phase 1: Exits ────────────────────────────────────────────────
        to_remove: list[str] = []
        for sym, pos in positions.items():
            df = all_data[sym]
            if date not in df.index:
                continue
            row = df.loc[date]
            price = row["close"]
            pnl_pct = (price - pos.entry_price) / pos.entry_price

            # Stop-loss (overrides profit-taking)
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
                continue

            # Profit-taking (check from highest level down)
            if pnl_pct >= TP3_PCT:
                sell_n = pos.remaining_shares
                if sell_n > 0:
                    cash += sell_n * price
                    trade_log.append(dict(
                        symbol=sym, date=date, side="SELL",
                        shares=sell_n, price=price,
                        reason="TP3_20%", pnl_pct=pnl_pct,
                        entry_price=pos.entry_price, entry_date=pos.entry_date,
                    ))
                    pos.remaining_shares = 0
                to_remove.append(sym)
                closed_today.add(sym)

            elif pnl_pct >= TP2_PCT and not pos.took_tp2:
                sell_n = max(1, int(pos.initial_shares * TP2_EXIT))
                sell_n = min(sell_n, pos.remaining_shares)
                if sell_n > 0:
                    cash += sell_n * price
                    pos.remaining_shares -= sell_n
                    trade_log.append(dict(
                        symbol=sym, date=date, side="SELL",
                        shares=sell_n, price=price,
                        reason="TP2_10%", pnl_pct=pnl_pct,
                        entry_price=pos.entry_price, entry_date=pos.entry_date,
                    ))
                pos.took_tp2 = True
                if not pos.took_tp1:
                    pos.took_tp1 = True
                if pos.remaining_shares <= 0:
                    to_remove.append(sym)
                    closed_today.add(sym)

            elif pnl_pct >= TP1_PCT and not pos.took_tp1:
                sell_n = max(1, int(pos.initial_shares * TP1_EXIT))
                sell_n = min(sell_n, pos.remaining_shares)
                if sell_n > 0:
                    cash += sell_n * price
                    pos.remaining_shares -= sell_n
                    trade_log.append(dict(
                        symbol=sym, date=date, side="SELL",
                        shares=sell_n, price=price,
                        reason="TP1_5%", pnl_pct=pnl_pct,
                        entry_price=pos.entry_price, entry_date=pos.entry_date,
                    ))
                pos.took_tp1 = True
                if pos.remaining_shares <= 0:
                    to_remove.append(sym)
                    closed_today.add(sym)

        for sym in to_remove:
            positions.pop(sym, None)

        # ── Phase 2: Entries ──────────────────────────────────────────────
        for sym in sorted(all_data.keys()):
            if sym in positions or sym in closed_today:
                continue
            df = all_data[sym]
            if date not in df.index:
                continue
            row = df.loc[date]
            if not row["entry_ok"]:
                continue

            budget = min(alloc_amount, cash)
            if budget < 100:
                continue
            shares = int(budget / row["close"])
            if shares <= 0:
                continue

            cost = shares * row["close"]
            cash -= cost
            positions[sym] = Position(sym, row["close"], date, shares)
            trade_log.append(dict(
                symbol=sym, date=date, side="BUY",
                shares=shares, price=row["close"],
                reason="ENTRY", pnl_pct=0.0,
                entry_price=row["close"], entry_date=date,
            ))

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
# Performance Analytics (vectorbt + manual)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_overall_stats(equity_df: pd.DataFrame, round_trips: list[dict]) -> dict:
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

    return dict(
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
# Plotly Chart Generators
# ═══════════════════════════════════════════════════════════════════════════════
_PLOTLY_CFG = dict(full_html=False, include_plotlyjs=False)
_COLORS = dict(up="#26a69a", down="#ef5350", blue="#2196f3",
               gray="#9e9e9e", orange="#ff9800")


def _chart_equity(equity_df: pd.DataFrame) -> str:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
        vertical_spacing=0.04,
        subplot_titles=("Portfolio Equity Curve", "Active Positions"),
    )
    fig.add_trace(go.Scatter(
        x=equity_df.index, y=equity_df["equity"],
        name="Equity", line=dict(color=_COLORS["blue"], width=1.5),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.08)",
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


def _chart_yearly_returns(equity_df: pd.DataFrame) -> str:
    equity = equity_df["equity"]
    yearly = equity.resample("YE").last().pct_change().dropna()
    if yearly.empty:
        return ""
    years = yearly.index.year.astype(str)
    colors = [_COLORS["up"] if v > 0 else _COLORS["down"] for v in yearly.values]
    fig = go.Figure(go.Bar(
        x=years, y=yearly.values * 100,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in yearly.values * 100],
        textposition="outside",
    ))
    fig.update_layout(
        title="Annual Returns",
        height=350, margin=dict(l=50, r=30, t=40, b=30),
        yaxis_title="Return %",
    )
    return pio.to_html(fig, **_PLOTLY_CFG)


def _chart_exit_reasons(round_trips: list[dict]) -> str:
    if not round_trips:
        return ""
    reasons = pd.Series([t["exit_reason"] for t in round_trips]).value_counts()
    fig = go.Figure(go.Pie(
        labels=reasons.index, values=reasons.values,
        hole=0.4, textinfo="label+percent+value",
        marker_colors=["#ef5350", "#26a69a", "#ff9800", "#2196f3", "#9c27b0"],
    ))
    fig.update_layout(title="Exit Reason Breakdown", height=350,
                      margin=dict(l=30, r=30, t=40, b=30))
    return pio.to_html(fig, **_PLOTLY_CFG)


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
  <li><b>Entry:</b> UT Bot in <em>Buy</em> environment AND Close &gt; HMA({{ hma_len }}) AND Close &gt; EMA({{ ema_len }}) &mdash; allocate {{ pos_pct }}% of initial capital per position (max 20 concurrent)</li>
  <li><b>Exit (staged):</b> +5% &rarr; sell 30%, +10% &rarr; sell 50%, +20% &rarr; sell all remaining</li>
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
  <td><b>{{ row.symbol }}</b></td>
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


def generate_report(equity_df, trade_df, round_trips, stats, per_sym, all_data):
    """Render the full HTML report and write to disk."""
    # Generate all charts
    equity_chart = _chart_equity(equity_df)
    drawdown_chart = _chart_drawdown(equity_df)
    monthly_chart = _chart_monthly_returns(equity_df)
    yearly_chart = _chart_yearly_returns(equity_df)
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
    Path(OUTPUT_FILE).write_text(html, encoding="utf-8")
    print(f"[report] Written to {OUTPUT_FILE} ({len(html)/1024:.0f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  UT Bot + HMA/EMA Multi-Symbol Backtest")
    print("=" * 60)

    # 1. Load data & indicators
    all_data = load_all_data()

    # 2. Run simulation
    equity_df, trade_df = run_backtest(all_data)

    # 3. Build round-trip trades
    round_trips = build_round_trips(trade_df)
    print(f"[stat] {len(round_trips)} round-trip trades built")

    # 4. Compute stats
    stats = compute_overall_stats(equity_df, round_trips)
    per_sym = compute_per_symbol_stats(round_trips)

    # 5. Create vectorbt portfolio (for cross-validation)
    pf = create_vbt_portfolio(all_data, equity_df, trade_df)
    if pf is not None:
        try:
            print(f"[vbt]  VBT Total Return: {pf.total_return():.4f}")
            print(f"[vbt]  VBT Sharpe:       {pf.sharpe_ratio():.4f}")
            print(f"[vbt]  VBT Max DD:       {pf.max_drawdown():.4f}")
        except Exception as e:
            print(f"[vbt]  Stats extraction warning: {e}")

    # 6. Generate HTML report
    generate_report(equity_df, trade_df, round_trips, stats, per_sym, all_data)

    print("=" * 60)
    print(f"  DONE — Total Return: {stats['total_return']*100:.2f}%  |  "
          f"CAGR: {stats['cagr']*100:.2f}%  |  Max DD: {stats['max_drawdown']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
