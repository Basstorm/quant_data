#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pandas-ta", "numpy"]
# ///
"""
Stock Selection Analysis — Compare kept vs removed symbols to derive selection rules.
"""
import numpy as np
import pandas as pd
import pandas_ta as ta
from pathlib import Path

DATA_DIR = Path("data")
ATR_PERIOD = 10
UT_K = 2.5
HMA_LENGTH = 100
EMA_LENGTH = 200

ETF_SYMBOLS = {
    "DBA", "DIA", "EEM", "EWY", "FNDA", "FNDC", "FNDX",
    "GLD", "HYG", "IBIT", "IGV", "IWM", "KWEB",
    "PRF", "PRFZ", "QQQ", "SCHG", "SCHI", "SLV",
    "SMH", "SOXL", "SOXX", "SPY", "SQQQ", "TLT",
    "TQQQ", "TSLL",
}

# Current blacklist (user-removed)
BLACKLIST = {
    "BMNR", "BAC", "CLSK", "ERNA", "GME", "MRVL", "QBTS", "QUBT",
    "POET", "SGHC", "MARA", "NOVT", "NVTS", "SMBS", "NOK", "ALKS", "ARMK",
}

print("=" * 78)
print("  Stock Selection Analysis — Kept vs Removed")
print("=" * 78)

# ── Load ALL non-ETF stocks (including blacklisted) ───────────────────────────
records = []
csv_files = sorted(DATA_DIR.glob("*_25Y_daily.csv"))

for f in csv_files:
    symbol = f.stem.replace("_25Y_daily", "")
    if symbol in ETF_SYMBOLS:
        continue

    df = pd.read_csv(f, parse_dates=["date"], index_col="date").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < EMA_LENGTH + 60:
        continue

    # Indicators
    df["hma"] = ta.hma(df["close"], length=HMA_LENGTH)
    df["ema200"] = ta.ema(df["close"], length=EMA_LENGTH)
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=ATR_PERIOD)
    df["rsi"] = ta.rsi(df["close"], length=14)
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx_df["ADX_14"]

    df = df.dropna(subset=["hma", "ema200", "atr", "rsi", "adx"])
    if len(df) < 50:
        continue

    # ── Key Metrics ────────────────────────────────────────────────────────
    # Dollar volume = close × volume
    df["dollar_vol"] = df["close"] * df["volume"]
    # ATR as % of price
    df["atr_pct"] = df["atr"] / df["close"]

    # Last 30 trading days
    last30 = df.tail(30)
    # Last 60 trading days
    last60 = df.tail(60)
    # Last 250 trading days (~1 year)
    last250 = df.tail(250)
    # Full history
    full = df

    # ── Entry signal quality (simulate strategy) ──────────────────────────
    buy_env = df["close"] > df["hma"]  # simplified UT buy proxy
    df["entry_signal"] = (
        (df["close"] > df["hma"])
        & (df["close"] > df["ema200"])
        & (df["adx"] > 20)
    )
    total_signals = df["entry_signal"].sum()
    total_days = len(df)
    signal_density = total_signals / total_days if total_days > 0 else 0

    # ── Trend quality: % of time in uptrend (close > EMA200) ──────────────
    pct_above_ema = (df["close"] > df["ema200"]).mean()
    pct_above_hma = (df["close"] > df["hma"]).mean()

    # ── Return characteristics ────────────────────────────────────────────
    daily_rets = df["close"].pct_change().dropna()
    annual_ret = (df["close"].iloc[-1] / df["close"].iloc[0]) ** (252 / len(df)) - 1
    annual_vol = daily_rets.std() * np.sqrt(252)
    sharpe_raw = annual_ret / annual_vol if annual_vol > 0 else 0

    # Max drawdown
    peak = df["close"].cummax()
    dd = (df["close"] - peak) / peak
    max_dd = dd.min()

    # ── Upside/downside ratio ─────────────────────────────────────────────
    up_days = daily_rets[daily_rets > 0]
    down_days = daily_rets[daily_rets < 0]
    up_down_ratio = up_days.mean() / abs(down_days.mean()) if len(down_days) > 0 and down_days.mean() != 0 else 1.0

    rec = dict(
        symbol=symbol,
        status="KEPT" if symbol not in BLACKLIST else "REMOVED",
        # Price & size
        last_price=df["close"].iloc[-1],
        avg_price_1y=last250["close"].mean() if len(last250) > 0 else np.nan,
        # Dollar volume
        dvol_30d_mean=last30["dollar_vol"].mean() if len(last30) > 0 else np.nan,
        dvol_30d_median=last30["dollar_vol"].median() if len(last30) > 0 else np.nan,
        dvol_60d_mean=last60["dollar_vol"].mean() if len(last60) > 0 else np.nan,
        dvol_250d_mean=last250["dollar_vol"].mean() if len(last250) > 0 else np.nan,
        dvol_full_mean=full["dollar_vol"].mean(),
        dvol_full_median=full["dollar_vol"].median(),
        # Volume
        vol_30d_mean=last30["volume"].mean() if len(last30) > 0 else np.nan,
        vol_30d_median=last30["volume"].median() if len(last30) > 0 else np.nan,
        # Volatility
        atr_pct_30d=last30["atr_pct"].mean() if len(last30) > 0 else np.nan,
        atr_pct_250d=last250["atr_pct"].mean() if len(last250) > 0 else np.nan,
        atr_pct_full=full["atr_pct"].mean(),
        annual_vol=annual_vol,
        # Trend quality
        pct_above_ema200=pct_above_ema,
        pct_above_hma100=pct_above_hma,
        # Returns
        annual_ret=annual_ret,
        sharpe_raw=sharpe_raw,
        max_dd=max_dd,
        up_down_ratio=up_down_ratio,
        # Signal density
        signal_density=signal_density,
        total_signals=total_signals,
        # Data length
        n_bars=len(df),
        start_date=df.index[0],
        end_date=df.index[-1],
    )
    records.append(rec)

all_df = pd.DataFrame(records)
kept = all_df[all_df["status"] == "KEPT"]
removed = all_df[all_df["status"] == "REMOVED"]

print(f"\n  Total non-ETF symbols: {len(all_df)}")
print(f"  Kept: {len(kept)}   Removed: {len(removed)}")

# ── 1. Group comparison ───────────────────────────────────────────────────────
print(f"\n{'─'*78}")
print("  1. KEPT vs REMOVED — KEY METRICS COMPARISON")
print(f"{'─'*78}")

compare_cols = [
    ("dvol_30d_mean",    "30d Avg Dollar Vol",   "${:>14,.0f}"),
    ("dvol_30d_median",  "30d Med Dollar Vol",   "${:>14,.0f}"),
    ("dvol_250d_mean",   "250d Avg Dollar Vol",  "${:>14,.0f}"),
    ("dvol_full_mean",   "Full Avg Dollar Vol",  "${:>14,.0f}"),
    ("dvol_full_median", "Full Med Dollar Vol",  "${:>14,.0f}"),
    ("vol_30d_mean",     "30d Avg Volume",       "{:>14,.0f}"),
    ("vol_30d_median",   "30d Med Volume",       "{:>14,.0f}"),
    ("atr_pct_30d",      "30d ATR%",             "{:>14.2%}"),
    ("atr_pct_full",     "Full ATR%",            "{:>14.2%}"),
    ("annual_vol",       "Annual Volatility",    "{:>14.2%}"),
    ("pct_above_ema200", "% Time > EMA200",      "{:>14.1%}"),
    ("pct_above_hma100", "% Time > HMA100",      "{:>14.1%}"),
    ("annual_ret",       "Annualized Return",    "{:>14.2%}"),
    ("sharpe_raw",       "Raw Sharpe",           "{:>14.2f}"),
    ("max_dd",           "Max Drawdown",         "{:>14.2%}"),
    ("up_down_ratio",    "Up/Down Day Ratio",    "{:>14.2f}"),
    ("signal_density",   "Signal Density",       "{:>14.3f}"),
    ("last_price",       "Last Price",           "${:>13.2f}"),
]

print(f"\n  {'Metric':<25s}  {'KEPT (mean)':>16s}  {'REMOVED (mean)':>16s}  {'KEPT (med)':>16s}  {'REMOVED (med)':>16s}")
print(f"  {'─'*25}  {'─'*16}  {'─'*16}  {'─'*16}  {'─'*16}")
for col, label, fmt in compare_cols:
    km = kept[col].mean()
    rm = removed[col].mean()
    kmed = kept[col].median()
    rmed = removed[col].median()
    print(f"  {label:<25s}  {fmt.format(km):>16s}  {fmt.format(rm):>16s}  {fmt.format(kmed):>16s}  {fmt.format(rmed):>16s}")

# ── 2. All symbols ranked by dollar volume ────────────────────────────────────
print(f"\n{'─'*78}")
print("  2. ALL SYMBOLS RANKED BY 30-DAY MEDIAN DOLLAR VOLUME")
print(f"{'─'*78}")
ranked = all_df.sort_values("dvol_30d_median", ascending=False)
print(f"\n  {'#':>3s}  {'Symbol':>8s}  {'Status':>8s}  {'30d MedDVol':>14s}  {'30d AvgDVol':>14s}  "
      f"{'ATR%':>7s}  {'%>EMA200':>8s}  {'AnnRet':>8s}  {'Sharpe':>7s}  {'MaxDD':>8s}")
for i, (_, r) in enumerate(ranked.iterrows(), 1):
    print(f"  {i:>3d}  {r['symbol']:>8s}  {r['status']:>8s}  "
          f"${r['dvol_30d_median']:>13,.0f}  ${r['dvol_30d_mean']:>13,.0f}  "
          f"{r['atr_pct_full']:>6.1%}  {r['pct_above_ema200']:>7.0%}  "
          f"{r['annual_ret']:>+7.1%}  {r['sharpe_raw']:>6.2f}  {r['max_dd']:>+7.1%}")

# ── 3. Full-history average dollar volume (important for pre-2020 stocks) ─────
print(f"\n{'─'*78}")
print("  3. ALL SYMBOLS RANKED BY FULL-HISTORY MEDIAN DOLLAR VOLUME")
print(f"{'─'*78}")
ranked2 = all_df.sort_values("dvol_full_median", ascending=False)
print(f"\n  {'#':>3s}  {'Symbol':>8s}  {'Status':>8s}  {'FullMedDVol':>14s}  {'FullAvgDVol':>14s}  "
      f"{'ATR%':>7s}  {'%>EMA200':>8s}  {'Signals':>7s}  {'SigDens':>7s}")
for i, (_, r) in enumerate(ranked2.iterrows(), 1):
    print(f"  {i:>3d}  {r['symbol']:>8s}  {r['status']:>8s}  "
          f"${r['dvol_full_median']:>13,.0f}  ${r['dvol_full_mean']:>13,.0f}  "
          f"{r['atr_pct_full']:>6.1%}  {r['pct_above_ema200']:>7.0%}  "
          f"{r['total_signals']:>7.0f}  {r['signal_density']:>6.3f}")

# ── 4. Identify threshold candidates ─────────────────────────────────────────
print(f"\n{'─'*78}")
print("  4. POTENTIAL SELECTION THRESHOLDS")
print(f"{'─'*78}")

# Test various dollar volume thresholds
print(f"\n  A) FULL-HISTORY MEDIAN DOLLAR VOLUME thresholds:")
print(f"     {'Threshold':>14s}  {'Kept pass':>10s}  {'Removed pass':>12s}  {'Would cut':>10s}")
for thresh in [1e6, 5e6, 10e6, 20e6, 50e6, 100e6]:
    k_pass = (kept["dvol_full_median"] >= thresh).sum()
    r_pass = (removed["dvol_full_median"] >= thresh).sum()
    would_cut = len(kept) - k_pass
    print(f"     ${thresh:>13,.0f}  {k_pass:>10d}  {r_pass:>12d}  {would_cut:>10d}")

print(f"\n  B) % TIME ABOVE EMA(200) thresholds:")
print(f"     {'Threshold':>10s}  {'Kept pass':>10s}  {'Removed pass':>12s}  {'Would cut':>10s}")
for thresh in [0.40, 0.45, 0.50, 0.55, 0.60]:
    k_pass = (kept["pct_above_ema200"] >= thresh).sum()
    r_pass = (removed["pct_above_ema200"] >= thresh).sum()
    would_cut = len(kept) - k_pass
    print(f"     {thresh:>9.0%}  {k_pass:>10d}  {r_pass:>12d}  {would_cut:>10d}")

print(f"\n  C) RAW SHARPE RATIO thresholds:")
print(f"     {'Threshold':>10s}  {'Kept pass':>10s}  {'Removed pass':>12s}  {'Would cut':>10s}")
for thresh in [-0.1, 0.0, 0.05, 0.10, 0.15, 0.20]:
    k_pass = (kept["sharpe_raw"] >= thresh).sum()
    r_pass = (removed["sharpe_raw"] >= thresh).sum()
    would_cut = len(kept) - k_pass
    print(f"     {thresh:>10.2f}  {k_pass:>10d}  {r_pass:>12d}  {would_cut:>10d}")

print(f"\n  D) FULL-HISTORY ATR% thresholds (lower = less volatile):")
print(f"     {'Threshold':>10s}  {'Kept pass':>10s}  {'Removed pass':>12s}  {'Would cut':>10s}")
for thresh in [0.02, 0.03, 0.04, 0.05, 0.06]:
    k_pass = (kept["atr_pct_full"] <= thresh).sum()
    r_pass = (removed["atr_pct_full"] <= thresh).sum()
    would_cut = len(kept) - k_pass
    print(f"     ≤{thresh:>8.1%}  {k_pass:>10d}  {r_pass:>12d}  {would_cut:>10d}")

# ── 5. Recommended filters (multi-criteria) ──────────────────────────────────
print(f"\n{'─'*78}")
print("  5. MULTI-CRITERIA FILTER TEST")
print(f"{'─'*78}")

# Test combinations
criteria_sets = [
    ("DolVol≥$10M", lambda r: r["dvol_full_median"] >= 10e6),
    ("DolVol≥$10M + Sharpe≥0", lambda r: (r["dvol_full_median"] >= 10e6) & (r["sharpe_raw"] >= 0)),
    ("DolVol≥$10M + %EMA>50%", lambda r: (r["dvol_full_median"] >= 10e6) & (r["pct_above_ema200"] >= 0.50)),
    ("DolVol≥$5M + Sharpe≥0.05", lambda r: (r["dvol_full_median"] >= 5e6) & (r["sharpe_raw"] >= 0.05)),
    ("DolVol≥$5M + Sharpe≥0 + %EMA>50%", lambda r: (r["dvol_full_median"] >= 5e6) & (r["sharpe_raw"] >= 0) & (r["pct_above_ema200"] >= 0.50)),
]

print(f"\n  {'Criteria':<40s}  {'Kept✓':>6s}  {'Rmvd✓':>6s}  {'Kept✗':>6s}  {'Rmvd✗':>6s}  {'Precision':>9s}")
for name, fn in criteria_sets:
    k_pass = fn(kept).sum()
    r_pass = fn(removed).sum()
    k_fail = len(kept) - k_pass
    r_fail = len(removed) - r_pass
    prec = k_pass / (k_pass + r_pass) if (k_pass + r_pass) > 0 else 0
    print(f"  {name:<40s}  {k_pass:>6d}  {r_pass:>6d}  {k_fail:>6d}  {r_fail:>6d}  {prec:>8.0%}")

# ── 6. Symbols that would be cut by each filter ──────────────────────────────
print(f"\n{'─'*78}")
print("  6. KEPT SYMBOLS THAT FAIL EACH FILTER")
print(f"{'─'*78}")

print(f"\n  A) Kept symbols with Full Median Dollar Vol < $10M:")
low_dvol = kept[kept["dvol_full_median"] < 10e6].sort_values("dvol_full_median")
for _, r in low_dvol.iterrows():
    print(f"     {r['symbol']:>8s}  MedDVol=${r['dvol_full_median']:>11,.0f}  "
          f"Sharpe={r['sharpe_raw']:>5.2f}  %EMA200={r['pct_above_ema200']:.0%}  AnnRet={r['annual_ret']:+.1%}")

print(f"\n  B) Kept symbols with raw Sharpe < 0:")
low_sharpe = kept[kept["sharpe_raw"] < 0].sort_values("sharpe_raw")
for _, r in low_sharpe.iterrows():
    print(f"     {r['symbol']:>8s}  Sharpe={r['sharpe_raw']:>5.2f}  "
          f"MedDVol=${r['dvol_full_median']:>11,.0f}  %EMA200={r['pct_above_ema200']:.0%}  AnnRet={r['annual_ret']:+.1%}")

print(f"\n  C) Kept symbols with % above EMA200 < 50%:")
low_trend = kept[kept["pct_above_ema200"] < 0.50].sort_values("pct_above_ema200")
for _, r in low_trend.iterrows():
    print(f"     {r['symbol']:>8s}  %EMA200={r['pct_above_ema200']:.0%}  "
          f"MedDVol=${r['dvol_full_median']:>11,.0f}  Sharpe={r['sharpe_raw']:>5.2f}  AnnRet={r['annual_ret']:+.1%}")

print(f"\n{'='*78}")
print("  ANALYSIS COMPLETE")
print(f"{'='*78}")
