#!/usr/bin/env python3
"""
Analyze backtest round-trip trades to find improvement opportunities.
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import load_all_data, run_backtest, build_round_trips, run_single_backtest

print("=" * 70)
print("  Trade Analysis — Finding Improvement Opportunities")
print("=" * 70)

# ── Load & Run ────────────────────────────────────────────────────────────────
all_data = load_all_data(exclude_etfs=True)

# Portfolio-level
equity_df, trade_df = run_backtest(all_data)
rts = build_round_trips(trade_df)
df_rt = pd.DataFrame(rts)

# Per-symbol single backtests
sym_rts = {}
for sym in sorted(all_data.keys()):
    eq, tdf, rt_list = run_single_backtest(all_data[sym], sym)
    sym_rts[sym] = {
        "equity": eq,
        "round_trips": rt_list,
        "final_ret": eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1,
    }

print(f"\n{'='*70}")
print(f"  PORTFOLIO TRADE ANALYSIS  ({len(df_rt)} round-trips)")
print(f"{'='*70}")

# ── 1. Holding Period Analysis ────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  1. HOLDING PERIOD DISTRIBUTION")
print(f"{'─'*70}")
for label, sub in [("ALL", df_rt),
                   ("WINNERS", df_rt[df_rt["pnl"] > 0]),
                   ("LOSERS", df_rt[df_rt["pnl"] <= 0])]:
    if len(sub) == 0:
        continue
    hd = sub["hold_days"]
    print(f"\n  {label} ({len(sub)} trades):")
    print(f"    Mean:   {hd.mean():6.1f} days")
    print(f"    Median: {hd.median():6.1f} days")
    print(f"    P25:    {hd.quantile(0.25):6.1f} days")
    print(f"    P75:    {hd.quantile(0.75):6.1f} days")
    print(f"    P90:    {hd.quantile(0.90):6.1f} days")
    print(f"    P95:    {hd.quantile(0.95):6.1f} days")
    print(f"    Max:    {hd.max():6.0f} days")

# Bucket analysis
print(f"\n  Hold-period buckets (portfolio):")
buckets = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 120), (120, 9999)]
print(f"  {'Bucket':>12s}  {'Trades':>7s}  {'WinRate':>8s}  {'AvgPnL%':>9s}  {'TotPnL$':>12s}  {'AvgPnL$':>10s}")
for lo, hi in buckets:
    mask = (df_rt["hold_days"] >= lo) & (df_rt["hold_days"] < hi)
    sub = df_rt[mask]
    if len(sub) == 0:
        continue
    wr = (sub["pnl"] > 0).mean()
    label = f"{lo}-{hi}d" if hi < 9999 else f"{lo}d+"
    print(f"  {label:>12s}  {len(sub):>7d}  {wr:>7.1%}  {sub['pnl_pct'].mean()*100:>+8.2f}%  "
          f"${sub['pnl'].sum():>11,.0f}  ${sub['pnl'].mean():>9,.0f}")

# ── 2. Exit Reason Analysis ──────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  2. EXIT REASON ANALYSIS")
print(f"{'─'*70}")
print(f"  {'Reason':>12s}  {'Trades':>7s}  {'WinRate':>8s}  {'AvgPnL%':>9s}  {'AvgHold':>8s}  {'TotPnL$':>12s}")
for reason in sorted(df_rt["exit_reason"].unique()):
    sub = df_rt[df_rt["exit_reason"] == reason]
    wr = (sub["pnl"] > 0).mean()
    print(f"  {reason:>12s}  {len(sub):>7d}  {wr:>7.1%}  {sub['pnl_pct'].mean()*100:>+8.2f}%  "
          f"{sub['hold_days'].mean():>7.1f}d  ${sub['pnl'].sum():>11,.0f}")

# ── 3. PnL Distribution ──────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  3. PNL DISTRIBUTION")
print(f"{'─'*70}")
pnl_pcts = df_rt["pnl_pct"] * 100
print(f"  Mean:   {pnl_pcts.mean():+.2f}%")
print(f"  Median: {pnl_pcts.median():+.2f}%")
print(f"  Stdev:  {pnl_pcts.std():.2f}%")
print(f"  Min:    {pnl_pcts.min():+.2f}%")
print(f"  Max:    {pnl_pcts.max():+.2f}%")
print(f"  Win rate: {(df_rt['pnl'] > 0).mean():.1%}")
print(f"  Avg winner: {df_rt[df_rt['pnl']>0]['pnl_pct'].mean()*100:+.2f}%")
print(f"  Avg loser:  {df_rt[df_rt['pnl']<=0]['pnl_pct'].mean()*100:+.2f}%")
print(f"  Profit factor: {df_rt[df_rt['pnl']>0]['pnl'].sum() / max(1, abs(df_rt[df_rt['pnl']<=0]['pnl'].sum())):.2f}")

# ── 4. Time-in-trade: winners that overstayed ────────────────────────────────
print(f"\n{'─'*70}")
print("  4. WINNERS THAT OVERSTAYED — did holding longer help?")
print(f"{'─'*70}")
winners = df_rt[df_rt["pnl"] > 0].copy()
for threshold in [10, 20, 30, 50]:
    short_w = winners[winners["hold_days"] <= threshold]
    long_w  = winners[winners["hold_days"] > threshold]
    if len(short_w) > 5 and len(long_w) > 5:
        print(f"\n  Winners ≤{threshold}d:  n={len(short_w):4d}  avg_pnl={short_w['pnl_pct'].mean()*100:+.2f}%")
        print(f"  Winners >{threshold}d:  n={len(long_w):4d}  avg_pnl={long_w['pnl_pct'].mean()*100:+.2f}%")

# ── 5. Losers that could have been cut earlier ────────────────────────────────
print(f"\n{'─'*70}")
print("  5. LOSER ANALYSIS — could earlier stop help?")
print(f"{'─'*70}")
losers = df_rt[df_rt["pnl"] <= 0].copy()
for threshold in [5, 10, 15, 20, 30]:
    short_l = losers[losers["hold_days"] <= threshold]
    long_l  = losers[losers["hold_days"] > threshold]
    if len(long_l) > 0:
        print(f"  Losers >{threshold:2d}d:  n={len(long_l):4d}  avg_pnl={long_l['pnl_pct'].mean()*100:+.2f}%  "
              f"total_loss=${long_l['pnl'].sum():>10,.0f}")

# ── 6. Per-symbol summary ────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  6. PER-SYMBOL SUMMARY (single-symbol 100% backtest)")
print(f"{'─'*70}")
sym_stats = []
for sym, info in sym_rts.items():
    rts_list = info["round_trips"]
    if not rts_list:
        sym_stats.append(dict(sym=sym, ret=info["final_ret"], n=0, wr=0, avg_hold=0, avg_pnl=0))
        continue
    rdf = pd.DataFrame(rts_list)
    wr = (rdf["pnl"] > 0).mean()
    sym_stats.append(dict(
        sym=sym, ret=info["final_ret"], n=len(rdf), wr=wr,
        avg_hold=rdf["hold_days"].mean(),
        avg_pnl=rdf["pnl_pct"].mean(),
        med_hold=rdf["hold_days"].median(),
        stop_pct=(rdf["exit_reason"] == "STOP").mean(),
    ))

sdf = pd.DataFrame(sym_stats).sort_values("ret", ascending=False)
print(f"\n  {'Symbol':>8s}  {'Return':>9s}  {'Trades':>6s}  {'WinR':>6s}  {'AvgHold':>8s}  {'MedHold':>8s}  {'StopExit%':>9s}  {'AvgPnL%':>8s}")
for _, r in sdf.iterrows():
    print(f"  {r['sym']:>8s}  {r['ret']*100:>+8.1f}%  {r['n']:>6.0f}  {r['wr']:>5.1%}  "
          f"{r['avg_hold']:>7.1f}d  {r.get('med_hold',0):>7.1f}d  {r.get('stop_pct',0):>8.1%}  {r['avg_pnl']*100:>+7.2f}%")

# ── 7. Correlation: holding period vs PnL ─────────────────────────────────────
print(f"\n{'─'*70}")
print("  7. HOLDING PERIOD vs PNL CORRELATION")
print(f"{'─'*70}")
corr = df_rt["hold_days"].corr(df_rt["pnl_pct"])
print(f"  Pearson corr(hold_days, pnl_pct) = {corr:.4f}")

# ── 8. Consecutive-loss streaks ───────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  8. CONSECUTIVE LOSS STREAKS")
print(f"{'─'*70}")
sorted_rt = df_rt.sort_values("entry_date")
is_loss = (sorted_rt["pnl"] <= 0).values
max_streak = 0
cur_streak = 0
streaks = []
for v in is_loss:
    if v:
        cur_streak += 1
    else:
        if cur_streak > 0:
            streaks.append(cur_streak)
        cur_streak = 0
if cur_streak > 0:
    streaks.append(cur_streak)
print(f"  Max consecutive losses: {max(streaks) if streaks else 0}")
print(f"  Avg consecutive losses: {np.mean(streaks):.1f}")
print(f"  Streak distribution: {pd.Series(streaks).describe().to_dict()}")

# ── 9. Re-entry analysis: same symbol re-entered quickly ─────────────────────
print(f"\n{'─'*70}")
print("  9. RE-ENTRY ANALYSIS — same symbol quick re-entry")
print(f"{'─'*70}")
re_entries = []
for sym in df_rt["symbol"].unique():
    sym_trips = df_rt[df_rt["symbol"] == sym].sort_values("entry_date")
    if len(sym_trips) < 2:
        continue
    for i in range(1, len(sym_trips)):
        gap = (sym_trips.iloc[i]["entry_date"] - sym_trips.iloc[i-1]["exit_date"]).days
        prev_pnl = sym_trips.iloc[i-1]["pnl_pct"]
        curr_pnl = sym_trips.iloc[i]["pnl_pct"]
        re_entries.append(dict(sym=sym, gap=gap, prev_pnl=prev_pnl, curr_pnl=curr_pnl))

redf = pd.DataFrame(re_entries)
if len(redf) > 0:
    for gap_max in [1, 3, 5, 10]:
        quick = redf[redf["gap"] <= gap_max]
        if len(quick) > 5:
            wr = (quick["curr_pnl"] > 0).mean()
            print(f"  Re-entry within {gap_max}d: n={len(quick):4d}  win_rate={wr:.1%}  "
                  f"avg_pnl={quick['curr_pnl'].mean()*100:+.2f}%")
    # After a loss, re-enter quickly?
    after_loss = redf[(redf["prev_pnl"] <= 0) & (redf["gap"] <= 5)]
    if len(after_loss) > 5:
        print(f"\n  Re-entry ≤5d AFTER A LOSS: n={len(after_loss)}  "
              f"win_rate={(after_loss['curr_pnl']>0).mean():.1%}  "
              f"avg_pnl={after_loss['curr_pnl'].mean()*100:+.2f}%")

# ── 10. Yearly breakdown ─────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print("  10. YEARLY PERFORMANCE BREAKDOWN (portfolio)")
print(f"{'─'*70}")
df_rt["year"] = pd.to_datetime(df_rt["entry_date"]).dt.year
print(f"  {'Year':>6s}  {'Trades':>7s}  {'WinRate':>8s}  {'AvgPnL%':>9s}  {'TotPnL$':>12s}  {'AvgHold':>8s}")
for yr in sorted(df_rt["year"].unique()):
    sub = df_rt[df_rt["year"] == yr]
    wr = (sub["pnl"] > 0).mean()
    print(f"  {yr:>6d}  {len(sub):>7d}  {wr:>7.1%}  {sub['pnl_pct'].mean()*100:>+8.2f}%  "
          f"${sub['pnl'].sum():>11,.0f}  {sub['hold_days'].mean():>7.1f}d")

# ── 11. Proposed max-hold-days simulation ─────────────────────────────────────
print(f"\n{'─'*70}")
print("  11. WHAT-IF: MAX HOLDING PERIOD (force-exit at N days)")
print(f"{'─'*70}")
# Approximate: if hold_days > N, cap pnl at some estimate
# More precisely, trades that exited after N days — what if we exited at day N?
# This is approximate since we don't have intra-trade data, but we can
# compare: trades within N days vs trades that went beyond.
for max_days in [10, 15, 20, 25, 30, 40, 50]:
    within = df_rt[df_rt["hold_days"] <= max_days]
    beyond = df_rt[df_rt["hold_days"] > max_days]
    total_kept = within["pnl"].sum()
    # Trades beyond max_days: we don't know exact pnl at day N,
    # but we can flag the opportunity cost
    print(f"  MaxHold={max_days:2d}d: {len(within):4d} trades kept (PnL=${total_kept:>10,.0f})  |  "
          f"{len(beyond):4d} would be force-closed (their actual PnL=${beyond['pnl'].sum():>10,.0f}, avg={beyond['pnl_pct'].mean()*100:+.1f}%)")

print(f"\n{'='*70}")
print("  ANALYSIS COMPLETE")
print(f"{'='*70}")
