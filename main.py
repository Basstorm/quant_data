"""
IBKR Historical K-Line Fetcher
使用 ib_insync 从 Interactive Brokers 获取历史K线数据

前置条件:
  - IB TWS 或 IB Gateway 需在本地运行
  - TWS 端口:     7497 (模拟) / 7496 (实盘)
  - Gateway 端口: 4002 (模拟) / 4001 (实盘)
  - 需在 TWS 中开启 Socket API:
      Edit → Global Configuration → API → Settings
      → Enable ActiveX and Socket Clients ✓

关于 TWS"写入请求"弹窗:
  ib_insync 默认连接时会发送 reqMarketDataType(4)（修改行情数据模式），
  TWS 将其视为写入操作并弹窗确认。
  代码层面: connect(..., readonly=True) 可跳过该调用。
  TWS 层面: API Settings → 勾选 "Read-Only API" 可彻底屏蔽所有写入弹窗。
"""

import asyncio
import json
import logging
from pathlib import Path

from ib_insync import IB, ScannerSubscription, Stock, util
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _bars_to_df(bars) -> "pd.DataFrame | None":
    """将 ib_insync BarData 列表转为整洁的 DataFrame。"""
    df = util.df(bars)
    if df is None or df.empty:
        return None
    df = df[["date", "open", "high", "low", "close", "volume", "barCount", "average"]]
    df.rename(columns={"barCount": "bar_count", "average": "vwap"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)
    return df


# ── 单标的历史K线 ─────────────────────────────────────────────────────────────

def fetch_historical_bars(
    host: str = "127.0.0.1",
    port: int = 7496,
    client_id: int = 1,
    symbol: str = "AAPL",
    exchange: str = "SMART",
    currency: str = "USD",
    duration: str = "1 Y",
    bar_size: str = "1 day",
    what_to_show: str = "TRADES",
) -> pd.DataFrame:
    """
    连接 IBKR 并获取单个标的的历史K线。

    参数:
        duration     历史时长, 如 "1 Y", "6 M", "30 D"
        bar_size     K线周期, 如 "1 day", "1 hour", "5 mins"
        what_to_show 数据类型: TRADES / MIDPOINT / BID / ASK
    """
    ib = IB()
    # readonly=True: 跳过 reqMarketDataType 等写入类调用，消除 TWS 写入请求弹窗
    ib.connect(host, port, clientId=client_id, readonly=True)
    try:
        contract = Stock(symbol, exchange, currency)
        ib.qualifyContracts(contract)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1,
        )
    finally:
        ib.disconnect()

    df = _bars_to_df(bars)
    if df is None:
        raise RuntimeError(f"未获取到 {symbol} 的历史数据，请检查连接和合约信息")
    return df


# ── 获取美股标的列表 ───────────────────────────────────────────────────────────

# 覆盖不同市场特征的扫描维度，每维最多 50 条，合并去重以扩大覆盖面
# 格式: (scan_code, 说明)
_SCAN_CODES: list[tuple[str, str]] = [
    # ("MOST_ACTIVE",          "成交量最大（按成交股数排名）"),
    ("MOST_ACTIVE_USD",      "成交额最大（按成交金额 USD 排名）"),
    # ("TOP_VOLUME_RATE",      "成交量/平均成交量比率最高（放量异动）"),
    ("TOP_TRADE_COUNT",      "成交笔数最多（高频成交活跃度）"),
    # ("TOP_PERC_GAIN",        "当日涨幅最大（百分比涨幅榜）"),
    # ("TOP_PERC_LOSE",        "当日跌幅最大（百分比跌幅榜）"),
    ("HIGH_VS_52W_HL",       "价格创52周新高（突破年线压力）"),
    # ("LOW_VS_52W_HL",        "价格创52周新低（跌破年线支撑）"),
    ("HIGH_VS_13W_HL",       "价格创13周新高（突破季线压力）"),
    # ("LOW_VS_13W_HL",        "价格创13周新低（跌破季线支撑）"),
    ("HOT_BY_VOLUME",        "按成交量排名的热门股"),
    # ("HOT_BY_PRICE",         "按价格变动排名的热门股"),
    ("OPT_VOLUME_MOST_ACTIVE","期权成交量最大的正股（期权市场关注度高）"),
    # ("HALTED",               "当日被交易所暂停交易的股票"),
]


def get_us_symbols(
    ib: IB,
    save_dir: "Path | None" = None,
    save_path: "Path | None" = None,
    min_usd_volume: float = 20_000_000,
    min_listing_bars: int = 250,
    requests_per_10min: int = 45,
) -> list[str]:
    """
    通过 IBKR Scanner 扫描美股主要交易所（NYSE / NASDAQ / AMEX），
    并应用三重过滤条件:

      条件 1 — 股价 > $10        : Scanner 原生 abovePrice 字段（免费）
      条件 2 — 上市超1年         : 二次过滤，1Y日线bar数 ≥ min_listing_bars
      条件 3 — 近10日均成交额    : 二次过滤，avg(close×volume) ≥ min_usd_volume

    每个 scan code 的原始结果单独写入 {save_dir}/{SCAN_CODE}.txt。

    参数:
        min_usd_volume      近10日平均日成交额门槛（默认 2000万 USD）
        min_listing_bars    1Y日线最少bar数，约252=1年（默认250）
        requests_per_10min  二次过滤限速（默认45，低于IBKR上限60）

    注意:
        IBKR Scanner 每次最多返回 50 条；如需更完整的标的宇宙，
        可事先准备 symbols.txt 并直接传给 download_all_us_daily()。
    """
    symbols: set[str] = set()
    for code, desc in _SCAN_CODES:
        try:
            sub = ScannerSubscription(
                instrument="STK",
                locationCode="STK.US.MAJOR",
                scanCode=code,
                numberOfRows=50,
                # ── 扫描层原生过滤（免费，不占 API 配额） ──
                abovePrice=10,          # 股价 > $10
            )
            results = ib.reqScannerData(sub)
            code_symbols = sorted(
                item.contractDetails.contract.symbol for item in results
            )

            # 写入单独文件
            if save_dir is not None:
                code_file = save_dir / f"{code}.txt"
                code_file.write_text("\n".join(code_symbols))
                log.info("[%s] %s → %d 个，已保存 %s", code, desc, len(code_symbols), code_file.name)
            else:
                log.info("[%s] %s → %d 个", code, desc, len(code_symbols))

            before = len(symbols)
            symbols.update(code_symbols)
            log.info("  新增 %d 个，合计 %d 个", len(symbols) - before, len(symbols))
        except Exception as exc:
            log.warning("[%s] 跳过: %s", code, exc)

    sorted_symbols = sorted(symbols)

    # ── 二次过滤: 上市超1年 + 近10日均成交额 > min_usd_volume ─────────────────
    if min_usd_volume > 0 or min_listing_bars > 0:
        log.info("开始二次过滤 (上市>1年 & 成交额>%.0fM USD)，共 %d 个候选...",
                 min_usd_volume / 1e6, len(sorted_symbols))
        sorted_symbols = _post_filter_symbols(
            ib, sorted_symbols,
            min_usd_volume=min_usd_volume,
            min_listing_bars=min_listing_bars,
            requests_per_10min=requests_per_10min,
        )
        log.info("二次过滤后剩余 %d 个标的", len(sorted_symbols))

    if save_path is not None:
        save_path.write_text("\n".join(sorted_symbols))
        log.info("全量标的列表已保存: %s", save_path)
    return sorted_symbols


# ── 二次过滤（上市时长 + 成交额）─────────────────────────────────────────────

def _post_filter_symbols(
    ib: IB,
    symbols: list[str],
    min_usd_volume: float = 20_000_000,
    min_listing_bars: int = 250,
    requests_per_10min: int = 45,
) -> list[str]:
    """
    对候选标的做二次筛选（单次 reqHistoricalData 同时判断两个条件）:

      条件 2 — 上市超1年:
        请求 "1 Y" 日线数据，若 bar 数 ≥ min_listing_bars (默认250 ≈ 1年交易日)
        则认为上市超1年。

      条件 3 — 近10日平均成交额 ≥ min_usd_volume (默认 2000万 USD):
        取最近10根 bar 的 close × volume 均值。

    限速: requests_per_10min 控制每10分钟最大请求数（建议 ≤ 45）。
    """
    interval = 600.0 / requests_per_10min
    passed: list[str] = []

    for i, symbol in enumerate(symbols):
        try:
            contract = Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(contract)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="1 Y",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            if not bars:
                log.debug("post-filter ✗ %s: 无历史数据", symbol)
                continue

            df = util.df(bars)

            # 条件 2: 上市超1年
            if len(df) < min_listing_bars:
                log.debug(
                    "post-filter ✗ %s: 仅 %d 根K线 (< %d, 不足1年)",
                    symbol, len(df), min_listing_bars,
                )
                continue

            # 条件 3: 近10日平均成交额
            avg_usd_vol = (df["close"].tail(10) * df["volume"].tail(10)).mean()
            if avg_usd_vol < min_usd_volume:
                log.debug(
                    "post-filter ✗ %s: 近10日均成交额 %.1fM < %.0fM",
                    symbol, avg_usd_vol / 1e6, min_usd_volume / 1e6,
                )
                continue

            passed.append(symbol)
            log.info(
                "post-filter ✓ %s  成交额=%.1fM  K线数=%d",
                symbol, avg_usd_vol / 1e6, len(df),
            )

        except Exception as exc:
            log.warning("post-filter %s 异常 (跳过): %s", symbol, exc)

        # 限速
        if i < len(symbols) - 1:
            ib.sleep(interval)

    return passed


# ── 异步批量下载核心 ──────────────────────────────────────────────────────────

async def _bulk_download_async(
    ib: IB,
    symbols: list[str],
    output_dir: Path,
    progress_file: Path,
    requests_per_10min: int = 50,
) -> list[str]:
    """
    异步令牌桶限速下载。
    IBKR 上限: 60 次 / 10 分钟；默认取 50 次留出安全余量。
    最大并发: 45 个未完成请求（IBKR 允许 50）。
    """
    # 断点续传
    done: set[str] = set()
    if progress_file.exists():
        done = set(json.loads(progress_file.read_text()))

    pending = [s for s in symbols if s not in done]
    log.info("待下载: %d / %d (已完成 %d)", len(pending), len(symbols), len(done))
    if not pending:
        return []

    interval = 600.0 / requests_per_10min   # 每次请求最小间隔（秒）
    semaphore = asyncio.Semaphore(45)        # 最大并发数
    rate_lock = asyncio.Lock()
    next_slot: list[float] = [0.0]          # 下一个可用时间槽
    failed: list[str] = []
    progress_counter = [0]

    async def fetch_one(symbol: str) -> tuple[str, bool]:
        # ① 领取时间槽（串行），确定本次请求的发出时刻
        async with rate_lock:
            loop_time = asyncio.get_event_loop().time()
            slot = max(loop_time, next_slot[0])
            next_slot[0] = slot + interval
            wait = slot - loop_time

        if wait > 0:
            await asyncio.sleep(wait)

        # ② 并发执行实际请求（受 semaphore 限制）
        async with semaphore:
            try:
                contract = Stock(symbol, "SMART", "USD")
                qualified = await ib.qualifyContractsAsync(contract)
                if not qualified:
                    log.warning("%s: qualifyContracts 失败", symbol)
                    return symbol, False

                bars = await ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr="25 Y",
                    barSizeSetting="1 day",
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                )
                df = _bars_to_df(bars)
                if df is None:
                    log.warning("%s: 返回空数据", symbol)
                    return symbol, False

                df.to_csv(output_dir / f"{symbol}_25Y_daily.csv")
                return symbol, True

            except Exception as exc:
                log.warning("%s: %s", symbol, exc)
                return symbol, False

    tasks = [asyncio.ensure_future(fetch_one(s)) for s in pending]
    total = len(tasks)

    for future in asyncio.as_completed(tasks):
        symbol, ok = await future
        progress_counter[0] += 1
        n = progress_counter[0]

        if ok:
            done.add(symbol)
            log.info("[%d/%d] ✓ %s", n, total, symbol)
        else:
            failed.append(symbol)
            log.warning("[%d/%d] ✗ %s", n, total, symbol)

        # 每完成 20 个保存一次进度
        if n % 20 == 0:
            progress_file.write_text(json.dumps(sorted(done), indent=2))

    progress_file.write_text(json.dumps(sorted(done), indent=2))
    return failed


# ── 批量下载入口 ──────────────────────────────────────────────────────────────

def download_all_us_daily(
    host: str = "127.0.0.1",
    port: int = 7496,
    client_id: int = 2,
    output_dir: str = "data",
    symbols: "list[str] | None" = None,
    requests_per_10min: int = 50,
) -> None:
    """
    获取美股标的列表，批量下载每个标的近 25 年日K数据。

    文件输出:
        {output_dir}/{SYMBOL}_25Y_daily.csv   每个标的的 K 线数据
        {output_dir}/us_symbols.txt           全量合并去重的标的列表
        {output_dir}/{SCAN_CODE}.txt          每个 scan code 单独的标的列表
        {output_dir}/.progress.json           下载进度（支持断点续传）
        {output_dir}/failed_symbols.txt       失败标的列表

    参数:
        symbols             手动传入标的列表；None 则通过 IBKR Scanner 自动获取
        requests_per_10min  限速（IBKR 上限 60，建议 ≤ 55）
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    progress_file = out / ".progress.json"

    ib = IB()
    ib.connect(host, port, clientId=client_id, readonly=True)
    try:
        if symbols is None:
            log.info("正在通过 IBKR Scanner 获取美股标的列表...")
            symbols = get_us_symbols(
                ib,
                save_dir=out,
                save_path=out / "us_symbols.txt",
            )
            log.info("共找到 %d 个标的", len(symbols))

        est_minutes = len(symbols) * 600 / requests_per_10min / 60
        log.info(
            "开始批量下载 25年日K，共 %d 个标的，预估耗时约 %.0f 分钟",
            len(symbols), est_minutes,
        )

        failed = util.run(
            _bulk_download_async(ib, symbols, out, progress_file, requests_per_10min)
        )
    finally:
        ib.disconnect()

    if failed:
        failed_file = out / "failed_symbols.txt"
        failed_file.write_text("\n".join(failed))
        log.warning("失败标的 %d 个，已保存: %s", len(failed), failed_file)

    log.info("下载完成！数据保存在: %s/", out)


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main():
    # ── 测试: AAPL 近 1 年日K ─────────────────────────────────────────────
    # print("=== 测试: AAPL 近1年日K ===")
    # df = fetch_historical_bars(symbol="AAPL", duration="1 Y")
    # print(f"共 {len(df)} 根K线\n{df.tail()}")
    # df.to_csv("AAPL_1Y_daily.csv")
    # print("已保存 AAPL_1Y_daily.csv\n")

    # ── 批量下载所有美股 25 年日K（取消注释以运行，耗时较长）─────────────
    download_all_us_daily(output_dir="data")


if __name__ == "__main__":
    main()
