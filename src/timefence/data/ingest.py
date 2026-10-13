"""Download A-share daily bars from Eastmoney through akshare and build the local store.

Needs network access to *.eastmoney.com and the optional `data` dependencies
(`uv sync --extra data`). Each stock is cached as its own Parquet file, so an interrupted
download resumes where it stopped.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from timefence.data.schema import board_of, with_available_at
from timefence.data.store import MarketStore

log = logging.getLogger(__name__)

A_SHARE_PREFIXES = (
    "600",
    "601",
    "603",
    "605",
    "688",
    "689",
    "000",
    "001",
    "002",
    "003",
    "300",
    "301",
)
COLUMN_MAP = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def _ak():
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError("akshare is not installed; run `uv sync --extra data`") from exc
    return ak


def list_codes(include_delisted: bool = True) -> pd.DataFrame:
    """Current A-share list plus delisted stocks, so the history has no survivorship bias."""
    ak = _ak()
    frames = [ak.stock_zh_a_spot_em()[["代码", "名称"]].assign(delisted=False)]
    if include_delisted:
        frames.append(ak.stock_zh_a_stop_em()[["代码", "名称"]].assign(delisted=True))
    codes = pd.concat(frames).rename(columns={"代码": "code", "名称": "name"})
    codes["code"] = codes["code"].astype(str).str.zfill(6)
    codes = codes[codes["code"].str.startswith(A_SHARE_PREFIXES)]
    return codes.drop_duplicates("code", keep="first").reset_index(drop=True)


def clean_adj_factor(ratio: pd.Series, tol: float = 2e-3) -> pd.Series:
    """Turn the noisy ratio hfq_close / raw_close into a step function.

    Eastmoney rounds both prices to cents, so the ratio wobbles by up to ~0.2% even when
    nothing happened. Real adjustments (dividends, splits) are permanent level shifts. We
    keep a change only if the level after it stays away from the level before it.
    """
    values = ratio.to_numpy(dtype=float)
    if len(values) == 0:
        return ratio
    out = np.empty_like(values)
    level = values[0]
    out[0] = level
    for i in range(1, len(values)):
        ahead = np.median(values[i : i + 5])
        if abs(ahead / level - 1) > tol and abs(values[i] / level - 1) > tol:
            level = ahead
        out[i] = level
    return pd.Series(out, index=ratio.index)


def fetch_one(code: str, start: str, end: str, fetch: Callable | None = None) -> pd.DataFrame:
    """Raw and backward-adjusted bars for one stock, merged into our bar schema."""
    fetch = fetch or _ak().stock_zh_a_hist
    raw = fetch(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
    if raw is None or raw.empty:
        return pd.DataFrame()
    hfq = fetch(symbol=code, period="daily", start_date=start, end_date=end, adjust="hfq")
    raw = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())]
    hfq = hfq.rename(columns=COLUMN_MAP)[["date", "close"]].rename(columns={"close": "hfq_close"})
    bars = raw.merge(hfq, on="date", how="left")
    bars["date"] = pd.to_datetime(bars["date"])
    bars = bars.sort_values("date").reset_index(drop=True)
    ratio = (bars["hfq_close"] / bars["close"]).ffill().bfill().fillna(1.0)
    bars["adj_factor"] = clean_adj_factor(ratio)
    bars["volume"] = bars["volume"] * 100  # Eastmoney reports lots of 100 shares
    bars["code"] = code
    return bars.drop(columns=["hfq_close"])


def _fetch_with_retry(code: str, start: str, end: str, retries: int, pause: float) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            time.sleep(pause * random.uniform(0.5, 1.5))
            return fetch_one(code, start, end)
        except Exception as exc:  # network errors vary by library version
            wait = 2**attempt
            log.warning("fetch %s failed (%s), retry in %ss", code, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {code} after {retries} attempts")


def download(
    root: str | Path,
    start: str = "20160101",
    end: str | None = None,
    workers: int = 4,
    retries: int = 4,
    pause: float = 0.3,
    limit: int | None = None,
) -> MarketStore:
    """Download every A-share (listed and delisted) and build `root/bars.parquet`."""
    root = Path(root)
    cache = root / "raw"
    cache.mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y%m%d")

    codes = list_codes()
    if limit:
        codes = codes.head(limit)
    todo = [c for c in codes["code"] if not (cache / f"{c}.parquet").exists()]
    log.info("%d stocks listed, %d to download", len(codes), len(todo))

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_with_retry, c, start, end, retries, pause): c for c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            code = futures[fut]
            try:
                bars = fut.result()
            except RuntimeError:
                failed.append(code)
                continue
            # An empty file marks "no data in range" so we do not ask again.
            bars.to_parquet(cache / f"{code}.parquet", index=False)
            if i % 200 == 0:
                log.info("downloaded %d / %d", i, len(todo))
    if failed:
        log.warning("%d stocks failed: %s", len(failed), failed[:20])

    return build_store(root, codes)


def build_store(root: str | Path, codes: pd.DataFrame) -> MarketStore:
    root = Path(root)
    parts = [pd.read_parquet(p) for p in sorted((root / "raw").glob("*.parquet"))]
    bars = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    bars = with_available_at(bars)

    first = bars.groupby("code")["date"].min()
    last = bars.groupby("code")["date"].max()
    meta = codes.set_index("code").reindex(first.index)
    stocks = pd.DataFrame(
        {
            "code": first.index,
            "name": meta["name"].to_numpy(),
            "board": [board_of(c) for c in first.index],
            "list_date": first.to_numpy(),
            "delist_date": np.where(
                meta["delisted"].fillna(False).to_numpy(), last.to_numpy(), None
            ),
        }
    )
    stocks["delist_date"] = pd.to_datetime(stocks["delist_date"])
    store = MarketStore(bars, stocks)
    store.save(root)
    return store
