#!/usr/bin/env python3
"""Fetch daily adjusted-close prices from the Yahoo Finance chart API directly.

yfinance tends to time out from this environment and stooq is bot-blocked,
so we call query1.finance.yahoo.com/v8/finance/chart/<symbol> with a
User-Agent header (the approach that worked for the sector-crowding check).

Output: data/prices.csv  (date, NVDA, SOX, SPX, AVGO, MU, MSFT, GOOGL, META, AMZN, TSM, US10Y)
US10Y is the ^TNX index level (= yield x 10), used for the 10y yield change control.
"""
import json
import time
import urllib.request
import urllib.parse
from datetime import datetime, timezone

import pandas as pd

SYMBOLS = {
    "NVDA": "NVDA",
    "^SOX": "SOX",
    "^GSPC": "SPX",
    "AVGO": "AVGO",
    "MU": "MU",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "META": "META",
    "AMZN": "AMZN",
    "TSM": "TSM",
    "^TNX": "US10Y",
}

P1 = int(datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp())
P2 = int(datetime(2026, 10, 31, tzinfo=timezone.utc).timestamp())

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def fetch(symbol: str) -> pd.Series:
    q = urllib.parse.quote(symbol, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{q}"
        f"?period1={P1}&period2={P2}&interval=1d&events=div,split"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.load(r)
    result = payload["chart"]["result"][0]
    ts = result["timestamp"]
    adj = result["indicators"]["adjclose"][0]["adjclose"]
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/New_York").date
    s = pd.Series(adj, index=pd.Index(idx, name="date"), dtype="float64")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def main() -> None:
    frames = {}
    for sym, col in SYMBOLS.items():
        for attempt in range(3):
            try:
                s = fetch(sym)
                frames[col] = s
                print(f"{sym:8s} -> {col:6s} rows={len(s)} "
                      f"{s.index.min()} .. {s.index.max()}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"{sym}: attempt {attempt+1} failed: {e}")
                time.sleep(5)
        else:
            raise RuntimeError(f"failed to fetch {sym}")
        time.sleep(1)
    df = pd.DataFrame(frames)
    df.index.name = "date"
    out = "data/prices.csv"
    df.to_csv(out)
    print(f"saved {out}: shape={df.shape}, NaNs per col:\n{df.isna().sum()}")


if __name__ == "__main__":
    main()
