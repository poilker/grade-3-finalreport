from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
from pybit.unified_trading import HTTP

session = HTTP(testnet=False)  # 公開資料不需 key

def fetch_kline(symbol="BTCUSDT", category="linear", interval="60", days=365, chunk_days=10, out_csv="BTCUSDT_1h.csv"):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    cur = start
    parts = []

    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        resp = session.get_kline(
            category=category,
            symbol=symbol,
            interval=interval,
            start=int(cur.timestamp() * 1000),
            end=int(nxt.timestamp() * 1000),
            limit=1000
        )
        if resp.get("retCode") != 0:
            raise RuntimeError(resp)

        rows = resp["result"]["list"]
        if rows:
            df = pd.DataFrame(rows, columns=["startTime","open","high","low","close","volume","turnover"])
            df["time"] = pd.to_datetime(df["startTime"].astype("int64"), unit="ms", utc=True)
            for c in ["open","high","low","close","volume","turnover"]:
                df[c] = df[c].astype(float)
            df = df.drop(columns=["startTime"])
            parts.append(df)

        cur = nxt
        time.sleep(0.1)

    out = (pd.concat(parts, ignore_index=True)
           .drop_duplicates(subset=["time"])
           .sort_values("time")
           .reset_index(drop=True))

    out.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} rows={len(out)}")

if __name__ == "__main__":
    fetch_kline(interval="60", out_csv="BTCUSDT_1h.csv", days=365)
