import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator

# ===== 檔案路徑 =====
CSV_PATH = "BTCUSDT_1h.csv"        # 你的 1H K 線
TRADES_PATH = "trades.csv"         # run_backtest 後輸出的交易紀錄
EQUITY_PATH = "equity_curve.csv"   # run_backtest 後輸出的 equity 曲線

# ===== 圖表視窗大小 =====
LAST_N = 2000     # 建議 500~3000；8760 也行但比較重
N_SR = 48         # 支撐/壓力 rolling window（1H: 48=2天）

# ===== 交易標記設定 =====
SHOW_TRADE_LINES = False  # True: 會畫每筆交易的進出連線（很多筆會很亂）


def _read_ohlcv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("CSV 必須有 time 欄位")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _read_trades(trades_path: str) -> pd.DataFrame:
    if not os.path.exists(trades_path):
        return pd.DataFrame()

    tdf = pd.read_csv(trades_path)

    # 最常見欄位：side, entry_time, entry_price, exit_time, exit_price, pnl, reason
    for col in ["entry_time", "exit_time"]:
        if col in tdf.columns:
            tdf[col] = pd.to_datetime(tdf[col], utc=True, errors="coerce")

    # side 可能是字串，轉成 int
    if "side" in tdf.columns:
        tdf["side"] = pd.to_numeric(tdf["side"], errors="coerce").astype("Int64")

    return tdf.dropna(subset=["entry_time", "exit_time"]) if len(tdf) else tdf


def _filter_trades_to_window(tdf: pd.DataFrame, t_start: pd.Timestamp, t_end: pd.Timestamp) -> pd.DataFrame:
    if tdf.empty:
        return tdf

    # 只保留進場時間落在圖表視窗內的交易
    m = (tdf["entry_time"] >= t_start) & (tdf["entry_time"] <= t_end)
    return tdf.loc[m].copy()


def _add_trade_markers(fig: go.Figure, df: pd.DataFrame, tdf: pd.DataFrame):
    if tdf.empty:
        return

    # 若 trades.csv 沒有 entry_price/exit_price，就用 K 線 close 近似補上（保底）
    time_to_close = pd.Series(df["close"].values, index=df["time"]).sort_index()

    def ensure_price(col_time: str, col_price: str) -> np.ndarray:
        if col_price in tdf.columns and pd.notna(tdf[col_price]).all():
            return tdf[col_price].astype(float).values
        # fallback：用該時間最近的 close
        return time_to_close.reindex(tdf[col_time], method="nearest").values

    entry_px = ensure_price("entry_time", "entry_price")
    exit_px = ensure_price("exit_time", "exit_price")

    side = tdf["side"].fillna(0).astype(int).values if "side" in tdf.columns else np.zeros(len(tdf), dtype=int)

    # hover text
    def _hover(prefix: str, idx: int) -> str:
        parts = [prefix]
        if "pnl" in tdf.columns:
            try:
                parts.append(f"pnl={float(tdf['pnl'].iat[idx]):.6f}")
            except Exception:
                parts.append(f"pnl={tdf['pnl'].iat[idx]}")
        if "reason" in tdf.columns:
            parts.append(f"reason={tdf['reason'].iat[idx]}")
        return " | ".join(parts)

    # 分組：Long / Short entry & exit
    long_idx = np.where(side == 1)[0]
    short_idx = np.where(side == -1)[0]

    # Entry/Exit markers
    if len(long_idx):
        fig.add_trace(
            go.Scatter(
                x=tdf.loc[tdf.index[long_idx], "entry_time"],
                y=entry_px[long_idx],
                mode="markers",
                name="Long Entry",
                marker=dict(symbol="triangle-up", size=10),
                hovertext=[_hover("LONG entry", i) for i in long_idx],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=tdf.loc[tdf.index[long_idx], "exit_time"],
                y=exit_px[long_idx],
                mode="markers",
                name="Long Exit",
                marker=dict(symbol="x", size=10),
                hovertext=[_hover("LONG exit", i) for i in long_idx],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    if len(short_idx):
        fig.add_trace(
            go.Scatter(
                x=tdf.loc[tdf.index[short_idx], "entry_time"],
                y=entry_px[short_idx],
                mode="markers",
                name="Short Entry",
                marker=dict(symbol="triangle-down", size=10),
                hovertext=[_hover("SHORT entry", i) for i in short_idx],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=tdf.loc[tdf.index[short_idx], "exit_time"],
                y=exit_px[short_idx],
                mode="markers",
                name="Short Exit",
                marker=dict(symbol="x", size=10),
                hovertext=[_hover("SHORT exit", i) for i in short_idx],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

    # Optional: connect entry->exit lines (many trades may clutter)
    if SHOW_TRADE_LINES:
        xs = []
        ys = []
        for i in range(len(tdf)):
            xs += [tdf["entry_time"].iat[i], tdf["exit_time"].iat[i], None]
            ys += [entry_px[i], exit_px[i], None]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name="Trade lines",
                hoverinfo="skip",
                line=dict(width=1),
            ),
            row=1,
            col=1,
        )


def _compute_backtest_stats_html(trades_all: pd.DataFrame, equity_path: str) -> str:
    # Trades
    trades = int(len(trades_all)) if trades_all is not None else 0

    # Final equity multiple (從 equity_curve.csv 最後一筆 equity)
    final_equity = np.nan
    if os.path.exists(equity_path):
        try:
            edf = pd.read_csv(equity_path)
            if len(edf) and "equity" in edf.columns:
                final_equity = float(edf["equity"].iloc[-1])
        except Exception:
            final_equity = np.nan

    # Win rate & Avg trade pnl（從 trades.csv 的 pnl 欄）
    win_rate = np.nan
    avg_pnl = np.nan
    if trades_all is not None and (not trades_all.empty) and ("pnl" in trades_all.columns):
        pnl = pd.to_numeric(trades_all["pnl"], errors="coerce")
        if pnl.notna().any():
            win_rate = float((pnl > 0).mean())
            avg_pnl = float(pnl.mean())

    lines = [
        f"Trades: {trades}",
        f"Final equity multiple: {final_equity:.12f}" if np.isfinite(final_equity) else "Final equity multiple: NA",
        f"Win rate: {win_rate:.12f}" if np.isfinite(win_rate) else "Win rate: NA",
        f"Avg trade pnl: {avg_pnl:.12f}" if np.isfinite(avg_pnl) else "Avg trade pnl: NA",
    ]
    return "<br>".join(lines)  # Plotly annotation 用 <br> 換行


def main():
    df_all = _read_ohlcv(CSV_PATH)

    # 只畫最後 LAST_N 根，互動更順
    df = df_all.tail(LAST_N).copy()

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # ===== 指標（BB/ATR/ADX/EMA20 + 支撐壓力） =====
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_up"] = bb.bollinger_hband()
    df["bb_dn"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_up"] - df["bb_dn"]) / df["bb_mid"]

    df["atr"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    df["adx"] = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    df["ema20"] = EMAIndicator(close=close, window=20).ema_indicator()

    # 支撐/壓力（shift(1) 避免偷看同一根K）
    df["support"] = low.rolling(N_SR).min().shift(1)
    df["resist"] = high.rolling(N_SR).max().shift(1)

    # ===== 建圖（只保留主圖） =====
    fig = make_subplots(
        rows=1,
        cols=1,
        shared_xaxes=True,
    )

    # --- 主圖: Candlestick ---
    fig.add_trace(
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="BTCUSDT",
        ),
        row=1,
        col=1,
    )

    # --- 主圖 overlays: BB, EMA, Support/Resist ---
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_up"], mode="lines", name="BB Upper"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_mid"], mode="lines", name="BB Mid"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["bb_dn"], mode="lines", name="BB Lower"), row=1, col=1)

    fig.add_trace(go.Scatter(x=df["time"], y=df["ema20"], mode="lines", name="EMA20"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["support"], mode="lines", name=f"Support({N_SR})"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["resist"], mode="lines", name=f"Resist({N_SR})"), row=1, col=1)

    # ===== 交易買賣點（注意：統計用全交易；畫點用視窗內交易） =====
    tdf_all = _read_trades(TRADES_PATH)
    tdf_view = tdf_all

    if not tdf_all.empty:
        t_start = df["time"].iloc[0]
        t_end = df["time"].iloc[-1]
        tdf_view = _filter_trades_to_window(tdf_all, t_start, t_end)
        _add_trade_markers(fig, df, tdf_view)

    # ===== 標題（仍保留 trades in view）=====
    title = "BTCUSDT — Kline + Indicators + Trades"
    if tdf_view is not None and (not tdf_view.empty):
        title += f" | trades in view: {len(tdf_view)}"

    # ===== 底部統計文字 =====
    stats_html = _compute_backtest_stats_html(tdf_all, EQUITY_PATH)

    fig.add_annotation(
        x=0, y=-0.22,
        xref="paper", yref="paper",
        text=stats_html,
        showarrow=False,
        align="left",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(0,0,0,0.25)",
        borderwidth=1,
        borderpad=6,
    )

    # ===== 互動設定（拖曳平移、滾輪縮放、slider） =====
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        dragmode="pan",
        legend_orientation="h",
        legend_y=1.02,
        # 底部要加大，才放得下 4 行統計
        margin=dict(l=40, r=20, t=60, b=150),
    )

    fig.update_layout(xaxis=dict(fixedrange=False))
    fig.update_yaxes(fixedrange=False)

    out_html = "kline_with_indicators_and_trades.html"
    fig.write_html(out_html, auto_open=False)
    print("Saved:", os.path.abspath(out_html))


if __name__ == "__main__":
    main()
