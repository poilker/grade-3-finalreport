from __future__ import annotations  # 讓型別註記（type hints）可以用「尚未定義的類名」或更彈性的寫法（含 | None 等）
from dataclasses import dataclass   # 匯入 dataclass：用來快速建立「只裝資料」的類別
import pandas as pd                # 匯入 pandas，回測資料基本都用 DataFrame/Series/Timestamp

@dataclass                         # 告訴 Python：這個 class 是資料容器，幫你自動生成 __init__、__repr__ 等
class Trade:                       # 定義「一筆交易」的資料結構（進出場資訊 + 損益 + 原因）
    side: int  # 1 long, -1 short  # 方向：1=做多，-1=做空
    entry_time: pd.Timestamp       # 進場時間（pandas 的時間戳）
    entry_price: float             # 進場價格
    exit_time: pd.Timestamp        # 出場時間
    exit_price: float              # 出場價格
    pnl: float                     # 這筆交易的損益（這裡是「報酬率型」：例如 0.02 = +2%）
    reason: str                    # 出場原因（例如 stop_long / exit_short）

def run_backtest(                  # 定義回測主函式
    df: pd.DataFrame,              # K 線資料表（至少要有 time/close/high/low 欄位）
    entry_long: pd.Series,         # 做多進場訊號（每根 bar 一個 True/False）
    entry_short: pd.Series,        # 做空進場訊號
    exit_long: pd.Series,          # 做多出場訊號
    exit_short: pd.Series,         # 做空出場訊號
    stop_long: pd.Series | None = None,   # 做多停損價（每根 bar 一個價格；沒有就傳 None）
    stop_short: pd.Series | None = None,  # 做空停損價
    fee_rate: float = 0.0006,      # 手續費率（單邊），例如 0.0006 = 0.06%
    slippage_bp: float = 2.0,      # 滑價（bp），2bp = 0.02% = 0.0002
    leverage: float = 5.0,         # 槓桿倍數（這裡 pnl 會乘上 leverage）
) -> tuple[pd.DataFrame, list[Trade]]:     # 回傳：(equity curve 的 DataFrame, trades 清單)

    df = df.copy()                 # 複製一份 df，避免改到外面原本傳進來的 df
    df["time"] = pd.to_datetime(df["time"], utc=True)  # 確保 time 欄位轉成 Timestamp（並指定 UTC）
    df = df.sort_values("time").reset_index(drop=True) # 按時間排序，並把 index 重排成 0..n-1

    close = df["close"].astype(float)  # 把 close 欄位轉成 float（避免字串/物件型導致計算出錯）
    high = df["high"].astype(float)    # high 同理
    low = df["low"].astype(float)      # low 同理

    slippage = slippage_bp / 10000.0   # bp → 比例：2bp = 2/10000 = 0.0002

    pos = 0                            # 目前持倉：0=空倉，1=多單，-1=空單
    entry_price = None                 # 當前倉位的進場價（空倉時為 None）
    entry_time = None                  # 當前倉位的進場時間（空倉時為 None）

    equity = 1.0                       # 初始淨值用 1.0 當基準（最後 1.35 代表 +35%）
    equity_curve = []                  # 用來存每根 bar 的 (time, equity)
    trades: list[Trade] = []           # 用來存每筆完成交易（進 + 出 一筆）

    for i in range(len(df)):           # 逐根 K 線跑回測
        t = df["time"].iat[i]          # 取第 i 根的時間（iat 是「用位置取值」比較快）
        c = close.iat[i]               # 取第 i 根收盤價
        h = high.iat[i]                # 取第 i 根最高價
        l = low.iat[i]                 # 取第 i 根最低價

        # stop first                  # 規則：先判斷停損（因為停損是硬性風控，優先級最高）
        if pos == 1 and stop_long is not None and pd.notna(stop_long.iat[i]):  # 若目前多單，且有多單停損價
            st = float(stop_long.iat[i])   # 取出停損價
            if l <= st:                    # 若最低價 <= 停損價，代表這根 K 線打到停損
                exit_px = st * (1 - slippage)  # 多單停損出場：對你不利 → 出場價比 st 更低（扣滑價）
                pnl = (exit_px - entry_price) / entry_price * leverage  # 多單報酬率 * 槓桿
                pnl -= fee_rate * 2         # 扣手續費（進場一次 + 出場一次 = 兩邊）
                equity *= (1 + pnl)         # 更新淨值：用乘法累乘
                trades.append(Trade(1, entry_time, entry_price, t, exit_px, pnl, "stop_long"))  # 記錄交易
                pos = 0                     # 清倉
                entry_price = entry_time = None  # 清掉進場資訊

        if pos == -1 and stop_short is not None and pd.notna(stop_short.iat[i]):  # 若目前空單，且有空單停損價
            st = float(stop_short.iat[i])  # 取出停損價
            if h >= st:                    # 若最高價 >= 停損價，代表空單被掃停損
                exit_px = st * (1 + slippage)  # 空單停損出場：對你不利 → 出場價比 st 更高（加滑價）
                pnl = (-1) * (exit_px - entry_price) / entry_price * leverage  # 空單報酬：用 -1 把方向反過來
                pnl -= fee_rate * 2         # 扣兩邊手續費
                equity *= (1 + pnl)         # 更新淨值
                trades.append(Trade(-1, entry_time, entry_price, t, exit_px, pnl, "stop_short"))  # 記錄交易
                pos = 0                     # 清倉
                entry_price = entry_time = None

        # normal exit                  # 一般出場（不是停損）
        if pos == 1 and bool(exit_long.iat[i]):   # 若目前多單，且多單出場訊號為 True
            exit_px = c * (1 - slippage)          # 多單一般出場：用收盤價出，仍然對你不利（扣滑價）
            pnl = (exit_px - entry_price) / entry_price * leverage  # 多單報酬 * 槓桿
            pnl -= fee_rate * 2                   # 扣兩邊手續費
            equity *= (1 + pnl)                   # 更新淨值
            trades.append(Trade(1, entry_time, entry_price, t, exit_px, pnl, "exit_long"))  # 記錄交易
            pos = 0                               # 清倉
            entry_price = entry_time = None

        if pos == -1 and bool(exit_short.iat[i]): # 若目前空單，且空單出場訊號為 True
            exit_px = c * (1 + slippage)          # 空單一般出場：用收盤價出，對你不利（加滑價）
            pnl = (-1) * (exit_px - entry_price) / entry_price * leverage  # 空單報酬 * 槓桿
            pnl -= fee_rate * 2                   # 扣兩邊手續費
            equity *= (1 + pnl)                   # 更新淨值
            trades.append(Trade(-1, entry_time, entry_price, t, exit_px, pnl, "exit_short"))  # 記錄交易
            pos = 0                               # 清倉
            entry_price = entry_time = None

        # entries (only if flat)       # 進場：只有空倉時才允許進
        if pos == 0:                   # 空倉才檢查進場訊號
            if bool(entry_long.iat[i]):            # 若做多進場訊號 True
                entry_px = c * (1 + slippage)      # 多單進場：對你不利（加滑價）
                pos = 1                            # 設為多單持倉
                entry_price = entry_px             # 記錄進場價
                entry_time = t                     # 記錄進場時間
            elif bool(entry_short.iat[i]):         # 否則若做空進場訊號 True
                entry_px = c * (1 - slippage)      # 空單進場：對你不利（扣滑價；因為你希望賣更高）
                pos = -1                           # 設為空單持倉
                entry_price = entry_px             # 記錄進場價
                entry_time = t                     # 記錄進場時間

        equity_curve.append((t, equity))           # 不管有沒有交易，每根 bar 都記錄當下 equity

    eq = pd.DataFrame(equity_curve, columns=["time", "equity"])  # 把 equity_curve 變成 DataFrame
    return eq, trades                         # 回傳：equity 曲線 + 交易清單
