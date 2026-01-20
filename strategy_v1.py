import numpy as np
import pandas as pd

from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator

from backtest_engine import run_backtest

# ============================
# Helpers
# ============================

def rolling_linreg_slope(y: pd.Series, window: int) -> pd.Series:
    """Rolling linear regression slope (price units per bar)."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(arr: np.ndarray) -> float:
        yv = arr.astype(float)
        y_mean = yv.mean()
        num = ((x - x_mean) * (yv - y_mean)).sum()
        return float(num / denom) if denom != 0 else 0.0

    return y.rolling(window).apply(_slope, raw=True)


def confirmed_pivots(high: pd.Series, low: pd.Series, L: int = 3):
    """Return pivot prices that are only known after confirmation (shifted by L bars).

    At time t, we confirm if bar (t-L) was a pivot based on surrounding bars.
    Output series have values at confirmation time t (not at the pivot bar).
    """
    # Raw pivots (uses future bars; we shift by L to avoid lookahead)
    piv_hi_raw = (high.shift(L) < high) & (high.shift(-L) < high)
    piv_hi_raw &= high.eq(high.rolling(2 * L + 1, center=True).max())

    piv_lo_raw = (low.shift(L) > low) & (low.shift(-L) > low)
    piv_lo_raw &= low.eq(low.rolling(2 * L + 1, center=True).min())

    # Confirmed at t (pivot happened at t-L)
    piv_hi_conf = piv_hi_raw.shift(L).fillna(False)
    piv_lo_conf = piv_lo_raw.shift(L).fillna(False)

    piv_hi_price = high.shift(L).where(piv_hi_conf)
    piv_lo_price = low.shift(L).where(piv_lo_conf)
    return piv_hi_price, piv_lo_price


def build_daily_allow(df_1h: pd.DataFrame,
                      drop_k_atr: float = 2.0,
                      stable_days: int = 5) -> pd.Series:
    """Daily gate: after a big down day, wait until we go `stable_days` without making a new low."""
    d = (
        df_1h.set_index("time")
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )

    atr_d = AverageTrueRange(high=d["high"], low=d["low"], close=d["close"], window=14).average_true_range()
    ret_d = d["close"].pct_change()

    # big drop if return <= -k * (ATR/price)
    big_drop = ret_d <= -(drop_k_atr * (atr_d / d["close"]))

    allow = []
    waiting = False
    last_min_low = None
    stable_count = 0

    for i in range(len(d)):
        lo = float(d["low"].iat[i])
        is_drop = bool(big_drop.iat[i]) if pd.notna(big_drop.iat[i]) else False

        if is_drop:
            waiting = True
            last_min_low = lo
            stable_count = 0

        if waiting:
            # new lower low resets the counter
            if last_min_low is None or lo < last_min_low:
                last_min_low = lo
                stable_count = 0
            else:
                stable_count += 1

            if stable_count >= stable_days:
                waiting = False

        allow.append(not waiting)

    allow_s = pd.Series(allow, index=d.index, name="allow")
    return allow_s


# ============================
# Strategy script
# ============================

df = pd.read_csv("BTCUSDT_1h.csv")
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values("time").reset_index(drop=True)

close = df["close"].astype(float)
high = df["high"].astype(float)
low = df["low"].astype(float)

# --- Indicators (1H)
bb = BollingerBands(close=close, window=20, window_dev=2)
bb_mid = bb.bollinger_mavg()
bb_up = bb.bollinger_hband()
bb_dn = bb.bollinger_lband()
bb_width = (bb_up - bb_dn) / bb_mid

atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()
ema20 = EMAIndicator(close=close, window=20).ema_indicator()

# --- Daily gate (big drop -> wait stable)
allow_d = build_daily_allow(df, drop_k_atr=2.0, stable_days=5)
df["date"] = df["time"].dt.floor("D")
allow_map = allow_d.to_dict()
df["allow"] = df["date"].map(allow_map).fillna(True)
allow = df["allow"]

# --- 1H pivots -> range lines (平頂平底 proxy)
L = 3
piv_hi_price, piv_lo_price = confirmed_pivots(high, low, L=L)

# If pivots are sparse at the beginning, use a fallback rolling min/max.
N_fallback = 48
support_fb = low.rolling(N_fallback).min()
resist_fb = high.rolling(N_fallback).max()

win = 250  # about ~10 days of 1H bars
range_high = piv_hi_price.rolling(win, min_periods=1).median().combine_first(resist_fb)
range_low = piv_lo_price.rolling(win, min_periods=1).median().combine_first(support_fb)

# near top/bottom zones
tol = 0.6
near_bottom = close <= (range_low + tol * atr)
near_top = close >= (range_high - tol * atr)

# --- Regime: RANGE vs SQUEEZE
is_range = adx < 20

lookback = 240  # ~10 days
w_th = bb_width.rolling(lookback).quantile(0.2)
roll_range = (high.rolling(20).max() - low.rolling(20).min()) / bb_mid
r_th = roll_range.rolling(lookback).quantile(0.2)

is_squeeze = (bb_width <= w_th) & (roll_range <= r_th)

# --- Drift direction during squeeze
slope = rolling_linreg_slope(close, window=48)

# --- Weekend bias (UTC): use return since Saturday 00:00
# This signal is only used on weekend bars (Sat/Sun).
ts = df["time"]
is_weekend = ts.dt.weekday >= 5  # Sat=5, Sun=6
is_sat_00 = (ts.dt.weekday == 5) & (ts.dt.hour == 0)

sat_anchor = close.where(is_sat_00).ffill()
weekend_ret = (close / sat_anchor - 1.0).where(is_weekend)

dead = 0.3 * (atr / close)
weekend_bias = pd.Series(
    np.where(weekend_ret > dead, 1, np.where(weekend_ret < -dead, -1, 0)),
    index=df.index,
)
weekend_bias = weekend_bias.fillna(0)

# ============================
# Entries / Exits
# ============================

bb_buf = 0.10  # ATR-scaled buffer around BB

# Range mean reversion (needs BOTH: BB touch + near top/bottom zone)
range_long_ok = (~is_weekend) | (weekend_bias >= 0)
range_short_ok = (~is_weekend) | (weekend_bias <= 0)

entry_long_range = (
    allow
    & is_range
    & (~is_squeeze)
    & near_bottom
    & (close <= (bb_dn + bb_buf * atr))
    & range_long_ok
)

entry_short_range = (
    allow
    & is_range
    & (~is_squeeze)
    & near_top
    & (close >= (bb_up - bb_buf * atr))
    & range_short_ok
    & (slope <= 0)  # avoid shorting when price is drifting up
)

# Exit range at midline (simple, you can later do scale-out)
exit_long_range = close >= bb_mid
exit_short_range = close <= bb_mid

b = 0.8
stop_long_range = (range_low - b * atr)
stop_short_range = (range_high + b * atr)

# Breakout during squeeze
buf = 0.2

break_long_ok = (~is_weekend) | (weekend_bias == 1)  # weekend: need up; flat -> no trade
break_short_ok = (~is_weekend) | (weekend_bias == -1)

entry_long_break = (
    allow
    & is_squeeze
    & (slope > 0)
    & (close > (range_high + buf * atr))
    & break_long_ok
)

entry_short_break = (
    allow
    & is_squeeze
    & (slope < 0)
    & (close < (range_low - buf * atr))
    & break_short_ok
)

# Exit breakout when momentum fades
exit_long_break = close < ema20
exit_short_break = close > ema20

stop_long_break = (range_high - b * atr)
stop_short_break = (range_low + b * atr)

# Combine
entry_long = (entry_long_range | entry_long_break).fillna(False)
entry_short = (entry_short_range | entry_short_break).fillna(False)
exit_long = (exit_long_range | exit_long_break).fillna(False)
exit_short = (exit_short_range | exit_short_break).fillna(False)

# stops: long pick tighter (higher), short pick tighter (lower)
stop_long = pd.concat([stop_long_range, stop_long_break], axis=1).max(axis=1)
stop_short = pd.concat([stop_short_range, stop_short_break], axis=1).min(axis=1)

# ============================
# Backtest
# ============================

eq, trades = run_backtest(
    df=df,
    entry_long=entry_long,
    entry_short=entry_short,
    exit_long=exit_long,
    exit_short=exit_short,
    stop_long=stop_long,
    stop_short=stop_short,
    fee_rate=0.0006,
    slippage_bp=2.0,
    leverage=5.0,
)

trades_df = pd.DataFrame([t.__dict__ for t in trades])
eq.to_csv("equity_curve.csv", index=False)
trades_df.to_csv("trades.csv", index=False)

print("Trades:", len(trades))
print("Final equity multiple:", float(eq["equity"].iat[-1]) if len(eq) else None)
if len(trades_df):
    print("Win rate:", float((trades_df["pnl"] > 0).mean()))
    print("Avg trade pnl:", float(trades_df["pnl"].mean()))
