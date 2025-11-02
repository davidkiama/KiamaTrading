import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import pandas_ta as ta

def load_data_yf(symbol: str, start: str, end: str, interval: str = "5m") -> pd.DataFrame:
    """
    Download *one* ticker and return a tidy OHLCV DataFrame.
    """
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        group_by="ticker",   # yfinance ≥0.2.30 keeps MultiIndex
        auto_adjust=False,
    )

    if raw.empty:
        raise RuntimeError(f"No data for {symbol}")

    return raw[symbol]

def add_pivots(
    df: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """
    Add boolean columns 'pivoth' and 'pivotl' to *df*.
    A pivot-high is a high that is the maximum within ±window bars.
    """
    df = df.copy()
    df["pivoth"] = (
        df["High"]
        .rolling(window * 2 + 1, center=True)
        .apply(lambda x: x[window] == x.max(), raw=True)
        .fillna(0)
        .astype(bool)
    )
    df["pivotl"] = (
        df["Low"]
        .rolling(window * 2 + 1, center=True)
        .apply(lambda x: x[window] == x.min(), raw=True)
        .fillna(0)
        .astype(bool)
    )
    return df

def plot_candles_with_pivots(
    df,
    start_idx: int,
    end_idx: int,
    time_col: str = None,
    body_width: float = 0.6,
    up_color: str = "#26a69a",
    down_color: str = "#ef5350",
):
    """
    Uses Plotly graph_objects (go) Candlestick.
    df must contain:  'Open' 'High' 'Low' 'Close' + pivot columns.
    """
    data_slice = df.iloc[start_idx : end_idx + 1]

    # Candlestick trace
    candle = go.Candlestick(
        x=data_slice.index,
        open=data_slice["Open"],
        high=data_slice["High"],
        low=data_slice["Low"],
        close=data_slice["Close"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
        increasing_fillcolor="#26a69a",
        decreasing_fillcolor="#ef5350",
        name="Price",
    )

    # Pivot-low ▼ markers (triangle-up placed slightly under low)
    pivot_lows = data_slice[data_slice["pivotl"]]
    trace_lows = go.Scatter(
        x=pivot_lows.index,
        y=pivot_lows["Low"] * 0.9998,          # tiny offset below wick
        mode="markers",
        marker=dict(symbol="triangle-up", size=10, color="red"),
        name="Pivot Low",
    )

    # Pivot-high ▲ markers (triangle-down placed slightly above high)
    pivot_highs = data_slice[data_slice["pivoth"]]
    trace_highs = go.Scatter(
        x=pivot_highs.index,
        y=pivot_highs["High"] * 1.0002,         # tiny offset above wick
        mode="markers",
        marker=dict(symbol="triangle-down", size=10, color="red"),
        name="Pivot High",
    )

    fig = go.Figure(data=[candle, trace_lows, trace_highs])
    fig.update_layout(
        title= f"Candles {start_idx} → {end_idx}",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600,
        width=1000,
    )
    fig.show()


def _slope(xs: np.ndarray, ys: np.ndarray) -> float:
    """Return slope of best-fit line."""
    if len(xs) < 2:
        return np.nan
    model = LinearRegression().fit(xs.reshape(-1, 1), ys)
    x2d = xs.reshape(-1, 1)
    slope = model.coef_[0]           # β₁
    r2    = model.score(x2d, ys)     # identical to r2_score(y, reg.predict(x))

    return slope, r2

# ────────────────────────────────────────────────────────────────
# flag detector (past-only because we slice first)
# ────────────────────────────────────────────────────────────────
def detect_flag(
    df: pd.DataFrame,
    idx: int,
    lookback: int = 18,
    pivot_window: int = 3,
    max_high_slope: float = 0.0001,
    min_low_slope:  float = 0.003,
    min_r2: float   = 0.70,
):
    if idx < lookback or idx < pivot_window * 2 + 1:
        return 0

    flag_start       = idx - lookback
    flag_end         = idx - 1
    last_confirmable = idx - pivot_window

    # ---------- use POSITION array instead of DateTime index ----------
    pos = np.arange(len(df))                               # NEW
    mask_slice = (pos >= flag_start) & (pos <= last_confirmable)

    abs_hi_idx = pos[mask_slice & df["pivoth"].values].tolist()
    abs_lo_idx = pos[mask_slice & df["pivotl"].values].tolist()
    # -------------------------------------------------------------------

    abs_hi_idx = abs_hi_idx[-4:]
    abs_lo_idx = abs_lo_idx[-4:]

    if (
        len(abs_hi_idx) < 2
        or len(abs_lo_idx) < 2
        or (len(abs_hi_idx) + len(abs_lo_idx)) < 5
    ):
        return 0

    xh = np.arange(len(abs_hi_idx)); yh = df["High"].iloc[abs_hi_idx].values
    xl = np.arange(len(abs_lo_idx)); yl = df["Low"].iloc[abs_lo_idx].values

    slope_h_raw, r2_h = _slope(xh, yh)
    slope_l_raw, r2_l = _slope(xl, yl)

    slope_h = slope_h_raw / yh.mean()
    slope_l = slope_l_raw / yl.mean()

    if not (
        abs(slope_h) <= max_high_slope
        and slope_l   >= min_low_slope
        and r2_h >= min_r2
        and r2_l >= min_r2
    ):
        return 0

    highs_coords = [(int(i), float(df["High"].iat[i])) for i in abs_hi_idx]
    lows_coords  = [(int(i), float(df["Low"].iat[i]))  for i in abs_lo_idx]

    return {
    "highs": highs_coords,
    "lows" : lows_coords,
    "slope_high": slope_h,
    "slope_low" : slope_l
    }

# ────────────────────────────────────────────────────────────────
# Precompute flags (breakout bar + next 2 bars)
# ────────────────────────────────────────────────────────────────
def precompute_triangle_flags(
    df: pd.DataFrame,
    lookback: int = 30,
    pivot_window: int = 4,
    max_high_slope: float = 0.0004,
    min_low_slope:  float = 0.001,
    min_r2: float   = 0.70,
    signal_len: int = 3,
) -> pd.DataFrame:
    """
    Adds/overwrites:
      - triangle_flag  (bool)
      - triangle_highs (object: list[tuple[int, float]] or None)
      - triangle_lows  (object: list[tuple[int, float]] or None)
    """
    df_out = df.copy()

    # (Re)create columns with correct dtypes
    n = len(df_out)
    df_out["triangle_flag"] = False
    df_out["triangle_highs"] = pd.Series([None] * n, dtype="object")
    df_out["triangle_lows"]  = pd.Series([None] * n, dtype="object")

    # sanity: pivots needed by detect_flag
    if not {"pivoth", "pivotl"} <= set(df_out.columns):
        raise ValueError("Missing 'pivoth'/'pivotl' in DataFrame")

    start_i = max(lookback, pivot_window * 2 + 1)
    for i in range(start_i, n):
        flag = detect_flag(
            df_out, i,
            lookback=lookback,
            pivot_window=pivot_window,
            max_high_slope=max_high_slope,
            min_low_slope=min_low_slope,
            min_r2=min_r2,
        )
        if not flag:
            continue

        # mark breakout + next bars
        end_pos = min(i + signal_len, n)
        df_out.iloc[i:end_pos, df_out.columns.get_loc("triangle_flag")] = True

        # ---- store highs/lows ONLY on breakout bar (as Python tuples) ----
        def _to_tuples(seq):
            # seq can be list of tuples or ndarray shape (k,2)
            arr = np.asarray(seq)
            if arr.ndim != 2 or arr.shape[1] != 2:
                # fallback: try iterating as pairs
                return [ (int(a), float(b)) for (a,b) in map(tuple, seq) ]
            return [ (int(a), float(b)) for a, b in arr.tolist() ]

        highs_list = _to_tuples(flag["highs"])
        lows_list  = _to_tuples(flag["lows"])

        # scalar write by *label* to avoid broadcasting errors
        row_label = df_out.index[i]
        df_out.at[row_label, "triangle_highs"] = highs_list
        df_out.at[row_label, "triangle_lows"]  = lows_list

    return df_out

# ────────────────────────────────────────────────────────────────
# Plot: candles + pivots + triangle rails & transparent fill
# ────────────────────────────────────────────────────────────────
def _interp_band_from_points(points_hi, points_lo):
    """
    Given lists of (abs_index, price) tuples for highs & lows,
    build a dense integer x grid and piecewise-linear y_top, y_bot.
    """
    # sort & split to arrays
    xh, yh = zip(*sorted(points_hi, key=lambda t: t[0]))
    xl, yl = zip(*sorted(points_lo, key=lambda t: t[0]))
    xh = np.asarray(xh, dtype=int); yh = np.asarray(yh, dtype=float)
    xl = np.asarray(xl, dtype=int); yl = np.asarray(yl, dtype=float)

    xmin = int(min(xh.min(), xl.min()))
    xmax = int(max(xh.max(), xl.max()))
    xgrid = np.arange(xmin, xmax + 1, dtype=int)

    # piecewise-linear interpolation along indices
    y_top = np.interp(xgrid, xh, yh)
    y_bot = np.interp(xgrid, xl, yl)

    return xgrid, y_top, y_bot

def _fit_band_from_points(
    points_hi, points_lo,
    x_min=None, x_max=None,
    extend_left: int = 0,
    extend_right: int = 0,
    degree: int = 1,
):
    """
    Fit y = m*x + b (or higher degree) through high/low pivot points separately.
    Returns integer x grid plus y_top, y_bot without corners.
    """
    def _xy(points):
        xs, ys = zip(*sorted(points, key=lambda t: t[0]))
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    xh, yh = _xy(points_hi)
    xl, yl = _xy(points_lo)

    # handle degenerate cases (1 point -> flat line)
    def _fit(x, y):
        if len(x) == 1:
            return np.poly1d([0.0, float(y[0])])  # flat at that y
        deg = min(degree, len(x) - 1)
        coeffs = np.polyfit(x, y, deg=deg)
        return np.poly1d(coeffs)

    ph = _fit(xh, yh)
    pl = _fit(xl, yl)

    # span to draw
    xmin_span = int(min(xh.min(), xl.min()))
    xmax_span = int(max(xh.max(), xl.max()))
    xmin = xmin_span if x_min is None else int(x_min)
    xmax = xmax_span if x_max is None else int(x_max)

    xmin -= int(max(0, extend_left))
    xmax += int(max(0, extend_right))
    if xmax < xmin:
        xmax = xmin

    xgrid = np.arange(xmin, xmax + 1, dtype=int)
    y_top = ph(xgrid)
    y_bot = pl(xgrid)

    return xgrid, y_top, y_bot

def plot_candles_with_pivots_and_flags(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    body_width: float = 0.6,
    up_color: str = "#26a69a",
    down_color: str = "#ef5350",
    rail_width: float = 2.0,
    fill_rgba: str = "rgba(255, 215, 0, 0.15)",  # soft golden fill
    rail_color: str = "#ffd700",                  # golden rails
    use_int_index: bool = True,                   # << new option
):
    """
    Extends your pivot plot by overlaying triangle flag rails + translucent fill.

    Requires df to have:
      'Open','High','Low','Close','pivoth','pivotl',
      'triangle_flag','triangle_highs','triangle_lows'
    """
    data_slice = df.iloc[start_idx:end_idx+1]

    # helper: choose x values
    def _xvals(index):
        return (index - start_idx) if use_int_index else df.index[index]

    # ── Candle trace
    candle = go.Candlestick(
        x=list(range(len(data_slice))) if use_int_index else data_slice.index,  # <-- wrap in list()
        open=data_slice["Open"], high=data_slice["High"],
        low=data_slice["Low"],  close=data_slice["Close"],
        increasing_line_color=up_color,  increasing_fillcolor=up_color,
        decreasing_line_color=down_color, decreasing_fillcolor=down_color,
        name="Price",
    )

    # ── Pivot markers
    pivot_lows  = data_slice[data_slice["pivotl"]]
    pivot_highs = data_slice[data_slice["pivoth"]]
    # positions within the visible slice
    low_pos  = data_slice.index.get_indexer(pivot_lows.index)   # e.g., [3, 27, 58, ...]
    high_pos = data_slice.index.get_indexer(pivot_highs.index)

    trace_lows = go.Scatter(
    x=(low_pos if use_int_index else pivot_lows.index),
    y=pivot_lows["Low"] * 0.9998,
    mode="markers",
    marker=dict(symbol="triangle-up", size=10, color="red"),
    name="Pivot Low",
    )
    trace_highs = go.Scatter(
        x=(high_pos if use_int_index else pivot_highs.index),
        y=pivot_highs["High"] * 1.0002,
        mode="markers",
        marker=dict(symbol="triangle-down", size=10, color="red"),
        name="Pivot High",
    )

    fig = go.Figure(data=[candle, trace_lows, trace_highs])

    # ── Triangle rails & translucent band
    plotted_spans = []

    slice_abs_range = range(start_idx, end_idx + 1)
    for abs_i in slice_abs_range:
        row = df.iloc[abs_i]
        if not bool(row.get("triangle_flag", False)):
            continue
        highs = row.get("triangle_highs", False)
        lows  = row.get("triangle_lows",  False)
        if not (isinstance(highs, (list, tuple)) and isinstance(lows, (list, tuple)) and highs and lows):
            continue

        xgrid, y_top, y_bot = _fit_band_from_points(
            highs, lows,
            x_min=None, x_max=None,
            extend_left=0, extend_right=5,
            degree=1
        )
        xmin, xmax = int(xgrid.min()), int(xgrid.max())
        overlap = any(not (xmax < a or xmin > b) for (a, b) in plotted_spans)
        if overlap:
            continue
        plotted_spans.append((xmin, xmax))

        mask = (xgrid >= start_idx) & (xgrid <= end_idx)
        if not mask.any():
            continue
        xgrid_vis = xgrid[mask]

        # choose x coords based on flag
        x_vals = (xgrid_vis - start_idx) if use_int_index else df.index[xgrid_vis]
        y_top_vis = y_top[mask]
        y_bot_vis = y_bot[mask]

        upper = go.Scatter(
            x=x_vals, y=y_top_vis,
            mode="lines",
            line=dict(width=rail_width, color=rail_color),
            name="Flag Upper",
            showlegend=False,
        )
        lower = go.Scatter(
            x=x_vals, y=y_bot_vis,
            mode="lines",
            line=dict(width=rail_width, color=rail_color),
            fill="tonexty",
            fillcolor=fill_rgba,
            name="Flag Lower",
            showlegend=False,
        )
        fig.add_trace(upper)
        fig.add_trace(lower)

    fig.update_layout(
        title=f"Candles {start_idx} → {end_idx} (pivots + flag rails)",
        xaxis_title="Index" if use_int_index else "Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=650, width=1100,
    )
    fig.show()

# def plot_candles_with_pivots_and_flags(
#     df: pd.DataFrame,
#     start_idx: int,
#     end_idx: int,
#     body_width: float = 0.6,
#     up_color: str = "#26a69a",
#     down_color: str = "#ef5350",
#     rail_width: float = 2.0,
#     fill_rgba: str = "rgba(255, 215, 0, 0.15)",  # soft golden fill
#     rail_color: str = "#ffd700",                  # golden rails
# ):
#     """
#     Extends your pivot plot by overlaying triangle flag rails + translucent fill.

#     Requires df to have:
#       'Open','High','Low','Close','pivoth','pivotl',
#       'triangle_flag','triangle_highs','triangle_lows'
#     """
#     data_slice = df.iloc[start_idx:end_idx+1]
#     idx_offset = start_idx  # needed to map absolute int indices to timestamps

#     # ── Candle trace
#     candle = go.Candlestick(
#         x=data_slice.index,
#         open=data_slice["Open"], high=data_slice["High"],
#         low=data_slice["Low"],  close=data_slice["Close"],
#         increasing_line_color=up_color,  increasing_fillcolor=up_color,
#         decreasing_line_color=down_color, decreasing_fillcolor=down_color,
#         name="Price",
#     )

#     # ── Pivot markers
#     pivot_lows  = data_slice[data_slice["pivotl"]]
#     pivot_highs = data_slice[data_slice["pivoth"]]
#     trace_lows = go.Scatter(
#         x=pivot_lows.index,
#         y=pivot_lows["Low"] * 0.9998,
#         mode="markers",
#         marker=dict(symbol="triangle-up", size=10, color="red"),
#         name="Pivot Low",
#     )
#     trace_highs = go.Scatter(
#         x=pivot_highs.index,
#         y=pivot_highs["High"] * 1.0002,
#         mode="markers",
#         marker=dict(symbol="triangle-down", size=10, color="red"),
#         name="Pivot High",
#     )

#     fig = go.Figure(data=[candle, trace_lows, trace_highs])

#     # ── Triangle rails & translucent band
#     # We only draw once per pattern (even though triangle_flag covers 3 bars).
#     plotted_spans = []  # list of (xmin, xmax) absolute index ranges already drawn

#     slice_abs_range = range(start_idx, end_idx + 1)
#     # iterate ONLY rows where the breakout bar (first of the 3) sits in view
#     for abs_i in slice_abs_range:
#         row = df.iloc[abs_i]
#         # only draw on the breakout bar where lists are stored
#         if not bool(row.get("triangle_flag", False)):
#             continue
#         highs = row.get("triangle_highs", False)
#         lows  = row.get("triangle_lows",  False)
#         if not (isinstance(highs, (list, tuple)) and isinstance(lows, (list, tuple)) and highs and lows):
#             continue

#         # build piecewise-linear band
#         # xgrid, y_top, y_bot = _interp_band_from_points(highs, lows)
#         # build fitted (no-corner) band; set extend_right if you want the rails to project forward a bit
#         xgrid, y_top, y_bot = _fit_band_from_points(
#             highs, lows,
#             x_min=None, x_max=None,
#             extend_left=0, extend_right=5,   # tweak as you like
#             degree=1                         # keep 1 for straight lines; try 2 for gentle curves
#         )
#         xmin, xmax = int(xgrid.min()), int(xgrid.max())

#         # skip if we already plotted an overlapping span
#         overlap = any(not (xmax < a or xmin > b) for (a, b) in plotted_spans)
#         if overlap:
#             continue
#         plotted_spans.append((xmin, xmax))

#         # map absolute integer indices to timestamps for visible window
#         # keep only points that lie within [start_idx, end_idx] so the band aligns
#         mask = (xgrid >= start_idx) & (xgrid <= end_idx)
#         if not mask.any():
#             continue
#         xgrid_vis = xgrid[mask]
#         x_ts = df.index[xgrid_vis]  # timestamps
#         y_top_vis = y_top[mask]
#         y_bot_vis = y_bot[mask]

#         # Upper rail
#         upper = go.Scatter(
#             x=x_ts, y=y_top_vis,
#             mode="lines",
#             line=dict(width=rail_width, color=rail_color),
#             name="Flag Upper",
#             showlegend=False,
#         )
#         # Lower rail with fill to previous trace
#         lower = go.Scatter(
#             x=x_ts, y=y_bot_vis,
#             mode="lines",
#             line=dict(width=rail_width, color=rail_color),
#             fill="tonexty",
#             fillcolor=fill_rgba,
#             name="Flag Lower",
#             showlegend=False,
#         )
#         fig.add_trace(upper)
#         fig.add_trace(lower)

#     fig.update_layout(
#         title=f"Candles {start_idx} → {end_idx} (pivots + flag rails)",
#         xaxis_title="Time", yaxis_title="Price",
#         xaxis_rangeslider_visible=False,
#         template="plotly_dark",
#         legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
#         height=650, width=1100,
#     )
#     fig.show()

def add_spike_volume(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    For every candle, compare the CURRENT volume to the *average volume of the
    previous `lookback` bars* (exclusive).  
    Set `dry_volume = True` when that average is **lower** than the current
    candle's volume.

    Parameters
    ----------
    df : DataFrame
        Must contain a 'Volume' column.
    lookback : int
        Number of past bars to average.

    Returns
    -------
    df_out : DataFrame (new copy) with a Boolean column 'dry_volume'.
    """
    df_out = df.copy()

    # rolling mean of the PREVIOUS `lookback` volumes (shift(1) drops the current bar)
    prev_avg = (
        df_out["Volume"]
        .shift(1)
        .rolling(lookback, min_periods=lookback)
        .mean()
    )

    # True  → current vol > mean of previous vols
    df_out["volume_spike"] = df_out["Volume"] > prev_avg

    # For the first <lookback> rows, prev_avg is NaN → set dry_volume False
    df_out["volume_spike"].fillna(False, inplace=True)

    return df_out

def add_macd_signal(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Ensure MACD histogram is present and add Boolean column 'macd_signal'.

    macd_signal[t] is True  ⇢  hist[t-2] < hist[t-1]  AND  hist[t-1] > 0
    (Undefined / first two bars ⇒ False)

    Parameters
    ----------
    df : DataFrame with 'Close' prices.
    fast, slow, signal : int
        MACD parameters.

    Returns
    -------
    df_out : DataFrame copy with new columns:
        * 'machist'     (if not already present)
        * 'macd_signal' (Boolean)
    """
    df_out = df.copy()

    # MACD histogram – compute only if missing
    if "machist" not in df_out.columns:
        macd = ta.macd(df_out["Close"], fast=fast, slow=slow, signal=signal)
        df_out["machist"] = macd[f"MACD_{fast}_{slow}_{signal}"] - macd[f"MACDs_{fast}_{slow}_{signal}"]

    # Rising, positive histogram rule
    hist = df_out["machist"]
    cond = (hist.shift(2) < hist.shift(1)) & (hist.shift(1) > 0)

    df_out["macd_signal"] = cond.fillna(False)

    return df_out



