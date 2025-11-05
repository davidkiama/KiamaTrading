#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: FVG_Strategy with Support-Resistance.ipynb
Conversion Date: 2025-11-05T12:13:06.347Z
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from tqdm import tqdm

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

from backtesting import Strategy, Backtest

import yfinance as yf


dataF = yf.download("EURUSD=X", period="2y", interval="1h")

dataF

dataF = dataF.reset_index()            # Move Datetime from index to column
dataF.rename(columns={'Datetime': 'Gmt time'}, inplace=True)
dataF

my_data = dataF[['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']]


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the 'Ticker' row level from a Yahoo Finance DataFrame
    that has multi-index columns like ('EURUSD=X', 'Open').

    Returns a DataFrame with simple column names like 'Open', 'High', etc.
    """
    # If columns are MultiIndex (e.g. ('EURUSD=X', 'Open')), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


my_data = flatten_yf_columns(my_data)
my_data


def detect_fvg(data, lookback_period=10, body_multiplier=1.5):
    """
    Detects Fair Value Gaps (FVGs) in historical price data.

    Parameters:
        data (DataFrame): DataFrame with columns ['open', 'high', 'low', 'close'].
        lookback_period (int): Number of candles to look back for average body size.
        body_multiplier (float): Multiplier to determine significant body size.

    Returns:
        list of tuples: Each tuple contains ('type', start, end, index).
    """
    fvg_list = [None, None]

    for i in range(2, len(data)):
        first_high = data['High'].iloc[i-2]
        first_low = data['Low'].iloc[i-2]
        middle_open = data['Open'].iloc[i-1]
        middle_close = data['Close'].iloc[i-1]
        third_low = data['Low'].iloc[i]
        third_high = data['High'].iloc[i]

        # Calculate the average absolute body size over the lookback period
        prev_bodies = (data['Close'].iloc[max(0, i-1-lookback_period):i-1] -
                       data['Open'].iloc[max(0, i-1-lookback_period):i-1]).abs()
        avg_body_size = prev_bodies.mean()

        # Ensure avg_body_size is nonzero to avoid false positives
        avg_body_size = avg_body_size if avg_body_size > 0 else 0.001

        middle_body = abs(middle_close - middle_open)

        # Check for Bullish FVG
        if third_low > first_high and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bullish', first_high, third_low, i))

        # Check for Bearish FVG
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bearish', first_low, third_high, i))

        else:
            fvg_list.append(None)

    return fvg_list

my_data['FVG'] = detect_fvg(my_data)
my_data.head(20)


df = my_data

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
dfpl = df[450:540]
# Create the figure
fig = go.Figure()

# Add candlestick chart
fig.add_trace(go.Candlestick(
    x=dfpl.index,
    open=dfpl["Open"],
    high=dfpl["High"],
    low=dfpl["Low"],
    close=dfpl["Close"],
    name="Candles"
))

# Add FVG zones
for _, row in dfpl.iterrows():
    if isinstance(row["FVG"], tuple):
        fvg_type, start, end, index = row["FVG"]
        color = "rgba(0,255,0,0.3)" if fvg_type == "bullish" else "rgba(255,0,0,0.3)"
        fig.add_shape(
            type="rect",
            x0=index - 2,
            x1=index + 30,
            y0=start,
            y1=end,
            fillcolor=color,
            opacity=0.8,
            layer="below",
            line=dict(width=0),
        )

# Show the chart
fig.update_layout(width=1200, height=800,
                  xaxis=dict(showgrid=False),
                  yaxis=dict(showgrid=False),
                  plot_bgcolor='black',
                  paper_bgcolor='black')
fig.show()

def detect_key_levels(df, current_candle, backcandles=50, test_candles=10):
    """
    Detects key support and resistance levels in a given backcandles window.
    
    A level is identified if a candle's high is the highest or its low is the lowest 
    compared to `test_candles` before and after it.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        current_candle (int): The index of the current candle (latest available candle).
        backcandles (int): Number of candles to look back.
        test_candles (int): Number of candles before and after each candle to check.

    Returns:
        dict: A dictionary with detected 'support' and 'resistance' levels.
    """
    key_levels = {"support": [], "resistance": []}

    # Define the last candle that can be tested to avoid lookahead bias
    last_testable_candle = current_candle - test_candles

    # Ensure we have enough data
    if last_testable_candle < backcandles + test_candles:
        return key_levels  # Not enough historical data

    # Iterate through the backcandles window
    for i in range(current_candle - backcandles, last_testable_candle):
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]

        # Get surrounding window of test_candles before and after
        before = df.iloc[max(0, i - test_candles):i]
        after = df.iloc[i + 1: min(len(df), i + test_candles + 1)]

        # Check if current high is the highest among before & after candles
        if high > before['High'].max() and high > after['High'].max():
            key_levels["resistance"].append((i, high))

        # Check if current low is the lowest among before & after candles
        if low < before['Low'].min() and low < after['Low'].min():
            key_levels["support"].append((i, low))

    return key_levels

def fill_key_levels(df, backcandles=50, test_candles=10):
    """
    Adds a 'key_levels' column to the DataFrame where each row contains all
    key support and resistance levels detected up to that candle (including
    both the level value and the index of the candle that generated it).
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        backcandles (int): Lookback window for detecting key levels.
        test_candles (int): Number of candles before/after for validation.

    Returns:
        pd.DataFrame: Updated DataFrame with the new 'key_levels' column.
    """
    df["key_levels"] = None  # Initialize the column

    from tqdm import tqdm
    for current_candle in tqdm(range(backcandles + test_candles, len(df))):
        # Detect key levels for the current candle
        key_levels = detect_key_levels(
            df, current_candle, backcandles, test_candles)

        # Collect support and resistance levels (with their indices) up to current_candle
        support_levels = [(idx, level) for (idx, level) in key_levels["support"]
                          if idx < current_candle]
        resistance_levels = [(idx, level) for (idx, level) in key_levels["resistance"]
                             if idx < current_candle]

        # Store the levels along with the originating candle index
        if support_levels or resistance_levels:
            df.at[current_candle, "key_levels"] = {
                "support": support_levels,
                "resistance": resistance_levels
            }

    return df


df = fill_key_levels(df, backcandles=50, test_candles=10)

df

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_fvg_and_key_levels(df, start_idx, end_idx, extension=30):
    """
    Plots candlesticks, FVG zones, and key levels (support/resistance) for a
    subset of a DataFrame from `start_idx` to `end_idx`.
    
    The FVG column is expected to have tuples of the form:
        (fvg_type, start_price, end_price, trigger_index)

    The key_levels column is expected to have dictionaries of the form:
        {
          "support": [(idx, price), (idx, price), ...],
          "resistance": [(idx, price), (idx, price), ...]
        }

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain: "Open", "High", "Low", "Close", "FVG", "key_levels".
    start_idx : int
        Starting row index for plotting.
    end_idx : int
        Ending row index for plotting.
    extension : int
        How far (in x-axis units/index steps) to extend the FVG rectangles
        and key-level lines.
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        A Plotly Figure with the candlesticks, FVG, and key-level lines.
    """

    # Slice the DataFrame to the desired plotting range
    dfpl = df.loc[start_idx:end_idx]

    # Create the figure
    fig = go.Figure()

    # -- 1) Add Candlestick Chart --
    fig.add_trace(go.Candlestick(
        x=dfpl.index,
        open=dfpl["Open"],
        high=dfpl["High"],
        low=dfpl["Low"],
        close=dfpl["Close"],
        name="Candles"
    ))

    # -- 2) Add FVG Zones --
    for i, row in dfpl.iterrows():
        # Check if "FVG" is a valid tuple: (fvg_type, start_price, end_price, trigger_index)
        if isinstance(row.get("FVG"), tuple):
            fvg_type, start_price, end_price, trigger_idx = row["FVG"]

            # Choose a fill color based on bullish vs. bearish
            if fvg_type == "bullish":
                color = "rgba(0, 255, 0, 0.3)"   # greenish
            else:
                color = "rgba(255, 0, 0, 0.3)"   # reddish

            fig.add_shape(
                type="rect",
                x0=trigger_idx,
                x1=trigger_idx + extension,
                y0=start_price,
                y1=end_price,
                fillcolor=color,
                opacity=0.4,
                layer="below",
                line=dict(width=0),
            )

    # -- 3) Add Key Levels as Horizontal Lines --
    for i, row in dfpl.iterrows():
        key_levels = row.get("key_levels", None)
        if key_levels:
            # key_levels is a dict: {"support": [(idx, val), ...], "resistance": [(idx, val), ...]}
            support_levels = key_levels.get("support", [])
            resistance_levels = key_levels.get("resistance", [])

            # Plot support levels
            for (gen_idx, s_price) in support_levels:
                # We only draw the line if gen_idx is in (start_idx, end_idx)
                # You can decide to relax/omit this check if you want lines from outside the window.
                if start_idx <= gen_idx <= end_idx:
                    fig.add_shape(
                        type="line",
                        x0=gen_idx,
                        x1=gen_idx + extension,
                        y0=s_price,
                        y1=s_price,
                        line=dict(color="blue", width=2),
                        layer="below"
                    )

            # Plot resistance levels
            for (gen_idx, r_price) in resistance_levels:
                if start_idx <= gen_idx <= end_idx:
                    fig.add_shape(
                        type="line",
                        x0=gen_idx,
                        x1=gen_idx + extension,
                        y0=r_price,
                        y1=r_price,
                        line=dict(color="orange", width=2),
                        layer="below"
                    )

    # -- 4) Figure Aesthetics --
    fig.update_layout(
        width=1200,
        height=800,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        plot_bgcolor='black',
        paper_bgcolor='black'
    )

    fig.show()
    return fig


fig = plot_fvg_and_key_levels(df, start_idx=440, end_idx=590, extension=30)


def detect_break_signal(df):
    """
    Detects if the current candle carries an FVG signal and,
    at the same time, the previous candle has crossed a key level
    in the expected direction (up for bullish, down for bearish).

    - If FVG is bullish and previous candle crosses ABOVE a level -> signal = 2
    - If FVG is bearish and previous candle crosses BELOW a level -> signal = 1
    - Otherwise -> signal = 0

    The 'FVG' column is expected to have tuples like:
        (fvg_type, lower_price, upper_price, trigger_index)
      where fvg_type is either "bullish" or "bearish".

    The 'key_levels' column is expected to be a dictionary with:
        {
            'support': [(level_candle_idx, level_price), ...],
            'resistance': [(level_candle_idx, level_price), ...]
        }
    """

    # Initialize the new signal column to 0
    df["break_signal"] = 0

    # We start at 1 because we compare candle i with its previous candle (i-1)
    for i in range(1, len(df)):
        fvg = df.loc[i, "FVG"]
        key_levels = df.loc[i, "key_levels"]

        # We only proceed if there's an FVG tuple and some key_levels dict
        if isinstance(fvg, tuple) and isinstance(key_levels, dict):
            fvg_type = fvg[0]  # "bullish" or "bearish"

            # Previous candle's OHLC
            prev_open = df.loc[i-1, "Open"]
            prev_close = df.loc[i-1, "Close"]

            # -----------------------
            # 1) Bullish FVG check
            # -----------------------
            if fvg_type == "bullish":
                # Typically you'd check crossing a "resistance" level
                # crossing means the previous candle goes from below -> above
                resistance_levels = key_levels.get("resistance", [])

                for (lvl_idx, lvl_price) in resistance_levels:
                    # Condition: previously below, ended above
                    # simplest check is: prev_open < lvl_price < prev_close
                    if prev_open < lvl_price and prev_close > lvl_price:
                        df.loc[i, "break_signal"] = 2
                        break  # No need to check more levels

            # -----------------------
            # 2) Bearish FVG check
            # -----------------------
            elif fvg_type == "bearish":
                # Typically you'd check crossing a "support" level
                support_levels = key_levels.get("support", [])

                for (lvl_idx, lvl_price) in support_levels:
                    # Condition: previously above, ended below
                    # simplest check is: prev_open > lvl_price and prev_close < lvl_price
                    if prev_open > lvl_price and prev_close < lvl_price:
                        df.loc[i, "break_signal"] = 1
                        break  # No need to check more levels

    return df


df = detect_break_signal(df)

# Now df["break_signal"] is set to:
#  - 2 if the candle's FVG is bullish and previous candle crosses up,
#  - 1 if the candle's FVG is bearish and previous candle crosses down,
#  - 0 otherwise.

df[df["break_signal"] != 0]




def pointpos(x):
    if x['break_signal'] == 2:
        return x['Low']-1e-4
    elif x['break_signal'] == 1:
        return x['High']+1e-4
    else:
        return np.nan


df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)

st = 30
end = 250
fig = plot_fvg_and_key_levels(df, start_idx=st, end_idx=end, extension=30)

fig.add_scatter(x=df.index[st:end], y=df['pointpos'][st:end], mode="markers",
                marker=dict(size=8, color="MediumPurple"),
                name="pivot")

from backtesting import Strategy, Backtest
import numpy as np

spread_threshold_value = 0.000  # Placeholder for spread/commission


def SIGNAL():
    return df.break_signal


class MyStrat(Strategy):
    risk_percent = 0.01  # Risk % of equity per trade
    tp_sl_ratio = 1.8  # Take-profit to stop-loss ratio

    def init(self):
        super().init()
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next()
        spread_threshold = spread_threshold_value  # Add spread buffer if applicable
        equity = self.equity  # Current account equity

        pip_size = 0.0001  # For EURUSD, 1 pip = 0.0001
        exchange_rate = self.data.Close[-1]
        # pip_value_per_unit = pip_size / exchange_rate  # Value of 1 pip per unit of asset

        # -------------------------------------------------------
        # LONG POSITION LOGIC
        # -------------------------------------------------------
        if self.signal1[-1] == 2 and not self.position:
            previous_low = self.data.Low[-2]
            current_close = self.data.Close[-1]
            sl = previous_low  # Stop-loss at the low of the current candle
            tp = current_close + self.tp_sl_ratio * \
                (current_close - previous_low)  # TP calculation

            sl_distance = current_close - sl  # Stop-loss distance in price terms
            if sl_distance <= 5e-4:
                return  # Avoid invalid SL/TP configuration

            # Dollar risk: % of equity
            risk_amount = equity * self.risk_percent

            # Calculate position size in units of the asset
            size_in_units = risk_amount * exchange_rate / sl_distance

            # print(sl_distance, exchange_rate, equity, risk_amount, size_in_units)

            # Check the condition to open a position
            if tp > current_close + spread_threshold > sl + 2 * spread_threshold:
                self.buy(size=int(size_in_units), sl=sl, tp=tp)

        # -------------------------------------------------------
        # SHORT POSITION LOGIC
        # -------------------------------------------------------
        elif self.signal1[-1] == 1 and not self.position:
            previous_high = self.data.High[-2]
            current_close = self.data.Close[-1]
            sl = previous_high  # Stop-loss at the high of the current candle
            tp = current_close - self.tp_sl_ratio * \
                (previous_high - current_close)  # TP calculation

            sl_distance = sl - current_close  # Stop-loss distance in price terms
            if sl_distance <= 5e-4:
                return  # Avoid invalid SL/TP configuration

            # Dollar risk: % of equity
            risk_amount = equity * self.risk_percent

            # Calculate position size in units of the asset
            size_in_units = risk_amount * exchange_rate / sl_distance

            # print(sl_distance, size_in_units)

            # Check the condition to open a position
            if tp + 2 * spread_threshold < current_close + spread_threshold < sl:
                self.sell(size=int(size_in_units), sl=sl, tp=tp)


df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='%d.%m.%Y %H:%M:%S.%f')

# New: Set the 'Gmt time' column as the DataFrame index
# This resolves the 'Data index is not datetime' UserWarning.
df.set_index('Gmt time', inplace=True)

# Also, rename columns to the expected capitalization (optional but recommended for clarity)
df.rename(columns={
    'Open': 'Open',
    'High': 'High',
    'Low': 'Low',
    'Close': 'Close',
    'Volume': 'Volume'
}, inplace=True)

bt_df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'break_signal']].copy()


# -------------------------------------------------------
# RUN THE BACKTEST
# -------------------------------------------------------
bt = Backtest(bt_df, MyStrat, cash=10000, margin=1 /
              50, commission=spread_threshold_value)
stats = bt.optimize(tp_sl_ratio=np.arange(1.0, 2.2, 0.1).tolist(),  # 1.0 to 3.0 in steps of 0.1
                    # or whichever metric you want to maximize
                    maximize='Return [%]'
                    )
stats

stats._strategy

bt.plot()

from backtesting import Strategy, Backtest
import numpy as np

spread_threshold_value = 0.000


def SIGNAL():
    return df.break_signal


class MyStrat(Strategy):
    mysize = 0.05  # Trade size 5% of the account
    tp_sl_ratio = 1.5

    def init(self):
        super().init()
        # Assuming SIGNAL is a function that returns signals
        self.signal1 = self.I(SIGNAL)

    def next(self):
        super().next()
        spread_threshold = spread_threshold_value
        if self.signal1[-1] == 2 and not self.position:
            # Open a new long position with calculated SL
            previous_low = self.data.Low[-2]
            current_close = self.data.Close[-1]
            sl = previous_low  # SL at the low of the current candle
            tp = current_close + self.tp_sl_ratio * \
                (current_close - previous_low)

            # Check the TP > Close > SL condition
            if tp > current_close+spread_threshold > sl + 2*spread_threshold:
                self.buy(size=self.mysize, sl=sl, tp=tp)

        elif self.signal1[-1] == 1 and not self.position:
            # Open a new short position with calculated SL
            previous_high = self.data.High[-2]
            current_close = self.data.Close[-1]
            sl = previous_high  # SL at the high of the current candle
            tp = current_close - self.tp_sl_ratio * \
                (previous_high - current_close)

            # Check the TP < Close < SL condition
            if tp + 2*spread_threshold < current_close + spread_threshold < sl:
                self.sell(size=self.mysize, sl=sl, tp=tp)


# bt = Backtest(df, MyStrat, cash=10000, margin=1/50, commission=spread_threshold_value)
bt = Backtest(bt_df, MyStrat, cash=10000, margin=1 /
              50, commission=spread_threshold_value)
stats = bt.optimize(tp_sl_ratio=np.arange(1.0, 2.5, 0.1).tolist(),  # 1.0 to 3.0 in steps of 0.1
                    # or whichever metric you want to maximize
                    maximize='Return [%]'
                    )

stats