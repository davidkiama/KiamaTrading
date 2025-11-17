#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FVG + Key Level Breakout Strategy → Live OANDA Trading
Optimized with caching to process only new candles
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from oanda_candles import Pair, Gran, CandleClient
from config import access_token, accountID

# ========================================
# CONFIGURATION (EASY TO TWEAK)
# ========================================

INSTRUMENT = "USD_CHF"
LOT_SIZE = 1000
STOP_LOSS_PCT = 0.0002
TAKE_PROFIT_PCT = 0.0006
REAL_TRADING = False

# ========================================
# GLOBAL CACHE
# ========================================


class StrategyCache:
    def __init__(self):
        self.dfstream = None
        self.last_candle_time = None
        self.initialized = False

    def needs_full_calculation(self, new_candles):
        """Check if we need full recalculation or just incremental"""
        if not self.initialized or self.dfstream is None:
            return True
        # Check if data structure changed significantly
        if len(new_candles) < len(self.dfstream) - 10:
            return True
        return False

    def update(self, dfstream):
        """Update cache with new dataframe"""
        self.dfstream = dfstream.copy()
        self.initialized = True


cache = StrategyCache()

# ========================================
# 1. FVG DETECTION (Optimized for single candle)
# ========================================


def detect_fvg_single(data, index, lookback_period=14, body_multiplier=1.5):
    """Detect FVG for a single candle at given index"""
    if index < 2:
        return None

    first_high = data['High'].iloc[index-2]
    first_low = data['Low'].iloc[index-2]
    middle_open = data['Open'].iloc[index-1]
    middle_close = data['Close'].iloc[index-1]
    third_low = data['Low'].iloc[index]
    third_high = data['High'].iloc[index]

    prev_bodies = (data['Close'].iloc[max(0, index-1-lookback_period):index-1] -
                   data['Open'].iloc[max(0, index-1-lookback_period):index-1]).abs()
    avg_body_size = prev_bodies.mean()
    avg_body_size = avg_body_size if avg_body_size > 0 else 0.0001
    middle_body = abs(middle_close - middle_open)

    if third_low > first_high and middle_body > avg_body_size * body_multiplier:
        return ('bullish', first_high, third_low, index)
    elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
        return ('bearish', first_low, third_high, index)
    return None


def detect_fvg_bulk(data, lookback_period=14, body_multiplier=1.5):
    """Detect FVG for entire dataframe (initial load only)"""
    fvg_list = [None, None]
    for i in range(2, len(data)):
        fvg_list.append(detect_fvg_single(
            data, i, lookback_period, body_multiplier))
    return fvg_list


# ========================================
# 2. KEY LEVEL DETECTION (Optimized)
# ========================================

def detect_key_levels_single(df, current_candle, backcandles=50, test_candles=10):
    """Detect key levels for a single candle"""
    key_levels = {"support": [], "resistance": []}
    last_testable_candle = current_candle - test_candles

    if last_testable_candle < backcandles + test_candles:
        return key_levels

    for i in range(current_candle - backcandles, last_testable_candle):
        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]
        before = df.iloc[max(0, i - test_candles):i]
        after = df.iloc[i + 1: min(len(df), i + test_candles + 1)]

        if high > before['High'].max() and high > after['High'].max():
            key_levels["resistance"].append((i, high))
        if low < before['Low'].min() and low < after['Low'].min():
            key_levels["support"].append((i, low))

    return key_levels


def fill_key_levels_bulk(df, backcandles=50, test_candles=10):
    """Fill key levels for entire dataframe (initial load only)"""
    df = df.copy()
    df["key_levels"] = None

    for current_candle in range(backcandles + test_candles, len(df)):
        key_levels = detect_key_levels_single(
            df, current_candle, backcandles, test_candles)
        support_levels = [(idx, level) for (idx, level)
                          in key_levels["support"] if idx < current_candle]
        resistance_levels = [(idx, level) for (
            idx, level) in key_levels["resistance"] if idx < current_candle]

        if support_levels or resistance_levels:
            df.at[current_candle, "key_levels"] = {
                "support": support_levels,
                "resistance": resistance_levels
            }
    return df


# ========================================
# 3. BREAK SIGNAL DETECTION (Single candle)
# ========================================

def detect_break_signal_single(df, index):
    """Detect break signal for a single candle"""
    if index < 1:
        return 0

    fvg = df.loc[index, "FVG"]
    key_levels = df.loc[index, "key_levels"]

    if not isinstance(fvg, tuple) or not isinstance(key_levels, dict):
        return 0

    fvg_type = fvg[0]
    prev_open = df.loc[index-1, "Open"]
    prev_close = df.loc[index-1, "Close"]

    if fvg_type == "bullish":
        for (_, lvl_price) in key_levels.get("resistance", []):
            if prev_open < lvl_price < prev_close:
                return 2  # BUY
    elif fvg_type == "bearish":
        for (_, lvl_price) in key_levels.get("support", []):
            if prev_open > lvl_price > prev_close:
                return 1  # SELL
    return 0


# ========================================
# 4. FETCH LIVE CANDLES
# ========================================

def get_candles(n=200):
    client = CandleClient(access_token, real=REAL_TRADING)
    pair_map = getattr(Pair, INSTRUMENT, None)
    if pair_map is None:
        raise ValueError(f"Unsupported instrument: {INSTRUMENT}")
    collector = client.get_collector(pair_map, Gran.M15)
    candles = collector.grab(n)
    return candles


# ========================================
# 5. PROCESS DATA (Incremental or Full)
# ========================================

def process_data_incremental(new_candles):
    """Process only new candles using cached data"""
    global cache

    # Convert new candles to dataframe
    dfstream = pd.DataFrame([{
        'Open': float(str(c.bid.o)),
        'High': float(str(c.bid.h)),
        'Low': float(str(c.bid.l)),
        'Close': float(str(c.bid.c)),
    } for c in new_candles])
    dfstream.reset_index(drop=True, inplace=True)

    # Check if we need full recalculation
    if cache.needs_full_calculation(new_candles):
        print("Performing full calculation (first run or data mismatch)...")
        dfstream['FVG'] = detect_fvg_bulk(dfstream)
        dfstream = fill_key_levels_bulk(
            dfstream, backcandles=50, test_candles=10)
        dfstream['break_signal'] = 0
        for i in range(1, len(dfstream)):
            dfstream.loc[i, 'break_signal'] = detect_break_signal_single(
                dfstream, i)
        cache.update(dfstream)
        return dfstream

    # Incremental update: only process new candles
    print("Performing incremental update (processing new candle only)...")

    # Get the number of new candles
    num_existing = len(cache.dfstream)
    num_new = len(dfstream)

    if num_new <= num_existing:
        # No new candles, just return cached data
        return cache.dfstream

    # Start with cached data
    result_df = cache.dfstream.copy()

    # Append new rows
    new_rows_start = num_existing
    for i in range(new_rows_start, num_new):
        new_row = dfstream.iloc[i:i+1].copy()
        result_df = pd.concat([result_df, new_row], ignore_index=True)

        # Calculate FVG for new candle
        new_index = len(result_df) - 1
        result_df.at[new_index, 'FVG'] = detect_fvg_single(
            result_df, new_index)

        # Calculate key levels for new candle
        key_levels = detect_key_levels_single(
            result_df, new_index, backcandles=50, test_candles=10)
        support_levels = [(idx, level) for (idx, level)
                          in key_levels["support"] if idx < new_index]
        resistance_levels = [(idx, level) for (idx, level)
                             in key_levels["resistance"] if idx < new_index]

        if support_levels or resistance_levels:
            result_df.at[new_index, "key_levels"] = {
                "support": support_levels,
                "resistance": resistance_levels
            }
        else:
            result_df.at[new_index, "key_levels"] = None

        # Calculate break signal for new candle
        result_df.at[new_index, 'break_signal'] = detect_break_signal_single(
            result_df, new_index)

    # Update cache
    cache.update(result_df)
    return result_df


# ========================================
# 6. TRADING JOB
# ========================================

def trading_job():
    try:
        print(
            f"\n[{datetime.now(pytz.timezone('America/Chicago'))}] Running FVG Strategy on {INSTRUMENT}...")

        candles = get_candles(n=200)
        if len(candles) < 50:
            print("Not enough candles. Skipping.")
            return

        # Process data (incremental if possible)
        dfstream = process_data_incremental(candles)

        latest_signal = dfstream['break_signal'].iloc[-1]
        current_price = dfstream['Close'].iloc[-1]

        print(f"Latest Signal: {latest_signal} | Price: {current_price:.5f}")

        if latest_signal not in [1, 2]:
            print("No valid breakout signal.")
            return

        # Calculate SL/TP dynamically using percentage
        if latest_signal == 2:  # BUY
            sl = current_price * (1 - STOP_LOSS_PCT)
            tp = current_price * (1 + TAKE_PROFIT_PCT)
            mo = MarketOrderRequest(
                instrument=INSTRUMENT,
                units=LOT_SIZE,
                takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
                stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
            )
            print(
                f"BUY SIGNAL: Entry={current_price:.5f}, TP={tp:.5f}, SL={sl:.5f}")

        elif latest_signal == 1:  # SELL
            sl = current_price * (1 + STOP_LOSS_PCT)
            tp = current_price * (1 - TAKE_PROFIT_PCT)
            mo = MarketOrderRequest(
                instrument=INSTRUMENT,
                units=-LOT_SIZE,
                takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
                stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
            )
            print(
                f"SELL SIGNAL: Entry={current_price:.5f}, TP={tp:.5f}, SL={sl:.5f}")

        # Execute trade
        client = API(access_token)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print("Order executed:", rv)

    except Exception as e:
        print(f"Error in trading_job: {str(e)}")
        import traceback
        traceback.print_exc()


# ========================================
# 7. SCHEDULER
# ========================================
if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(
        trading_job,
        'cron',
        day_of_week='mon-fri',
        hour='0-23',
        minute='1,16,31,46',
        timezone='America/Chicago',
        misfire_grace_time=120,
        coalesce=True,  # Skip duplicate missed runs
        max_instances=2  # Only one instance at a time
    )
    print(
        f"FVG OANDA Trader Started for {INSTRUMENT}. Waiting for next 15m candle...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Trader stopped.")
