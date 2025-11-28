#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FVG + Key Level Breakout Strategy → Live OANDA Trading
Uses real-time 15m data from OANDA with configurable instrument and risk settings.
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
from tqdm import tqdm
from trade_logger import log_executed_trade

# ========================================
# CONFIGURATION (EASY TO TWEAK)
# ========================================

# OANDA instrument (e.g., "XAU_USD", "EUR_USD", "USD_JPY")
INSTRUMENT = "USD_CHF"
LOT_SIZE = 1000             # Number of units (positive integer)
STOP_LOSS_PCT = 0.001
TAKE_PROFIT_PCT = 0.002
REAL_TRADING = False       # True = live trading, False = practice/sim

# ========================================
# 1. FVG DETECTION
# ========================================


def detect_fvg(data, lookback_period=14, body_multiplier=1.5):
    fvg_list = [None, None]
    for i in range(2, len(data)):
        first_high = data['High'].iloc[i-2]
        first_low = data['Low'].iloc[i-2]
        middle_open = data['Open'].iloc[i-1]
        middle_close = data['Close'].iloc[i-1]
        third_low = data['Low'].iloc[i]
        third_high = data['High'].iloc[i]

        prev_bodies = (data['Close'].iloc[max(0, i-1-lookback_period):i-1] -
                       data['Open'].iloc[max(0, i-1-lookback_period):i-1]).abs()
        avg_body_size = prev_bodies.mean()
        avg_body_size = avg_body_size if avg_body_size > 0 else 0.0001
        middle_body = abs(middle_close - middle_open)

        if third_low > first_high and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bullish', first_high, third_low, i))
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bearish', first_low, third_high, i))
        else:
            fvg_list.append(None)
    return fvg_list


# ========================================
# 2. KEY LEVEL DETECTION
# ========================================

def detect_key_levels(df, current_candle, backcandles=50, test_candles=10):
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


def fill_key_levels(df, backcandles=50, test_candles=10):
    df = df.copy()
    df["key_levels"] = None
    for current_candle in tqdm(range(backcandles + test_candles, len(df)), desc="Filling Key Levels"):
        key_levels = detect_key_levels(
            df, current_candle, backcandles, test_candles)
        support_levels = [(idx, level) for (idx, level)
                          in key_levels["support"] if idx < current_candle]
        resistance_levels = [(idx, level) for (idx, level)
                             in key_levels["resistance"] if idx < current_candle]
        if support_levels or resistance_levels:
            df.at[current_candle, "key_levels"] = {
                "support": support_levels,
                "resistance": resistance_levels
            }
    return df


# ========================================
# 3. BREAK SIGNAL DETECTION
# ========================================

def detect_break_signal(df):
    df = df.copy()
    df["break_signal"] = 0
    for i in range(1, len(df)):
        fvg = df.loc[i, "FVG"]
        key_levels = df.loc[i, "key_levels"]
        if isinstance(fvg, tuple) and isinstance(key_levels, dict):
            fvg_type = fvg[0]
            prev_open = df.loc[i-1, "Open"]
            prev_close = df.loc[i-1, "Close"]

            if fvg_type == "bullish":
                for (_, lvl_price) in key_levels.get("resistance", []):
                    if prev_open < lvl_price < prev_close:
                        df.loc[i, "break_signal"] = 2
                        break
            elif fvg_type == "bearish":
                for (_, lvl_price) in key_levels.get("support", []):
                    if prev_open > lvl_price > prev_close:
                        df.loc[i, "break_signal"] = 1
                        break
    return df


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
# 5. TRADING JOB
# ========================================

def trading_job():
    print(f"\n[{datetime.now(pytz.timezone('America/Chicago'))}] Running FVG Strategy on {INSTRUMENT}...")

    candles = get_candles(n=200)
    if len(candles) < 50:
        print("Not enough candles. Skipping.")
        return

    dfstream = pd.DataFrame([{
        'Open': float(str(c.bid.o)),
        'High': float(str(c.bid.h)),
        'Low': float(str(c.bid.l)),
        'Close': float(str(c.bid.c)),
    } for c in candles])

    dfstream.reset_index(drop=True, inplace=True)

    # Apply strategy logic
    dfstream['FVG'] = detect_fvg(dfstream)
    dfstream = fill_key_levels(dfstream, backcandles=50, test_candles=10)
    dfstream = detect_break_signal(dfstream)

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
    try:
        client = API(access_token)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        log_executed_trade(
            instrument=INSTRUMENT,
            signal="BUY",
            entry=current_price,
            sl=sl,
            tp=tp,
            timeframe="15M"
        )
        print("Order executed:", rv)
    except Exception as e:
        print("Order failed:", str(e))


# ========================================
# 6. SCHEDULER
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
        max_instances=2
    )
    print(
        f"FVG OANDA Trader Started for {INSTRUMENT}. Waiting for next 15m candle...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Trader stopped.")
