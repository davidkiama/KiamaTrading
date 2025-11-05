#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FVG + Key Level Breakout Strategy → Live OANDA Trading
Uses real-time 15m EURUSD data from OANDA
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

# ========================================
# 1. FVG DETECTION
# ========================================


def detect_fvg(data, lookback_period=10, body_multiplier=1.5):
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
        resistance_levels = [(idx, level) for (
            idx, level) in key_levels["resistance"] if idx < current_candle]
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
                for (lvl_idx, lvl_price) in key_levels.get("resistance", []):
                    if prev_open < lvl_price < prev_close:
                        df.loc[i, "break_signal"] = 2
                        break
            elif fvg_type == "bearish":
                for (lvl_idx, lvl_price) in key_levels.get("support", []):
                    if prev_open > lvl_price > prev_close:
                        df.loc[i, "break_signal"] = 1
                        break
    return df

# ========================================
# 4. FETCH LIVE CANDLES
# ========================================


def get_candles(n=200):
    client = CandleClient(access_token, real=False)  # Set real=True for live
    collector = client.get_collector(Pair.EUR_USD, Gran.M15)
    candles = collector.grab(n)
    return candles

# ========================================
# 5. TRADING JOB (Runs every 15 mins)
# ========================================


def trading_job():
    print(
        f"\n[{datetime.now(pytz.timezone('America/Chicago'))}] Running FVG Strategy...")

    # Fetch recent 15m candles
    candles = get_candles(n=200)
    if len(candles) < 50:
        print("Not enough candles. Skipping.")
        return

    # Build DataFrame
    dfstream = pd.DataFrame([{
        'Open': float(str(c.bid.o)),
        'High': float(str(c.bid.h)),
        'Low': float(str(c.bid.l)),
        'Close': float(str(c.bid.c)),
        'Volume': c.volume
    } for c in candles])

    dfstream.reset_index(drop=True, inplace=True)

    # ========================================
    # Run Strategy Logic
    # ========================================
    dfstream['FVG'] = detect_fvg(dfstream)
    dfstream = fill_key_levels(dfstream, backcandles=50, test_candles=10)
    dfstream = detect_break_signal(dfstream)

    latest_signal = dfstream['break_signal'].iloc[-1]
    current_price = dfstream['Close'].iloc[-1]
    prev_low = dfstream['Low'].iloc[-2]
    prev_high = dfstream['High'].iloc[-2]

    print(f"Latest Signal: {latest_signal} | Price: {current_price:.5f}")

    if latest_signal not in [1, 2]:
        print("No valid breakout signal.")
        return

    # ========================================
    # Risk Management
    # ========================================
    SLTPRatio = 1.8
    risk_units = 1000  # Fixed risk per trade (adjust as needed)

    if latest_signal == 2:  # BUY
        sl = prev_low
        tp = current_price + SLTPRatio * (current_price - sl)
        if tp <= current_price or sl >= current_price:
            print("Invalid SL/TP for BUY.")
            return

        mo = MarketOrderRequest(
            instrument="EUR_USD",
            units=risk_units,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
        )
        print(
            f"BUY SIGNAL: Entry ~{current_price:.5f}, TP={tp:.5f}, SL={sl:.5f}")

    elif latest_signal == 1:  # SELL
        sl = prev_high
        tp = current_price - SLTPRatio * (sl - current_price)
        if tp >= current_price or sl <= current_price:
            print("Invalid SL/TP for SELL.")
            return

        mo = MarketOrderRequest(
            instrument="EUR_USD",
            units=-risk_units,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
        )
        print(
            f"SELL SIGNAL: Entry ~{current_price:.5f}, TP={tp:.5f}, SL={sl:.5f}")

    # ========================================
    # Execute Order
    # ========================================
    try:
        client = API(access_token)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print("Order executed:", rv)
    except Exception as e:
        print("Order failed:", str(e))


# ========================================
# 6. SCHEDULER (Every 15 mins)
# ========================================
if __name__ == "__main__":
    scheduler = BlockingScheduler()
    # Run at 1,16,31,46 minutes past the hour (aligned with 15m candle close)
    scheduler.add_job(
        trading_job,
        'cron',
        day_of_week='mon-fri',
        hour='0-23',
        minute='1,16,31,46',
        timezone='America/Chicago'
    )
    print("FVG OANDA Trader Started. Waiting for next 15m candle...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Trader stopped.")
