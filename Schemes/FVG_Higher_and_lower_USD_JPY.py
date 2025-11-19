#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FVG + Multi-Timeframe Breakout Strategy → Live OANDA Trading
Higher TF (1D, 4H) for trend, 15M for precise entry with 2 consecutive FVGs
"""

from config import access_token, accountID
from oanda_candles import Pair, Gran, CandleClient
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
import oandapyV20.endpoints.orders as orders
from oandapyV20 import API
from apscheduler.schedulers.blocking import BlockingScheduler
import pytz
from datetime import datetime
import pandas as pd
import numpy as np


# ========================================
# CONFIGURATION VARIABLES
# ========================================

INSTRUMENT = "USD_JPY"              # Trading instrument
N = 15                              # Run interval in minutes

# Timeframe Configuration
HIGHER_TF_1 = Gran.D                # 1 Day (primary trend)
HIGHER_TF_2 = Gran.H4               # 4 Hour (secondary trend)
ENTRY_TF = Gran.M15                 # 15 Minute (entry timeframe)

# FVG Detection Parameters
HTF_BODY_MULTIPLIER = 1.5           # Body size multiplier for higher timeframes
# Lower threshold for 15M FVGs (allows smaller FVGs)
ENTRY_TF_BODY_MULTIPLIER = 1.0

# Risk Management
SLTP_RATIO = 1.8                    # Risk-to-reward ratio
RISK_UNITS = 1000                   # Position size
# Buffer below swing low for stop loss (adjust for instrument)
SL_BUFFER_PIPS = 0.005


# ========================================
# 1. FVG DETECTION
# ========================================

def detect_fvg(data, lookback_period=14, body_multiplier=1.5):
    """
    Detects Fair Value Gaps in price data
    Returns list of FVG tuples: (direction, upper_bound, lower_bound, index)
    """
    fvg_list = [None, None]
    for i in range(2, len(data)):
        first_high = data['High'].iloc[i - 2]
        first_low = data['Low'].iloc[i - 2]
        middle_open = data['Open'].iloc[i - 1]
        middle_close = data['Close'].iloc[i - 1]
        third_low = data['Low'].iloc[i]
        third_high = data['High'].iloc[i]

        prev_bodies = (data['Close'].iloc[max(0, i - 1 - lookback_period):i - 1] -
                       data['Open'].iloc[max(0, i - 1 - lookback_period):i - 1]).abs()
        avg_body_size = prev_bodies.mean()
        avg_body_size = avg_body_size if avg_body_size > 0 else 0.0001
        middle_body = abs(middle_close - middle_open)

        # Bullish FVG: Gap between first candle high and third candle low
        if third_low > first_high and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bullish', first_high, third_low, i))
        # Bearish FVG: Gap between first candle low and third candle high
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bearish', first_low, third_high, i))
        else:
            fvg_list.append(None)
    return fvg_list


# ========================================
# 2. CHECK FOR TWO CONSECUTIVE FVGs ON 15M
# ========================================

def has_two_consecutive_fvgs(df_15m, direction='bullish'):
    """
    Checks if the last two candles both have FVGs in the same direction
    Returns: (bool, first_fvg, second_fvg)
    """
    latest_fvg = df_15m['FVG'].iloc[-1]
    previous_fvg = df_15m['FVG'].iloc[-2]

    if (latest_fvg and latest_fvg[0] == direction and
            previous_fvg and previous_fvg[0] == direction):
        return True, previous_fvg, latest_fvg

    return False, None, None


# ========================================
# 3. MULTI-TIMEFRAME ALIGNMENT CHECK
# ========================================

def check_bullish_alignment(df_1d, df_4h, df_15m):
    """
    Checks for bullish alignment:
    - 1D has bullish FVG (can be bigger)
    - 4H has bullish FVG (can be bigger)
    - 15M has TWO consecutive bullish FVGs (can be smaller)
    """
    latest_fvg_1d = df_1d['FVG'].iloc[-1]
    latest_fvg_4h = df_4h['FVG'].iloc[-1]

    # Check higher timeframes for bullish FVG
    htf_bullish = (latest_fvg_1d and latest_fvg_1d[0] == 'bullish' and
                   latest_fvg_4h and latest_fvg_4h[0] == 'bullish')

    if not htf_bullish:
        return False, None, None

    # Check for two consecutive bullish FVGs on 15M
    two_fvgs, first_fvg, second_fvg = has_two_consecutive_fvgs(
        df_15m, 'bullish')

    return two_fvgs, first_fvg, second_fvg


def check_bearish_alignment(df_1d, df_4h, df_15m):
    """
    Checks for bearish alignment:
    - 1D has bearish FVG (can be bigger)
    - 4H has bearish FVG (can be bigger)
    - 15M has TWO consecutive bearish FVGs (can be smaller)
    """
    latest_fvg_1d = df_1d['FVG'].iloc[-1]
    latest_fvg_4h = df_4h['FVG'].iloc[-1]

    # Check higher timeframes for bearish FVG
    htf_bearish = (latest_fvg_1d and latest_fvg_1d[0] == 'bearish' and
                   latest_fvg_4h and latest_fvg_4h[0] == 'bearish')

    if not htf_bearish:
        return False, None, None

    # Check for two consecutive bearish FVGs on 15M
    two_fvgs, first_fvg, second_fvg = has_two_consecutive_fvgs(
        df_15m, 'bearish')

    return two_fvgs, first_fvg, second_fvg


# ========================================
# 4. FETCH LIVE CANDLES
# ========================================

def fetch_candles_multi_granularity(granularity, body_multiplier=1.5, n=50):
    """
    Fetches candles for specified granularity and adds FVG detection
    """
    client = CandleClient(access_token, real=False)
    pair = getattr(Pair, INSTRUMENT)
    collector = client.get_collector(pair, granularity)
    candles = collector.grab(n)
    df = pd.DataFrame([{
        'Open': float(str(c.bid.o)),
        'High': float(str(c.bid.h)),
        'Low': float(str(c.bid.l)),
        'Close': float(str(c.bid.c)),
    } for c in candles])
    df.reset_index(drop=True, inplace=True)
    df['FVG'] = detect_fvg(df, body_multiplier=body_multiplier)

    return df


# ========================================
# 5. TRADING JOB (Runs every N mins)
# ========================================

def trading_job():
    print(f"\n[{datetime.now(pytz.timezone('America/Chicago'))}] Running {INSTRUMENT} Multi-Timeframe FVG Strategy...")

    # Fetch data from all timeframes
    df_1d = fetch_candles_multi_granularity(
        HIGHER_TF_1, body_multiplier=HTF_BODY_MULTIPLIER)
    df_4h = fetch_candles_multi_granularity(
        HIGHER_TF_2, body_multiplier=HTF_BODY_MULTIPLIER)
    df_15m = fetch_candles_multi_granularity(
        ENTRY_TF, body_multiplier=ENTRY_TF_BODY_MULTIPLIER)

    # Check for alignment
    bullish_signal, first_bull_fvg, second_bull_fvg = check_bullish_alignment(
        df_1d, df_4h, df_15m)
    bearish_signal, first_bear_fvg, second_bear_fvg = check_bearish_alignment(
        df_1d, df_4h, df_15m)

    # Current price is the open of the candle that created the second FVG (last closed candle's close)
    entry_price = df_15m['Close'].iloc[-1]

    print(f"Current Price (15M): {entry_price:.5f}")

    # BULLISH ENTRY
    if bullish_signal:
        # Stop loss: slightly below the swing low of the FIRST FVG on 15M
        # The first FVG's lower bound is at index -2
        # Low of the candle where first FVG formed
        swing_low = df_15m['Low'].iloc[-2]
        sl = swing_low - SL_BUFFER_PIPS

        # Take profit based on risk-reward ratio
        risk = entry_price - sl
        tp = entry_price + (SLTP_RATIO * risk)

        if tp <= entry_price or sl >= entry_price:
            print("Invalid SL/TP for BUY.")
            return

        mo = MarketOrderRequest(
            instrument=INSTRUMENT,
            units=RISK_UNITS,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
        )
        print(f"✅ BUY SIGNAL [{INSTRUMENT}]:")
        print(f"   Entry: {entry_price:.5f}")
        print(f"   Stop Loss: {sl:.5f} (Swing Low: {swing_low:.5f})")
        print(f"   Take Profit: {tp:.5f}")
        print(f"   Risk/Reward: 1:{SLTP_RATIO}")
        print(f"   1D FVG: Bullish | 4H FVG: Bullish | 15M: 2 Consecutive Bullish FVGs")

    # BEARISH ENTRY
    elif bearish_signal:
        # Stop loss: slightly above the swing high of the FIRST FVG on 15M
        swing_high = df_15m['High'].iloc[-2]
        sl = swing_high + SL_BUFFER_PIPS

        # Take profit based on risk-reward ratio
        risk = sl - entry_price
        tp = entry_price - (SLTP_RATIO * risk)

        if tp >= entry_price or sl <= entry_price:
            print("Invalid SL/TP for SELL.")
            return

        mo = MarketOrderRequest(
            instrument=INSTRUMENT,
            units=-RISK_UNITS,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
        )
        print(f"✅ SELL SIGNAL [{INSTRUMENT}]:")
        print(f"   Entry: {entry_price:.5f}")
        print(f"   Stop Loss: {sl:.5f} (Swing High: {swing_high:.5f})")
        print(f"   Take Profit: {tp:.5f}")
        print(f"   Risk/Reward: 1:{SLTP_RATIO}")
        print(f"   1D FVG: Bearish | 4H FVG: Bearish | 15M: 2 Consecutive Bearish FVGs")

    else:
        print("❌ No valid trade signal at this time.")
        return

    # Execute the order
    try:
        client = API(access_token)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print("✅ Order executed successfully:", rv)
    except Exception as e:
        print("❌ Order failed:", str(e))


# ========================================
# 6. SCHEDULER (Every N mins)
# ========================================

if __name__ == "__main__":
    run_minutes = ",".join(str(i) for i in range(1, 60, N))

    scheduler = BlockingScheduler()
    scheduler.add_job(
        trading_job,
        'cron',
        day_of_week='mon-fri',
        hour='0-23',
        minute=run_minutes,
        timezone='America/Chicago',
        misfire_grace_time=120,
        max_instances=2
    )

    print(f"🚀 {INSTRUMENT} Multi-Timeframe FVG OANDA Trader Started")
    print(
        f"📊 Timeframes: {HIGHER_TF_1} (HTF1), {HIGHER_TF_2} (HTF2), {ENTRY_TF} (Entry)")
    print(f"⏰ Running every {N} minutes at [{run_minutes}] past each hour")
    print(f"💰 Risk Units: {RISK_UNITS} | R:R Ratio: 1:{SLTP_RATIO}")
    print("="*60)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Trader stopped.")
