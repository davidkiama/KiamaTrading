#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FVG + Multi-Timeframe Breakout Strategy → Live OANDA Trading
Trades a configurable instrument (default USD_JPY) using 1H, 30M, and 15M candle data with FVG signals
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

# ← Change this to any instrument (e.g., "EUR_USD", "GBP_USD")
INSTRUMENT = "USD_JPY"
N = 15                   # Run interval in minutes


# ========================================
# 1. FVG DETECTION
# ========================================

def detect_fvg(data, lookback_period=14, body_multiplier=1.5):
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

        if third_low > first_high and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bullish', first_high, third_low, i))
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bearish', first_low, third_high, i))
        else:
            fvg_list.append(None)
    return fvg_list


# ========================================
# 2. MULTI-TIMEFRAME FVG BREAKOUT / SELL SIGNAL
# ========================================

def bullish_fvg_breakout_3tf(df_1h, df_30m, df_15m):
    latest_fvg_1h = df_1h['FVG'].iloc[-1]
    latest_fvg_30m = df_30m['FVG'].iloc[-1]
    latest_fvg_15m = df_15m['FVG'].iloc[-1]

    if (latest_fvg_1h and latest_fvg_1h[0] == 'bullish' and
        latest_fvg_30m and latest_fvg_30m[0] == 'bullish' and
            latest_fvg_15m and latest_fvg_15m[0] == 'bullish'):
        return True
    return False


def bearish_fvg_sell_signal_3tf(df_1h, df_30m, df_15m):
    latest_fvg_1h = df_1h['FVG'].iloc[-1]
    latest_fvg_30m = df_30m['FVG'].iloc[-1]
    latest_fvg_15m = df_15m['FVG'].iloc[-1]

    if (latest_fvg_1h and latest_fvg_1h[0] == 'bearish' and
        latest_fvg_30m and latest_fvg_30m[0] == 'bearish' and
            latest_fvg_15m and latest_fvg_15m[0] == 'bearish'):
        return True
    return False


# ========================================
# 3. FETCH LIVE CANDLES
# ========================================

def fetch_candles_multi_granularity(granularity, n=200):
    client = CandleClient(access_token, real=False)
    pair = getattr(Pair, INSTRUMENT)  # Dynamically use the instrument
    collector = client.get_collector(pair, granularity)
    candles = collector.grab(n)
    df = pd.DataFrame([{
        'Open': float(str(c.bid.o)),
        'High': float(str(c.bid.h)),
        'Low': float(str(c.bid.l)),
        'Close': float(str(c.bid.c)),
    } for c in candles])
    df.reset_index(drop=True, inplace=True)
    df['FVG'] = detect_fvg(df)

    return df


# ========================================
# 4. TRADING JOB (Runs every N mins)
# ========================================

def trading_job():
    print(f"\n[{datetime.now(pytz.timezone('America/Chicago'))}] Running {INSTRUMENT} Multi-Timeframe FVG Strategy...")

    df_1h = fetch_candles_multi_granularity(Gran.H1)
    df_30m = fetch_candles_multi_granularity(Gran.M30)
    df_15m = fetch_candles_multi_granularity(Gran.M15)

    bullish_buy = bullish_fvg_breakout_3tf(df_1h, df_30m, df_15m)
    bearish_sell = bearish_fvg_sell_signal_3tf(df_1h, df_30m, df_15m)

    current_price = df_15m['Close'].iloc[-1]
    prev_low = df_15m['Low'].iloc[-2]
    prev_high = df_15m['High'].iloc[-2]

    print(f"Current Price (15m): {current_price:.3f}")

    SLTPRatio = 1.8
    risk_units = 1000

    if bullish_buy:
        sl = prev_low
        tp = current_price + SLTPRatio * (current_price - sl)
        if tp <= current_price or sl >= current_price:
            print("Invalid SL/TP for BUY.")
            return

        mo = MarketOrderRequest(
            instrument=INSTRUMENT,
            units=risk_units,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp:.3f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl:.3f}").data
        )
        print(
            f"BUY SIGNAL [{INSTRUMENT}]: Entry ~{current_price:.3f}, TP={tp:.3f}, SL={sl:.3f}")

    elif bearish_sell:
        sl = prev_high
        tp = current_price - SLTPRatio * (sl - current_price)
        if tp >= current_price or sl <= current_price:
            print("Invalid SL/TP for SELL.")
            return

        mo = MarketOrderRequest(
            instrument=INSTRUMENT,
            units=-risk_units,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp:.3f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl:.3f}").data
        )
        print(
            f"SELL SIGNAL [{INSTRUMENT}]: Entry ~{current_price:.3f}, TP={tp:.3f}, SL={sl:.3f}")

    else:
        print("No valid trade signal at this time.")
        return

    try:
        client = API(access_token)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print("Order executed:", rv)
    except Exception as e:
        print("Order failed:", str(e))


# ========================================
# 5. SCHEDULER (Every N mins)
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

    print(f"{INSTRUMENT} Multi-Timeframe FVG OANDA Trader Started.")
    print(f"Running every {N} minutes at [{run_minutes}] past each hour...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Trader stopped.")
