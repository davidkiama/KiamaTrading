#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-11T07:41:47.692Z
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
# 2. MULTI-TIMEFRAME FVG BREAKOUT / SELL SIGNAL
# ========================================


def bullish_fvg_breakout_3tf(df_1h, df_15m):
    latest_fvg_1h = df_1h['FVG'].iloc[-1]
    # latest_fvg_30m = df_30m['FVG'].iloc[-1]
    latest_fvg_15m = df_15m['FVG'].iloc[-1]

    # if (latest_fvg_1h and latest_fvg_1h[0] == 'bullish' and
    #     latest_fvg_30m and latest_fvg_30m[0] == 'bullish' and
    #         latest_fvg_15m and latest_fvg_15m[0] == 'bullish'):
    #     return True
    if (latest_fvg_1h and latest_fvg_1h[0] == 'bullish' and
            latest_fvg_15m and latest_fvg_15m[0] == 'bullish'):
        return True
    return False


def bearish_fvg_sell_signal_3tf(df_1h, df_15m):
    latest_fvg_1h = df_1h['FVG'].iloc[-1]
    # latest_fvg_30m = df_30m['FVG'].iloc[-1]
    latest_fvg_15m = df_15m['FVG'].iloc[-1]

    # if (latest_fvg_1h and latest_fvg_1h[0] == 'bearish' and
    #     latest_fvg_30m and latest_fvg_30m[0] == 'bearish' and
    #         latest_fvg_15m and latest_fvg_15m[0] == 'bearish'):
    #     return True
    if (latest_fvg_1h and latest_fvg_1h[0] == 'bearish' and
            latest_fvg_15m and latest_fvg_15m[0] == 'bearish'):
        return True
    return False

# ========================================
# 3. FETCH LIVE CANDLES
# ========================================


def fetch_candles_multi_granularity(granularity, n=200):
    # Set real=True for live trading
    client = CandleClient(access_token, real=False)
    collector = client.get_collector(Pair.EUR_USD, granularity)
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
# 4. TRADING JOB (Runs every 15 mins)
# ========================================


def trading_job():
    print(f"\n[{datetime.now(pytz.timezone('America/Chicago'))}] Running multi-timeframe FVG Strategy...")

    # Fetch candles on 1H, 30M, and 15M
    df_1h = fetch_candles_multi_granularity(Gran.H1)
    # df_30m = fetch_candles_multi_granularity(Gran.M30)
    df_15m = fetch_candles_multi_granularity(Gran.M15)

    # bullish_buy = bullish_fvg_breakout_3tf(df_1h, df_30m, df_15m)
    # bearish_sell = bearish_fvg_sell_signal_3tf(df_1h, df_30m, df_15m)

    bullish_buy = bullish_fvg_breakout_3tf(df_1h,  df_15m)
    bearish_sell = bearish_fvg_sell_signal_3tf(df_1h,  df_15m)

    current_price = df_15m['Close'].iloc[-1]
    prev_low = df_15m['Low'].iloc[-2]
    prev_high = df_15m['High'].iloc[-2]

    print(f"Current Price (15m): {current_price:.5f}")

    SLTPRatio = 1.8
    risk_units = 4000  # Fixed units per trade, adjust as needed

    if bullish_buy:
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

    elif bearish_sell:
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
# 5. SCHEDULER (Every 15 mins)
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
    print("Multi-Timeframe FVG OANDA Trader Started. Waiting for next 15m candle...")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Trader stopped.")
