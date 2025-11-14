#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OANDA Live Trading Bot - Multi-Timeframe FVG Strategy
Instrument: GBP/USD
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
# CONFIGURATION
# ========================================
STRATEGY_NAME = "Multi_TF_FVG"
INSTRUMENT = "GBP_USD"
TRADE_SIZE = 1000
SLTP_RATIO = 1.8
USE_LIVE = False  # Set True for live trading

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
        avg_body_size = prev_bodies.mean() or 0.0001
        middle_body = abs(middle_close - middle_open)

        if third_low > first_high and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bullish', first_high, third_low, i))
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bearish', first_low, third_high, i))
        else:
            fvg_list.append(None)
    return fvg_list

# ========================================
# 2. MULTI-TIMEFRAME SIGNAL LOGIC
# ========================================


def bullish_fvg_breakout_3tf(df_1h, df_15m):
    latest_fvg_1h = df_1h['FVG'].iloc[-1]
    latest_fvg_15m = df_15m['FVG'].iloc[-1]
    if (latest_fvg_1h and latest_fvg_1h[0] == 'bullish' and
            latest_fvg_15m and latest_fvg_15m[0] == 'bullish'):
        return True
    return False


def bearish_fvg_sell_signal_3tf(df_1h, df_15m):
    latest_fvg_1h = df_1h['FVG'].iloc[-1]
    latest_fvg_15m = df_15m['FVG'].iloc[-1]
    if (latest_fvg_1h and latest_fvg_1h[0] == 'bearish' and
            latest_fvg_15m and latest_fvg_15m[0] == 'bearish'):
        return True
    return False

# ========================================
# 3. FETCH LIVE CANDLES
# ========================================


def fetch_candles_multi_granularity(instrument_pair, granularity, n=200):
    client = CandleClient(access_token, real=USE_LIVE)
    collector = client.get_collector(instrument_pair, granularity)
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
# 4. TRADING JOB
# ========================================


def trading_job():
    print(f"\n{'='*65}")
    print(f"[{datetime.now(pytz.timezone('America/Chicago'))}] Running {STRATEGY_NAME}")
    print(f"Instrument: {INSTRUMENT}")
    print(f"Timeframes: 1H + 15M")
    print(f"{'='*65}")

    # Fetch candles for 1H and 15M
    instrument_pair = getattr(Pair, INSTRUMENT)
    df_1h = fetch_candles_multi_granularity(instrument_pair, Gran.H1)
    df_15m = fetch_candles_multi_granularity(instrument_pair, Gran.M15)

    bullish_buy = bullish_fvg_breakout_3tf(df_1h, df_15m)
    bearish_sell = bearish_fvg_sell_signal_3tf(df_1h, df_15m)

    current_price = df_15m['Close'].iloc[-1]
    prev_low = df_15m['Low'].iloc[-2]
    prev_high = df_15m['High'].iloc[-2]

    print(f"Current Price (15m): {current_price:.5f}")
    print(f"Bullish Signal: {bullish_buy} | Bearish Signal: {bearish_sell}")

    if not bullish_buy and not bearish_sell:
        print("No valid trade signal at this time.")
        return

    # Compute SL/TP
    if bullish_buy:
        sl = prev_low
        tp = current_price + SLTP_RATIO * (current_price - sl)
        units = TRADE_SIZE
        direction = "BUY"
        print(f"\n🟢 BUY SIGNAL DETECTED ({INSTRUMENT})")
        print(f"Entry: {current_price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
        print(f"Risk/Reward ≈ 1:{SLTP_RATIO:.2f}")

    elif bearish_sell:
        sl = prev_high
        tp = current_price - SLTP_RATIO * (sl - current_price)
        units = -TRADE_SIZE
        direction = "SELL"
        print(f"\n🔴 SELL SIGNAL DETECTED ({INSTRUMENT})")
        print(f"Entry: {current_price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}")
        print(f"Risk/Reward ≈ 1:{SLTP_RATIO:.2f}")

    # SL/TP validation
    if (direction == "BUY" and (tp <= current_price or sl >= current_price)) or \
       (direction == "SELL" and (tp >= current_price or sl <= current_price)):
        print("❌ Invalid SL/TP detected. Trade skipped.")
        return

    # Prepare order
    mo = MarketOrderRequest(
        instrument=INSTRUMENT,
        units=units,
        takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
        stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
    )

    try:
        client = API(access_token)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print(f"\n✅ Order executed successfully!")
        print(f"Details: {rv}")
    except Exception as e:
        print(f"❌ Order failed: {str(e)}")

# ========================================
# 5. SCHEDULER
# ========================================


if __name__ == "__main__":
    print("="*65)
    print("OANDA MULTI-TIMEFRAME FVG STRATEGY BOT")
    print("="*65)
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Instrument: {INSTRUMENT}")
    print(f"Trade Size: {TRADE_SIZE} units")
    print(f"SL/TP Ratio: {SLTP_RATIO}")
    print(f"Mode: {'LIVE' if USE_LIVE else 'PRACTICE'}")
    print("="*65)
    print("🤖 Bot initialized. Waiting for next 15m candle...\n")

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

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Trader stopped by user.")
        print("="*65)
