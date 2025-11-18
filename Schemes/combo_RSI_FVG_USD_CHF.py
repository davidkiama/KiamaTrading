#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-Timeframe FVG + Support/Resistance Strategy → Live OANDA Trading
- FVG detection on 1H and 15M timeframes
- Support/Resistance identification on 30M timeframe
- RSI confirmation on 30M for entry timing
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

INSTRUMENT = "USD_CHF"
N = 15  # Run interval in minutes

# Support/Resistance Parameters
WICK_THRESHOLD = 0.0001
N1 = 8  # Candles before
N2 = 6  # Candles after
BACK_CANDLES = 140
PROXIMITY = 0.0001  # For merging close levels

# Risk Management
SLTP_RATIO = 2.0  # 1:2 risk-reward
RISK_UNITS = 1000
SL_BUFFER = 0.0005  # Buffer above resistance / below support


# ========================================
# 1. FVG DETECTION
# ========================================

def detect_fvg(data, lookback_period=14, body_multiplier=1.5):
    """Detect Fair Value Gaps in price data"""
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

        # Bullish FVG
        if third_low > first_high and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bullish', first_high, third_low, i))
        # Bearish FVG
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
            fvg_list.append(('bearish', first_low, third_high, i))
        else:
            fvg_list.append(None)

    return fvg_list


# ========================================
# 2. SUPPORT & RESISTANCE DETECTION
# ========================================

def support(df, l, n1, n2):
    """Check if candle at index l is a support level"""
    if l < n1 or l + n2 >= len(df):
        return 0

    # Check if it's the lowest point in the range
    if (df['Low'].iloc[l-n1:l].min() < df['Low'].iloc[l] or
            df['Low'].iloc[l+1:l+n2+1].min() < df['Low'].iloc[l]):
        return 0

    # Check for significant lower wick
    candle_body = abs(df['Open'].iloc[l] - df['Close'].iloc[l])
    lower_wick = min(df['Open'].iloc[l],
                     df['Close'].iloc[l]) - df['Low'].iloc[l]

    if (lower_wick > candle_body) and (lower_wick > WICK_THRESHOLD):
        return 1

    return 0


def resistance(df, l, n1, n2):
    """Check if candle at index l is a resistance level"""
    if l < n1 or l + n2 >= len(df):
        return 0

    # Check if it's the highest point in the range
    if (df['High'].iloc[l-n1:l].max() > df['High'].iloc[l] or
            df['High'].iloc[l+1:l+n2+1].max() > df['High'].iloc[l]):
        return 0

    # Check for significant upper wick
    candle_body = abs(df['Open'].iloc[l] - df['Close'].iloc[l])
    upper_wick = df['High'].iloc[l] - \
        max(df['Open'].iloc[l], df['Close'].iloc[l])

    if (upper_wick > candle_body) and (upper_wick > WICK_THRESHOLD):
        return 1

    return 0


def find_support_resistance_levels(df, n1, n2, back_candles):
    """Find all support and resistance levels in recent candles"""
    ss = []  # Support levels
    rr = []  # Resistance levels

    current_idx = len(df) - 1

    for i in range(current_idx - back_candles, current_idx - n2):
        if i < n1:
            continue

        if support(df, i, n1, n2):
            ss.append(df['Low'].iloc[i])
        if resistance(df, i, n1, n2):
            rr.append(df['High'].iloc[i])

    # Merge close support levels
    ss.sort()
    i = 0
    while i < len(ss) - 1:
        if abs(ss[i] - ss[i+1]) <= PROXIMITY:
            del ss[i+1]
        else:
            i += 1

    # Merge close resistance levels
    rr.sort(reverse=True)
    i = 0
    while i < len(rr) - 1:
        if abs(rr[i] - rr[i+1]) <= PROXIMITY:
            del rr[i]
        else:
            i += 1

    return ss, rr


def find_nearest_level(price, levels):
    """Find the nearest support or resistance level to current price"""
    if not levels:
        return None
    return min(levels, key=lambda x: abs(x - price))


def is_level_tested(df, level, lookback=6):
    """Check if price has recently tested a level"""
    recent_lows = df['Low'].iloc[-lookback:]
    recent_highs = df['High'].iloc[-lookback:]

    # Check if any recent candle touched the level
    for low, high in zip(recent_lows, recent_highs):
        if low <= level <= high:
            return True
    return False


# ========================================
# 3. RSI CALCULATION
# ========================================

def calculate_rsi(data, period=14):
    """Calculate RSI indicator"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ========================================
# 4. FETCH LIVE CANDLES
# ========================================

def fetch_candles_multi_granularity(granularity, n=200):
    """Fetch candles and add technical indicators"""
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
    df['FVG'] = detect_fvg(df)
    df['RSI'] = calculate_rsi(df)

    return df


# ========================================
# 5. TRADING SIGNAL GENERATION
# ========================================

def generate_trading_signal(df_1h, df_30m, df_15m):
    """
    Generate trading signals based on:
    1. FVG alignment on 1H and 15M
    2. Support/Resistance on 30M
    3. RSI confirmation on 30M
    """

    # Get latest FVG signals
    latest_fvg_1h = df_1h['FVG'].iloc[-1]
    latest_fvg_15m = df_15m['FVG'].iloc[-1]

    # Get RSI from 30M
    current_rsi_30m = df_30m['RSI'].iloc[-1]

    # Find support/resistance on 30M
    supports, resistances = find_support_resistance_levels(
        df_30m, N1, N2, BACK_CANDLES
    )

    current_price = df_15m['Close'].iloc[-1]

    print(f"\n=== Market Analysis ===")
    print(f"1H FVG: {latest_fvg_1h}")
    print(f"15M FVG: {latest_fvg_15m}")
    print(f"30M RSI: {current_rsi_30m:.2f}")
    print(f"Supports found: {len(supports)}")
    print(f"Resistances found: {len(resistances)}")

    # BULLISH SETUP
    if (latest_fvg_1h and latest_fvg_1h[0] == 'bullish' and
            latest_fvg_15m and latest_fvg_15m[0] == 'bullish'):

        print("\n✓ Bullish FVG alignment detected (1H + 15M)")

        # Find nearest support level
        nearest_support = find_nearest_level(current_price, supports)

        if nearest_support and nearest_support < current_price:
            # Check if support was recently tested
            if is_level_tested(df_30m, nearest_support):
                # RSI confirmation: oversold or recovering
                if current_rsi_30m < 55:
                    print(f"✓ Support level tested: {nearest_support:.5f}")
                    print(f"✓ RSI confirmation: {current_rsi_30m:.2f} < 55")

                    sl = nearest_support - SL_BUFFER
                    risk = current_price - sl
                    tp = current_price + (SLTP_RATIO * risk)

                    return 'BUY', current_price, sl, tp
                else:
                    print(f"✗ RSI too high: {current_rsi_30m:.2f} >= 55")
            else:
                print("✗ Support level not recently tested")
        else:
            print("✗ No valid support level found below price")

    # BEARISH SETUP
    elif (latest_fvg_1h and latest_fvg_1h[0] == 'bearish' and
          latest_fvg_15m and latest_fvg_15m[0] == 'bearish'):

        print("\n✓ Bearish FVG alignment detected (1H + 15M)")

        # Find nearest resistance level
        nearest_resistance = find_nearest_level(current_price, resistances)

        if nearest_resistance and nearest_resistance > current_price:
            # Check if resistance was recently tested
            if is_level_tested(df_30m, nearest_resistance):
                # RSI confirmation: overbought or declining
                if current_rsi_30m > 45:
                    print(
                        f"✓ Resistance level tested: {nearest_resistance:.5f}")
                    print(f"✓ RSI confirmation: {current_rsi_30m:.2f} > 45")

                    sl = nearest_resistance + SL_BUFFER
                    risk = sl - current_price
                    tp = current_price - (SLTP_RATIO * risk)

                    return 'SELL', current_price, sl, tp
                else:
                    print(f"✗ RSI too low: {current_rsi_30m:.2f} <= 45")
            else:
                print("✗ Resistance level not recently tested")
        else:
            print("✗ No valid resistance level found above price")

    return None, None, None, None


# ========================================
# 6. TRADING JOB
# ========================================

def trading_job():
    """Main trading logic executed every N minutes"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now(pytz.timezone('America/Chicago'))}]")
    print(f"Running Multi-TF Strategy for {INSTRUMENT}")
    print(f"{'='*60}")

    try:
        # Fetch data from all timeframes
        df_1h = fetch_candles_multi_granularity(Gran.H1)
        df_30m = fetch_candles_multi_granularity(Gran.M30)
        df_15m = fetch_candles_multi_granularity(Gran.M15)

        # Generate trading signal
        signal, entry, sl, tp = generate_trading_signal(df_1h, df_30m, df_15m)

        if signal is None:
            print("\n⊘ No valid trade signal at this time")
            return

        # Validate SL/TP levels
        if signal == 'BUY':
            if tp <= entry or sl >= entry:
                print("\n✗ Invalid SL/TP for BUY order")
                return
            units = RISK_UNITS
        else:  # SELL
            if tp >= entry or sl <= entry:
                print("\n✗ Invalid SL/TP for SELL order")
                return
            units = -RISK_UNITS

        # Create market order
        mo = MarketOrderRequest(
            instrument=INSTRUMENT,
            units=units,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
        )

        print(f"\n{'='*60}")
        print(f"🎯 {signal} SIGNAL [{INSTRUMENT}]")
        print(f"Entry: ~{entry:.5f}")
        print(f"Stop Loss: {sl:.5f}")
        print(f"Take Profit: {tp:.5f}")
        print(f"Risk: {abs(entry - sl):.5f}")
        print(f"Reward: {abs(tp - entry):.5f}")
        print(f"R:R Ratio: 1:{SLTP_RATIO}")
        print(f"{'='*60}")

        # Execute order
        client = API(access_token)
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = client.request(r)
        print("\n✓ Order executed successfully!")
        print(rv)

    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()


# ========================================
# 7. SCHEDULER
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

    print(f"\n{'='*60}")
    print(f"🤖 {INSTRUMENT} Multi-TF FVG + S/R Bot Started")
    print(f"{'='*60}")
    print(f"Strategy: FVG (1H+15M) + S/R (30M) + RSI (30M)")
    print(f"Interval: Every {N} minutes")
    print(f"Schedule: [{run_minutes}] past each hour")
    print(f"Risk:Reward: 1:{SLTP_RATIO}")
    print(f"{'='*60}\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n\n🛑 Trader stopped by user")
