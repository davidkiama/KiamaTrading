#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: multi_tf_combo.ipynb
Conversion Date: 2025-11-20T15:07:58.799Z
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FVG + Multi-Timeframe Bias + 1H S/R Strategy → Live OANDA Trading
Trades a configurable instrument using a multi-timeframe FVG bias and 1H S/R levels.
"""
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from config import access_token, accountID
from oanda_candles import Pair, Gran, CandleClient

# ========================================
# CONFIGURATION VARIABLES
# ========================================
OANDA_FX_ASSETS = [
    "EUR_USD", "USD_JPY", "GBP_USD", "USD_CHF",
    "AUD_USD", "NZD_USD", "USD_CAD", "EUR_GBP"
]


INSTRUMENT = "USD_JPY"
N = 15                   # Run interval in minutes
SLTPRatio = 2.0          # Your specified 1:2 Risk/Reward ratio
risk_units = 1000        # Units to trade
SR_LOOKBACK_CANDLES = 50  # Lookback for 1H S/R detection
SR_TEST_CANDLES = 10     # Candles before/after for 1H S/R validation
M15_CANDLES_FOR_CROSS = 200  # Check for S/R cross over last 200 15M candles
SL_BUFFER_FACTOR = 0.0001  # Small buffer for Stop Loss (e.g., 1 pip)

# ========================================
# 1. FVG DETECTION (RETAINED)
# ========================================


def detect_fvg(data, lookback_period=14, body_multiplier=1.5, min_gap_val=0.2):
    """
    Detects Fair Value Gaps (FVGs) in historical price data.
    """
    fvg_list = [None, None]
    for i in range(2, len(data)):
        first_high = data['High'].iloc[i-2]
        first_low = data['Low'].iloc[i-2]
        middle_open = data['Open'].iloc[i-1]
        middle_close = data['Close'].iloc[i-1]
        third_low = data['Low'].iloc[i]
        third_high = data['High'].iloc[i]
        middle_body = abs(middle_close - middle_open)
        prev_bodies = (data['Close'].iloc[max(0, i-1-lookback_period):i-1] -
                       data['Open'].iloc[max(0, i-1-lookback_period):i-1]).abs()
        avg_body_size = prev_bodies.mean()
        avg_body_size = avg_body_size if avg_body_size > 0 else 0.001
        min_gap = middle_body * min_gap_val

        # Check Bullish FVG
        if third_low > first_high:
            gap_size = third_low - first_high
            if (middle_body > avg_body_size * body_multiplier and gap_size > min_gap):
                fvg_list.append(('bullish', first_high, third_low, i))
                continue
        # Check Bearish FVG
        if third_high < first_low:
            gap_size = first_low - third_high
            if (middle_body > avg_body_size * body_multiplier and gap_size > min_gap):
                fvg_list.append(('bearish', first_low, third_high, i))
                continue
        fvg_list.append(None)
    return fvg_list

# ========================================
# 2. 1H SUPPORT/RESISTANCE DETECTION
# ========================================


def detect_key_levels(df, backcandles=SR_LOOKBACK_CANDLES, test_candles=SR_TEST_CANDLES):
    """
    Detects key support and resistance levels on the 1H timeframe.
    """
    key_levels = {"support": [], "resistance": []}
    current_candle = len(df) - 1  # Use the latest candle as the current point

    # Iterate through a window before the last 'test_candles'
    for i in range(current_candle - backcandles - test_candles, current_candle - test_candles + 1):
        if i < test_candles or i >= len(df) - test_candles:
            continue

        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]

        before_high = df['High'].iloc[max(0, i - test_candles):i].max()
        after_high = df['High'].iloc[i +
                                     1: min(len(df), i + test_candles + 1)].max()

        before_low = df['Low'].iloc[max(0, i - test_candles):i].min()
        after_low = df['Low'].iloc[i +
                                   1: min(len(df), i + test_candles + 1)].min()

        # Check if current high is the highest among before & after candles
        if high == max(df['High'].iloc[i - test_candles: i + test_candles + 1]):
            key_levels["resistance"].append(high)

        # Check if current low is the lowest among before & after candles
        if low == min(df['Low'].iloc[i - test_candles: i + test_candles + 1]):
            key_levels["support"].append(low)

    # Return only unique levels
    return {
        "support": sorted(list(set(key_levels["support"])), reverse=True),
        "resistance": sorted(list(set(key_levels["resistance"])))
    }

# ========================================
# 3. TRADING SIGNAL GENERATION
# ========================================


def generate_signal(df_d1, df_h4, df_h1, df_m30, df_m15, h1_levels):

    # Step 1: Check FVG in at least two or more of the higher timeframes
    higher_tfs = [df_d1, df_h4, df_h1, df_m30]

    bullish_count = 0
    bearish_count = 0

    for df in higher_tfs:
        latest_fvg = df['FVG'].iloc[-1]
        if latest_fvg and latest_fvg[0] == 'bullish':
            bullish_count += 1
        elif latest_fvg and latest_fvg[0] == 'bearish':
            bearish_count += 1

    # Determine trend/bias
    current_bias = None
    if bullish_count >= 2:
        current_bias = 'BULLISH'
    elif bearish_count >= 2:
        current_bias = 'BEARISH'

    if current_bias is None:
        return None, None, None, "No dominant multi-timeframe FVG bias (need >= 2)."

    # Step 2: Check 15M FVG (at least 2 FVGs in the same direction)
    m15_fvg_list = df_m15['FVG'].iloc[-4:-1].tolist()
    m15_directional_fvg_count = sum(
        1 for fvg in m15_fvg_list if fvg and fvg[0] == current_bias.lower()
    )

    if m15_directional_fvg_count < 2:
        return None, None, None, f"{current_bias} bias but < 2 directional 15M FVGs."

    # ============================
    # ✔ NO MORE S/R CROSS CHECKS
    # ============================

    # Entry = Current 15M Close
    current_close = df_m15['Close'].iloc[-1]
    entry_level = current_close

    # Use nearest opposite H1 S/R for stop loss
    stop_loss = None

    if current_bias == 'BULLISH':
        # SL under nearest support
        if h1_levels["support"]:
            sl_support = min(h1_levels["support"])
            stop_loss = sl_support - SL_BUFFER_FACTOR
        else:
            return None, None, None, "No H1 support available for SL."

        return 'BUY', entry_level, stop_loss, "BULLISH bias + 15M confirmation."

    elif current_bias == 'BEARISH':
        # SL above nearest resistance
        if h1_levels["resistance"]:
            sl_resistance = max(h1_levels["resistance"])
            stop_loss = sl_resistance + SL_BUFFER_FACTOR
        else:
            return None, None, None, "No H1 resistance available for SL."

        return 'SELL', entry_level, stop_loss, "BEARISH bias + 15M confirmation."

    return None, None, None, "Unexpected condition."

# ========================================
# 4. FETCH LIVE CANDLES (UPDATED FOR D1/H4)
# ========================================


def fetch_candles_multi_granularity(granularity, n=100):
    client = CandleClient(access_token, real=False)
    pair = getattr(Pair, INSTRUMENT)
    collector = client.get_collector(pair, granularity)
    # Fetch extra candles for FVG calculation and S/R lookback
    candles = collector.grab(n + SR_LOOKBACK_CANDLES)
    df = pd.DataFrame([{
        'Open': float(str(c.bid.o)),
        'High': float(str(c.bid.h)),
        'Low': float(str(c.bid.l)),
        'Close': float(str(c.bid.c)),
    } for c in candles])
    df.reset_index(drop=True, inplace=True)
    df['FVG'] = detect_fvg(df)
    return df.iloc[-n:]  # Return only the required number of candles

# ========================================
# 5. TRADING JOB (Runs every N mins - UPDATED)
# ========================================


def trading_job():
    print(f"\n[{datetime.now(pytz.timezone('America/Chicago'))}] Running {INSTRUMENT} Multi-Timeframe FVG Strategy...")

    # Fetch data for all required timeframes
    # We only need the latest 200 candles for analysis
    # NOTE: D1/H4/H1 are fetched with a lookback for S/R and FVG calc
    df_d1 = fetch_candles_multi_granularity(Gran.D, n=30)
    df_h4 = fetch_candles_multi_granularity(Gran.H4, n=70)
    df_h1 = fetch_candles_multi_granularity(Gran.H1, n=100)
    df_m30 = fetch_candles_multi_granularity(Gran.M30, n=100)
    df_m15 = fetch_candles_multi_granularity(Gran.M15, n=100)

    # Step 2: Determine 1H Support and Resistance
    h1_levels = detect_key_levels(df_h1)

    # Generate the signal
    signal, entry_level, sl_price, reason = generate_signal(
        df_d1, df_h4, df_h1, df_m30, df_m15, h1_levels)
    current_price = df_m15['Close'].iloc[-1]

    print(f"Bias Check: Bullish {len([df for df in [df_d1, df_h4, df_h1, df_m30] if df['FVG'].iloc[-1] and df['FVG'].iloc[-1][0] == 'bullish'])}, Bearish {len([df for df in [df_d1, df_h4, df_h1, df_m30] if df['FVG'].iloc[-1] and df['FVG'].iloc[-1][0] == 'bearish'])}")
    print(f"1H Support: {h1_levels['support']}")
    print(f"1H Resistance: {h1_levels['resistance']}")
    print(f"Current Price (15m): {current_price:.5f}")
    print(f"Signal Check: {reason}")

    if signal == 'BUY' and entry_level and sl_price:
        # Step 4: Calculate Take Profit (1:2 R:R)
        risk = current_price - sl_price
        tp_price = current_price + SLTPRatio * risk

        if tp_price <= current_price or sl_price >= current_price:
            print("Invalid SL/TP for BUY.")
            return

        mo = MarketOrderRequest(
            instrument=INSTRUMENT,
            units=risk_units,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp_price:.5f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl_price:.5f}").data
        )
        print(
            f"BUY SIGNAL [{INSTRUMENT}]: Entry ~{current_price:.5f}, TP={tp_price:.5f}, SL={sl_price:.5f}, R:R=1:{SLTPRatio}")
    elif signal == 'SELL' and entry_level and sl_price:
        # Step 4: Calculate Take Profit (1:2 R:R)
        risk = sl_price - current_price
        tp_price = current_price - SLTPRatio * risk

        if tp_price >= current_price or sl_price <= current_price:
            print("Invalid SL/TP for SELL.")
            return

        mo = MarketOrderRequest(
            instrument=INSTRUMENT,
            units=-risk_units,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp_price:.5f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl_price:.5f}").data
        )
        print(
            f"SELL SIGNAL [{INSTRUMENT}]: Entry ~{current_price:.5f}, TP={tp_price:.5f}, SL={sl_price:.5f}, R:R=1:{SLTPRatio}")
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
# 6. SCHEDULER (Every N mins - RETAINED)
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
        # Ensure your OANDA config is set up before running this live script
        # The script is currently set to use the practice/demo environment (real=False)
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Trader stopped.")
