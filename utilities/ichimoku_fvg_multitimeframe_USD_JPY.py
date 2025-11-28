#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ichimoku FVG Multi-Timeframe OANDA Trading System
Combines Ichimoku retracement detection with multi-timeframe FVG bias
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from oandapyV20 import API
import oandapyV20.endpoints.orders as orders
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from config import access_token, accountID
from oanda_candles import Pair, Gran, CandleClient
from trade_logger import log_executed_trade


# ========================================
# CONFIGURATION VARIABLES
# ========================================
INSTRUMENT = "USD_JPY"
N = 15  # Run interval in minutes
SLTPRatio = 2.0  # Risk/Reward ratio
risk_units = 1000  # Units to trade
SL_BUFFER_FACTOR = 0.0001  # Small buffer for Stop Loss

# Ichimoku parameters
TENKAN = 9
KIJUN = 26
SENKOU_B = 52
ATR_LEN = 14
ATR_MULT_SL = 1.5
ATR_MULT_TP = 2.0

# FVG parameters
FVG_LOOKBACK = 14
FVG_BODY_MULT = 1.5
FVG_MIN_GAP = 0.2

# S/R detection
SR_LOOKBACK_CANDLES = 50
SR_TEST_CANDLES = 10

# ========================================
# 1. ICHIMOKU CLOUD CALCULATION
# ========================================


def add_ichimoku(df, tenkan=TENKAN, kijun=KIJUN, senkou_b=SENKOU_B):
    """Build bias-safe Ichimoku columns for signal logic."""
    out = df.copy()

    h, l, c = out["High"], out["Low"], out["Close"]

    # Calculate Tenkan & Kijun
    tenkan_series = (h.rolling(tenkan).max() + l.rolling(tenkan).min()) / 2.0
    kijun_series = (h.rolling(kijun).max() + l.rolling(kijun).min()) / 2.0

    # Calculate raw spans (no forward shift)
    span_a_raw = (tenkan_series + kijun_series) / 2.0
    span_b_raw = (h.rolling(senkou_b).max() + l.rolling(senkou_b).min()) / 2.0

    out["ich_tenkan"] = tenkan_series
    out["ich_kijun"] = kijun_series
    out["ich_spanA"] = span_a_raw
    out["ich_spanB"] = span_b_raw

    # ATR for stop loss calculation
    out["ATR"] = ta.atr(out["High"], out["Low"], out["Close"], length=ATR_LEN)

    # Cloud boundaries
    cloud_top = out[["ich_spanA", "ich_spanB"]].max(axis=1)
    cloud_bot = out[["ich_spanA", "ich_spanB"]].min(axis=1)
    out["cloud_top"] = cloud_top
    out["cloud_bot"] = cloud_bot

    return out.dropna(subset=["ich_tenkan", "ich_kijun", "ich_spanA", "ich_spanB", "ATR"])


# ========================================
# 2. ICHIMOKU RETRACEMENT SIGNAL
# ========================================

def detect_ichimoku_retracement(df, lookback=7):
    """
    Detect Ichimoku retracement signals:
    - Price pulls back to touch/support cloud
    - Tenkan/Kijun alignment with cloud
    - Bounce potential indicated by cloud structure

    Returns: 'bullish', 'bearish', or None
    """
    if len(df) < lookback:
        return None

    latest = df.iloc[-1]
    cloud_top = latest["cloud_top"]
    cloud_bot = latest["cloud_bot"]
    close = latest["Close"]
    tenkan = latest["ich_tenkan"]
    kijun = latest["ich_kijun"]

    # Recent candles
    recent = df.iloc[-lookback:]

    # Check for bullish retracement:
    # Price pulled back into/near cloud, lines crossing up
    bullish_retrace = (
        (close > cloud_bot and close < cloud_top) and  # Price in cloud
        (tenkan > kijun) and  # Tenkan above Kijun (uptrend structure)
        (recent["Close"].min() >= cloud_bot) and  # Didn't break below support
        recent[recent["Close"] > close].shape[0] >= 1  # Some upside momentum
    )

    # Check for bearish retracement:
    # Price pulled back into/near cloud, lines crossing down
    bearish_retrace = (
        (close > cloud_bot and close < cloud_top) and  # Price in cloud
        (tenkan < kijun) and  # Tenkan below Kijun (downtrend structure)
        # Didn't break above resistance
        (recent["Close"].max() <= cloud_top) and
        recent[recent["Close"] < close].shape[0] >= 1  # Some downside momentum
    )

    if bullish_retrace:
        return 'bullish'
    elif bearish_retrace:
        return 'bearish'
    else:
        return None


# ========================================
# 3. FVG DETECTION
# ========================================

def detect_fvg(data, lookback_period=FVG_LOOKBACK, body_multiplier=FVG_BODY_MULT,
               min_gap_val=FVG_MIN_GAP):
    """Detects Fair Value Gaps (FVGs) in historical price data."""
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
        avg_body_size = prev_bodies.mean() if len(prev_bodies) > 0 else 0.001
        avg_body_size = max(avg_body_size, 0.001)
        min_gap = middle_body * min_gap_val

        # Bullish FVG
        if third_low > first_high:
            gap_size = third_low - first_high
            if middle_body > avg_body_size * body_multiplier and gap_size > min_gap:
                fvg_list.append(('bullish', first_high, third_low, i))
                continue

        # Bearish FVG
        if third_high < first_low:
            gap_size = first_low - third_high
            if middle_body > avg_body_size * body_multiplier and gap_size > min_gap:
                fvg_list.append(('bearish', first_low, third_high, i))
                continue

        fvg_list.append(None)

    return fvg_list


# ========================================
# 4. S/R DETECTION
# ========================================

def detect_key_levels(df, backcandles=SR_LOOKBACK_CANDLES, test_candles=SR_TEST_CANDLES):
    """Detects key support and resistance levels."""
    key_levels = {"support": [], "resistance": []}
    current_candle = len(df) - 1

    for i in range(max(0, current_candle - backcandles - test_candles),
                   min(len(df) - test_candles, current_candle + 1)):
        if i < test_candles or i >= len(df) - test_candles:
            continue

        high = df['High'].iloc[i]
        low = df['Low'].iloc[i]

        # Resistance: local maximum
        if high == df['High'].iloc[max(0, i - test_candles):min(len(df), i + test_candles + 1)].max():
            key_levels["resistance"].append(high)

        # Support: local minimum
        if low == df['Low'].iloc[max(0, i - test_candles):min(len(df), i + test_candles + 1)].min():
            key_levels["support"].append(low)

    return {
        "support": sorted(list(set(key_levels["support"])), reverse=True),
        "resistance": sorted(list(set(key_levels["resistance"])))
    }


# ========================================
# 5. MULTI-TIMEFRAME BIAS DETECTION
# ========================================

def check_multitf_bias(df_d1, df_h4, df_h1, df_m30):
    """
    Check for dominant bias from multiple timeframes (D1, H4, H1, 30M).
    Need 2 or more FVGs in the same direction.

    Returns: 'BULLISH', 'BEARISH', or None
    """
    higher_tfs = [df_d1, df_h4, df_h1, df_m30]
    bullish_count = 0
    bearish_count = 0

    for df in higher_tfs:
        latest_fvg = df['FVG'].iloc[-1]
        if latest_fvg is not None:
            if latest_fvg[0] == 'bullish':
                bullish_count += 1
            elif latest_fvg[0] == 'bearish':
                bearish_count += 1

    if bullish_count >= 2:
        return 'BULLISH'
    elif bearish_count >= 2:
        return 'BEARISH'
    else:
        return None


# ========================================
# 6. GENERATE TRADE SIGNAL
# ========================================

def generate_signal(df_d1, df_h4, df_h1, df_m30, df_m15, m30_levels):
    """
    Generate trading signal based on:
    1. Multi-TF bias (2+ FVGs in higher TFs)
    2. 15M FVG alignment with bias (1 FVG minimum)
    3. Ichimoku retracement confirmation on 15M
    4. 30M S/R for SL/TP placement
    """

    # Step 1: Check multi-TF bias
    bias = check_multitf_bias(df_d1, df_h4, df_h1, df_m30)
    if bias is None:
        return None, None, None, None, "No dominant multi-TF bias (need >= 2 FVGs)"

    # Step 2: Check 15M FVG (need at least 1 in same direction)
    m15_latest_fvg = df_m15['FVG'].iloc[-1]

    if m15_latest_fvg is None:
        return None, None, None, None, f"{bias} bias but no 15M FVG"

    if m15_latest_fvg[0] != bias.lower():
        return None, None, None, None, f"{bias} bias but 15M FVG is opposite direction"

    # Step 3: Check Ichimoku retracement on 15M
    retrace_signal = detect_ichimoku_retracement(df_m15, lookback=7)

    if retrace_signal is None:
        return None, None, None, None, f"{bias} bias + 15M FVG but no Ichimoku retrace"

    if retrace_signal != bias.lower():
        return None, None, None, None, f"{bias} bias but Ichimoku shows opposite retrace"

    # Step 4: Determine entry, SL, and TP
    current_close = df_m15['Close'].iloc[-1]
    current_atr = df_m15['ATR'].iloc[-1]
    entry_level = current_close

    if bias == 'BULLISH':
        # SL below nearest support
        if not m30_levels["support"]:
            return None, None, None, None, "No 30M support for SL"

        sl_support = min(m30_levels["support"])
        stop_loss = sl_support - SL_BUFFER_FACTOR
        risk = entry_level - stop_loss
        take_profit = entry_level + (risk * SLTPRatio)

        return 'BUY', entry_level, stop_loss, take_profit, \
               f"✓ BULLISH: Multi-TF bias + 15M FVG + Ichimoku retrace"

    elif bias == 'BEARISH':
        # SL above nearest resistance
        if not m30_levels["resistance"]:
            return None, None, None, None, "No 30M resistance for SL"

        sl_resistance = max(m30_levels["resistance"])
        stop_loss = sl_resistance + SL_BUFFER_FACTOR
        risk = stop_loss - entry_level
        take_profit = entry_level - (risk * SLTPRatio)

        return 'SELL', entry_level, stop_loss, take_profit, \
               f"✓ BEARISH: Multi-TF bias + 15M FVG + Ichimoku retrace"

    return None, None, None, None, "Unexpected condition"


# ========================================
# 7. FETCH CANDLES (MULTI-GRANULARITY)
# ========================================

def fetch_candles_multi_granularity(granularity, n=100):
    """Fetch candles and add technical indicators."""
    try:
        client = CandleClient(access_token, real=False)
        pair = getattr(Pair, INSTRUMENT)
        collector = client.get_collector(pair, granularity)

        # Fetch extra candles for indicator warmup
        fetch_count = n + max(SENKOU_B, SR_LOOKBACK_CANDLES)
        candles = collector.grab(fetch_count)

        df = pd.DataFrame([{
            'Open': float(str(c.bid.o)),
            'High': float(str(c.bid.h)),
            'Low': float(str(c.bid.l)),
            'Close': float(str(c.bid.c)),
        } for c in candles])

        df.reset_index(drop=True, inplace=True)

        # Add indicators
        df = add_ichimoku(df, TENKAN, KIJUN, SENKOU_B)
        df['FVG'] = detect_fvg(df)

        # Return only required count
        return df.iloc[-n:].reset_index(drop=True)

    except Exception as e:
        print(f"Error fetching {granularity} candles: {e}")
        return None


# ========================================
# 8. MAIN TRADING JOB
# ========================================

def trading_job():
    """Main trading logic - runs every N minutes."""
    print(f"\n{'='*80}")
    print(f"[{datetime.now(pytz.timezone('America/Chicago'))}] Running {INSTRUMENT} Strategy")
    print(f"{'='*80}")

    try:
        # Fetch data for all timeframes
        print("Fetching candles...")
        df_d1 = fetch_candles_multi_granularity(Gran.D, n=100)
        df_h4 = fetch_candles_multi_granularity(Gran.H4, n=100)
        df_h1 = fetch_candles_multi_granularity(Gran.H1, n=100)
        df_m30 = fetch_candles_multi_granularity(Gran.M30, n=100)
        df_m15 = fetch_candles_multi_granularity(Gran.M15, n=100)

        if any(df is None for df in [df_d1, df_h4, df_h1, df_m30, df_m15]):
            print("⚠️ Failed to fetch all timeframe data")
            return

        # Detect S/R levels on 30M
        m30_levels = detect_key_levels(df_m30)

        print(f"\n📊 Multi-Timeframe Analysis:")
        print(f"   D1 FVG: {df_d1['FVG'].iloc[-1]}")
        print(f"   H4 FVG: {df_h4['FVG'].iloc[-1]}")
        print(f"   H1 FVG: {df_h1['FVG'].iloc[-1]}")
        print(f"   M30 FVG: {df_m30['FVG'].iloc[-1]}")
        print(f"   M15 FVG: {df_m15['FVG'].iloc[-1]}")

        print(f"\n📍 30M S/R Levels:")
        print(
            f"   Support: {m30_levels['support'][:3] if m30_levels['support'] else 'None'}")
        print(
            f"   Resistance: {m30_levels['resistance'][:3] if m30_levels['resistance'] else 'None'}")

        print(f"\n💨 15M Current State:")
        print(f"   Close: {df_m15['Close'].iloc[-1]:.5f}")
        print(f"   Cloud Top: {df_m15['cloud_top'].iloc[-1]:.5f}")
        print(f"   Cloud Bot: {df_m15['cloud_bot'].iloc[-1]:.5f}")
        print(f"   Retrace Signal: {detect_ichimoku_retracement(df_m15)}")

        # Generate signal
        signal, entry, sl, tp, reason = generate_signal(
            df_d1, df_h4, df_h1, df_m30, df_m15, m30_levels
        )

        print(f"\n🔔 Signal: {reason}")

        if signal is None:
            print("❌ No valid trade signal")
            return

        # Validate SL/TP
        if signal == 'BUY':
            if tp <= entry or sl >= entry:
                print(
                    f"❌ Invalid BUY setup: Entry={entry:.5f}, SL={sl:.5f}, TP={tp:.5f}")
                return
        elif signal == 'SELL':
            if tp >= entry or sl <= entry:
                print(
                    f"❌ Invalid SELL setup: Entry={entry:.5f}, SL={sl:.5f}, TP={tp:.5f}")
                return

        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 0

        print(f"\n✅ TRADE SETUP:")
        print(f"   Signal: {signal}")
        print(f"   Entry: {entry:.5f}")
        print(f"   Stop Loss: {sl:.5f}")
        print(f"   Take Profit: {tp:.5f}")
        print(f"   Risk: {risk:.5f} pips")
        print(f"   Reward: {reward:.5f} pips")
        print(f"   R/R Ratio: 1:{rr:.2f}")

        # Place order
        try:
            client = API(access_token)

            if signal == 'BUY':
                mo = MarketOrderRequest(
                    instrument=INSTRUMENT,
                    units=risk_units,
                    takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
                    stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
                )
            else:  # SELL
                mo = MarketOrderRequest(
                    instrument=INSTRUMENT,
                    units=-risk_units,
                    takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
                    stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
                )

            r = orders.OrderCreate(accountID, data=mo.data)
            rv = client.request(r)
            log_executed_trade(
                instrument=INSTRUMENT,
                signal="BUY",
                entry=entry,
                sl=sl,
                tp=tp,
                timeframe="15M"
            )
            print(f"\n✓ Order placed successfully!")
            print(f"   Response: {rv}")

        except Exception as e:
            print(f"\n✗ Order failed: {str(e)}")

    except Exception as e:
        print(f"\n✗ Trading job error: {str(e)}")
        import traceback
        traceback.print_exc()


# ========================================
# 9. SCHEDULER
# ========================================

if __name__ == "__main__":
    run_minutes = ",".join(str(i) for i in range(0, 60, N))

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

    print(f"\n{'='*80}")
    print(f"🚀 {INSTRUMENT} Ichimoku FVG Multi-TF OANDA Trader Started")
    print(f"{'='*80}")
    print(f"Running every {N} minutes")
    print(f"Configuration:")
    print(f"  - Timeframes: D1, H4, H1, 30M (bias), 15M (entry)")
    print(f"  - Entry: Ichimoku retracement on 15M")
    print(f"  - S/R: 30M levels for SL/TP")
    print(f"  - R/R Ratio: 1:{SLTPRatio}")
    print(f"{'='*80}\n")

    try:
        # IMPORTANT: Set real=True when going live
        # Currently set to practice environment (real=False)
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n✓ Trader stopped gracefully")
