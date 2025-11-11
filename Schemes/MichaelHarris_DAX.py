#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OANDA Live Trading Bot - Michael Harris DAX Strategy
Configurable timeframe and instrument with comprehensive logging
"""

from config import access_token, accountID
from oanda_candles import Pair, Gran, CandleClient
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.positions as positions
from oandapyV20 import API
from apscheduler.schedulers.blocking import BlockingScheduler
import pytz
from datetime import datetime
import pandas as pd
import numpy as np
import os
import json

# ========================================
# CONFIGURATION
# ========================================
STRATEGY_NAME = "Michael_Harris_DAX"  # Used for log file naming
INSTRUMENT = "EUR_USD"  # Change to: BTC_USD, ETH_USD, etc.
# Options: Gran.M1, Gran.M5, Gran.M15, Gran.M30, Gran.H1, Gran.H4, etc.
TIMEFRAME = Gran.M15
TRADE_SIZE = 5000       # Units to trade
SL_PERCENTAGE = 0.04    # 4% Stop Loss
TP_PERCENTAGE = 0.1    # 2% Take Profit
USE_LIVE = False        # Set True for live trading, False for practice
LOG_DIRECTORY = "./trading_logs"  # Directory to store log files

# Timeframe to cron schedule mapping
TIMEFRAME_CRON = {
    Gran.M1: {'minute': '*'},
    Gran.M5: {'minute': '*/5'},
    Gran.M15: {'minute': '1,16,31,46'},
    Gran.M30: {'minute': '1,31'},
    Gran.H1: {'minute': '1'},
    Gran.H4: {'minute': '1', 'hour': '*/4'},
}

# ========================================
# LOGGING SYSTEM
# ========================================


class TradeLogger:
    """
    Comprehensive trade logging system that tracks all trades and their outcomes.
    Reusable across different strategies - just change STRATEGY_NAME.
    """

    def __init__(self, strategy_name, log_directory="./trading_logs"):
        self.strategy_name = strategy_name
        self.log_directory = log_directory

        # ✅ Create log directory before trying to create file
        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
            print(f"📁 Created log directory: {log_directory}")

        self.log_file = self._create_log_file()
        self.active_trades = {}  # Store active trade details

    def _create_log_file(self):
        """Create a unique log filename with strategy name and timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.strategy_name}_{timestamp}.csv"
        filepath = os.path.join(self.log_directory, filename)

        # Create CSV with headers if it doesn't exist
        if not os.path.exists(filepath):
            headers = [
                "Trade_ID", "Open_Time", "Close_Time", "Instrument", "Direction",
                "Entry_Price", "Exit_Price", "SL_Price", "TP_Price", "Units",
                "Outcome", "PnL", "PnL_Percent", "Duration_Minutes", "Signal_Type"
            ]
            df = pd.DataFrame(columns=headers)
            df.to_csv(filepath, index=False)
            print(f"📝 Created new log file: {filepath}")

        return filepath

    def log_trade_open(self, trade_id, instrument, direction, entry_price, sl_price, tp_price, units, signal_type):
        """Log when a trade is opened"""
        open_time = datetime.now(pytz.timezone('America/Chicago'))

        self.active_trades[trade_id] = {
            "Trade_ID": trade_id,
            "Open_Time": open_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Close_Time": None,
            "Instrument": instrument,
            "Direction": direction,
            "Entry_Price": entry_price,
            "Exit_Price": None,
            "SL_Price": sl_price,
            "TP_Price": tp_price,
            "Units": units,
            "Outcome": "OPEN",
            "PnL": 0,
            "PnL_Percent": 0,
            "Duration_Minutes": 0,
            "Signal_Type": signal_type
        }

        print(f"📊 Trade {trade_id} logged as OPEN")

    def log_trade_close(self, trade_id, exit_price, outcome, pnl):
        """Log when a trade is closed (TP, SL, or manual)"""
        if trade_id not in self.active_trades:
            print(f"⚠️ Warning: Trade {trade_id} not found in active trades")
            return

        close_time = datetime.now(pytz.timezone('America/Chicago'))
        trade = self.active_trades[trade_id]

        # Calculate duration
        open_time = datetime.strptime(trade["Open_Time"], "%Y-%m-%d %H:%M:%S")
        close_time_aware = close_time.replace(tzinfo=None)
        duration = (close_time_aware - open_time).total_seconds() / 60

        # Calculate PnL percentage
        entry_price = trade["Entry_Price"]
        pnl_percent = (pnl / (entry_price * abs(trade["Units"]))) * 100

        # Update trade record
        trade["Close_Time"] = close_time.strftime("%Y-%m-%d %H:%M:%S")
        trade["Exit_Price"] = exit_price
        trade["Outcome"] = outcome
        trade["PnL"] = pnl
        trade["PnL_Percent"] = pnl_percent
        trade["Duration_Minutes"] = duration

        # Write to CSV
        df = pd.DataFrame([trade])
        df.to_csv(self.log_file, mode='a', header=False, index=False)

        # Remove from active trades
        del self.active_trades[trade_id]

        print(
            f"✅ Trade {trade_id} logged as {outcome}: PnL = {pnl:.2f} ({pnl_percent:.2f}%)")

    def check_closed_trades(self, api_client, account_id):
        """
        Check OANDA for closed trades and log them automatically.
        Call this periodically to catch SL/TP hits.
        """
        try:
            # Get list of current open trades
            r = trades.TradesList(account_id)
            response = api_client.request(r)
            open_trade_ids = [t['id'] for t in response.get('trades', [])]

            # Check if any active trades are now closed
            for trade_id in list(self.active_trades.keys()):
                if trade_id not in open_trade_ids:
                    # Trade is closed, get details
                    try:
                        r_detail = trades.TradeDetails(
                            account_id, tradeID=trade_id)
                        trade_detail = api_client.request(r_detail)

                        # Determine outcome based on closing reason
                        closing_reason = trade_detail.get(
                            'trade', {}).get('state', 'CLOSED')
                        exit_price = float(trade_detail.get(
                            'trade', {}).get('price', 0))
                        realized_pl = float(trade_detail.get(
                            'trade', {}).get('realizedPL', 0))

                        if realized_pl > 0:
                            outcome = "WIN_TP"
                        elif realized_pl < 0:
                            outcome = "LOSS_SL"
                        else:
                            outcome = "CLOSED"

                        self.log_trade_close(
                            trade_id, exit_price, outcome, realized_pl)

                    except Exception as e:
                        # If we can't get details, log as unknown close
                        print(
                            f"⚠️ Could not fetch details for closed trade {trade_id}: {e}")
                        self.log_trade_close(trade_id, 0, "CLOSED_UNKNOWN", 0)

        except Exception as e:
            print(f"⚠️ Error checking closed trades: {e}")

    def get_statistics(self):
        """Get performance statistics from the log file"""
        try:
            df = pd.read_csv(self.log_file)

            if len(df) == 0:
                return "No trades logged yet."

            # Filter only closed trades
            closed_trades = df[df['Outcome'].isin(
                ['WIN_TP', 'LOSS_SL', 'CLOSED'])]

            if len(closed_trades) == 0:
                return "No closed trades yet."

            total_trades = len(closed_trades)
            wins = len(closed_trades[closed_trades['Outcome'] == 'WIN_TP'])
            losses = len(closed_trades[closed_trades['Outcome'] == 'LOSS_SL'])
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

            total_pnl = closed_trades['PnL'].sum()
            avg_pnl = closed_trades['PnL'].mean()
            avg_win = closed_trades[closed_trades['PnL']
                                    > 0]['PnL'].mean() if wins > 0 else 0
            avg_loss = closed_trades[closed_trades['PnL']
                                     < 0]['PnL'].mean() if losses > 0 else 0

            stats = f"""
📈 TRADING STATISTICS - {self.strategy_name}
{'='*50}
Total Trades: {total_trades}
Wins: {wins} | Losses: {losses}
Win Rate: {win_rate:.2f}%
Total PnL: {total_pnl:.2f}
Average PnL: {avg_pnl:.2f}
Average Win: {avg_win:.2f}
Average Loss: {avg_loss:.2f}
{'='*50}
"""
            return stats
        except Exception as e:
            return f"Error calculating statistics: {e}"


# Initialize logger
logger = TradeLogger(STRATEGY_NAME, LOG_DIRECTORY)

# ========================================
# MICHAEL HARRIS SIGNAL DETECTION
# ========================================


def detect_harris_signal(df):
    """
    Detects Michael Harris DAX pattern signals
    Returns: 2 for BUY, 1 for SELL, 0 for no signal
    """
    if len(df) < 4:
        return 0

    current_pos = len(df) - 1

    # Buy condition (Bullish DAX Pattern)
    c1 = df['High'].iloc[current_pos] > df['High'].iloc[current_pos-1]
    c2 = df['High'].iloc[current_pos-1] > df['Low'].iloc[current_pos]
    c3 = df['Low'].iloc[current_pos] > df['High'].iloc[current_pos-2]
    c4 = df['High'].iloc[current_pos-2] > df['Low'].iloc[current_pos-1]
    c5 = df['Low'].iloc[current_pos-1] > df['High'].iloc[current_pos-3]
    c6 = df['High'].iloc[current_pos-3] > df['Low'].iloc[current_pos-2]
    c7 = df['Low'].iloc[current_pos-2] > df['Low'].iloc[current_pos-3]

    if c1 and c2 and c3 and c4 and c5 and c6 and c7:
        return 2  # BUY signal

    # Sell condition (Bearish DAX Pattern)
    c1 = df['Low'].iloc[current_pos] < df['Low'].iloc[current_pos-1]
    c2 = df['Low'].iloc[current_pos-1] < df['High'].iloc[current_pos]
    c3 = df['High'].iloc[current_pos] < df['Low'].iloc[current_pos-2]
    c4 = df['Low'].iloc[current_pos-2] < df['High'].iloc[current_pos-1]
    c5 = df['High'].iloc[current_pos-1] < df['Low'].iloc[current_pos-3]
    c6 = df['Low'].iloc[current_pos-3] < df['High'].iloc[current_pos-2]
    c7 = df['High'].iloc[current_pos-2] < df['High'].iloc[current_pos-3]

    if c1 and c2 and c3 and c4 and c5 and c6 and c7:
        return 1  # SELL signal

    return 0  # No signal

# ========================================
# FETCH LIVE CANDLES
# ========================================


def fetch_candles(instrument_pair, granularity, n=200):
    """
    Fetch candles from OANDA
    """
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
    return df

# ========================================
# TRADING JOB
# ========================================


def trading_job():
    print(f"\n{'='*60}")
    print(f"[{datetime.now(pytz.timezone('America/Chicago'))}]")
    print(f"Running Michael Harris Strategy on {INSTRUMENT} ({TIMEFRAME})")
    print(f"{'='*60}")

    try:
        # Check for closed trades first (to log SL/TP hits)
        api_client = API(access_token)
        logger.check_closed_trades(api_client, accountID)

        # Convert instrument string to Pair enum
        instrument_pair = getattr(Pair, INSTRUMENT)

        # Fetch candles
        df = fetch_candles(instrument_pair, TIMEFRAME, n=200)

        # Detect signal
        signal = detect_harris_signal(df)

        current_price = df['Close'].iloc[-1]
        print(f"Current Price: {current_price:.5f}")
        print(f"Signal: {signal} (0=None, 1=SELL, 2=BUY)")

        if signal == 0:
            print("No valid trade signal at this time.")
            # Still print stats
            print(logger.get_statistics())
            return

        # Calculate SL and TP based on strategy parameters
        if signal == 2:  # BUY
            sl = current_price - (SL_PERCENTAGE * current_price)
            tp = current_price + (TP_PERCENTAGE * current_price)
            units = TRADE_SIZE
            direction = "BUY"
            signal_type = "BULLISH_DAX"

            print(f"\n🟢 BUY SIGNAL DETECTED")
            print(f"Entry: {current_price:.5f}")
            print(f"Stop Loss: {sl:.5f} ({SL_PERCENTAGE*100}% below entry)")
            print(f"Take Profit: {tp:.5f} ({TP_PERCENTAGE*100}% above entry)")
            print(f"Risk/Reward: 1:{TP_PERCENTAGE/SL_PERCENTAGE:.2f}")

        elif signal == 1:  # SELL
            sl = current_price + (SL_PERCENTAGE * current_price)
            tp = current_price - (TP_PERCENTAGE * current_price)
            units = -TRADE_SIZE
            direction = "SELL"
            signal_type = "BEARISH_DAX"

            print(f"\n🔴 SELL SIGNAL DETECTED")
            print(f"Entry: {current_price:.5f}")
            print(f"Stop Loss: {sl:.5f} ({SL_PERCENTAGE*100}% above entry)")
            print(f"Take Profit: {tp:.5f} ({TP_PERCENTAGE*100}% below entry)")
            print(f"Risk/Reward: 1:{TP_PERCENTAGE/SL_PERCENTAGE:.2f}")

        # Validate SL/TP
        if signal == 2 and (tp <= current_price or sl >= current_price):
            print("❌ Invalid SL/TP for BUY order.")
            return
        if signal == 1 and (tp >= current_price or sl <= current_price):
            print("❌ Invalid SL/TP for SELL order.")
            return

        # Create market order
        mo = MarketOrderRequest(
            instrument=INSTRUMENT,
            units=units,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
        )

        # Execute order
        r = orders.OrderCreate(accountID, data=mo.data)
        rv = api_client.request(r)

        # Extract trade details from response
        fill_transaction = rv.get('orderFillTransaction', {})
        trade_id = fill_transaction.get('id', 'UNKNOWN')
        fill_price = float(fill_transaction.get('price', current_price))

        print(f"\n✅ Order executed successfully!")
        print(f"Order ID: {trade_id}")
        print(f"Fill Price: {fill_price:.5f}")

        # Log the trade opening
        logger.log_trade_open(
            trade_id=trade_id,
            instrument=INSTRUMENT,
            direction=direction,
            entry_price=fill_price,
            sl_price=sl,
            tp_price=tp,
            units=units,
            signal_type=signal_type
        )

        # Print current statistics
        print(logger.get_statistics())

    except AttributeError:
        print(
            f"❌ Error: Invalid instrument '{INSTRUMENT}'. Check Pair enum in oanda_candles.")
    except Exception as e:
        print(f"❌ Order failed: {str(e)}")
        import traceback
        traceback.print_exc()

# ========================================
# SCHEDULER
# ========================================


if __name__ == "__main__":
    print("="*60)
    print("OANDA MICHAEL HARRIS DAX STRATEGY BOT")
    print("="*60)
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Instrument: {INSTRUMENT}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Trade Size: {TRADE_SIZE} units")
    print(f"Stop Loss: {SL_PERCENTAGE*100}%")
    print(f"Take Profit: {TP_PERCENTAGE*100}%")
    print(f"Mode: {'LIVE' if USE_LIVE else 'PRACTICE'}")
    print(f"Log File: {logger.log_file}")
    print("="*60)

    # Print existing statistics if any
    print(logger.get_statistics())

    scheduler = BlockingScheduler()

    # Get cron settings for the selected timeframe
    cron_settings = TIMEFRAME_CRON.get(TIMEFRAME, {'minute': '*/15'})

    scheduler.add_job(
        trading_job,
        'cron',
        day_of_week='mon-fri',
        hour=cron_settings.get('hour', '0-23'),
        minute=cron_settings.get('minute', '*/15'),
        timezone='America/Chicago'
    )

    print(f"\n🤖 Bot started. Waiting for next {TIMEFRAME} candle close...")
    print("Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Bot stopped by user.")
        print("\n" + "="*60)
        print("FINAL STATISTICS")
        print("="*60)
        print(logger.get_statistics())
