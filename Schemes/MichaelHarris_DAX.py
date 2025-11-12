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
STRATEGY_NAME = "Michael_Harris_DAX"
# 👈 Change this to the instrument you want to trade (e.g. EUR_USD, BTC_USD)
INSTRUMENT = "NZD_USD"
# Options: Gran.M1, Gran.M5, Gran.M15, Gran.M30, Gran.H1, Gran.H4, etc.
TIMEFRAME = Gran.M15
TRADE_SIZE = 1000
SL_PERCENTAGE = 0.04     # 4% Stop Loss
TP_PERCENTAGE = 0.10     # 10% Take Profit
USE_LIVE = False          # Set True for live trading, False for practice
LOG_DIRECTORY = "./trading_logs"

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
    def __init__(self, strategy_name, log_directory="./trading_logs"):
        self.strategy_name = strategy_name
        self.log_directory = log_directory

        if not os.path.exists(log_directory):
            os.makedirs(log_directory)
            print(f"📁 Created log directory: {log_directory}")

        self.log_file = self._create_log_file()
        self.active_trades = {}

    def _create_log_file(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.strategy_name}_{timestamp}.csv"
        filepath = os.path.join(self.log_directory, filename)

        if not os.path.exists(filepath):
            headers = [
                "Trade_ID", "Open_Time", "Close_Time", "Instrument", "Direction",
                "Entry_Price", "Exit_Price", "SL_Price", "TP_Price", "Units",
                "Outcome", "PnL", "PnL_Percent", "Duration_Minutes", "Signal_Type"
            ]
            pd.DataFrame(columns=headers).to_csv(filepath, index=False)
            print(f"📝 Created new log file: {filepath}")

        return filepath

    def log_trade_open(self, trade_id, instrument, direction, entry_price, sl_price, tp_price, units, signal_type):
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
        if trade_id not in self.active_trades:
            print(f"⚠️ Trade {trade_id} not found in active trades")
            return

        close_time = datetime.now(pytz.timezone('America/Chicago'))
        trade = self.active_trades[trade_id]
        open_time = datetime.strptime(trade["Open_Time"], "%Y-%m-%d %H:%M:%S")
        duration = (close_time.replace(tzinfo=None) -
                    open_time).total_seconds() / 60

        entry_price = trade["Entry_Price"]
        pnl_percent = (pnl / (entry_price * abs(trade["Units"]))) * 100

        trade.update({
            "Close_Time": close_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Exit_Price": exit_price,
            "Outcome": outcome,
            "PnL": pnl,
            "PnL_Percent": pnl_percent,
            "Duration_Minutes": duration
        })

        pd.DataFrame([trade]).to_csv(
            self.log_file, mode='a', header=False, index=False)
        del self.active_trades[trade_id]
        print(
            f"✅ Trade {trade_id} logged as {outcome}: PnL = {pnl:.2f} ({pnl_percent:.2f}%)")

    def check_closed_trades(self, api_client, account_id):
        try:
            r = trades.TradesList(account_id)
            response = api_client.request(r)
            open_trade_ids = [t['id'] for t in response.get('trades', [])]

            for trade_id in list(self.active_trades.keys()):
                if trade_id not in open_trade_ids:
                    try:
                        r_detail = trades.TradeDetails(
                            account_id, tradeID=trade_id)
                        trade_detail = api_client.request(r_detail)
                        trade_data = trade_detail.get('trade', {})
                        exit_price = float(trade_data.get('price', 0))
                        realized_pl = float(trade_data.get('realizedPL', 0))
                        outcome = "WIN_TP" if realized_pl > 0 else "LOSS_SL" if realized_pl < 0 else "CLOSED"
                        self.log_trade_close(
                            trade_id, exit_price, outcome, realized_pl)
                    except Exception as e:
                        print(
                            f"⚠️ Could not fetch closed trade {trade_id}: {e}")
                        self.log_trade_close(trade_id, 0, "CLOSED_UNKNOWN", 0)
        except Exception as e:
            print(f"⚠️ Error checking closed trades: {e}")

    def get_statistics(self):
        try:
            df = pd.read_csv(self.log_file)
            if len(df) == 0:
                return "No trades logged yet."

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

            return f"""
📈 TRADING STATS - {self.strategy_name} ({INSTRUMENT})
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
        except Exception as e:
            return f"Error calculating statistics: {e}"


logger = TradeLogger(STRATEGY_NAME, LOG_DIRECTORY)

# ========================================
# SIGNAL DETECTION
# ========================================


def detect_harris_signal(df):
    if len(df) < 4:
        return 0
    i = len(df) - 1
    c1 = df['High'].iloc[i] > df['High'].iloc[i-1]
    c2 = df['High'].iloc[i-1] > df['Low'].iloc[i]
    c3 = df['Low'].iloc[i] > df['High'].iloc[i-2]
    if c1 and c2 and c3:
        return 2  # BUY
    elif df['Low'].iloc[i] < df['Low'].iloc[i-1]:
        return 1  # SELL
    return 0

# ========================================
# FETCH CANDLES
# ========================================


def fetch_candles(instrument_pair, granularity, n=200):
    client = CandleClient(access_token, real=USE_LIVE)
    collector = client.get_collector(instrument_pair, granularity)
    candles = collector.grab(n)
    return pd.DataFrame([{
        'Open': float(str(c.bid.o)),
        'High': float(str(c.bid.h)),
        'Low': float(str(c.bid.l)),
        'Close': float(str(c.bid.c)),
    } for c in candles])

# ========================================
# TRADING JOB
# ========================================


def trading_job():
    print(f"\n{'='*60}")
    print(f"📅 [{datetime.now(pytz.timezone('America/Chicago'))}]")
    print(f"⚡ Running Michael Harris Strategy on 🟡 {INSTRUMENT} ({TIMEFRAME})")
    print(f"{'='*60}")

    try:
        api_client = API(access_token)
        logger.check_closed_trades(api_client, accountID)
        instrument_pair = getattr(Pair, INSTRUMENT)
        df = fetch_candles(instrument_pair, TIMEFRAME, n=200)

        signal = detect_harris_signal(df)
        current_price = df['Close'].iloc[-1]
        print(f"Current Price: {current_price:.5f}")
        print(f"Signal: {signal} (0=None, 1=SELL, 2=BUY)")

        if signal == 0:
            print("No valid trade signal.")
            print(logger.get_statistics())
            return

        if signal == 2:
            sl = current_price - (SL_PERCENTAGE * current_price)
            tp = current_price + (TP_PERCENTAGE * current_price)
            units = TRADE_SIZE
            direction = "BUY"
            signal_type = "BULLISH"
            print(
                f"🟢 BUY SIGNAL — Entry: {current_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
        else:
            sl = current_price + (SL_PERCENTAGE * current_price)
            tp = current_price - (TP_PERCENTAGE * current_price)
            units = -TRADE_SIZE
            direction = "SELL"
            signal_type = "BEARISH"
            print(
                f"🔴 SELL SIGNAL — Entry: {current_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")

        mo = MarketOrderRequest(
            instrument=INSTRUMENT,
            units=units,
            takeProfitOnFill=TakeProfitDetails(price=f"{tp:.5f}").data,
            stopLossOnFill=StopLossDetails(price=f"{sl:.5f}").data
        )

        r = orders.OrderCreate(accountID, data=mo.data)
        rv = api_client.request(r)

        trade_id = rv.get('orderFillTransaction', {}).get('id', 'UNKNOWN')
        fill_price = float(rv.get('orderFillTransaction',
                           {}).get('price', current_price))
        print(
            f"✅ Trade executed on {INSTRUMENT}: ID {trade_id}, Entry {fill_price:.5f}")

        logger.log_trade_open(trade_id, INSTRUMENT, direction,
                              fill_price, sl, tp, units, signal_type)
        print(logger.get_statistics())

    except AttributeError:
        print(f"❌ Invalid instrument '{INSTRUMENT}' — check Pair enums.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# ========================================
# SCHEDULER
# ========================================


if __name__ == "__main__":
    print("="*60)
    print("🤖 OANDA MICHAEL HARRIS DAX STRATEGY BOT")
    print("="*60)
    print(f"Strategy: {STRATEGY_NAME}")
    print(f"Trading: 🟡 {INSTRUMENT}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Trade Size: {TRADE_SIZE} units")
    print(
        f"Stop Loss: {SL_PERCENTAGE*100}% | Take Profit: {TP_PERCENTAGE*100}%")
    print(f"Mode: {'LIVE' if USE_LIVE else 'PRACTICE'}")
    print(f"Logs: {logger.log_file}")
    print("="*60)

    print(logger.get_statistics())

    scheduler = BlockingScheduler()
    cron = TIMEFRAME_CRON.get(TIMEFRAME, {'minute': '*/15'})

    scheduler.add_job(
        trading_job,
        'cron',
        day_of_week='mon-fri',
        hour=cron.get('hour', '0-23'),
        minute=cron.get('minute', '*/15'),
        timezone='America/Chicago',
        misfire_grace_time=120
    )

    print(
        f"\n🚀 Bot started for {INSTRUMENT}. Waiting for next {TIMEFRAME} candle...")
    print("Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n🛑 Bot stopped by user.")
        print(logger.get_statistics())
