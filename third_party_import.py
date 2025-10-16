import yfinance as yf
import pandas as pd
import numpy as np


# Example: get D1 (daily) data for BTC/USD, EUR/USD, etc.
symbols = {
    'EURUSD': 'EURUSD=X',
    'USDCHF': 'USDCHF=X',
    'GBPUSD': 'GBPUSD=X',
    'USDCAD': 'USDCAD=X',
    'BTCUSD': 'BTC-USD',
    'ETHUSD': 'ETH-USD',
    'XAUUSD': 'XAUUSD=X',
    'XAGUSD': 'XAGUSD=X',
    'SP500m': '^GSPC',
    'UK100': '^FTSE'
}

# Pick one asset
ticker = symbols['BTCUSD']
data = yf.download(ticker, start="2010-01-01", end="2025-10-15", interval="1d")

# Save to Excel
data.to_excel('my_data.xlsx')

# Later, load it back
my_data = pd.read_excel('my_data.xlsx')
my_data = np.array(my_data)

print('MY DATA', my_data)
