import yfinance as yf
import pandas as pd
import numpy as np


# TODO:COPIED CODE TO BE REMOVED


def add_column(data, times):
    """
    data-dataset that we want to manipulate
    times - number of columns we want to add
    """
    for i in range(1, times+1):
        new = np.zeros((len(data), 1), dtype=float)
        data = np.append(data, new, axis=1)

    return data


def delete_column(data, index, times):
    """
    data-dataset that we want to manipulate
    index - from which column to start to delete
    times - the number of columns we want to remove

    """
    for i in range(1, times+1):
        data = np.delete(data, index, axis=1)

    return data


def add_row(data, times):
    for i in range(1, times+1):
        # shape is to know the number of rows(0) & columns(1)
        columns = np.shape(data)[1]
        new = np.zeros((1, columns), dtype=float)
        data = np.append(data, new, axis=0)
    return data


def delete_row(data, number):
    """
    This simple function states that the data actually starts
    from the number index and continues to the end of the data.
    Basically, it ignores a select number of rows from the beginning
    """
    # data = data[number:, 1]
    data = data[number:, :]   # Keep all columns

    return data


"""
# An array has the following structure: array[row, column]
# Referring to the whole my_data array
my_data
# Referring to the first 100 rows inside the array
my_data[:100, ]
# Referring to the first 100 rows of the first two columns my_data[:100, 0:2]
# Referring to all the rows of the seventh column
my_data[:, 6]
# Referring to the last 500 rows of the array
my_data[-500:, ]
# Referring to the first row of the first column
my_data[0, 0]
# Referring to the last row of the last column
my_data[-1, -1]
"""


def rouding(data, how_far):
    data = data.round(decimals=how_far)
    return data


"""

Examples


Add two columns to the array:
my_data = add_column(my_data, 2)

Delete three columns starting from the second column:
my_data = delete_column(my_data, 1, 3)

Add 11 rows to the end of the array:
my_data = add_row(my_data, 11)

Delete the first four rows in the array:
my_data = delete_row(my_data, 4)

Round all the values in the array to four decimals:
my_data = rounding(my_data, 4)


"""


# # Example: get D1 (daily) data for BTC/USD, EUR/USD, etc.
# symbols = {
#     'EURUSD': 'EURUSD=X',
#     'USDCHF': 'USDCHF=X',
#     'GBPUSD': 'GBPUSD=X',
#     'USDCAD': 'USDCAD=X',
#     'BTCUSD': 'BTC-USD',
#     'ETHUSD': 'ETH-USD',
#     'XAUUSD': 'XAUUSD=X',
#     'XAGUSD': 'XAGUSD=X',
#     'SP500m': '^GSPC',
#     'UK100': '^FTSE'
# }

# # Pick one asset
# ticker = symbols['ETHUSD']
# data = yf.download(ticker, start="2020-01-01", end="2025-10-15", interval="1d")

# Save to Excel
# data.to_excel('my_data.xlsx')

# Later, load it back
my_data = pd.read_excel('my_data.xlsx')
my_data = np.array(my_data)


my_data_no_volume = delete_column(my_data, 4, 1)
# Convert back to a Pandas DataFrame before saving
my_data_no_volume = pd.DataFrame(my_data_no_volume)

# Save to Excel
my_data_no_volume.to_excel('my_data_no_volume.xlsx', index=False)


# ALPHA PATTERN
"""
A long (buy) signal is generated on the next open whenever
the current low is lower than the low price 5 periods ago
and the low price 13 periods ago but higher than the low
price 21 periods ago. Simultaneously, the close price of
the current bar must be higher than the close price 3
periods ago.


A short (sell) signal is generated on the next open whenever
the current high price is higher than the high price 5
periods ago and the high price 13 periods ago but lower
than the high price 21 periods ago. Simultaneously, the
close price of the current bar must be lower than the close
price 3 periods ago.
"""


def signal(data):

    # from the 6th column since we have volume column
    data = add_column(data, 5)
    for i in range(len(data)):
        try:
            # Bullish Alpha
            if data[i, 2] < data[i - 5, 2] and data[i, 2] < data[i - 13, 2] and data[i, 2] > data[i - 21, 2] and data[i, 3] > data[i - 1, 3] and data[i, 4] == 0:
                data[i+1, 4] = 1
            # Bearish Alpha
            elif data[i, 1] > data[i - 5, 1] and data[i, 1] > data[i - 13, 1] and data[i, 1] < data[i - 21, 1] and data[i, 3] < data[i - 1, 3] and data[i, 5] == 0:
                data[i+1, 5] = -1
        except IndexError:
            pass

    return data


"""
The statement data[i, 4] == 0 is the condition that to
have a buy signal, you must not have had a buy signal
in the previous row.3 This is to avoid successive signals
in case the pattern is repetitive in nature.

"""

my_data = pd.read_excel('my_data.xlsx')

my_data = np.array(my_data)

my_data = delete_row(my_data, 3)

signal(my_data_no_volume)
