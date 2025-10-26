import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# -----------------------------------------------------
# Utility functions
# -----------------------------------------------------
def add_column(data, times):
    """Adds a number of empty (zero) columns to a NumPy array."""
    new_cols = np.zeros((len(data), times), dtype=float)
    data = np.hstack((data, new_cols))
    return data


def delete_column(data, index, times):
    """Deletes `times` columns starting from `index`."""
    for _ in range(times):
        data = np.delete(data, index, axis=1)
    return data


def delete_row(data, number):
    """Deletes the first `number` rows."""
    return data[number:, :]


# -----------------------------------------------------
# Alpha pattern signal logic
# -----------------------------------------------------
def signal(data):
    """
    Generates trading signals based on Alpha pattern rules.
    data = [open, high, low, close]
    """
    data = add_column(data, 1)  # add a new column for signals

    for i in range(21, len(data) - 1):
        try:
            # Bullish (buy) condition
            if (data[i, 2] < data[i - 5, 2] and  # low < low[-5]
                data[i, 2] < data[i - 13, 2] and  # low < low[-13]
                data[i, 2] > data[i - 21, 2] and  # low > low[-21]
                data[i, 3] > data[i - 3, 3] and   # close > close[-3]
                    data[i, -1] == 0):
                data[i + 1, -1] = 8888  # signal on next open

            # Bearish (sell) condition
            elif (data[i, 1] > data[i - 5, 1] and  # high > high[-5]
                  data[i, 1] > data[i - 13, 1] and  # high > high[-13]
                  data[i, 1] < data[i - 21, 1] and  # high < high[-21]
                  data[i, 3] < data[i - 3, 3] and   # close < close[-3]
                  data[i, -1] == 0):
                data[i + 1, -1] = 7777
        except IndexError:
            pass

    return data


# -----------------------------------------------------
# OHLC plot function
# -----------------------------------------------------
def ohlc_plot_bars(ax, data, window):
    sample = data[-window:, ]

    for i in range(len(sample)):
        o, h, l, c = sample[i, :4]

        # Draw high–low wick
        ax.vlines(x=i, ymin=l, ymax=h, color='black', linewidth=1)

        # Candle body: green (bullish), red (bearish)
        if c > o:
            ax.vlines(x=i, ymin=o, ymax=c, color='black', linewidth=2)
        elif c < o:
            ax.vlines(x=i, ymin=c, ymax=o, color='black', linewidth=2)
        else:
            ax.vlines(x=i, ymin=c, ymax=c + 0.00003,
                      color='black', linewidth=1)

    ax.grid()


# -----------------------------------------------------
# Plot with signal arrows
# -----------------------------------------------------
def signal_chart_arrow(data, window=250):
    sample = data[-window:, ]
    fig, ax = plt.subplots(figsize=(10, 5))
    ohlc_plot_bars(ax, data, window)

    for i in range(len(sample)):
        # 8888 → buy signal
        if sample[i, -1] == 8888:
            x, y = i, sample[i, 2]  # draw arrow from the low
            ax.annotate('', xy=(x, y), xytext=(x, y - (y * 0.001)),
                        arrowprops=dict(facecolor='green', width=2,
                                        headlength=10, headwidth=5))

        # 7777 → sell signal
        elif sample[i, -1] == 7777:
            x, y = i, sample[i, 1]  # draw arrow from the high
            ax.annotate('', xy=(x, y), xytext=(x, y + (y * 0.001)),
                        arrowprops=dict(facecolor='red', width=2,
                                        headlength=10, headwidth=5))

    plt.show()


# -----------------------------------------------------
# MAIN WORKFLOW
# -----------------------------------------------------
# 1. Fetch data
my_data = yf.download("USDCHF=X", period="60d", interval="1h")

# 2. Keep only OHLC
my_data = my_data[['Open', 'High', 'Low', 'Close']]

# 3. Convert to NumPy for signal logic
my_data_np = my_data.to_numpy()

# 4. Apply Alpha signal logic
data_with_signals = signal(my_data_np)

# 5. Plot
signal_chart_arrow(data_with_signals, window=250)
