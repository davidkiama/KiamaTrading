import matplotlib.pyplot as plt

import yfinance as yf


def ohlc_plot_bars(data, window):
    sample = data[-window:, ]

    for i in range(len(sample)):
        # sample[i] = [open, high, low, close]
        o, h, l, c = sample[i]

        # Draw high–low wick
        plt.vlines(x=i, ymin=l, ymax=h, color='black', linewidth=1)

        # Bullish candle → green
        if c > o:
            plt.vlines(x=i, ymin=o, ymax=c, color='green', linewidth=2)

        # Bearish candle → red
        elif c < o:
            plt.vlines(x=i, ymin=c, ymax=o, color='red', linewidth=2)

        # Neutral candle → black
        else:
            plt.vlines(x=i, ymin=c, ymax=c + 0.00003,
                       color='black', linewidth=1)

    plt.grid()
    plt.show()


# --- Fetch and prepare data ---
my_data = yf.download("EURUSD=X", period="30d", interval="1h")

# Extract only Open, High, Low, Close columns
my_data = my_data[['Open', 'High', 'Low', 'Close']]

# Convert to NumPy array so it matches what ohlc_plot_bars expects
my_data_np = my_data.to_numpy()

# --- Plot last 500 bars ---
ohlc_plot_bars(my_data_np, 500)
