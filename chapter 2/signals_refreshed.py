import yfinance as yf
import pandas as pd
import numpy as np


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
# Signal logic
# -----------------------------------------------------
def signal(data):
    """
    Generates trading signals based on Alpha pattern rules.
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
# Main workflow
# -----------------------------------------------------
# 1. Load Excel and skip header rows
my_data = pd.read_excel('my_data.xlsx', skiprows=3)

# 2. Keep only numeric columns (ignore Date or Symbol)
numeric_data = my_data.select_dtypes(include=[np.number]).to_numpy()

# 3. Remove the Volume column (last one)
numeric_data = numeric_data[:, :-1]

print(
    f"First row - High: {numeric_data[0, 1]}, Low: {numeric_data[0, 2]}, Close: {numeric_data[0, 3]}")
# 4. Apply signal logic
data_with_signals = signal(numeric_data)

# 5. Convert back to DataFrame for easier export
columns = list(my_data.select_dtypes(
    include=[np.number]).columns[:-1]) + ["Signal"]
df_with_signals = pd.DataFrame(data_with_signals, columns=columns)

# 6. Save output
df_with_signals.to_excel("my_data_with_signals.xlsx", index=False)

print("✅ Done! Signals saved to my_data_with_signals.xlsx")
print(df_with_signals.tail())
