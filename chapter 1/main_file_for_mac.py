import datetime
import pytz
import pandas as pd
import numpy as np
import yfinance as yf

# Equivalent timeframe mappings
frame_M1 = '1m'
frame_M15 = '15m'
frame_M30 = '30m'
frame_H1 = '1h'
frame_H4 = '4h'
frame_D1 = '1d'
frame_W1 = '1wk'

now = datetime.datetime.now()

# Matching your MT5 asset list with Yahoo tickers
assets = {
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


# ✅ Replacement for get_quotes()
def get_quotes(time_frame, year=2005, month=1, day=1, asset='EURUSD'):
    try:
        timezone = pytz.timezone('Europe/Paris')
        time_from = datetime.datetime(year, month, day, tzinfo=timezone)
        time_to = datetime.datetime.now(timezone)

        symbol = assets[asset]
        df = yf.download(
            symbol,
            start=time_from.strftime('%Y-%m-%d'),
            end=time_to.strftime('%Y-%m-%d'),
            interval=time_frame,
            progress=False
        )

        if df.empty:
            print(f"No data for {asset}. Check ticker symbol or timeframe.")
            return pd.DataFrame()

        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'tick_volume'
        })

        df.reset_index(inplace=True)
        return df

    except Exception as e:
        print(f"Error fetching data for {asset}: {e}")
        return pd.DataFrame()


# ✅ Replacement for mass_import()
def mass_import(asset_index, time_frame):
    # Reverse map your numerical index to the assets list order
    asset_list = list(assets.keys())
    asset = asset_list[asset_index]

    if time_frame == 'H1':
        data = get_quotes(frame_H1, 2013, 1, 1, asset=asset)
    elif time_frame == 'D1':
        data = get_quotes(frame_D1, 2000, 1, 1, asset=asset)
    else:
        data = get_quotes(frame_D1, 2000, 1, 1, asset=asset)

    if data.empty:
        print(f"No data found for {asset} on timeframe {time_frame}")
        return np.array([])

    # Return only OHLC like MT5 version
    data = data.loc[:, ['open', 'high', 'low', 'close']].values
    data = np.round(data, decimals=5)
    return data


# ✅ Example usage
my_data = mass_import(5, 'D1')  # 5 = ETHUSD

# Save to Excel (optional)
if my_data.size > 0:
    df = pd.DataFrame(my_data, columns=['Open', 'High', 'Low', 'Close'])
    df.to_excel('my_data.xlsx', index=False)
    print('DATA AS DATAFRAME', df)
    #print("Data saved as my_data.xlsx")
else:
    print("No data to save.")
