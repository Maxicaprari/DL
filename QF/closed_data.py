data = yf.download(tickers, start="2020-01-01", end="2023-01-01", group_by='ticker', auto_adjust=True)
# Construir DataFrame limpio de precios de cierre ajustados
close_data = pd.DataFrame()
for ticker in tickers:
    if ticker in data.columns.levels[0]:  # chequea que el ticker esté en los datos
        close_data[ticker] = data[ticker]['Close']

# print(close_data.head())
