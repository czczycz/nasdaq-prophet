import yfinance as yf
import pandas as pd

# download historical data of Nasdaq (^IXIC)
df = yf.download("^IXIC", period="max")
df.to_csv("dataset/nasdaq_index_raw.csv")
print("CSV 文件已生成：nasdaq_index_raw.csv")
