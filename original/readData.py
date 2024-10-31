import pandas_datareader.data as pdr
import datetime as dt
# import matplotlib.pyplot as plt
#取得データの開始日, 終了日
start= dt.datetime(2023, 1, 1)
end= dt.datetime(2023, 12, 31)

with open("stock.txt", "r") as f:
    for line in f:
        df = pdr.DataReader(line.rstrip() + ".JP", "stooq", start, end)
        df.to_csv(line.rstrip() + ".csv")
