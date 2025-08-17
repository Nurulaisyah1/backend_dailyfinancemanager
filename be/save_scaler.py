import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv("data/dataset_transaksi.csv", sep=";")
df.columns = [c.strip().lower() for c in df.columns]
df['tanggal'] = pd.to_datetime(df['tanggal'])
df = df[df['jenis_catatan'].str.lower() == 'pengeluaran']

df_grouped = df.groupby(['tanggal', 'kategori']).sum().reset_index()
pivot_df = df_grouped.pivot(index='tanggal', columns='kategori', values='nominal').fillna(0)

scaler = MinMaxScaler()
scaler.fit(pivot_df)

joblib.dump(scaler, "model/scaler.save")
