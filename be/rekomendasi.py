import numpy as np
import pandas as pd
from tensorflow.lite.python.interpreter import Interpreter
import joblib
from sklearn.preprocessing import MinMaxScaler

# === Load Scaler dan Data ===
scaler = joblib.load("scaler.save")
df = pd.read_csv("data/dataset_transaksi.csv", sep=";")
df.columns = [c.strip().lower() for c in df.columns]
df["tanggal"] = pd.to_datetime(df["tanggal"])
df = df[df["jenis_catatan"].str.lower() == "pengeluaran"]

# Preprocessing
df_grouped = df.groupby(["tanggal", "kategori"]).sum().reset_index()
pivot_df = df_grouped.pivot(index="tanggal", columns="kategori", values="nominal").fillna(0)
scaled_data = scaler.transform(pivot_df)

# Ambil sample terakhir untuk prediksi
input_data = scaled_data[-1:]  # (1, n_features)
input_data = np.expand_dims(input_data, axis=0)  # (1, 1, n_features)

# === Load TFLite Model ===
interpreter = Interpreter(model_path="model_lstm_keuangan.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.float32))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# Inverse transform hasil
predicted_nominal = scaler.inverse_transform(output_data)
print("\n=== Prediksi Pengeluaran per Kategori ===")
for kategori, value in zip(pivot_df.columns, predicted_nominal[0]):
    print(f"{kategori}: Rp {int(value):,}")

# === Status dan Rekomendasi Keuangan ===
total_prediksi = predicted_nominal.sum()
total_pemasukan = 1_000_000  # asumsi pemasukan tetap
selisih = total_pemasukan - total_prediksi

# Status keseimbangan
if selisih > 100_000:
    keseimbangan = "Surplus 💰"
    rekom_keseimbangan = "Keuangan kamu sehat! Masih ada ruang untuk menabung atau investasi."
elif selisih < -100_000:
    keseimbangan = "Defisit 🔻"
    rekom_keseimbangan = "Pengeluaran melebihi pemasukan. Coba atur ulang prioritas belanja!"
else:
    keseimbangan = "Seimbang ⚖️"
    rekom_keseimbangan = "Pengeluaran dan pemasukanmu hampir seimbang. Tetap waspada ya!"

# Status konsumsi
if total_prediksi < 300_000:
    status = "Hemat Banget 🎉"
    saran = "Keuangan kamu sangat stabil, tetap lanjutkan kebiasaan baik ini!"
elif total_prediksi < 600_000:
    status = "Cukup Stabil 🙂"
    saran = "Pengeluaran masih terkendali. Perhatikan pengeluaran tidak perlu."
elif total_prediksi < 1_000_000:
    status = "Mulai Boros ⚠️"
    saran = "Review kategori yang paling besar, mungkin ada pengeluaran tidak perlu."
else:
    status = "Boros Banget 😅"
    saran = "Evaluasi ulang anggaran dan mulai kurangi pengeluaran yang kurang penting."

# === Print Summary ===
print("\n=== Ringkasan Analisis Keuangan ===")
print(f"Total Pemasukan      : Rp {int(total_pemasukan):,}")
print(f"Total Pengeluaran    : Rp {int(total_prediksi):,}")
print(f"Selisih              : Rp {int(selisih):,}")
print(f"Status Keuangan      : {keseimbangan}")
print(f"Rekomendasi Keseimbangan : {rekom_keseimbangan}")
print(f"Status Konsumsi      : {status}")
print(f"Saran Friendly        : {saran}")
