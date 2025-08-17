import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
import os

# === 1. Load Data ===
df = pd.read_csv("data/dataset_transaksi.csv", sep=";")
df.columns = [c.strip().lower() for c in df.columns]
df["tanggal"] = pd.to_datetime(df["tanggal"])

# Filter hanya pengeluaran
df = df[df["jenis_catatan"].str.lower() == "pengeluaran"]

# === 2. Preprocessing ===
# Group dan pivot per kategori
df_grouped = df.groupby(["tanggal", "kategori"]).sum().reset_index()
pivot_df = df_grouped.pivot(index="tanggal", columns="kategori", values="nominal").fillna(0)

# Simpan kategori
os.makedirs("data", exist_ok=True)
with open("data/kategori.txt", "w") as f:
    for k in pivot_df.columns.tolist():
        f.write(k + "\n")

# Normalisasi
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(pivot_df)

# === 3. Siapkan Sequence Data ===
time_steps = 1
X, y = [], []

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i - time_steps:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 4. Model LSTM ===
model = Sequential()
model.add(LSTM(200, activation="relu", input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(200, activation="relu", return_sequences=True))
model.add(LSTM(200, activation="relu"))
model.add(Dense(y.shape[1]))

model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# === 5. Save Model & Scaler ===
model.save("model_lstm_keuangan_new.h5")
model.save("model_lstm_keuangan.keras")
joblib.dump(scaler, "scaler.save")

# Simpan fitur yang digunakan
with open("data/fit_features.txt", "w") as f:
    for feature in pivot_df.columns.tolist():
        f.write(feature + "\n")

# === 6. Evaluasi Model ===
loss = model.evaluate(X_test, y_test)
print(f"\nTest MSE: {loss:.4f}")

# === 7. Konversi ke TFLite ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS  # <== WAJIB untuk LSTM kompleks
]
converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("model_lstm_keuangan_new.tflite", "wb") as f:
    f.write(tflite_model)