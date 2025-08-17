import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf

# === 1. Load Data ===
df = pd.read_csv("data/dataset_transaksi.csv", sep=";")
df.columns = [c.strip().lower() for c in df.columns]
df["tanggal"] = pd.to_datetime(df["tanggal"])

# dibuat pengeluaran dan pemasukan
# df["jenis_catatan"] = df["jenis_catatan"].str.lower()
df = df[df["jenis_catatan"].str.lower() == "pengeluaran"]




# === 2. Preprocessing ===
# Group dan pivot per kategori
df_grouped = df.groupby(["tanggal", "kategori"]).sum().reset_index()
pivot_df = df_grouped.pivot(
    index="tanggal", 
    columns="kategori", 
    values="nominal").fillna(0)




# Save kategori
kategori = pivot_df.columns.tolist()
# Simpan kategori ke file
with open("data/kategori.txt", "w") as f: 
    for k in kategori:
        f.write(k + "\n")



# Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(pivot_df)

# === 3. Split Data ===
# Siapkan input sequence (sliding window)
time_steps = 1 # 1 bulan terakhir
X, y = [], []

for i in range(time_steps, len(scaled_data)):
    X.append(scaled_data[i - time_steps:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# Bagi train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# === 4. Training LSTM ===
model = Sequential()

# Layer pertama LSTM
model.add(LSTM(200, activation="relu", 
               input_shape=(1, scaled_data.shape[1]), 
               return_sequences=True))  # Menambahkan return_sequences=True

# Layer kedua LSTM
model.add(LSTM(200, activation="relu", return_sequences=True))  

# Layer ketiga LSTM
model.add(LSTM(200, activation="relu"))  # Layer LSTM ketiga

# Layer Dense output
model.add(Dense(y.shape[1]))

# Gantilah loss "mse" dengan tf.keras.losses.MeanSquaredError()
model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Save model dan scaler
model.save("model_lstm_keuangan_1.h5")
model.save("model_lstm_keuangan_1.keras")
joblib.dump(scaler, "scaler.save")

# === 5. Evaluasi ===
loss = model.evaluate(X_test, y_test)
print(f"Test MSE: {loss:.4f}")

# Simpan fitur yang digunakan
fit_features = pivot_df.columns.tolist()
with open("data/fit_features.txt", "w") as f:
    for feature in fit_features:
        f.write(feature + "\n")
