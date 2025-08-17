import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import StringIO
import logging

# === Konfigurasi logger ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# === Inisialisasi FastAPI ===
app = FastAPI()

# === 1. Memuat model H5 dan scaler ===
model = tf.keras.models.load_model("model/model_lstm_keuangan.h5")  # Memuat model H5
scaler = joblib.load("scaler.save")  # Memuat scaler yang digunakan saat pelatihan

# Membaca kategori yang digunakan selama pelatihan
with open("data/kategori.txt") as f:
    kategori = [line.strip() for line in f]

# === 2. Fungsi untuk memproses data transaksi ===
def preprocess_data(df):
    # Normalisasi nama kolom
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df = df[df['jenis_catatan'].str.lower() == 'pengeluaran']  # Filter pengeluaran saja

    # Grouping dan Pivot per kategori
    df_grouped = df.groupby(['tanggal', 'kategori']).sum().reset_index()
    pivot_df = df_grouped.pivot(index='tanggal', columns='kategori', values='nominal').fillna(0)

    # Scaling
    scaled_data = scaler.transform(pivot_df)
    return scaled_data, pivot_df

# === 3. Endpoint untuk menerima file CSV dan mengembalikan rekomendasi ===
@app.post("/predict/")
async def predict(transaksi: UploadFile = File(...)):
    try:
        # Membaca file CSV
        contents = await transaksi.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), sep=";")
        
        # Memproses data untuk prediksi
        scaled_data, pivot_df = preprocess_data(df)

        # Membuat input sequence untuk model LSTM
        time_steps = 1  # Misalnya menggunakan 1 bulan terakhir untuk prediksi
        last_5_scaled = scaled_data[-time_steps:]
        X_input = last_5_scaled.reshape(1, time_steps, scaled_data.shape[1]).astype(np.float32)

        # Prediksi dengan model H5
        y_pred_scaled = model.predict(X_input)

        # Balikkan ke bentuk nominal asli
        y_pred_original = scaler.inverse_transform(y_pred_scaled)

        # Hitung statistik dan rekomendasi
        avg_pengeluaran_per_kategori = pivot_df.mean().values
        total_prediksi = np.sum(y_pred_original)
        total_rata2 = np.sum(avg_pengeluaran_per_kategori)

        if total_prediksi > total_rata2:
            status = "Defisit"
            saran_status = (
                "Pengeluaran Anda diperkirakan lebih tinggi dari biasanya. "
                "Pertimbangkan mengurangi belanja yang tidak mendesak."
            )
        else:
            status = "Surplus"
            saran_status = (
                "Pengeluaran Anda diperkirakan lebih rendah dari biasanya. "
                "Ini kesempatan bagus untuk menabung lebih banyak."
            )

        # Detail per kategori
        detail = []
        for k, pred, avg in zip(kategori, y_pred_original[0], avg_pengeluaran_per_kategori):
            # Menghitung persentase perubahan dalam rentang 1-100%
            persen = ((pred - avg) / avg * 100) if avg != 0 else 0

            # Menentukan status per kategori
            if persen > 50:
                catatan = f"Naik signifikan dibanding biasanya ({round(persen, 2)}%). Disarankan untuk mengurangi pengeluaran di kategori ini."
            elif persen > 10:
                catatan = f"Naik dibanding biasanya ({round(persen, 2)}%). Pertimbangkan untuk memonitor pengeluaran lebih ketat."
            elif persen < -10:
                catatan = f"Turun dibanding biasanya ({round(persen, 2)}%). Anda bisa menyesuaikan anggaran di kategori ini."
            else:
                catatan = f"Stabil ({round(persen, 2)}%). Pengelolaan pengeluaran dapat tetap seperti biasa."

            detail.append({
                "kategori": k,
                "prediksi": round(pred),
                "status": catatan
            })

        # Menyusun hasil rekomendasi
        result = {
            "status": status,
            "saran": saran_status,
            "total_prediksi": round(total_prediksi),
            "detail": detail
        }

        # Mengembalikan hasil dalam format JSON
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Terjadi error: {e}")
        return {"error": str(e)}

# === 4. Health check endpoint untuk memastikan API berjalan ===
@app.get("/health")
def health_check():
    return {"status": "ok"}
