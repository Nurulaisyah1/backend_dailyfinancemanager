import os
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from io import StringIO
import logging
import uvicorn

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
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df = df[df['jenis_catatan'].str.lower() == 'pengeluaran']  # Filter pengeluaran saja

    df_grouped = df.groupby(['tanggal', 'kategori']).sum().reset_index()
    pivot_df = df_grouped.pivot(index='tanggal', columns='kategori', values='nominal').fillna(0)

    scaled_data = scaler.transform(pivot_df)
    return scaled_data, pivot_df

# === 3. Endpoint untuk menerima file CSV dan mengembalikan rekomendasi ===
@app.post("/predict/")
async def predict(transaksi: UploadFile = File(...)):
    try:
        contents = await transaksi.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")), sep=";")
        
        scaled_data, pivot_df = preprocess_data(df)

        time_steps = 1
        last_5_scaled = scaled_data[-time_steps:]
        X_input = last_5_scaled.reshape(1, time_steps, scaled_data.shape[1]).astype(np.float32)

        y_pred_scaled = model.predict(X_input)
        y_pred_original = scaler.inverse_transform(y_pred_scaled)

        avg_pengeluaran_per_kategori = pivot_df.mean().values
        total_prediksi = np.sum(y_pred_original)
        total_rata2 = np.sum(avg_pengeluaran_per_kategori)

        if total_prediksi > total_rata2:
            status = "Defisit"
            saran_status = "Pengeluaran Anda diperkirakan lebih tinggi dari biasanya. Pertimbangkan mengurangi belanja yang tidak mendesak."
        else:
            status = "Surplus"
            saran_status = "Pengeluaran Anda diperkirakan lebih rendah dari biasanya. Ini kesempatan bagus untuk menabung lebih banyak."

        detail = []
        for k, pred, avg in zip(kategori, y_pred_original[0], avg_pengeluaran_per_kategori):
            persen = ((pred - avg) / avg * 100) if avg != 0 else 0

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

        result = {
            "status": status,
            "saran": saran_status,
            "total_prediksi": round(total_prediksi),
            "detail": detail
        }

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Terjadi error: {e}")
        return {"error": str(e)}

# === 4. Health check endpoint untuk memastikan API berjalan ===
@app.get("/health")
def health_check():
    return {"status": "ok"}

# === 5. Menjalankan aplikasi ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default 5000 di lokal, dari Railway kalau di server
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
