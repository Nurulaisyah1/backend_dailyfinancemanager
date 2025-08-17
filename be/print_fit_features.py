import os
import pandas as pd
import joblib

# --- Parameter ---
csv_path   = "data/dataset_transaksi.csv"
output_dir = "model"
output_pkl = os.path.join(output_dir, "fit_features.pkl")

# === Daftar semua kategori (urutan disesuaikan kebutuhan) ===
all_kategori = [
    "Bayar Hutang",
    "Bonus",
    "Cicilan",
    "Freelance",
    "Gaji",
    "Hiburan",
    "Internet",
    "Investasi",
    "Kasbon Diberikan",
    "Kesehatan",
    "Makanan",
    "Menabung",
    "Penarikan Tabungan",
    "Pendidikan",
    "Penerimaan Cicilan",
    "Penjualan",
    "Transportasi",
    "Listrik"
]

# --- 1. Load CSV ---
df = pd.read_csv(csv_path, sep=";")
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# --- 2. Konversi tanggal & filter pengeluaran ---
df["tanggal"] = pd.to_datetime(df["tanggal"])
df = df[df["jenis_catatan"].str.lower() == "pengeluaran"]

if df.empty:
    raise ValueError("Dataset tidak berisi baris pengeluaran!")

# --- 3. Group dan Pivot ---
df_grouped = df.groupby(["tanggal", "kategori"]).sum().reset_index()
pivot_df = df_grouped.pivot(index="tanggal", columns="kategori", values="nominal").fillna(0)

# --- 4. Tambahkan kategori yang belum muncul ---
for k in all_kategori:
    if k not in pivot_df.columns:
        pivot_df[k] = 0

# --- 5. Urutkan kolom sesuai all_kategori ---
pivot_df = pivot_df[all_kategori]

# --- 6. Simpan fitur ke file ---
fit_features = pivot_df.columns.tolist()
os.makedirs(output_dir, exist_ok=True)
joblib.dump(fit_features, output_pkl)

print(f"Sukses menyimpan {len(fit_features)} fitur ke: {output_pkl}")
print("Daftar fitur:")
for i, f in enumerate(fit_features, 1):
    print(f"{i:02d}. {f}")
