import joblib
import json

# Load scaler dari file
scaler = joblib.load("scaler.save")

# Ambil nilai min dan max dari scaler
scaler_data = {
    "min": scaler.data_min_.tolist(),
    "max": scaler.data_max_.tolist()
}

# Simpan ke file JSON
with open("scaler.json", "w") as f:
    json.dump(scaler_data, f, indent=2)
