import tensorflow as tf

# Memuat model Keras (model .h5)
model = tf.keras.models.load_model("model_lstm_keuangan.h5")

# Mengonversi model Keras ke model TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Menentukan operasi yang didukung untuk model kompleks (misalnya LSTM)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,     # Operasi standar TensorFlow Lite
    tf.lite.OpsSet.SELECT_TF_OPS       # Operasi TensorFlow yang lebih kompleks seperti LSTM
]

# Menghindari penurunan tensor list ops (untuk operasi kompleks seperti LSTM)
converter._experimental_lower_tensor_list_ops = False

# Mengonversi model ke format TensorFlow Lite (.tflite)
tflite_model = converter.convert()

# Menyimpan model yang sudah dikonversi ke file .tflite
with open("model_lstm_keuangan_new.tflite", "wb") as f:
    f.write(tflite_model)

print("Model berhasil dikonversi menjadi TFLite!")
