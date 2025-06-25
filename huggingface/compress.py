import tensorflow as tf

# 1. Load model .h5 Anda yang sudah ada
model = tf.keras.models.load_model("model/model_klasifikasi_buah.h5")

# 2. Buat converter ke TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. Aktifkan Opsi Optimisasi (ini adalah langkah kuantisasi)
# Optimisasi default ini akan mencoba mengurangi ukuran dengan kuantisasi dinamis.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 4. Lakukan konversi
tflite_quant_model = converter.convert()

# 5. Simpan model .tflite yang baru dan jauh lebih kecil
with open("model_buah_quant.tflite", "wb") as f:
    f.write(tflite_quant_model)

print("Model telah berhasil dikonversi dan dikuantisasi ke .tflite!")
