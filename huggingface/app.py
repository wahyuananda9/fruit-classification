import gradio as gr

# Ganti baris ini
# import tflite_runtime.interpreter as tflite

# Dengan baris ini
import tensorflow as tf  # Import full TensorFlow
from PIL import Image
import numpy as np
import json


TFLITE_MODEL_PATH = "model/model_buah_quant.tflite"
CLASSES_PATH = "model/classes.json"

# Ganti baris ini
# interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)

# Dengan baris ini
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(CLASSES_PATH) as f:
    class_indices = json.load(f)
classes = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]


def predict_fruit(input_image: Image.Image):
    """
    Fungsi untuk memprediksi kelas buah menggunakan model TFLite.
    Input: gambar dari Gradio (format PIL.Image)
    Output: dictionary berisi label kelas dan skor kepercayaan
    """
    img = input_image.resize((500, 500))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]["index"])[0]
    confidences = {classes[i]: float(preds[i]) for i in range(len(classes))}

    return confidences


title = "Klasifikasi Gambar Buah (Versi TFLite) 🍓🍌🍎"
description = """
Unggah gambar buah untuk diklasifikasikan. 
Model ini adalah versi ringan (.tflite) yang telah dioptimalkan (dikuantisasi), 
sehingga lebih cepat dan lebih kecil ukurannya.
"""

iface = gr.Interface(
    fn=predict_fruit,
    inputs=gr.Image(type="pil", label="Upload Gambar Buah"),
    outputs=gr.Label(num_top_classes=3, label="Hasil Prediksi"),
    title=title,
    description=description,
)

iface.launch()