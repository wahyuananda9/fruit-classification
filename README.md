# Proyek Klasifikasi Gambar Buah dengan CNN

Repositori ini berisi kode dan sumber daya untuk membangun, melatih, dan men-deploy model *Convolutional Neural Network* (CNN) untuk mengklasifikasikan 90 jenis buah yang berbeda. Model ini dilatih menggunakan dataset "Fruits 360" dan mencapai akurasi yang sangat tinggi.

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/whyyou50/fruits_classification)

## Daftar Isi
1.  [Deskripsi Proyek](#deskripsi-proyek)
2.  [Dataset](#dataset)
3.  [Arsitektur Model](#arsitektur-model)
4.  [Hasil](#hasil)
5.  [Instalasi](#instalasi)
6.  [Cara Menggunakan](#cara-menggunakan)
7.  [Deployment](#deployment)
8.  [Struktur File](#struktur-file)

## Deskripsi Proyek
Proyek ini mengimplementasikan CNN menggunakan TensorFlow dan Keras untuk melakukan klasifikasi gambar. Tujuannya adalah untuk membuat model yang dapat secara akurat mengidentifikasi jenis buah dari sebuah gambar. Proyek ini mencakup semua langkah mulai dari pra-pemrosesan data, pembangunan model, pelatihan, evaluasi, hingga deployment ke platform publik.

## Dataset
* **Nama Dataset**: Fruits 360.
* **Sumber**: Diunduh melalui `kagglehub` dari dataset `moltean/fruits`.
* **Spesifikasi**:
    * **Jumlah Kelas**: 90.
    * **Data Latih**: 26.335 gambar.
    * **Data Validasi**: 2.887 gambar.
    * **Data Uji**: 14.527 gambar.
    * **Resolusi Input**: Gambar diubah ukurannya menjadi `500x500` piksel.

## Arsitektur Model
Model ini menggunakan arsitektur `Sequential` dari Keras. Lapisan-lapisan utamanya adalah sebagai berikut:
* `Conv2D(32, (3,3), activation='relu', input_shape=(500, 500, 3))`
* `MaxPooling2D(2, 2)`
* `Conv2D(64, (3,3), activation='relu')`
* `MaxPooling2D(2,2)`
* `Flatten()`
* `Dense(128, activation='relu')`
* `Dropout(0.5)`
* `Dense(90, activation='softmax')`

Model ini dikompilasi menggunakan optimizer `adam` dan loss function `categorical_crossentropy`.

## Hasil
Setelah melalui pelatihan selama 30 epoch, model dievaluasi pada set data uji dan mencapai hasil yang sangat memuaskan.
* **Akurasi pada Data Uji**: **99.62%**.

## Instalasi
Untuk menjalankan proyek ini secara lokal, pastikan Anda memiliki Python dan `pip` terinstal. Kemudian, instal pustaka yang diperlukan:
```bash
pip install tensorflow keras pillow scikit-learn matplotlib
