# ☕ Coffee Bean Image Classification using Deep Learning

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk melakukan klasifikasi tingkat kematangan biji kopi berdasarkan citra digital menggunakan pendekatan *Deep Learning*. Sistem dikembangkan dengan membandingkan tiga model berbeda, yaitu:
1. Convolutional Neural Network (CNN) yang dilatih dari awal (*scratch*)
2. ResNet18 (pretrained)
3. MobileNetV2 (pretrained)

Hasil pelatihan dan evaluasi divisualisasikan melalui sebuah **dashboard interaktif berbasis Streamlit** yang memungkinkan pengguna untuk:
- Melihat ringkasan performa pelatihan model
- Menguji gambar biji kopi secara langsung menggunakan model pilihan

---

## 📂 Dataset dan Preprocessing

### 🔹 Dataset
Dataset yang digunakan merupakan dataset citra biji kopi yang terdiri dari **4 kelas**, yaitu:
- **Dark**
- **Green**
- **Light**
- **Medium**

Distribusi dataset akhir:
| Dataset | Jumlah |
|------|------|
| Train | 4.200 |
| Validation | 400 |
| Test | 400 |
| **Total** | **5.000** |

Dataset awal terdiri dari 1.600 citra, kemudian dilakukan **augmentasi data** untuk meningkatkan jumlah dan keberagaman data latih.

---

### 🔹 Preprocessing & Augmentasi
Tahapan preprocessing yang diterapkan:
- Resize citra ke ukuran `224 × 224`
- Normalisasi citra
- Konversi ke tensor

Augmentasi data **hanya diterapkan pada data training**, dengan teknik:
- Random Rotation
- Horizontal Flip
- Zoom
- Color Jitter

Augmentasi bertujuan untuk:
- Mengurangi overfitting
- Meningkatkan generalisasi model

---

## 🧠 Model yang Digunakan

### 1️⃣ CNN Scratch
Model CNN dibangun dan dilatih dari awal tanpa menggunakan bobot pretrained.  
Model ini digunakan sebagai **baseline** untuk membandingkan efektivitas transfer learning.

**Karakteristik:**
- 4 convolutional layers
- MaxPooling dan ReLU activation
- Fully connected layer di akhir
- Early Stopping untuk mencegah overfitting

---

### 2️⃣ ResNet18 (Pretrained)
ResNet18 menggunakan bobot pretrained dari ImageNet dan hanya melakukan penyesuaian pada *fully connected layer*.

**Keunggulan:**
- Residual connection membantu mengatasi vanishing gradient
- Ekstraksi fitur yang lebih kuat
- Konvergensi lebih cepat

---

### 3️⃣ MobileNetV2 (Pretrained)
MobileNetV2 merupakan model ringan yang dioptimalkan untuk efisiensi komputasi.

**Keunggulan:**
- Arsitektur depthwise separable convolution
- Ukuran model lebih kecil
- Cocok untuk deployment

---

## 📊 Hasil Evaluasi dan Analisis Perbandingan

Evaluasi dilakukan menggunakan metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 🔹 Tabel Perbandingan Model

| Nama Model | Akurasi | Hasil Analisis |
|----------|---------|---------------|
| CNN Scratch | 96,75% | Memberikan performa yang cukup baik, namun masih terbatas dalam mengekstraksi fitur kompleks dibandingkan model pretrained. |
| ResNet18 | 98,75% | Memberikan performa terbaik berkat kemampuan transfer learning dan arsitektur residual yang kuat. |
| MobileNetV2 | 99,5% | Menawarkan keseimbangan antara akurasi dan efisiensi komputasi, cocok untuk sistem dengan resource terbatas. |

**Kesimpulan:**  
Model pretrained secara umum menghasilkan performa yang lebih baik dibandingkan CNN yang dilatih dari awal, dengan ResNet18 menunjukkan akurasi tertinggi.

---

## 🖥️ Panduan Menjalankan Sistem Website Secara Lokal

### 🔹 1. Clone Repository
```bash
git clone https://github.com/username/coffee-bean-classification.git
cd coffee-bean-classification

### 🔹 2. Clone Repository

