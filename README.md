# Klasifikasi Sentimen Ulasan Aplikasi Gojek Menggunakan Support Vector Machine (SVM)

Repositori ini berisi kode sumber eksperimen dan dataset untuk tugas **Case Method Mata Kuliah Data Mining (Bab 6: Text Mining)** pada Program Studi Informatika, Fakultas Teknologi Informasi dan Sains Data (FATISDA), Universitas Sebelas Maret.

* **Nama:** Syafiq Nafil Arkan
* **NIM:** L0124143
* **Kelas:** A
* **Dosen Pengampu:** Dr. Umi Salamah, S.Si., M.Kom.
* **Tahun:** 2026

---

## Ringkasan Metodologi

1. **Dataset & Balanced Sampling:** Menggunakan dataset publik *Gojek App Reviews Bahasa Indonesia* (`GojekAppReviewV4.0.0-V4.9.3_Cleaned.csv`, 225.002 baris). Ulasan rating 3 dieliminasi, rating 1–2 dilabeli **Negatif (-1)**, dan rating 4–5 dilabeli **Positif (+1)** dengan pengambilan sampel acak seimbang sebanyak **500 dokumen** (250 Positif, 250 Negatif; `random_state=42`).
2. **Text Preprocessing & Slang Normalization:** Mencakup *case folding*, pembersihan karakter non-alfabet, pemangkasan huruf berulang ($\ge 3$ karakter), normalisasi kata tidak baku menggunakan kamus khusus (`SLANG_DICT`), penghapusan *stopword*, dan *stemming* Nazief-Adriani menggunakan `PySastrawi`.
3. **Arsitektur Pipeline Bebas *Data Leakage*:** Ekstraksi fitur `TfidfVectorizer` (`max_features=1000`, `ngram_range=(1, 2)`) dan klasifikasi `SVC` dibungkus ke dalam `sklearn.pipeline.Pipeline` di dalam setiap lipatan validasi silang.
4. **Evaluasi & Uji Ablasi:**
   * *Nested GridSearchCV* (10-Fold Stratified Outer CV, 5-Fold Inner CV) untuk penyetelan parameter regularisasi $C \in \{0.01, 0.1, 1, 10, 100\}$.
   * Perbandingan *generalization gap* (*Training* vs. *Validation*) antara **Linear SVM** dan **Polynomial SVM** ($d=2, 3$).
   * Uji ablasi fitur N-Gram (`(1,1)`, `(2,2)`, `(1,2)`) dan stabilitas skema validasi (`5-Fold` vs. `10-Fold`).
   * Ekstraksi sampel kesalahan aktual (*False Positive* dan *False Negative*) beserta bobot fitur dominan.

---

## Ringkasan Hasil Eksperimen

| Skenario Pengujian | Konfigurasi Utama | Accuracy (Mean ± SD) | F1-Score (Mean ± SD) | Out-of-Fold AUC |
| :--- | :--- | :---: | :---: | :---: |
| **Nested GridSearchCV** | Linear SVM ($C \in \{0.01..100\}$), 10-Outer / 5-Inner | 86,80% ± 6,20% | 86,97% ± 5,54% | 0,9194 |
| **Linear SVM (Tuned)** | Kernel Linear ($C=1,0$), Unigram+Bigram, 10-Fold CV | **88,40% ± 5,06%** | **87,79% ± 5,90%** | **0,9346** |
| **Polynomial SVM ($d=2$)** | Kernel Poly ($d=2, C=1,0$), Unigram+Bigram, 10-Fold CV | 82,80% ± 6,34% | — | 0,9101 |
| **Polynomial SVM ($d=3$)** | Kernel Poly ($d=3, C=1,0$), Unigram+Bigram, 10-Fold CV | 82,20% ± 6,56% | — | 0,8877 |

---

## Cara Menjalankan Program

1. Kloning repositori ini dan masuk ke direktori proyek:
   ```bash
   git clone [https://github.com/username-kamu/gojek-sentiment-analysis-svm.git](https://github.com/username-kamu/gojek-sentiment-analysis-svm.git)
   cd gojek-sentiment-analysis-svm
   python svm_gojek_v2.py
   
## requirements
* numpy
* pandas
* matplotlib
* seaborn
* prettytable
* PySastrawi
* scikit-learn
