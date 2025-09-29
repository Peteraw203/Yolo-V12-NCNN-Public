# Deteksi Jalan & Halangan Real-time untuk Navigasi Kursi Roda Otomatis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Sebuah proyek Computer Vision yang diimplementasikan pada **Android Native** untuk mendeteksi kondisi permukaan jalan dan berbagai halangan secara real-time. Sistem ini dirancang khusus untuk meningkatkan keselamatan navigasi **kursi roda otomatis ♿**, dengan memanfaatkan kecepatan inferensi dari model **YOLOv12** yang dioptimalkan menggunakan *framework* **NCNN**.

![Demo Aplikasi](https://via.placeholder.com/600x350.png?text=Tempatkan+GIF+Demo+Aplikasi+Anda+di+Sini)

## Latar Belakang

Navigasi yang aman bagi pengguna kursi roda otomatis sangat bergantung pada kemampuan sistem untuk memahami lingkungan sekitarnya. Permukaan jalan yang tidak rata, berlubang, atau adanya halangan tak terduga dapat menjadi risiko. Proyek ini hadir sebagai solusi dengan menyediakan sistem deteksi visual yang cepat, efisien, dan berjalan sepenuhnya di perangkat (*on-device*), sehingga tidak memerlukan koneksi internet dan memiliki latensi minimal.

---

## Fitur Utama

-   **Deteksi Real-time**: Menganalisis *feed* video dari kamera secara langsung dengan *frame rate* yang tinggi.
-   **Klasifikasi Ganda**: Mampu mengidentifikasi beberapa kelas objek sekaligus:
-   **Inferensi Cepat**: Dioptimalkan dengan **NCNN (NCNN Is Not CNN)**, sebuah *framework* inferensi neural network ultra-cepat dari Tencent yang dirancang untuk mobile.
-   **Model Canggih**: Menggunakan arsitektur **YOLOv12**, generasi terbaru dari YOLO yang menawarkan akurasi dan kecepatan superior.
-   **Implementasi Penuh di Android**: Berjalan secara *native* di Android dengan *C++* untuk performa maksimal dan integrasi sistem yang mendalam.

---

## Teknologi yang Digunakan

-   **Platform**: Android
-   **Computer Vision**: OpenCV
-   **Deep Learning Framework**: NCNN (by Tencent)
-   **Model Deteksi Objek**: YOLOv12 (`.param` & `.bin`)
-   **Development Environment**: Android Studio

---

## Arsitektur Sistem

Sistem bekerja dalam alur yang sederhana namun efektif:

1.  **Input Kamera**: Aplikasi mengakses *stream* video dari kamera perangkat Android.
2.  **Pra-pemrosesan**: Setiap *frame* dari video diubah ukurannya dan dinormalisasi sesuai dengan kebutuhan input model YOLOv12 menggunakan OpenCV.
3.  **Inferensi Model**: *Frame* yang telah diproses dimasukkan ke *engine* NCNN untuk dideteksi oleh model YOLOv12.
4.  **Pasca-pemrosesan**: Hasil inferensi seperti koordinat bounding box, skor kepercayaan, dan kelas diolah untuk menyaring deteksi yang tidak relevan.
5.  **Visualisasi**: *Bounding box* dan label kelas digambar di atas *frame* asli dan ditampilkan di layar aplikasi secara *real-time*.



---

## Memulai

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut.

### Prasyarat

-   [Android Studio](https://developer.android.com/studio) (versi terbaru direkomendasikan).
-   Android SDK dengan API level yang sesuai.
-   NDK (Native Development Kit) terinstal melalui SDK Manager di Android Studio.
-   Perangkat Android fisik untuk pengujian (atau emulator yang mendukung akselerasi kamera).

### Instalasi & Konfigurasi

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/username-anda/nama-repo-anda.git](https://github.com/username-anda/nama-repo-anda.git)
    ```

2.  **Buka proyek** di Android Studio.

3.  **Unduh Model YOLOv12:**
    Pastikan Anda memiliki file model YOLOv12 yang telah dikonversi ke format NCNN:
    -   `yolov12.param`
    -   `yolov12.bin`

4.  **Tempatkan file model** ke dalam direktori `app/src/main/assets/` pada struktur proyek.

5.  **Sinkronkan Proyek**: Biarkan Android Studio mengunduh semua dependensi Gradle yang diperlukan.

6.  **Build dan Jalankan** aplikasi pada perangkat Android Anda.

---

## Informasi Model

-   **Arsitektur**: YOLOv12
-   **Format**: NCNN
-   **Kelas yang Dideteksi**:
    -   `Aspal`
    -   `Paving_block`
    -   `Keramik`
    -   `Tangga`

---

## Lisensi

Proyek ini dilisensikan di bawah **Lisensi MIT**. Lihat file `LICENSE` untuk detail lebih lanjut.

---

##  Credit

-   Tim **Tencent** untuk pengembangan *framework* NCNN.
-   Para pengembang di balik arsitektur YOLO.
-   Komunitas *open-source* yang telah menyediakan berbagai *library* dan pengetahuan.