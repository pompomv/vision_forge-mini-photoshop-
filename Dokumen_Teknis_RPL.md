# DOKUMEN TEKNIS REKAYASA PERANGKAT LUNAK

## HALAMAN JUDUL
- **Nama Proyek**: Lumina Synth (Mini Photoshop) - Aplikasi Pengolahan Citra Digital
- **Nama Organisasi/Instansi**: Politeknik Negeri Jakarta (PNJ)
- **Versi Dokumen**: 1.0
- **Tanggal Penyusunan**: 9 Juni 2026
- **Penyusun**: Tim Pengembang Lumina Synth

## RIWAYAT REVISI
| Versi | Tanggal | Deskripsi Perubahan | Penyusun |
|---|---|---|---|
| 1.0 | 9 Juni 2026 | Pembuatan Dokumen Awal | Tim Pengembang |

---

## DAFTAR ISI
- BAB 1. PENDAHULUAN
- BAB 2. GAMBARAN UMUM SISTEM
- BAB 3. ANALISIS KEBUTUHAN
- BAB 4. DESAIN SISTEM
- BAB 5. DESAIN ANTARMUKA
- BAB 6. IMPLEMENTASI
- BAB 7. PENGUJIAN
- BAB 8. PENUTUP
- LAMPIRAN

---

## BAB 1. PENDAHULUAN

### 1.1 Latar Belakang
Pengolahan Citra Digital (PCD) merupakan salah satu bidang krusial dalam ilmu komputer dengan penerapan luas di berbagai sektor seperti medis, keamanan, dan desain grafis. Kebutuhan akan alat yang praktis untuk mempelajari, menguji, dan menerapkan algoritma PCD seperti perbaikan citra (enhancement), transformasi geometri, restorasi, segmentasi, hingga deteksi objek cerdas menjadi sangat tinggi. Proyek **Lumina Synth (Mini Photoshop)** dikembangkan untuk menyediakan antarmuka web interaktif yang memungkinkan pengguna memproses citra secara _real-time_ tanpa perlu menginstal perangkat lunak desktop yang berat.

### 1.2 Tujuan Pengembangan
- Mengembangkan aplikasi berbasis web interaktif untuk mempraktikkan operasi-operasi Pengolahan Citra Digital.
- Menyediakan antarmuka (UI) yang _user-friendly_ dan modern bagi pengguna untuk mengatur parameter gambar secara instan.
- Mengimplementasikan fitur kecerdasan buatan, spesifiknya Convolutional Neural Network (CNN) untuk kemampuan klasifikasi dan deteksi objek.

### 1.3 Ruang Lingkup Sistem
Batasan dari sistem Lumina Synth mencakup:
- Dukungan format gambar: JPG, PNG, dan BMP (hingga 50MB).
- Fitur utama yang disediakan meliputi operasi Enhancement, Geometric Transformation, Image Restoration, Edge & Binary Processing, Color Processing, Segmentation, Image Compression, dan AI Object Detection.
- Aplikasi bersifat _stateless_ per sesi, menyimpan gambar secara _temporary_ di penyimpanan lokal server selama pemrosesan dan tidak menggunakan database relasional.

### 1.4 Definisi, Akronim, dan Singkatan
- **PCD**: Pengolahan Citra Digital
- **CNN**: Convolutional Neural Network (Arsitektur Deep Learning)
- **UI / UX**: User Interface / User Experience
- **API**: Application Programming Interface
- **MobileNetV2**: Model arsitektur CNN yang ringan dan efisien untuk pemrosesan gambar _mobile_ maupun _web_.

### 1.5 Referensi
- Dokumentasi Flask Python
- Dokumentasi OpenCV (Open Source Computer Vision Library)
- Dokumentasi TensorFlow dan Keras

---

## BAB 2. GAMBARAN UMUM SISTEM

### 2.1 Deskripsi Sistem
Lumina Synth adalah aplikasi web interaktif _Mini Photoshop_ yang didesain untuk pengolahan citra digital. Aplikasi ini memisahkan logika antarmuka yang dinamis dengan komputasi pengolahan citra di sisi backend (server). Dengan interaksi yang responsif, setiap perubahan pada nilai parameter akan langsung tercermin pada hasil _preview_ gambar secara _side-by-side_ (sebelum dan sesudah), dilengkapi dengan kalkulasi histogram untuk analisis visual mendalam.

### 2.2 Stakeholder Sistem
- **Administrator**: Memelihara server dan mengelola operasional dari aplikasi Flask, folder `uploads`, dan `outputs`.
- **Operator**: (Tidak relevan secara khusus, karena pengguna bertindak secara mandiri pada web).
- **Pengguna**: End-user yang mengunggah gambar, bereksperimen dengan berbagai filter dan algoritma, lalu mengunduh hasil gambar atau kompresi gambar.
- **Manajemen**: Tim dosen pengampu PCD atau tim supervisor yang melakukan _assessment_ terhadap kapabilitas fitur algoritma sistem.

### 2.3 Arsitektur Sistem
Sistem ini menggunakan arsitektur **Client-Server**.
1. **Client (Frontend)**: Dibangun dengan HTML5, CSS3, dan Vanilla JavaScript (dibantu dengan Bootstrap 5) untuk menangkap input pengguna, merender UI/UX yang modern, serta berkomunikasi via RESTful API ke server.
2. **Server (Backend)**: Menggunakan _framework_ web Flask (Python). Mengatur pengelolaan rute API (_endpoints_), memanajemen _session files_, dan memanggil modul pengolahan spesifik (seperti `image_processor.py`, `compressor.py`, dan `cnn_detector.py`).

---

## BAB 3. ANALISIS KEBUTUHAN

### 3.1 Kebutuhan Fungsional

**KF-01: Image Upload & Management**
- **Deskripsi**: Sistem dapat menerima unggahan file citra dari pengguna.
- **Input**: File gambar (JPG/PNG/BMP).
- **Proses**: Menyimpan secara sementara sebagai file `_original` dan `_current`.
- **Output**: Preview gambar tertampil di _canvas_ dengan statistik lebar/tinggi.

**KF-02: Image Enhancement**
- **Deskripsi**: Penyesuaian kualitas visual dasar.
- **Input**: Parameter brightness, contrast, kernel size, metode _smoothing_.
- **Proses**: Backend menghitung piksel matriks berdasar input algoritma.
- **Output**: Gambar dengan tingkat pencerahan, ketajaman, maupun _smoothness_ yang baru.

**KF-03: Edge Detection & Morphology**
- **Deskripsi**: Deteksi garis tepi dan operasi morfologi matematika.
- **Input**: Threshold value, Canny (T1, T2), Kernel Size operasi erode/dilate.
- **Proses**: Menggunakan filter kernel matematis spesifik (Sobel, Prewitt, Canny).
- **Output**: Hasil citra biner (hitam-putih) atau bergaris tepi.

**KF-04: AI Object Detection (CNN)**
- **Deskripsi**: Klasifikasi dan pengenalan objek dalam gambar.
- **Input**: Kategori filter (contoh: Semua, Hewan, Kendaraan) dan jumlah prediksi (Top N).
- **Proses**: Memuat model MobileNetV2 dan merender _bounding_box_/anotasi prediksi objek.
- **Output**: Citra yang ditandai dengan deteksi objek dan metrik probabilitas dalam bentuk JSON.

**KF-05: Image Compression**
- **Deskripsi**: Kompresi ukuran data gambar dengan berbagai metode algoritma (Huffman, RLE, LZW, dll).
- **Input**: Metode kompresi yang dipilih.
- **Proses**: Mentransformasikan data matriks piksel ke ukuran disk yang efisien.
- **Output**: Laporan statistik rasio kompresi dan file terkompresi.

### 3.2 Kebutuhan Non-Fungsional
- **Performa**: Pemrosesan setiap filter citra secara rata-rata di bawah 2 detik per _request_ API (untuk resolusi gambar wajar).
- **Usability**: UI didesain bersifat asinkron (AJAX). Render UI tidak _reload_ setiap ada pemrosesan.
- **Keamanan**: Maksimal limit _file upload_ 50MB untuk mencegah memori server berlebih (_Overload_).

### 3.3 Kebutuhan Perangkat Keras
- **Server**: CPU 4-Cores, 4GB RAM, Storage Minimum 10GB untuk menampung file temporer dan instalasi modul Machine Learning.
- **Client**: Komputer/Laptop biasa dengan tetikus, CPU 2-cores.

### 3.4 Kebutuhan Perangkat Lunak
- **Sistem Operasi**: Windows, macOS, atau Distribusi Linux.
- **Backend Software**: Python 3.11+, Flask >= 3.0.
- **Machine Learning / Pengolahan Citra Library**: OpenCV-Python >=4.8, NumPy >=1.24, TensorFlow >=2.15, scikit-image, scikit-learn.
- **Browser Client**: Google Chrome, Mozilla Firefox, Microsoft Edge terbaru (dengan dukungan Canvas API dan ES6 JavaScript).

---

## BAB 4. DESAIN SISTEM

### 4.1 Flow Sistem
1. Pengguna membuka aplikasi pada web browser.
2. Pengguna mengunggah citra (Unggahan diproses via API `/api/upload`).
3. Server merespon dengan data base64 citra serta menginisiasi histori penyimpanan temporer (menggunakan UUID Session).
4. Pengguna memodifikasi slider parameter filter di sisi _sidebar_ (contoh: Blur Gaussian).
5. _Client_ mengirim POST request asinkronus dengan format JSON ke `/api/restore` atau API relevan.
6. Server OpenCV/Python menerima data, melakukan kalkulasi matriks citra, menyimpan hasil _state_ baru di `/uploads/`, lalu mengirim respon JSON base64 ke pengguna.
7. Aplikasi _client_ merender perubahan visual beserta grafik histogram aktual ke layar pengguna.

### 4.2 Entity Relationship Diagram (ERD) (Opsional)
*(Sistem saat ini belum memerlukan basis data relasional karena manajemen gambar bergantung murni pada Sistem File di folder `uploads/` dan Session yang dinamis, dibersihkan per life-cycle aplikasi.)*

### 4.3 Desain API
Sistem memiliki beragam API mikro yang mengkapsulkan masing-masing algoritma komputer visi:

**1. Endpoint Image Transform**
- **Endpoint**: `/api/transform`
- **Method**: `POST`
- **Request**: JSON `{ "operation": "rotate", "angle": 90, "interpolation": "bilinear" }`
- **Response**: JSON `{ "success": true, "image": "<base64_string>", "info": {"width": 800, "height": 600} }`

**2. Endpoint Object Detection**
- **Endpoint**: `/api/detect`
- **Method**: `POST`
- **Request**: JSON `{ "category": "animal", "top_n": 5 }`
- **Response**: JSON `{ "success": true, "image": "<base64_string>", "predictions": [...], "category": "animal" }`

**3. Endpoint Download**
- **Endpoint**: `/api/download?format=png`
- **Method**: `GET`
- **Request**: Parameter query (format)
- **Response**: Attachment Blob Binary image file.

---

## BAB 5. DESAIN ANTARMUKA

### 5.1 Standar Desain UI/UX
- **Skema Warna**: Tema _Modern Dark Glassmorphism_ (`#0f172a` latar belakang utama, dengan aksen biru/ungu metalik khas perangkat lunak desain tingkat lanjut).
- **Tipografi**: _Sans-serif_ interaktif, mudah dibaca.
- **Komposisi Layout**: Terbagi menjadi area Navbar fungsional di atas, Sidebar _collapsible_ alat pengolahan di kiri, dan Canvas Preview (komparasi _Before / After_) yang besar di sisi tengah/kanan, dilengkapi modul Histogram di panel bawah.

### 5.2 Mockup Halaman
- **Halaman Login**: (Aplikasi ini berjalan sebagai utilitas _stateless_ yang bersifat _open-access_, tidak ada portal login khusus).
- **Dashboard / Ruang Kerja Utama (Workspace)**: 
  - Terdapat _Dropzone upload file_.
  - Setelah gambar masuk, antarmuka _workspace_ akan menampilkan gambar, memperbolehkan pergerakan nilai _slider_ paramater yang langsung terefleksi.
- **Manajemen Data**: Tombol navigasi "File" (Buka, Simpan Sebagai, Unduh, Reset Ke Original).
- **Laporan**: Terdapat representasi grafis melalui *Canvas Chart.js* yang memberikan informasi Histogram Merah, Hijau, Biru (RGB) sebelum dan sesudah operasi. Modul kompresi akan menyoroti _pop-up/status text_ mengenai data pengurangan byte data.

### 5.3 Navigasi Sistem
- **Navbar Utama**: File, Edit, Filter, Transform.
- **Sidebar Menu Accordion**: Enhancement, Transform, Restoration, Edge & Binary, Color, Segmentation, Compression, AI Detection (CNN).

---

## BAB 6. IMPLEMENTASI

### 6.1 Teknologi yang Digunakan
- **Frontend**: 
  - HTML5 & CSS3 Vanilla.
  - Bootstrap 5 (CSS Framework untuk navigasi dan struktur modal).
  - Bootstrap Icons.
  - JavaScript Fetch API (AJAX).
  - Chart.js (Untuk merender grafik Histogram).
- **Backend**:
  - Python 3.11.
  - Flask 3.0.
- **Framework & Libraries Komputasi Visi**:
  - `opencv-python` >= 4.8
  - `numpy` >= 1.24
  - `Pillow` >= 10.0
  - `scikit-image` >= 0.21
  - `scikit-learn` >= 1.3
  - `tensorflow` >= 2.15 (Integrasi Deep Learning CNN MobileNetV2).
- **Database**:
  - OS File System (`/uploads/` & `/outputs/` terhubung via Flask session ID UUID).

---

## BAB 7. PENGUJIAN

### 7.1 Strategi Pengujian
- Menggunakan pengujian kualitatif visual dan fungsional UI. Memastikan pengiriman data antara front-end JavaScript dan API back-end berjalan aman tanpa _Memory Leak_ dengan tipe _multipart_ dan Base64 encoding.

### 7.2 Skenario Pengujian
- **Unit Testing**: Memverifikasi setiap fungsi di kelas `ImageProcessor` (`image_processor.py`) tidak mengeluarkan galat (error) jika diberikan matriks piksel nol atau citra sangat kecil (e.g., 1x1 px).
- **Integration Testing**: Memastikan API endpoint Flask sukses memanggil fungsi kompresi dari `compressor.py` dan menerima respons waktu (elapsed time) kalkulasi statistik dengan _status-code_ 200.
- **System Testing**: Mensimulasikan pengguna menekan slider berulang kali (_Stress Test_), memvalidasi bahwa antarmuka memiliki penanganan _loading overlay_ yang efektif dan server sanggup melayani _queue_ HTTP.
- **User Acceptance Testing (UAT)**: Menugaskan sekelompok mahasiswa untuk melakukan manipulasi citra menggunakan Lumina Synth untuk menyelesaikan lembar kerja praktik (PCD), memverifikasi kebergunaan UX dan ketepatan output algoritma filter dengan ekspektasi teori matematika citra.

---

## BAB 8. PENUTUP

### 8.1 Kesimpulan
Proyek Lumina Synth telah berhasil mengimplementasikan berbagai modul Pengolahan Citra Digital dasar hingga mahir (termasuk kompresi dan integrasi kecerdasan buatan/CNN) dalam sebuah _platform_ berbasis web tunggal. Arsitektur Client-Server memungkinkan eksekusi algoritma OpenCV/TensorFlow di server lokal secara _power-efficient_, selagi menghadirkan UI responsif tanpa beban perangkat lunak desktop.

### 8.2 Rencana Pengembangan Lanjutan
1. **Dukungan Format RAW / Video**: Memperluas tipe MIME untuk mengolah file RAW fotografi dan pemrosesan urutan video.
2. **Koneksi Database Relasional**: Mengimplementasikan PostgreSQL/MySQL untuk fitur menyimpan Histori Editan Gambar (Undo-Tree) persisten untuk _user account_.
3. **Penyempurnaan Model AI**: Penambahan integrasi jaringan semantik model kustom berbasis YOLO (You Only Look Once) untuk akurasi pendeteksian dan _masking_ (Segmentasi _Instance_ AI) yang lebih baik.

---

## LAMPIRAN

### Lampiran A. Diagram Lengkap
(Dokumen terpisah/diagram disisipkan di luar text-markdown)

### Lampiran B. Struktur Basis Data
Aplikasi tidak menggunakan basis data konvensional. Struktur File Management:
- `/uploads/<session_id>_original.png`
- `/uploads/<session_id>_current.png`
- `/outputs/<filename>.<ext>`

### Lampiran C. Dokumentasi API
Sebagian dokumentasi disematkan pada Bab 4.3. Semua permintaan POST mengembalikan JSON format utama:
```json
{
  "success": true,
  "image": "data:image/jpeg;base64,...",
  "info": {
    "width": 1920,
    "height": 1080
  }
}
```

### Lampiran D. Hasil Pengujian
Sistem berfungsi optimal di atas lingkungan Python >=3.11. Model MobileNetV2 mengunduh otomatis _weights pre-trained_ pada permulaan sistem pertama.

### Lampiran E. Manual Pengguna
1. Buka laman `http://localhost:5000` via web browser.
2. Unggah gambar atau seret-jatuhkan di area _Dropzone_.
3. Buka menu kategori alat di *Sidebar Kiri*.
4. Geser _slider_ pada tiap alat yang ingin diterapkan.
5. Klik tanda Centang (*Apply*) untuk mengimplementasikan filter secara bertumpuk, atau gunakan tombol _Reset_ di menu atas untuk memulai ulang.
6. Simpan gambar dari menu **File -> Save As**.
