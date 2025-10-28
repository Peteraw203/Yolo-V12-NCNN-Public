import cv2
from ultralytics import YOLO
import time  # <-- TAMBAHKAN INI

# --- Konfigurasi ---

# Muat model YOLO 'best.pt' Anda
# Asumsi 'YOLO 12' adalah model yang Anda train dengan framework Ultralytics (spt YOLOv8, v9, dll)
model = YOLO('best.pt')

# Daftar nama kelas (HARUS SAMA PERSIS urutannya dengan saat training)
# Anda memberikan 6 kelas:
class_names = [
    "aspal",
    "keramik",
    "lubang",
    "paving",
    "uneven",
    "tangga"
]

# ID Webcam. 0 biasanya webcam internal.
# Ganti ke 1, 2, ... jika Anda punya banyak webcam
# Ganti ke "path/to/video.mp4" jika ingin memproses file video
WEBCAM_ID = 0

# --- Selesai Konfigurasi ---


# Inisialisasi webcam
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print(f"Error: Tidak bisa membuka webcam ID {WEBCAM_ID}.")
    exit()

# Atur resolusi frame (opsional, bisa mempercepat proses)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# --- Variabel untuk FPS ---
prev_frame_time = 0  # <-- TAMBAHKAN INI
new_frame_time = 0   # <-- TAMBAHKAN INI
# ---------------------------

print("Menjalankan deteksi real-time... Tekan 'q' untuk keluar.")

# Looping utama untuk membaca frame
while True:
    # Baca satu frame dari webcam
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame. Mungkin webcam terputus.")
        break

    # 1. Lakukan Deteksi dengan Model YOLO
    # stream=True direkomendasikan untuk video/real-time agar lebih efisien
    # verbose=False agar tidak menampilkan log di console
    results = model(frame, stream=True,
                     verbose=False,
                     iou=0.45)

    # 2. Loop melalui hasil deteksi
    for r in results:
        boxes = r.boxes  # Dapatkan objek 'boxes' dari hasil

        for box in boxes:
            # --- Dapatkan Data Deteksi ---
            
            # Koordinat Bounding Box (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Konversi ke integer

            # Confidence Score (skor kepercayaan)
            confidence = float(box.conf[0])
            
            # Class ID (angka/indeks kelas)
            cls_id = int(box.cls[0])

            # --- Gambar di Frame ---
            
            try:
                # Ambil nama kelas dari daftar berdasarkan ID
                class_name = class_names[cls_id]
            except IndexError:
                # Jika ID kelas di luar jangkauan (error di daftar class_names)
                class_name = f"Kelas {cls_id}"

            # Buat label teks (misal: "aspal: 85%")
            label = f"{class_name}: {confidence*100:.0f}%"

            # Tentukan warna (Anda bisa buat lebih canggih)
            color = (255, 0, 255) # Ungu (BGR)

            # 3. Gambar Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 4. Gambar Background Label
            # Hitung ukuran teks untuk membuat background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1) # -1 = filled

            # 5. Tulis Teks Label
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # Teks Putih

    # --- Menghitung dan Menampilkan FPS ---
    new_frame_time = time.time()  # Ambil waktu saat ini setelah pemrosesan frame
    
    # Hitung FPS
    # Hindari pembagian dengan nol di frame pertama
    if (new_frame_time - prev_frame_time) > 0:
        fps = 1 / (new_frame_time - prev_frame_time)
    else:
        fps = 0
        
    prev_frame_time = new_frame_time  # Simpan waktu frame ini untuk iterasi berikutnya

    # Ubah FPS menjadi string, format 2 angka desimal
    fps_text = f"FPS: {fps:.2f}"

    # Gambar teks FPS di frame (pojok kiri atas)
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # ------------------------------------

    # Tampilkan frame yang sudah digambar
    cv2.imshow("YOLO Real-Time Detection (Tekan 'q' untuk keluar)", frame)

    # Cek jika tombol 'q' ditekan untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Selesai: Rilis webcam dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
print("Deteksi dihentikan.")