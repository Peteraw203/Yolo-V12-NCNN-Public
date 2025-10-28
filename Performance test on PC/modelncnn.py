import cv2
import ncnn
import numpy as np
import time
import os


class Object:
    """ Meniru struct Object dari C++ """
    def __init__(self, rect=None, label=0, prob=0.0):
        self.rect = rect  # rect akan menjadi [x, y, w, h]
        self.label = label
        self.prob = prob

def intersection_area(a: Object, b: Object):
    """ Meniru static inline float intersection_area """
    x1 = max(a.rect[0], b.rect[0])
    y1 = max(a.rect[1], b.rect[1])
    x2 = min(a.rect[0] + a.rect[2], b.rect[0] + b.rect[2])
    y2 = min(a.rect[1] + a.rect[3], b.rect[1] + b.rect[3])
    
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return w * h

def nms_sorted_bboxes(faceobjects: list, picked: list, nms_threshold: float, agnostic: bool = False):
    """ Meniru static void nms_sorted_bboxes """
    picked.clear()
    n = len(faceobjects)
    areas = [obj.rect[2] * obj.rect[3] for obj in faceobjects]

    for i in range(n):
        a = faceobjects[i]
        keep = 1
        for j in range(len(picked)):
            b = faceobjects[picked[j]]

            if not agnostic and a.label != b.label:
                continue

            # intersection over union
            inter_area = intersection_area(a, b)
            union_area = areas[i] + areas[picked[j]] - inter_area
            
            if union_area == 0:
                continue

            if inter_area / union_area > nms_threshold:
                keep = 0
        
        if keep:
            picked.append(i)

def parse_yolo_detections(inputs: np.ndarray, confidence_threshold: float,
                          num_channels: int, num_anchors: int, num_labels: int,
                          infer_img_width: int, infer_img_height: int,
                          is_v5: bool = False) -> list:
    """ 
    Meniru static void parse_yolo_detections
    PERHATIKAN: Logika C++ Anda secara eksplisit mengatur is_v5 = false
    """
    detections = []
    
    # C++: cv::Mat output = cv::Mat((int) num_channels, (int) num_anchors, CV_32F, inputs);
    # Di Python, 'inputs' sudah menjadi array numpy
    output = inputs.reshape((num_channels, num_anchors))

    # C++: if (!is_v5) { output = output.t(); }
    if not is_v5:
        output = output.T  # Transpose

    # C++: int classes_index = 4; if (is_v5) { classes_index = 5; }
    classes_index = 5 if is_v5 else 4

    num_rows = output.shape[0]

    for i in range(num_rows):
        row = output[i, :]
        bboxes = row[:classes_index]
        classes = row[classes_index:]
        
        # C++: const float *max_s_ptr = std::max_element(...)
        max_s_ptr_index = np.argmax(classes)
        score = classes[max_s_ptr_index]

        # C++: if (is_v5) score = score * (*(row_ptr + 4));
        if is_v5:
            score = score * row[4] # row[4] adalah objectness

        if score > confidence_threshold:
            # C++: float x = *bboxes_ptr++; ...
            x, y, w, h = bboxes[0], bboxes[1], bboxes[2], bboxes[3]

            # C++: float x0 = clampf((x - 0.5f * w), 0.f, (float) infer_img_width);
            x0 = np.clip((x - 0.5 * w), 0.0, float(infer_img_width))
            y0 = np.clip((y - 0.5 * h), 0.0, float(infer_img_height))
            x1 = np.clip((x + 0.5 * w), 0.0, float(infer_img_width))
            y1 = np.clip((y + 0.5 * h), 0.0, float(infer_img_height))

            obj = Object()
            obj.label = max_s_ptr_index
            obj.prob = score
            obj.rect = [x0, y0, x1 - x0, y1 - y0] # [x, y, w, h]
            detections.append(obj)
            
    return detections


class Yolo:
    """ Meniru class Yolo dari C++ """
    def __init__(self):
        self.yolo = ncnn.Net()
        self.target_size = 0
        self.mean_vals = []
        self.norm_vals = []
        self.last_inference_time = 0
        self.class_names = []
        self.colors = []

    def load(self, modeltype: str, _target_size: int, _mean_vals: list, 
             _norm_vals: list, class_names: list, colors: list, use_gpu: bool = False):
        
        self.yolo.clear()
        
        # C++: yolo.opt = ncnn::Option();
        self.yolo.opt = ncnn.Option()

        # C++: yolo.opt.use_vulkan_compute = true;
        self.yolo.opt.use_vulkan_compute = use_gpu
        
        # C++: ncnn::set_cpu_powersave(2); ...
        if not use_gpu:
            # Mengatur thread di Python/ncnn bisa jadi rumit, 
            # kita serahkan pada default ncnn untuk kesederhanaan di laptop
            # Jika perlu, Anda bisa uncomment ini:
            # self.yolo.opt.num_threads = os.cpu_count()
            pass 

        # C++: sprintf(parampath, "%s.ncnn.param", modeltype);
        parampath = f"{modeltype}.ncnn.param"
        modelpath = f"{modeltype}.ncnn.bin"
        
        if not os.path.exists(parampath) or not os.path.exists(modelpath):
            print(f"Error: Model file tidak ditemukan di {parampath} atau {modelpath}")
            return -1

        self.yolo.load_param(parampath)
        self.yolo.load_model(modelpath)

        self.target_size = _target_size
        self.mean_vals = _mean_vals
        self.norm_vals = _norm_vals
        self.class_names = class_names
        self.colors = colors
        
        print(f"Model {modeltype} loaded. Target size: {self.target_size}, GPU: {use_gpu}")
        return 0

    def detect(self, rgb: np.ndarray, prob_threshold: float, nms_threshold: float) -> list:
        
        objects = []
        
        # C++: int width = rgb.cols; int height = rgb.rows;
        height, width, _ = rgb.shape

        # === 1. Pre-processing (Meniru C++ Letterbox) ===
        
        # C++: int w = width; int h = height; float scale; ...
        w = width
        h = height
        if w > h:
            scale = float(self.target_size) / w
            w = self.target_size
            h = int(h * scale)
        else:
            scale = float(self.target_size) / h
            h = self.target_size
            w = int(w * scale)

        # C++: ncnn::Mat in = ncnn::Mat::from_pixels_resize(...)
        # Di Python, kita resize dulu dengan OpenCV, lalu konversi
        resized_img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # C++: int wpad = (target_size + 31) / 32 * 32 - w;
        # Logika C++ Anda menyiratkan target_size adalah kelipatan 32
        # (320 + 31) / 32 * 32 = (351 // 32) * 32 = 10 * 32 = 320
        # Jadi ini sama dengan:
        padded_size = self.target_size 
        wpad = padded_size - w
        hpad = padded_size - h
        
        top = hpad // 2
        bottom = hpad - (hpad // 2)
        left = wpad // 2
        right = wpad - (wpad // 2)

        # C++: ncnn::copy_make_border(..., 114.f);
        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=[114, 114, 114])
        
        # C++: ncnn::Mat::from_pixels(..., ncnn::Mat::PIXEL_RGB, ...)
        # OpenCV adalah BGR, C++ Anda adalah RGB. Kita harus konversi.
        # Kita asumsikan 'rgb' adalah BGR dari cv2.VideoCapture
        
        rgb_padded_img = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
        
        # C++: in_pad = ncnn::Mat::from_pixels(...)
        # C++: in_pad.substract_mean_normalize(mean_vals, norm_vals);
        # PyNCNN bisa melakukan keduanya sekaligus
        mat_in = ncnn.Mat.from_pixels(rgb_padded_img, ncnn.Mat.PixelType.PIXEL_RGB, 
                                      padded_size, padded_size)
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)

        # === 2. Inferensi ===
        
        # C++: ncnn::Extractor ex = yolo.create_extractor();
        ex = self.yolo.create_extractor()
        ex.input("in0", mat_in)
        
        # C++: double start_time = ncnn::get_current_time();
        start_time = time.time()
        
        # C++: ex.extract("out0", out);
        ret, out_mat = ex.extract("out0")
        
        # C++: this->last_inference_time = end_time - start_time;
        end_time = time.time()
        self.last_inference_time = (end_time - start_time) * 1000 # Konversi ke milidetik

        # === 3. Post-processing ===
        
        # C++: bool is_v5 = false;
        is_v5 = False # HARUS SAMA DENGAN C++
        
        # C++: int num_labels = out.h - 4; ...
        if is_v5:
            num_labels = out_mat.w - 5
        else:
            num_labels = out_mat.h - 4

        # Konversi ncnn.Mat ke numpy array
        # out_mat.h = num_channels, out_mat.w = num_anchors
        out_array = np.array(out_mat).reshape((out_mat.h, out_mat.w))
        
        # C++: parse_yolo_detections(...)
        proposals = parse_yolo_detections(
            out_array, prob_threshold,
            out_mat.h, out_mat.w, num_labels,
            padded_size, padded_size,
            is_v5=is_v5
        )

        # C++: std::sort(proposals.begin(), ...);
        proposals.sort(key=lambda x: x.prob, reverse=True)

        # C++: nms_sorted_bboxes(proposals, picked, nms_threshold);
        picked = []
        nms_sorted_bboxes(proposals, picked, nms_threshold)

        count = len(picked)
        if count == 0:
            return []

        objects = [proposals[i] for i in picked]

        # === 4. Rescale Coordinates (Meniru C++) ===
        final_objects = []
        for obj in objects:
            # C++: float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
            x0 = (obj.rect[0] - left) / scale
            y0 = (obj.rect[1] - top) / scale
            x1 = (obj.rect[0] + obj.rect[2] - left) / scale
            y1 = (obj.rect[1] + obj.rect[3] - top) / scale

            # C++: clip
            x0 = np.clip(x0, 0.0, float(width - 1))
            y0 = np.clip(y0, 0.0, float(height - 1))
            x1 = np.clip(x1, 0.0, float(width - 1))
            y1 = np.clip(y1, 0.0, float(height - 1))
            
            final_obj = Object()
            final_obj.label = obj.label
            final_obj.prob = obj.prob
            final_obj.rect = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
            final_objects.append(final_obj)

        return final_objects


    def draw(self, bgr: np.ndarray, objects: list):
        """ Meniru int Yolo::draw """
        
        for obj in objects:
            try:
                color = self.colors[obj.label % len(self.colors)]
                class_name = self.class_names[obj.label]
            except (IndexError, TypeError):
                # Tangani jika obj.label di luar jangkauan
                color = (128, 128, 128) # Abu-abu
                class_name = f"Kelas {obj.label}"
                pass

            # C++: cv::Scalar cc(color[0], color[1], color[2]);
            # Catatan: Warna C++ mungkin RGB, OpenCV Python adalah BGR
            # Kita asumsikan 'colors' didefinisikan sebagai BGR
            cc = color 

            # C++: cv::rectangle(rgb, obj.rect, cc, 2);
            x, y, w, h = obj.rect
            cv2.rectangle(bgr, (x, y), (x + w, y + h), cc, 2)

            # C++: sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
            text = f"{class_name} {obj.prob * 100:.1f}%"

            # C++: cv::getTextSize(...)
            (label_width, label_height), baseLine = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # C++: int x = obj.rect.x; int y = obj.rect.y - ...
            tx = x
            ty = y - label_height - baseLine
            if ty < 0:
                ty = 0
            if tx + label_width > bgr.shape[1]:
                tx = bgr.shape[1] - label_width
            
            # C++: cv::rectangle(..., -1);
            cv2.rectangle(bgr, (tx, ty), (tx + label_width, ty + label_height + baseLine), cc, -1)
            
            # C++: cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? ...
            text_color = (0, 0, 0) if (color[0] + color[1] + color[2] >= 381) else (255, 255, 255)
            
            # C++: cv::putText(...)
            cv2.putText(bgr, text, (tx, ty + label_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # 4. Menampilkan waktu inferensi (Meniru C++)
        text_inf = f"Inference: {self.last_inference_time:.2f} ms"
        
        # Tulis teksnya
        cv2.putText(bgr, text_inf, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Shadow
        cv2.putText(bgr, text_inf, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Text

        return bgr


# ==================================================================
# MAIN EXECUTION (Menjalankan di Laptop)
# ==================================================================

if __name__ == "__main__":

    # --- KONFIGURASI (HARUS SAMA DENGAN C++) ---
    
    # 1. Path ke model NCNN Anda (tanpa .ncnn.param atau .ncnn.bin)
    #    Pastikan 'yolov12n.ncnn.param' dan 'yolov12n.ncnn.bin' ada di folder yang sama
    MODEL_TYPE = "yolov12n"  

    # 2. Ukuran input
    TARGET_SIZE = 320 # Samakan dengan _target_size di C++

    # 3. Normalisasi (Samakan dengan C++)
    #    Model Ultralytics/YOLO biasanya: mean=[0,0,0], norm=[1/255, 1/255, 1/255]
    MEAN_VALS = [0.0, 0.0, 0.0]
    NORM_VALS = [1/255.0, 1/255.0, 1/255.0]
    
    # 4. Gunakan GPU? (Vulkan)
    USE_GPU = False # Ganti ke 'True' untuk GPU, 'False' untuk CPU

    # 5. Threshold (Samakan dengan C++)
    PROB_THRESHOLD = 0.25
    NMS_THRESHOLD = 0.45

    # 6. Daftar Kelas dan Warna (Samakan dengan C++)
    CLASS_NAMES = [
        "aspal", "keramik", "lubang", "paving", "uneven", "tangga"
    ]
    
    # Daftar 19 warna dari C++ (dalam format BGR untuk OpenCV Python)
    COLORS = [
        (54, 67, 244), (99, 30, 233), (176, 39, 156), (183, 58, 103),
        (181, 81, 63), (243, 150, 33), (244, 169, 3), (212, 188, 0),
        (136, 150, 0), (80, 175, 76), (74, 195, 139), (57, 220, 205),
        (59, 235, 255), (7, 193, 255), (0, 152, 255), (34, 87, 255),
        (72, 85, 121), (158, 158, 158), (139, 125, 96)
    ]
    # ---------------------------------------------

    # Inisialisasi Yolo
    yolo_net = Yolo()
    ret = yolo_net.load(
        modeltype=MODEL_TYPE,
        _target_size=TARGET_SIZE,
        _mean_vals=MEAN_VALS,
        _norm_vals=NORM_VALS,
        class_names=CLASS_NAMES,
        colors=COLORS,
        use_gpu=USE_GPU
    )

    if ret != 0:
        print("Gagal memuat model. Keluar.")
        exit()

    # Buka Webcam
    cap = cv2.VideoCapture(0) # 0 untuk webcam laptop
    if not cap.isOpened():
        print("Error: Tidak bisa membuka webcam.")
        exit()
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    # --- Variabel untuk FPS dan Waktu Total ---
    prev_loop_time = time.time()
    total_loop_ms = 0
    fps = 0
    # ----------------------------------------

    print("Menjalankan deteksi real-time... Tekan 'q' untuk keluar.")

    while True:
        success, frame = cap.read()
        if not success:
            print("Gagal membaca frame.")
            break
            
        # Deteksi objek
        # 'yolo_net.detect' secara internal sudah menghitung 'last_inference_time'
        objects = yolo_net.detect(frame, PROB_THRESHOLD, NMS_THRESHOLD)
        
        # Gambar hasil
        # 'yolo_net.draw' akan menggambar box DAN 'Inference: ... ms'
        drawn_frame = yolo_net.draw(frame, objects)
        
        # --- Hitung & Tampilkan FPS dan Total MS ---
        new_loop_time = time.time()
        delta_time = new_loop_time - prev_loop_time
        prev_loop_time = new_loop_time

        if delta_time > 0:
            fps = 1.0 / delta_time
            total_loop_ms = delta_time * 1000
        else:
            fps = 0.0
            total_loop_ms = 0.0

        # Siapkan teks
        fps_text = f"Total FPS: {fps:.1f}"
        total_ms_text = f"Total Komputasi: {total_loop_ms:.1f} ms"
        
        # Tampilkan Teks FPS (di atas teks inferensi)
        cv2.putText(drawn_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Shadow
        cv2.putText(drawn_frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) # Text (Hijau)

        # Tampilkan Teks Total Komputasi (di bawah teks inferensi)
        cv2.putText(drawn_frame, total_ms_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2) # Shadow
        cv2.putText(drawn_frame, total_ms_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # Text (Putih)
        # ---------------------------------------------

        # Tampilkan
        cv2.imshow("NCNN Python Laptop (Replika C++)", drawn_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()