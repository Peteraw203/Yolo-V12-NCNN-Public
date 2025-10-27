#include "yolo.h"
#include "benchmark.h" // tambah header untuk mengkhitung waktu inferensi

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

//header ini untuk std::sort
#include <algorithm>



static inline float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

// Fungsi qsort_descent_inplace (2 fungsi) dihapus.
// diganti dengan std::sort di dalam Yolo::detect karena lebih cepat.


static void nms_sorted_bboxes(const std::vector<Object> &faceobjects, std::vector<int> &picked,
                              float nms_threshold, bool agnostic = false) {
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++) {
        const Object &a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label)
                continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float clampf(float d, float min, float max) {
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

static void parse_yolo_detections(
        float *inputs, float confidence_threshold,
        int num_channels, int num_anchors, int num_labels,
        int infer_img_width, int infer_img_height,
        std::vector<Object> &objects, bool is_v5) {
    std::vector<Object> detections;
    cv::Mat output = cv::Mat((int) num_channels, (int) num_anchors, CV_32F, inputs);
    if (!is_v5) {
        output = output.t();
    }
    int classes_index = 4;
    if (is_v5) {
        classes_index = 5;
    }

    for (int i = 0; i < output.rows; i++) {
        const float *row_ptr = output.row(i).ptr<float>();
        const float *bboxes_ptr = row_ptr;
        const float *classes_ptr = row_ptr + classes_index;
        const float *max_s_ptr = std::max_element(classes_ptr, classes_ptr + num_labels);
        float score = *max_s_ptr;
        if (is_v5) score = score * (*(row_ptr + 4));
        if (score > confidence_threshold) {
            float x = *bboxes_ptr++;
            float y = *bboxes_ptr++;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clampf((x - 0.5f * w), 0.f, (float) infer_img_width);
            float y0 = clampf((y - 0.5f * h), 0.f, (float) infer_img_height);
            float x1 = clampf((x + 0.5f * w), 0.f, (float) infer_img_width);
            float y1 = clampf((y + 0.5f * h), 0.f, (float) infer_img_height);

            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;
            Object object;
            object.label = max_s_ptr - classes_ptr;
            object.prob = score;
            object.rect = bbox;
            detections.push_back(object);
        }
    }
    objects = detections;
}


Yolo::Yolo() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    // 2. inisiasi variable timer
    this->last_inference_time = 0;
}


int Yolo::load(AAssetManager *mgr, const char *modeltype, int _target_size, const float *_mean_vals,
               const float *_norm_vals, bool use_gpu) {
    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN

    // GPU menggunakan vulkan
    if (use_gpu)
    {
        yolo.opt.use_vulkan_compute = true;
    }
#endif
    //Edited Optimized code
    // Hanya atur thread CPU jika kita tidak menggunakan GPU
    if (!use_gpu)
    {
        ncnn::set_cpu_powersave(2);
        ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
        yolo.opt.num_threads = ncnn::get_big_cpu_count();
    }

    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s.ncnn.param", modeltype);
    sprintf(modelpath, "%s.ncnn.bin", modeltype);

    yolo.load_param(mgr, parampath);
    yolo.load_model(mgr, modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}

int Yolo::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                 float nms_threshold) {
    if (rgb.empty()) {
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "image is empty"); // INI YANG DIPERBAIKI
        return false;
    }
    objects.clear();

    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale;
    if (w > h) {
        scale = (float) target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float) target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width, height,
                                                 w, h);

    // pad to target_size rectangle
    int wpad = (target_size + 31) / 32 * 32 - w;
    int hpad = (target_size + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                           ncnn::BORDER_CONSTANT, 114.f);
    in_pad.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("in0", in_pad);
    ncnn::Mat out;

    // 3. Menambah timer sebelum ex.extract
    double start_time = ncnn::get_current_time();

    ex.extract("out0", out); // Ini adalah proses inferensi murni

    double end_time = ncnn::get_current_time();
    this->last_inference_time = end_time - start_time;


    bool is_v5 = false;
    int num_labels = out.h - 4;
    if (is_v5) {
        // v5
        num_labels = out.w - 5;
    }
    std::vector<Object> proposals;
    parse_yolo_detections(
            (float *) out.data, prob_threshold,
            out.h, out.w, num_labels,
            in_pad.w, in_pad.h,
            proposals, is_v5);

    // sort all proposals by score from highest to lowest
    // OPTIMASI
    // Mengganti fungsi qsort rekursif (qsort_descent_inplace) dengan std::sort.
    // std::sort (bawaan C++) jauh lebih cepat dan efisien untuk pengurutan.
    std::sort(proposals.begin(), proposals.end(), [](const Object& a, const Object& b) {
        return a.prob > b.prob;
    });

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();
    if (count == 0) {
        return true;
    }
    objects.resize(count);
    for (int i = 0; i < count; i++) {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // sort objects by area
//    struct {
//        bool operator()(const Object &a, const Object &b) const {
//            return a.rect.area() > b.rect.area();
//        }
//    } objects_area_greater;
//    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}

int Yolo::draw(cv::Mat &rgb, const std::vector<Object> &objects) {

//  int color_index = 0;

    for (const auto &obj: objects) {

//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char *color = colors[obj.label % 19];
//     color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                    cv::Size(label_size.width, label_size.height + baseLine)), cc,
                      -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0)
                                                                    : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    textcc, 1);
    }

    //4. Menampilkan waktu inferensi

    char text_inf[256];
    sprintf(text_inf, "Inference: %.2f ms", this->last_inference_time);

    // Tentukan posisi (kiri atas, di bawah FPS/ms total)
    int x_inf = 10;
    int y_inf = 50; // Y-koordinat untuk baseline teks (di bawah FPS/ms total)

    cv::Scalar text_color(255, 255, 255); // Warna putih
    int baseLine_inf = 0;
    cv::Size label_size_inf = cv::getTextSize(text_inf, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine_inf);

    // Buat background hitam tipis
    cv::rectangle(rgb, cv::Rect(cv::Point(x_inf, y_inf - label_size_inf.height - baseLine_inf), // Titik kiri atas rect
                                cv::Size(label_size_inf.width, label_size_inf.height + baseLine_inf + 5)), // Lebar dan tinggi rect
                  cv::Scalar(0, 0, 0), -1);

    // Tulis teksnya
    cv::putText(rgb, text_inf, cv::Point(x_inf, y_inf), // Titik kiri-baseline teks
                cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1);


    return 0;
}
