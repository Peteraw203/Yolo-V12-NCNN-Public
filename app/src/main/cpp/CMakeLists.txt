cmake_minimum_required(VERSION 3.22.1)

project("yolov12")

set(OpenCV_DIR "C:/Users/peter/Desktop/Tugas/Tugas kuliah/ComVis/opencv-4.12.0-android-sdk/OpenCV-android-sdk/sdk/native/jni")
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR "C:/Users/peter/Desktop/Tugas/Tugas kuliah/ComVis/ncnn-20250503-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

add_library(${CMAKE_PROJECT_NAME} SHARED
		ndkcamera.cpp ndkcamera.h
		yolo.cpp yolo.h
		native-lib.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
		ncnn ${OpenCV_LIBS} camera2ndk mediandk
		android
		log)