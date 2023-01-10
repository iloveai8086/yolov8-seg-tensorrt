
// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include <app_yolov8_seg/yolo8seg.h>

using namespace std;
using namespace cv;

static const char *cocolabels[] = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
        "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
};

static bool exists(const string &path) {

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}


static bool build_yolov8_seg_model() {

    if (exists("yolov8s-seg-sim.trtmodel")) {
        printf("yolov8s-seg-sim.trtmodel has exists.\n");
        return true;
    }

    //SimpleLogger::set_log_level(SimpleLogger::LogLevel::Verbose);
    // iLogger::set_log_level(iLogger::LogLevel::Info);
    TRT::compile(
            TRT::Mode::FP16,
            1,
            "yolov8s-seg-sim.onnx",
            "yolov8s-seg-sim.trtmodel"
    );
    INFO("Done.");
    return true;
}



int main() {

    // 新的实现
//    if (!build_model()) {
//        return -1;
//    }
//    if (!build_seg_model()) {
//        return -1;
//    }
    if (!build_yolov8_seg_model()){
        return -1;
    }

    Yolov8Seg::yolov8Seg_inference();
    return 0;
}
