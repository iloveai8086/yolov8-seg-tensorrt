//
// Created by ros on 12/8/22.
//

#ifndef PRO_YOLOSEG_H
#define PRO_YOLOSEG_H

#include <string>
#include <future>
#include <memory>
#include <common/matrix.hpp>
#include <opencv2/opencv.hpp>

static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int segWidth = 160;
static const int segHeight = 160;
static const int segChannels = 32;
static const int CLASSES = 80;
static const int Box_col = 116;
static const int Num_box = 8400;
static const int OUTPUT_SIZE = Num_box * (CLASSES+5 + segChannels);  // det output
static const int OUTPUT_SIZE1 = segChannels * segWidth * segHeight ;//seg output

static const float CONF_THRESHOLD = 0.25;
static const float NMS_THRESHOLD = 0.5;
static const float MASK_THRESHOLD = 0.5;



namespace Yolov8Seg
{
    struct detBox{
        float left, top, right, bottom, confidence;
        int class_label;
        cv::Rect box;
        cv::Mat boxMask;
        Matrix mask_cofs;
        detBox() = default;
        detBox(float left, float top, float right, float bottom, float confidence, int class_label, Matrix mask_cofs, cv::Rect box)
                : left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label), mask_cofs(mask_cofs), box(box){}
    };
    void yolov8Seg_inference();
    void DrawPred(cv::Mat& img,std::vector<detBox> result);

};


#endif //PRO_YOLOSEG_H
