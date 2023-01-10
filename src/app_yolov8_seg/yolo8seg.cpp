//
// Created by ros on 12/8/22.
//
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include "yolo8seg.h"

namespace Yolov8Seg
{
    using namespace cv;
    using namespace std;
    void DrawPred(cv::Mat& img,std::vector<detBox> result) {
        std::vector<cv::Scalar> color;
        srand(time(0));
        for (int i = 0; i < CLASSES; i++)
        {
            int b = rand() % 256;
            int g = rand() % 256;
            int r = rand() % 256;
            color.push_back(cv::Scalar(b, g, r));
        }
        cv::Mat mask = img.clone();
        int count = 0;
        for (int i = 0; i < result.size(); i++) {
            count++;
            int left, top;
            left = result[i].box.x;
            top = result[i].box.y;
            int color_num = i;
            // color[result[i].class_label] -> color[count]
            rectangle(img, result[i].box,color[count], 2, 8);
            cv::Mat c = mask(result[i].box);

            cv::Mat a = result[i].boxMask;

            c.setTo(color[count], a);
            std::string label = std::to_string(result[i].class_label) + ":" + std::to_string(result[i].confidence);
            int baseLine;
            cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            top = std::max(top, labelSize.height);
            putText(img, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color[count], 2);
        }
        addWeighted(img, 0.5, mask, 0.5, 0, img);

    };

    void yolov8Seg_inference() {
        // 加载模型
        printf("load engine  \n");
        auto engine = TRT::load_infer("yolov8s-seg-sim.trtmodel");
        if(!engine){
            printf("load engine failed \n");
            return;
        }
        auto input       = engine->input();
        auto seg_out      = engine->output(0);
        auto det_out      = engine->output(1);

        int input_width  = input->width();
        int input_height = input->height();
        auto image = imread("street.jpg");
        auto img_o = image.clone();
        int img_w = image.cols;
        int img_h = image.rows;
        float scale_w = img_w / static_cast<float>(input_width); // 除法是真的坑
        float scale_h = img_h / static_cast<float>(input_height);
        Mat input_image;
        resize(image,input_image,Size(input_width,input_height));  // 直接resize 没有warpaffine
        Mat show_img = image.clone();
        input_image.convertTo(input_image, CV_32F);
        // 预处理
        Mat channel_based[3];
        for(int i = 0; i < 3; ++i)
            channel_based[i] = Mat(input_height, input_width, CV_32F, input->cpu<float>(0, 2-i));

        split(input_image, channel_based);
        for(int i = 0; i < 3; ++i)
            channel_based[i] = (channel_based[i] / 255.0f);

        engine->forward(true);

        // 检测结果
        float *det_res = det_out->cpu<float>();
        vector<detBox> boxes;
        for(int i = 0; i < Num_box; ++i){
            float* pitem = det_res + i * Box_col;
            // float objness = pitem[4];
            // if(objness < CONF_THRESHOLD)
            //     continue;

//            float* class_confidence = pitem + 4;
//            int label     = std::max_element(class_confidence, class_confidence + CLASSES) - class_confidence;
            float* class_confidence = pitem + 4;
            float confidence        = *class_confidence++;
            int label               = 0;
            for(int i = 1; i < CLASSES; ++i, ++class_confidence){
                if(*class_confidence > confidence){
                    confidence = *class_confidence;
                    label      = i;
                }
            }

            // float prob    = pclass[label];
            // float confidence = prob * objness;
            if(confidence < CONF_THRESHOLD)
                continue;

            float cx     = pitem[0] * scale_w;
            float cy     = pitem[1] * scale_h;
            float width  = pitem[2] * scale_w;
            float height = pitem[3] * scale_h;

            float left   = (cx - width * 0.5);
            float top    = (cy - height * 0.5);
            float right  = (cx + width * 0.5);
            float bottom = (cy + height * 0.5);
            Rect rect(left,top,width,height);
            // 每个box的mask系数
            vector<float> temp_proto(pitem + 4 + CLASSES, pitem + 4 + CLASSES + segChannels);
            Matrix tmp_cof(1, segChannels, temp_proto);

            boxes.emplace_back(left, top, right, bottom, confidence, (float)label,tmp_cof,rect);

        }

        // NMS
        std::sort(boxes.begin(), boxes.end(), [](detBox &a, detBox &b)
        { return a.confidence > b.confidence; });
        std::vector<bool> remove_flags(boxes.size());
        std::vector<detBox> box_result;
        box_result.reserve(boxes.size());

        auto iou = [](const detBox& a, const detBox& b){
            float cross_left   = std::max(a.left, b.left);
            float cross_top    = std::max(a.top, b.top);
            float cross_right  = std::min(a.right, b.right);
            float cross_bottom = std::min(a.bottom, b.bottom);

            float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
            float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top)
                               + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
            if(cross_area == 0 || union_area == 0) return 0.0f;
            return cross_area / union_area;
        };

        // nms
        for(int i = 0; i < boxes.size(); ++i){
            if(remove_flags[i]) continue;

            auto& ibox = boxes[i];
            box_result.emplace_back(ibox);
            for(int j = i + 1; j < boxes.size(); ++j){
                if(remove_flags[j]) continue;

                auto& jbox = boxes[j];
                if(ibox.class_label == jbox.class_label){
                    // class matched
                    if(iou(ibox, jbox) >= NMS_THRESHOLD)
                        remove_flags[j] = true;
                }
            }
        }

        // 原型mask 32 * 160 * 160
        float *seg_det = seg_out->cpu<float>();
        vector<float> mask(seg_det, seg_det + segChannels * segWidth * segHeight);
        // 矩阵表示
        Matrix seg_proto(segChannels, segWidth * segHeight, mask);
        for (int i = 0; i < box_result.size(); ++i) {
            // 可以将所有的mask系数放在一起，然后利用cuda或者其他库进行加速计算
            // 每个目标框的mask系数 乘以原型mask 并取sigmod
            auto start = std::chrono::system_clock::now();

//        Matrix resSeg = (mygemm(box_result[i].mask_cofs,seg_proto).exp(-1) + 1.0).power(-1);
            Matrix resSeg = (gemm(box_result[i].mask_cofs, false,seg_proto, false,1.0f,0.0f).exp(-1)+1.0).power(-1);
            auto end = std::chrono::system_clock::now();
            std::cout << "gemm time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                      << "us" << std::endl;
            Mat mask;
            Mat resMat(resSeg.data_);
            resMat = resMat.reshape(0,{segHeight,segWidth});
            // 如果图片预处理为直接resize,那么计算出来的resMat可以直接缩放回原图，
            // 如果是填充黑边的resize，可以参考原代码将原型mask恢复到原图大小
            // resize(resMat, resMat, Size(INPUT_H,INPUT_W), INTER_NEAREST);
            resize(resMat, resMat, Size(show_img.cols,show_img.rows), INTER_NEAREST);  // opencv这个行列真的坑，反了，什么row> col<
            // 获取原型mask里面目标框的区域
            Rect temp_rect = box_result[i].box;
            // 将目标框区域 大于0.5的值变为255
            cv::Mat binaryMat;
            inRange(resMat(temp_rect), 0.5, 1, binaryMat);  // 判断置信度
            box_result[i].boxMask = binaryMat;
            // cv::imwrite(to_string(i) + "_.jpg", b);
        }
        // 渲染
        DrawPred(show_img, box_result);
        cv::imwrite("output-seg.jpg", show_img);
        std::cout << "infer done！ " << std::endl;
    }
}

