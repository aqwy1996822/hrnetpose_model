//
// Created by leo on 2022/1/7.
//

#ifndef HRNET_HRNETPOSE_MODEL_H
#define HRNET_HRNETPOSE_MODEL_H

#include <iostream>
#include <chrono>
#include <cmath>
#include "hrnetpose/cuda_utils.h"
#include "hrnetpose/logging.h"
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <NvOnnxParser.h>
#include "NvInfer.h"
#include "calibrator.h"

using namespace nvinfer1;

class Hrnetpose_model
{
private:
#define DEVICE 0  // GPU id
#define CONF_THRESH 0.5

    const std::vector<cv::Scalar> colors={cv::Scalar(139,0,0),cv::Scalar(255,0,0),cv::Scalar(255,69,0),cv::Scalar(255,127,0),cv::Scalar(255,165,0),cv::Scalar(152,251,152),cv::Scalar(124,252,0),cv::Scalar(102,205,170),cv::Scalar(0,255,255),cv::Scalar(0,191,255),cv::Scalar(30,144,255),cv::Scalar(0,0,255),cv::Scalar(132,112,255),cv::Scalar(123,104,238),cv::Scalar(106,90,205)};
    const std::vector<std::vector<int>> PART_LINE={{0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, {17, 11}, {17, 12},{11, 13}, {12, 14}, {13, 15}, {14, 16}};
    static const int PART_NUM=17;
    static const int OUTPUT_H=64;
    static const int OUTPUT_W=48;

    static const int INPUT_H = 256;
    static const int INPUT_W = 192;
    static const int OUTPUT_SIZE = PART_NUM*OUTPUT_H*OUTPUT_W;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
    const int MAXBATCHSIZE = 4;

    Logger gLogger;
    IRuntime *runtime;
    ICudaEngine *engine;
    IExecutionContext *context;

    float *buffers[2];

    float * input=new float[4 * 3 * INPUT_H * INPUT_W];
    float * output=new float[4 * OUTPUT_SIZE];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex;
    int outputIndex;

    // Create stream
    cudaStream_t stream;
    //控制解析时是否需要释放内存
    bool predict_flag = false;
public:
    bool int8_flag= false;

    Hrnetpose_model();
    ~Hrnetpose_model();
    void serialize_engine();
    void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);
    void get_realpoint(cv::Rect rect, std::vector<cv::Point> &hint_poses);
    // letterbox; BGR to RGB
    cv::Mat preprocess_img(cv::Mat img, int input_w, int input_h);
    void deserialize_engine(std::string engine_name);
    void build_engine();
    void runtest2();
    void run_frame();
};



#endif //HRNET_HRNETPOSE_MODEL_H
