#include<iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "NvInferVersion.h"
#include <opencv2/opencv.hpp>
#include "spdlog/spdlog.h"

using namespace nvonnxparser;
using namespace nvinfer1;

#ifndef DETECTOR_H
#define DETECTOR_H

struct BBoxInfo{
    cv::Rect bbox;
    int label;
    float prob;
};

class SpdlogLogger : public nvinfer1::ILogger {
public:
    // Implement the log method
    void log(Severity severity, const char* msg) noexcept override {
        // Convert TensorRT severity to spdlog level
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                spdlog::error(msg);
                break;
            case Severity::kWARNING:
                spdlog::warn(msg);
                break;
            case Severity::kINFO:
                spdlog::info(msg);
                break;
            case Severity::kVERBOSE:
                spdlog::debug(msg);
                break;
            default:
                spdlog::info(msg);
                break;
        }
    }
};


class Detector{
    public:
        std::string onnxPath;
        std::string enginePath;
        int batchSize;
        int modelHWSize;
        int numberOfClasses;
        std::string inputBlobName;
        std::string outputBlobName;
        std::string precisionType;
        nvinfer1::IHostMemory* hostEngine = nullptr;
        nvinfer1::ICudaEngine* cudaEngine = nullptr;
        nvinfer1::IExecutionContext *nvContext;
        float confThreshold = 0.4;

        int imageHeight;
        int imageWidth;
        int imageChannels;

        std::string outputNumDetBlobName, outputDetBoxesBlobName, outputDetScoresBlobName, outputDetClassesBlobName;

    public:
        virtual void createEngine() = 0;
        virtual void loadEngine() = 0;
        virtual std::vector<BBoxInfo> postprocess(float* hostBuffersNMS[4]) = 0;
};

#endif // DETECTOR_H
