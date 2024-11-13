#include<iostream>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include "NvInferVersion.h"
#include "detector.h"

class YoloV7 : public Detector{
    public:
        void createEngine();
        void loadEngine();
        std::vector<BBoxInfo> postprocess(float* hostBuffersNMS[4]);
};