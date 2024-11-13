#pragma once
#include<iostream>
#include "yolov7.h"
#include "detector.h"
#include <thread>
#include <chrono>
#include <crow.h>
#include <boost/filesystem.hpp>
#include<string>
#include <fstream>
#include<json.hpp>
#include "utils.h"
#include <cuda_runtime_api.h>

using json = nlohmann::json;

using namespace std;
using namespace cv;
using namespace ccustomutils;

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif  // CUDA_CHECK

const std::vector<std::string> CLASSNAMES = {
        "person",         "bicycle",    "car",           "motorcycle",    "airplane",     "bus",           "train",
        "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",    "parking meter", "bench",
        "bird",           "cat",        "dog",           "horse",         "sheep",        "cow",           "elephant",
        "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",     "handbag",       "tie",
        "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball",  "kite",          "baseball bat",
        "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",       "wine glass",    "cup",
        "fork",           "knife",      "spoon",         "bowl",          "banana",       "apple",         "sandwich",
        "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",        "donut",         "cake",
        "chair",          "couch",      "potted plant",  "bed",           "dining table", "toilet",        "tv",
        "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",   "microwave",     "oven",
        "toaster",        "sink",       "refrigerator",  "book",          "clock",        "vase",          "scissors",
        "teddy bear",     "hair drier", "toothbrush"
    };

//Buffer
static void* deviceBuffersNMS[5];
static float* hostBuffersNMS[4];
static std::map<std::string, std::string> mp;

struct FrameInfo{

    cv::Mat frame;
    std::string type;
    std::string id;
    std::vector<BBoxInfo> bbox;

    FrameInfo(cv::Mat f, std::string id){
        this->frame = f;
        this->id = id;
        this->type = "";
        this->bbox = {};
    }

    FrameInfo(cv::Mat f, std::string id, std::string t){
        this->frame = f;
        this->id = id;
        this->type = t;
        this->bbox = {};
    }

    FrameInfo(cv::Mat f, std::string id, std::string t, std::vector<BBoxInfo> bb){
        this->frame = f;
        this->id = id;
        this->type = t;
        this->bbox = bb;
    }

};

void assignConfig(YoloV7* &yolov7){
    yolov7->onnxPath = "/app/models/yolov7.onnx";
    yolov7->enginePath = "/app/models/yolov7.trt";
    yolov7->batchSize = 1;
    yolov7->numberOfClasses = 80;
    yolov7->modelHWSize = 640;
    yolov7->inputBlobName = "images";
    yolov7->precisionType = "FP32";
    yolov7->confThreshold = 0.4;

    yolov7->outputNumDetBlobName = "num_dets";
    yolov7->outputDetBoxesBlobName = "det_boxes";
    yolov7->outputDetScoresBlobName = "det_scores";
    yolov7->outputDetClassesBlobName = "det_classes";

}

class App{
    private:
        cudaStream_t cudaStream;

        float letterbox(const cv::Mat& image, cv::Mat& out_image, const cv::Size& new_shape = cv::Size(640, 640), int stride = 32, const cv::Scalar& color = cv::Scalar(114, 114, 114), bool fixed_shape = true, bool scale_up = true) {
            cv::Size shape = image.size();
        float r = std::min(
            (float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);
        if (!scale_up) {
            r = std::min(r, 1.0f);
        }

        int newUnpad[2]{
            (int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

        cv::Mat tmp;
        if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
            cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
        } else {
            tmp = image.clone();
        }

        float dw = new_shape.width - newUnpad[0];
        float dh = new_shape.height - newUnpad[1];

        if (!fixed_shape) {
            dw = (float)((int)dw % stride);
            dh = (float)((int)dh % stride);
        }

        dw /= 2.0f;
        dh /= 2.0f;

        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

        return 1.0f / r;
        }

        void events(){
            while(true){
                if(rules2events.size() > 0){
                    FrameInfo objs = rules2events.front();
                    rules2events.pop();

                    cv::Mat image = objs.frame;

                    if(objs.type == "curl"){
                        json j;
                        if(objs.bbox.size() > 0){
                            
                            j["detections"] = json();
                            j["imageinfo"] = json();

                            for(auto bbox : objs.bbox){
                                if(j["detections"].count(CLASSNAMES[bbox.label]) == 0){
                                    j["detections"][CLASSNAMES[bbox.label]] = json::array();
                                }

                                json box;
                                box["xmin"] = bbox.bbox.x;
                                box["ymin"] = bbox.bbox.y;
                                box["width"] = bbox.bbox.width;
                                box["height"] = bbox.bbox.height;
                                box["conf"] = bbox.prob;

                                j["detections"][CLASSNAMES[bbox.label]].push_back(box);

                                // cv::rectangle(image, bbox.bbox, Scalar(255, 0, 0), 2, LINE_8); 
                                // std::cout<<bbox.bbox.x<<", "<<bbox.bbox.y<<", "<<bbox.bbox.width<<", "<<bbox.bbox.height<<", "<<bbox.prob<<", "<<CLASSNAMES[bbox.label]<<std::endl;
                            }
                            std::cout<<"Detections:"<<j<<std::endl;
                            // cv::imwrite("/app/drawn.jpg",image);
                        }
                        std::cout<<"here unique id: "<<objs.id<<std::endl;

                        json info;
                        info["json_path"] = "/app/dumps/"+objs.id+".json";
                        info["image_path"] = "/app/dumps/"+objs.id+".jpg";
                        info["image_height"] = image.rows;
                        info["image_width"] = image.cols;
                        j["imageinfo"] = info;
                        mp[objs.id] = j.dump();
                    }
                    std::cout<<"Received inside events a frame objs size:"<<objs.bbox.size()<<std::endl;
                }
                // std::cout<<"inside loop startRulesThread ..."<<std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

        }
        void countUseCase(int val){
            while(true){
                if(inference2rules.size() > 0){
                    FrameInfo objs = inference2rules.front();
                    inference2rules.pop();

                    if(objs.bbox.size() > val){
                        rules2events.push(objs);
                    }
                    std::cout<<"Received inside countUseCase a frame objs size:"<<objs.bbox.size()<<" id:"<<objs.id<<std::endl;
                }
                // std::cout<<"inside loop startRulesThread ..."<<std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        void inference(){
            while(true){
                if(this->input2Inference.size() > 0){
                    FrameInfo frameInfo = input2Inference.front();
                    input2Inference.pop();

                    //Resize
                    cv::Mat frame = frameInfo.frame;

                    cv::Mat pr_img;
                    float scale = this->letterbox(frame, pr_img, {640, 640}, 32, {114, 114, 114}, true);
                    cv::cvtColor(pr_img, pr_img, cv::COLOR_BGR2RGB);
                    float* blob = blobFromImage(pr_img);

                    this->detector->imageHeight = frame.rows;
                    this->detector->imageWidth = frame.cols;
                    this->detector->imageChannels = frame.channels();

                    int size_image = this->detector->modelHWSize * this->detector->modelHWSize * 3 * sizeof(float);

                    float* buffer = (float*)deviceBuffersNMS[0];

                    CUDA_CHECK(cudaMemcpyAsync(buffer,blob,size_image,cudaMemcpyHostToDevice,cudaStream));
                    
                    cudaStreamSynchronize(cudaStream);

                    if (!detector->cudaEngine || !detector->nvContext) {
                        std::cerr << "Error: Engine or context creation failed." << std::endl;
                    }

                    detector->nvContext->enqueue(1, deviceBuffersNMS, cudaStream, nullptr);

                    const int outputNumsDetIndex = this->detector->cudaEngine->getBindingIndex(this->detector->outputNumDetBlobName.c_str());
                    const int outputDetBoxesIndex = this->detector->cudaEngine->getBindingIndex(this->detector->outputDetBoxesBlobName.c_str());
                    const int outputDetScoresIndex = this->detector->cudaEngine->getBindingIndex(this->detector->outputDetScoresBlobName.c_str());
                    const int outputDetClassesIndex = this->detector->cudaEngine->getBindingIndex(this->detector->outputDetClassesBlobName.c_str());

                    std::cout<<"indices:"<<outputNumsDetIndex<<", "<<outputDetBoxesIndex<<", "<<outputDetScoresIndex<<", "<<outputDetClassesIndex<<std::endl; 
                    CUDA_CHECK(cudaMemcpyAsync(hostBuffersNMS[0], deviceBuffersNMS[1], 1 * this->detector->cudaEngine->getBindingDimensions(outputNumsDetIndex).d[1] * sizeof(float), cudaMemcpyDeviceToHost, cudaStream));
                    CUDA_CHECK(cudaMemcpyAsync(hostBuffersNMS[1], deviceBuffersNMS[2], 1 * this->detector->cudaEngine->getBindingDimensions(outputDetBoxesIndex).d[1] * this->detector->cudaEngine->getBindingDimensions(outputDetBoxesIndex).d[2] * sizeof(float), cudaMemcpyDeviceToHost, cudaStream));
                    CUDA_CHECK(cudaMemcpyAsync(hostBuffersNMS[2], deviceBuffersNMS[3], 1 * this->detector->cudaEngine->getBindingDimensions(outputDetScoresIndex).d[1] * sizeof(float),cudaMemcpyDeviceToHost, cudaStream));
                    CUDA_CHECK(cudaMemcpyAsync(hostBuffersNMS[3], deviceBuffersNMS[4], 1 * this->detector->cudaEngine->getBindingDimensions(outputDetClassesIndex).d[1] * sizeof(float), cudaMemcpyDeviceToHost, cudaStream));

                    cudaStreamSynchronize(cudaStream);
                    //postprocessing
                    std::vector<BBoxInfo> objs = this->detector->postprocess(hostBuffersNMS);

                    std::cout<<"Received inside inference a frame objs size:"<<objs.size()<<" id:"<<frameInfo.id<<std::endl;

                    //push to event rule queue
                    inference2rules.push(FrameInfo(frameInfo.frame, frameInfo.id, frameInfo.type, objs));
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                // std::cout<<"Waiting for new frame.."<<this->input2Inference.size()<<std::endl;
            }
        }

    public:
        std::queue<FrameInfo> input2Inference;
        std::queue<FrameInfo> inference2rules;
        std::queue<FrameInfo> rules2events;
        bool isModelLoaded = false;

        // static App* app;
        // Singletons should not be cloneable.
        // App() = delete;

        Detector* detector = nullptr;
        App( Detector *obj)
        {
            this->detector = obj;
        }

        // static App* getInstance(Detector* obj){
        //     if(app == nullptr){
        //         app = new App(obj);
        //         // this->detector = obj;
        //     }else{
        //         assert("This is singleton class");
        //     }
        //     return app;
        // }

        void memoryInit(){

            CUDA_CHECK(cudaStreamCreate(&cudaStream));

            const int inputIndex = this->detector->cudaEngine->getBindingIndex(this->detector->inputBlobName.c_str());
            assert(inputIndex == 0);
            CUDA_CHECK(cudaMalloc(&deviceBuffersNMS[0], this->detector->batchSize * 3 * this->detector->modelHWSize * this->detector->modelHWSize  * sizeof(float)));

            const int outputNumsDetIndex = this->detector->cudaEngine->getBindingIndex(detector->outputNumDetBlobName.c_str());
            const int outputDetBoxesIndex = this->detector->cudaEngine->getBindingIndex(detector->outputDetBoxesBlobName.c_str());
            const int outputDetScoresIndex = this->detector->cudaEngine->getBindingIndex(detector->outputDetScoresBlobName.c_str());
            const int outputDetClassesIndex = this->detector->cudaEngine->getBindingIndex(detector->outputDetClassesBlobName.c_str());

            CUDA_CHECK(cudaMalloc(&deviceBuffersNMS[1], this->detector->cudaEngine->getBindingDimensions(outputNumsDetIndex).d[0] * this->detector->cudaEngine->getBindingDimensions(outputNumsDetIndex).d[1] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&deviceBuffersNMS[2], this->detector->cudaEngine->getBindingDimensions(outputDetBoxesIndex).d[0] * this->detector->cudaEngine->getBindingDimensions(outputDetBoxesIndex).d[1] * this->detector->cudaEngine->getBindingDimensions(outputDetBoxesIndex).d[2] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&deviceBuffersNMS[3], this->detector->cudaEngine->getBindingDimensions(outputDetScoresIndex).d[0] * this->detector->cudaEngine->getBindingDimensions(outputDetScoresIndex).d[1] * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&deviceBuffersNMS[4], this->detector->cudaEngine->getBindingDimensions(outputDetClassesIndex).d[0] * this->detector->cudaEngine->getBindingDimensions(outputDetClassesIndex).d[1] * sizeof(float)));

            hostBuffersNMS[0] = new float[this->detector->cudaEngine->getBindingDimensions(outputNumsDetIndex).d[0] * this->detector->cudaEngine->getBindingDimensions(outputNumsDetIndex).d[1] * sizeof(float)];
            hostBuffersNMS[1] = new float[this->detector->cudaEngine->getBindingDimensions(outputDetBoxesIndex).d[0] * this->detector->cudaEngine->getBindingDimensions(outputDetBoxesIndex).d[1] * this->detector->cudaEngine->getBindingDimensions(outputDetBoxesIndex).d[2] * sizeof(float)];
            hostBuffersNMS[2] = new float[this->detector->cudaEngine->getBindingDimensions(outputDetScoresIndex).d[0] * this->detector->cudaEngine->getBindingDimensions(outputDetScoresIndex).d[1] * sizeof(float)];
            hostBuffersNMS[3] = new float[this->detector->cudaEngine->getBindingDimensions(outputDetClassesIndex).d[0] * this->detector->cudaEngine->getBindingDimensions(outputDetClassesIndex).d[1] * sizeof(float)];
        }

        void createEngine(){
            this->detector->createEngine();
            std::cout<<"Engine file is created"<<std::endl;
        }

        void loadEngine(){
            this->detector->loadEngine();
            this->memoryInit();
            this->isModelLoaded = true;
            std::cout<<"Engine file is loaded"<<std::endl;
        }

        void startInferenceThreads(){
            std::thread t(&App::inference, this);
            t.detach();
            std::cout<<"startInferenceThreads started"<<std::endl;
        }

        void startRulesThread(){
            std::thread t(&App::countUseCase, this, 0);
            t.detach();
            std::cout<<"startRulesThread started"<<std::endl;
        }

        void startEventsThread(){
            std::thread t(&App::events, this);
            t.detach();
            std::cout<<"startEventsThread started"<<std::endl;
        }
};
   

int main(){
    YoloV7* yolov7 = new YoloV7();
    assignConfig(yolov7);
    // App *app = App::getInstance(yolov7);
    
    App *app = new App(yolov7);

    if(!doesFileExist(app->detector->enginePath)){
        app->createEngine();
    }

    app->loadEngine();
    app->startInferenceThreads();
    app->startRulesThread();
    app->startEventsThread();

    crow::SimpleApp restapp;

    CROW_ROUTE(restapp, "/add_json")
    .methods("POST"_method)
    ([&app](const crow::request& req){
        auto x = crow::json::load(req.body);
        if (!x)
            return crow::response(crow::status::BAD_REQUEST); // same as crow::response(400)
        
        // std::cout<<"json_str:"<<x["image_str"].s()<<std::endl;

        cv::Mat receivedimg = base64ToMat(x["image_str"].s());

        // cv::imwrite("/app/build/receivedImg.jpg", receivedimg);
        // std::cout<<x["image_str"]<<std::endl;
        // cv::Mat img = cv::imread("/app/images/LM.jpg");


        std::string uniqueId = millisecondsToDateTimeString();

        FrameInfo payload(receivedimg, uniqueId, "curl");
        std::cout<<"unique id: "<<payload.id<<std::endl;
        app->input2Inference.push(payload);

        int timeout = 400;
        while(mp.find(uniqueId) ==  mp.end() && timeout){
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            timeout--;
        }
        
        if(mp.find(uniqueId) !=  mp.end()){
            std::string resData = mp[uniqueId];
            mp.erase(uniqueId);
            return crow::response(200, resData);
        }else{
            return crow::response(400, "timeout");
        }
        
        std::cout<<"Queue size:"<<app->input2Inference.size()<<std::endl;
        
    });

    // Start the server on port 8080
    restapp.port(8080).multithreaded().run();


    while (1)
    {
        std::this_thread::sleep_for(std::chrono::hours(5));
    }
    
    return 0;
}