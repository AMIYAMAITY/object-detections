
#include "yolov7.h"
#include <fstream>



inline bool doesFileExist(const std::string& filepath) {
    struct stat buffer;
    return (stat (filepath.c_str(), &buffer) == 0);
}

IHostMemory* buildEngineFromOnnx(const std::string &onnxModelPath, int maxBatchSize, int size, std::string inputName, std::string outputName, std::string precisionType) {
    IHostMemory* engine;

    if (!doesFileExist(onnxModelPath)) {
        std::cout << "Unable to find onnx model file" << std::endl;
        return engine;
    }

    SpdlogLogger trtLogger;
    bool plugin_initialized = initLibNvInferPlugins(&trtLogger, "");
    std::cout<<"plugin_initialized: "<<plugin_initialized<<std::endl;

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtLogger));
    if (!builder) {
        return engine;
    }

    builder->setMaxBatchSize(maxBatchSize);
    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return engine;
    }

    IParser*  parser = createParser(*network, trtLogger);
    // parser->parseFromFile(onnxModelPath.c_str(), 3);

    if (!parser->parseFromFile(onnxModelPath.c_str(), 3)) {
        std::cerr << "Failed to parse the ONNX file." << std::endl;
        return nullptr;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return engine;
    }

    // Specify the optimization profiles and the
    IOptimizationProfile* optProfile = builder->createOptimizationProfile();
    optProfile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN, Dims4(maxBatchSize, 3, size, size));
    optProfile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT, Dims4(maxBatchSize, 3, size, size));
    optProfile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX, Dims4(maxBatchSize, 3, size, size));
    config->addOptimizationProfile(optProfile);


    // Convert MB to bytes
    // config->setMaxWorkspaceSize(2000 * 1000000);
    config->setMaxWorkspaceSize(16 * (1 << 22));
    // config->setMaxWorkspaceSize(1U << 22);

    if(precisionType == "FP16")
        config->setFlag(BuilderFlag::kFP16);
    if(precisionType == "INT8")
        config->setFlag(BuilderFlag::kINT8);

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory*  serializedModel = builder->buildSerializedNetwork(*network, *config);
    return serializedModel;


    // Build the engine
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* cudaEngine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;
    engine = cudaEngine->serialize();

    return engine;
}

const char* getFileExtension(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if(!dot || dot == filename) return ""; // No extension found
    return dot + 1;
}

inline static float clamp(float val, float min, float max)
{
    return val > min ? (val < max ? val : max) : min;
}


std::vector<BBoxInfo> YoloV7::postprocess(float* hostBuffersNMS[4])
{
    int imgH = this->imageHeight;
    int imgW = this->imageWidth;

    int modelH = this->modelHWSize;
    int modelW = this->modelHWSize;

    int rows = imgH;
    int cols = imgW;

    const float inp_h = modelH;
    const float inp_w = modelW;

    float height = rows;
    float width = cols;

    float r = std::min(inp_h / height, inp_w / width);
    int padw = std::round(width * r);
    int padh = std::round(height * r);
    float dw = inp_w - padw;
    float dh = inp_h - padh;
    dw /= 2.0f;
    dh /= 2.0f;
    float ratio = 1 / r;


    std::vector<BBoxInfo> objs;

    auto* num_dets = reinterpret_cast<int*>(hostBuffersNMS[0]);
    auto* boxes =    reinterpret_cast<float*>(hostBuffersNMS[1]);
    auto* scores =   reinterpret_cast<float*>(hostBuffersNMS[2]);
    auto* labels =   reinterpret_cast<int*>(hostBuffersNMS[3]);

    for (int i = 0; i < num_dets[0]; i++)
    {
        float* ptr = boxes + i * 4;

        float x0 = *ptr++ - dw;
		float y0 = *ptr++ - dh;
		float x1 = *ptr++ - dw;
		float y1 = *ptr - dh;

		x0 = clamp(x0 * ratio, 0.f, width);
		y0 = clamp(y0 * ratio, 0.f, height);
		x1 = clamp(x1 * ratio, 0.f, width);
		y1 = clamp(y1 * ratio, 0.f, height);
        
        BBoxInfo newObj;
        newObj.bbox.x = x0;
        newObj.bbox.y = y0;
        newObj.bbox.width = x1 - x0;
        newObj.bbox.height = y1 - y0;

        newObj.label  = *(labels + i);
        newObj.prob = *(scores + i);

        if(this->precisionType == "FP16")
            newObj.prob =+ 1;

        if(newObj.prob >= this->confThreshold){
            objs.push_back(newObj);
        }
            
    }

    return objs;
}

void YoloV7::createEngine(){
    IHostMemory* engine = buildEngineFromOnnx(this->onnxPath, this->batchSize, this->modelHWSize, this->inputBlobName, this->outputBlobName, this->precisionType);
    this->hostEngine = engine;

    assert(engine != nullptr);
    std::ofstream engineFileDescriptor(this->enginePath, std::ios::binary);
    if (!engineFileDescriptor) {
        std::cerr << "could not open engine file" << std::endl;
        return;
    }

    engineFileDescriptor.write(reinterpret_cast<const char*>(engine->data()), engine->size());
    engine->destroy();
}


void YoloV7::loadEngine(){
    
    // deserialize the .engine and run inference
    std::ifstream file(this->enginePath, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << this->enginePath << " error!" << std::endl;
        return;
    }
    
    SpdlogLogger trtLogger;
    char *trtModelStream = nullptr;
    size_t size = 0;

    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();

    bool plugin_initialized = initLibNvInferPlugins(&trtLogger, "");
    std::cout<<"plugin_initialized: "<<plugin_initialized<<std::endl;

    IRuntime* runtime = createInferRuntime(trtLogger);
    assert(runtime != nullptr);
    this->cudaEngine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->cudaEngine != nullptr);
    this->nvContext = this->cudaEngine->createExecutionContext();
    assert(this->nvContext != nullptr);
    delete[] trtModelStream;

}



