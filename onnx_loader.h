#ifndef ONNX_LOADER_H
#define ONNX_LOADER_H

#include <string>
#include <vector>
#include <onnxruntime_cxx_api.h>

struct Config {
    std::string modelPath;
    bool useCUDA;
    
    // input detailes.
    int inputHeight = 32;
    int inputWidth = 32;
    int inputChannels = 3;
};

class ONNXLoader {
public:
    //constructor
    ONNXLoader(const Config& config);
    //destructor
    ~ONNXLoader();
    bool runInference();
    
private:
    Config config_;
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;
    std::unique_ptr<Ort::Session> session_;

    // input output node names and demension
    std::vector<const char*> inputNodeNames_{"input"};
    std::vector<const char*> outputNodeNames_{"output"};
    std::vector<int64_t> inputNodeDims_;

};
#endif // ONNX_LOADER_H
