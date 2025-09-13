#include "onnx_loader.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>

ONNXLoader::ONNXLoader(const Config& config)
    : config_(config), env_(ORT_LOGGING_LEVEL_WARNING, "ONNXLoader") {
    sessionOptions_.SetIntraOpNumThreads(1);
    std::cout << "[INFO] Loading model from: " << config_.modelPath << std::endl;
    try {
        session_ = std::make_unique<Ort::Session>(env_, config_.modelPath.c_str(), sessionOptions_);
    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] Failed to load the model: " << e.what() << std::endl;
        throw;
    }
    std::cout << "[INFO] Model loaded successfully." << std::endl;

    // Default input dims (NCHW)
    inputNodeDims_ = {1, config_.inputChannels, config_.inputHeight, config_.inputWidth};
}

ONNXLoader::~ONNXLoader() {
    // Let smart pointers clean up automatically
}

bool ONNXLoader::runInference() {
    std::cout << "[INFO] Running inference..." << std::endl;

    // Create dummy input tensor
    size_t inputTensorSize = config_.inputHeight * config_.inputWidth * config_.inputChannels;
    std::vector<float> inputTensorValues(inputTensorSize);
    for (size_t i = 0; i < inputTensorSize; i++) {
        inputTensorValues[i] = static_cast<float>(i) / inputTensorSize;
    }

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputNodeDims_.data(), inputNodeDims_.size()
    );

    try {
        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNodeNames_.data(),
            &inputTensor,
            1,
            outputNodeNames_.data(),
            1
        );

        const auto& outputTensor = outputTensors.front();

        // Extract output values
        size_t outputCount = outputTensor.GetTensorTypeAndShapeInfo().GetElementCount();
        const float* outputData = outputTensor.GetTensorData<float>();
        std::vector<float> outputTensorValues(outputData, outputData + outputCount);
        for (int i = 0; i < outputTensorValues.size() && i < 10; i++) {
            std::cout << outputTensorValues[i] << " ";
        }

        // Print shape
        auto outputShapeInfo = outputTensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputShape = outputShapeInfo.GetShape();

        std::cout << "[INFO] Output tensor shape: [";
        for (size_t i = 0; i < outputShape.size(); i++) {
            std::cout << outputShape[i];
            if (i + 1 < outputShape.size()) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "[INFO] Inference completed successfully." << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] Inference failed: " << e.what() << std::endl;
        return false;
    }
}
