#include "msnet_inference.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>

namespace fs = std::filesystem;

// Constructor implementation
MSNetInference::MSNetInference(const Config& config)
    : config_(config), env_(ORT_LOGGING_LEVEL_WARNING, "MSNetInference") {
    
    session_options_.SetIntraOpNumThreads(1);

    if (config_.use_cuda) {
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = config_.device_id;
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "INFO: CUDA execution provider enabled." << std::endl;
    }

    // Load Router Model
    std::cout << "INFO: Loading router model from: " << config_.router_model_path << std::endl;
    router_session_ = std::make_unique<Ort::Session>(env_, config_.router_model_path.c_str(), session_options_);

    // Load Expert Models
    std::cout << "INFO: Loading expert models from directory: " << config_.expert_model_dir << std::endl;
    for (const auto& entry : fs::directory_iterator(config_.expert_model_dir)) {
        if (entry.path().extension() == ".onnx") {
            std::string expert_key = entry.path().stem().string();
            std::string expert_path = entry.path().string();
            expert_sessions_[expert_key] = std::make_unique<Ort::Session>(env_, expert_path.c_str(), session_options_);
            std::cout << "  - Loaded expert: " << expert_key << std::endl;
        }
    }
    
    if (expert_sessions_.empty()) {
        throw std::runtime_error("No expert .onnx models found in the specified directory.");
    }
    
    parse_expert_class_map();
    
    // Define input tensor shape
    input_node_dims_ = {1, config_.input_channels, config_.input_height, config_.input_width};
}

// Prediction logic
int MSNetInference::predict(const std::vector<float>& input_image_data) {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        const_cast<float*>(input_image_data.data()), 
        input_image_data.size(), 
        input_node_dims_.data(), 
        input_node_dims_.size()
    );

    // 1. Run inference on the router
    std::vector<float> router_logits = run_inference(router_session_.get(), input_tensor);
    
    // 2. Get top-k predictions from the router
    auto top_router_preds = get_topk_predictions(router_logits, config_.topk);
    int pred1 = top_router_preds.first[0];
    int pred2 = top_router_preds.first[1];

    std::cout << "INFO: Router top-2 predictions: " << pred1 << ", " << pred2 << std::endl;

    // 3. Select an expert based on router predictions
    std::string selected_expert_key;
    for (const auto& pair : expert_class_map_) {
        const auto& classes = pair.second;
        // A more robust check is if the expert covers BOTH top predictions.
        bool covers_pred1 = std::find(classes.begin(), classes.end(), pred1) != classes.end();
        bool covers_pred2 = std::find(classes.begin(), classes.end(), pred2) != classes.end();

        if (covers_pred1 && covers_pred2) {
            selected_expert_key = pair.first;
            break; // Found the first matching expert
        }
    }

    std::vector<std::vector<float>> logits_to_average;
    logits_to_average.push_back(router_logits);

    // 4. If an expert is found, run inference and add its output for averaging
    if (!selected_expert_key.empty()) {
        std::cout << "INFO: Selected expert '" << selected_expert_key << "' for refinement." << std::endl;
        std::vector<float> expert_logits = run_inference(expert_sessions_[selected_expert_key].get(), input_tensor);
        logits_to_average.push_back(expert_logits);
    } else {
        std::cout << "WARN: No suitable expert found. Using router output only." << std::endl;
    }

    // 5. Average the logits and find the final prediction
    std::vector<float> final_logits = average_logits(logits_to_average);
    auto max_it = std::max_element(final_logits.begin(), final_logits.end());
    int final_prediction = std::distance(final_logits.begin(), max_it);

    return final_prediction;
}

std::vector<float> MSNetInference::run_inference(Ort::Session* session, const Ort::Value& input_tensor) {
    auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_node_names_.data(), &input_tensor, 1, output_node_names_.data(), 1);
    float* floatarr = output_tensors.front().GetTensorMutableData<float>();
    size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    return std::vector<float>(floatarr, floatarr + output_size);
}

std::pair<std::vector<int64_t>, std::vector<float>> MSNetInference::get_topk_predictions(const std::vector<float>& logits, int k) {
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return logits[a] > logits[b];
    });

    std::pair<std::vector<int64_t>, std::vector<float>> result;
    result.first.reserve(k);
    result.second.reserve(k);
    for (int i = 0; i < k; ++i) {
        result.first.push_back(indices[i]);
        result.second.push_back(logits[indices[i]]);
    }
    return result;
}

// Helper to average logits
std::vector<float> MSNetInference::average_logits(const std::vector<std::vector<float>>& all_logits) {
    if (all_logits.empty()) {
        return {};
    }
    size_t num_classes = all_logits[0].size();
    std::vector<float> avg_logits(num_classes, 0.0f);
    for (const auto& logits : all_logits) {
        for (size_t i = 0; i < num_classes; ++i) {
            avg_logits[i] += logits[i];
        }
    }
    float num_models = static_cast<float>(all_logits.size());
    for (size_t i = 0; i < num_classes; ++i) {
        avg_logits[i] /= num_models;
    }
    return avg_logits;
}

// Helper to parse expert names
void MSNetInference::parse_expert_class_map() {
    for (const auto& pair : expert_sessions_) {
        const std::string& key = pair.first;
        std::stringstream ss(key);
        std::string segment;
        std::vector<int> class_ids;

        while(std::getline(ss, segment, '_')) {
            try {
                class_ids.push_back(std::stoi(segment));
            } catch (const std::invalid_argument& e) {
                 std::cerr << "WARN: Could not parse class ID from expert name segment: " << segment << std::endl;
            }
        }
        expert_class_map_[key] = class_ids;
    }
}