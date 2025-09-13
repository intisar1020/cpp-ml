#ifndef MSNET_INFERENCE_H
#define MSNET_INFERENCE_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <onnxruntime_cxx_api.h>

// configuration struct to hold model paths and parameters
struct Config {
    std::string router_model_path;
    std::string expert_model_dir;
    int topk = 2; 
    bool use_cuda = false;
    int device_id = 0;

    int input_height = 32;
    int input_width = 32;
    int input_channels = 3;
};

class MSNetInference {
public:
    MSNetInference(const Config& config);
    int predict(const std::vector<float>& input_image_data);

private:
    std::vector<float> run_inference(Ort::Session* session, const Ort::Value& input_tensor);

    // Helper to get top-k predictions from raw logits
    std::pair<std::vector<int64_t>, std::vector<float>> get_topk_predictions(const std::vector<float>& logits, int k);

    // Helper to average logits from multiple model outputs
    std::vector<float> average_logits(const std::vector<std::vector<float>>& all_logits);

    // Helper to parse expert names like "5_23" into a vector of class IDs {5, 23}
    void parse_expert_class_map();

    Config config_;
    Ort::Env env_;
    Ort::SessionOptions session_options_;

    std::unique_ptr<Ort::Session> router_session_;
    std::map<std::string, std::unique_ptr<Ort::Session>> expert_sessions_;
    
    // Stores parsed class IDs for each expert for fast lookup
    std::map<std::string, std::vector<int>> expert_class_map_;

    // Input/Output node names (assuming standard names)
    std::vector<const char*> input_node_names_{"input"};
    std::vector<const char*> output_node_names_{"output"};
    std::vector<int64_t> input_node_dims_;
};

#endif // MSNET_INFERENCE_H