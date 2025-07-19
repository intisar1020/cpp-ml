// author: Intisar Chy.
// date: 2025-07-19
// description: Header file for the LogisticModel class, which implements logistic regression.


#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <numeric>

namespace ml_sim {
    template <typename T>
    class LogisticModel {
    public:
        // constructor
        LogisticModel();
        int predict(const std::vector<T>& input);

    private:
        // sigmoid, weights, bias
        std::vector<T> weights_;
        T bias_;
        double sigmoid(T z);
    };

    // its usually a good practice to implement
    // template class methods in the header file

    template <typename T>
    LogisticModel<T>::LogisticModel() {
        // Initialize with some hard-coded "trained" parameters, cast to type T.
        weights_ = {static_cast<T>(0.8), static_cast<T>(-1.2), static_cast<T>(0.3)};
        bias_ = static_cast<T>(0.5);
    }

    template <typename T>
    double LogisticModel<T>::sigmoid(T z) {
        return 1.0 / (1.0 + std::exp(-static_cast<double>(z)));
    }

    template <typename T>
    int LogisticModel<T>::predict(const std::vector<T>& input) {
        if (input.size() != weights_.size()) {
            throw std::invalid_argument("Input size must match the number of weights.");
        }
        // we will calcilate the dot product
        T z = bias_;
        for (size_t i = 0; i < input.size(); ++i) {
            z += weights_[i] * input[i];
        }

        double probability = sigmoid(z);
        if (probability >= 0.5) {
            return 1;
        }
        return 0;
    }
}
