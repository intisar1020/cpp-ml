#include <iostream>
#include <vector>
#include "logistic_model.h"

int main() {
    std:: cout << "Logistic Regression Model Simulation" << std::endl;

    ml_sim::LogisticModel<float> floatModel;
    std::vector<float> inputFloat = {1.0f, 2.0f, 3.0f};
    int resultFloat = floatModel.predict(inputFloat);
    std::cout << "Prediction for float input: " << resultFloat << std::endl;
    return 0;
}