#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip> // For formatting output
#include "mlmetrics.h"

template <typename T>
std::string vector_to_string(const std::vector<T>& vec) {
    std::stringstream ss;
    for (size_t i = 0; i < vec.size(); ++i) {
        ss << vec[i];
        if (i < vec.size() - 1) {
            ss << ",";
        }
    }
    return ss.str();
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <file.csv> <-r|-c>" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    std::string flag = argv[2];

    if (flag != "-r" && flag != "-c") {
        std::cerr << "Invalid flag. Use -r for regression or -c for classification." << std::endl;
        return 1;
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }

    std::vector<double> y_true;
    std::vector<double> y_pred;
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (const std::invalid_argument&) {
                std::cerr << "Invalid data format in line: " << line << std::endl;
                return 1;
            }
        }

        if (row.size() != 2) {
            std::cerr << "Incorrect number of columns in line: " << line << std::endl;
            return 1;
        }

        y_true.push_back(row[0]);
        y_pred.push_back(row[1]);
    }
    file.close();

    std::cout << "y_true: " << vector_to_string(y_true) << std::endl;
    std::cout << "y_pred: " << vector_to_string(y_pred) << std::endl;

    if (flag == "-r") {
        double mse = mlmetrics::get_mean_squared_error(y_true, y_pred);
        double mae = mlmetrics::get_mean_absolute_error(y_true, y_pred);
        double r2 = mlmetrics::get_r2_score(y_true, y_pred);
        double mape = mlmetrics::get_mape(y_true, y_pred);

        std::cout << "\nRegression mlmetrics:" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "MSE: " << mse << std::endl;
        std::cout << "MAE: " << mae << std::endl;
        std::cout << "R²: " << r2 << std::endl;
        std::cout << "MAPE: " << mape << "%" << std::endl;
    } else if (flag == "-c") {
        
        double accuracy = mlmetrics::get_accuracy_score(y_true, y_pred);
        double precision = mlmetrics::get_precision_score(y_true, y_pred);
        double recall = mlmetrics::get_recall_score(y_true, y_pred);
        double f1 = mlmetrics::get_f1_score(y_true, y_pred);
        double hamming = mlmetrics::get_hamming_loss(y_true, y_pred);

        std::cout << "\nClassification mlmetrics:" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Accuracy: " << accuracy << std::endl;
        std::cout << "Precision: " << precision << std::endl;
        std::cout << "Recall: " << recall << std::endl;
        std::cout << "F1-score: " << f1 << std::endl;
        std::cout << "Hamming loss: " << hamming << std::endl;
    }

    return 0;
}
