# mlmetrics.h

A **header-only C++ library** for computing machine learning metrics. Lightweight, easy to integrate, and designed for both classification and regression tasks.
True and predicted vectors must be of some **floating point type** (float, double, etc.)

![Header-only](https://img.shields.io/badge/header--only-lightgrey) ![C++11](https://img.shields.io/badge/C++-11-blue) ![License](https://img.shields.io/badge/license-MIT-brightgreen)

## 📦 Features

### Classification Metrics
| Metric               | Function                                            | Zero-Division Handling              |
|----------------------|-----------------------------------------------------|-------------------------------------|
| Weighted Precision   | `get_precision_score(y_true, y_test, zero_division)`| 0=return 0, 1=return 1, 2=skip class|
| Weighted Recall      | `get_recall_score(y_true, y_test, zero_division)`   | 0=return 0, 1=return 1, 2=skip class|
| Weighted F1 Score    | `get_f1_score(y_true, y_test, zero_division)`       | 0=return 0, 1=return 1, 2=skip class|
| Hamming Loss         | `get_hamming_loss(y_true, y_test)`                  | N/A                                 |


### Regression Metrics
| Metric                               | Function                                   |
|--------------------------------------|--------------------------------------------|
| Mean Squared Error (MSE)             | `get_mean_squared_error(y_true, y_test)`   |
| Mean Absolute Error (MAE)            | `get_mean_absolute_error(y_true, y_test)`  |
| R² Score                             | `get_r2_score(y_true, y_test)`             |
| Mean Absolute Percentage Error (MAPE)| `get_mape(y_true, y_test)`                 |

## 🚀 Installation
1. Download [`mlmetrics.h`](mlmetrics.h)
2. Include in your project:
   ```cpp
   #include "mlmetrics.h"
   
## Run test program. It simpy reads provided csv files with 2 columns and calculates regression or classification metrics

1. Compile: 
```bash
g++ test.cpp -o test
```
2. Run with regression data:
```bash
test data/test_regression.csv -r
```
3. Compare results with ```data/regression_sklearn_metrics.csv```
4. Run with classification data:
```bash
test data/test_classification.csv -c
```
5. Compare results with ```data/classification_sklearn_metrics.csv```

