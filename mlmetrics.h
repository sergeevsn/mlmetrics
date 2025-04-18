#ifndef MLMETRICS_H
#define MLMETRICS_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <cstddef>
#include <numeric>
#include <cmath>

namespace mlmetrics {

/// Computes weighted precision score for multiclass classification.
/// zero_division: {0->score=0, 1->score=1, 2->skip class in average}
template <typename T>
T get_precision_score(const std::vector<T>& y_true,
                      const std::vector<T>& y_pred,
                      int zero_division = 0) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Sizes of y_true and y_pred must match");
    }

    // Identify all classes from ground truth
    std::unordered_set<T> classes(y_true.begin(), y_true.end());

    // Count support (number of true instances) per class
    std::unordered_map<T, int> support;
    for (const T& t : y_true) {
        support[t]++;
    }

    T weighted_sum = T(0);
    int total_support = 0;

    // Compute per-class precision and accumulate weighted sum
    for (const T& cls : classes) {
        int tp = 0, fp = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_pred[i] == cls) {
                if (y_true[i] == cls) {
                    tp++;  // true positive
                } else {
                    fp++;  // false positive
                }
            }
        }

        int denom = tp + fp;
        T precision_i = T(0);

        if (denom == 0) {
            // No positive predictions for this class
            if (zero_division == 2) {
                continue;  // skip this class in the weighted average
            }
            // Use zero_division policy: 0 or 1
            precision_i = (zero_division == 1 ? T(1) : T(0));
        } else {
            precision_i = static_cast<T>(tp) / static_cast<T>(denom);
        }

        int sup = support[cls];
        weighted_sum += precision_i * static_cast<T>(sup);
        total_support += sup;
    }

    // If no classes contributed, return zero
    if (total_support == 0) {
        return T(0);
    }
    return weighted_sum / static_cast<T>(total_support);
}

/// Computes weighted recall score for multiclass classification.
/// zero_division: {0->score=0, 1->score=1, 2->skip class in average}
template <typename T>
T get_recall_score(const std::vector<T>& y_true,
                   const std::vector<T>& y_pred,
                   int zero_division = 0) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Sizes of y_true and y_pred must match");
    }

    // Identify all classes from ground truth
    std::unordered_set<T> classes(y_true.begin(), y_true.end());

    // Count support (number of true instances) per class
    std::unordered_map<T, int> support;
    for (const T& t : y_true) {
        support[t]++;
    }

    T weighted_sum = T(0);
    int total_support = 0;

    // Compute per-class recall and accumulate weighted sum
    for (const T& cls : classes) {
        int tp = 0, fn = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == cls) {
                if (y_pred[i] == cls) {
                    tp++;  // true positive
                } else {
                    fn++;  // false negative
                }
            }
        }

        int denom = tp + fn;
        T recall_i = T(0);

        if (denom == 0) {
            // No true instances for this class
            if (zero_division == 2) {
                continue;  // skip this class in the weighted average
            }
            // Use zero_division policy: 0 or 1
            recall_i = (zero_division == 1 ? T(1) : T(0));
        } else {
            recall_i = static_cast<T>(tp) / static_cast<T>(denom);
        }

        int sup = support[cls];
        weighted_sum += recall_i * static_cast<T>(sup);
        total_support += sup;
    }

    // If no classes contributed, return zero
    if (total_support == 0) {
        return T(0);
    }
    return weighted_sum / static_cast<T>(total_support);
}

/// Computes weighted F1 score for multiclass classification.
/// zero_division: {0->score=0, 1->score=1, 2->skip class in average}
template <typename T>
T get_f1_score(const std::vector<T>& y_true,
               const std::vector<T>& y_pred,
               int zero_division = 0) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Sizes of y_true and y_pred must match");
    }

    // Identify all classes from ground truth
    std::unordered_set<T> classes(y_true.begin(), y_true.end());

    // Count support (number of true instances) per class
    std::unordered_map<T, int> support;
    for (const T& t : y_true) {
        support[t]++;
    }

    T weighted_sum = T(0);
    int total_support = 0;

    // Compute per-class precision, recall and F1, then accumulate weighted sum
    for (const T& cls : classes) {
        int tp = 0, fp = 0, fn = 0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            if (y_true[i] == cls) {
                if (y_pred[i] == cls) {
                    tp++;  // true positive
                } else {
                    fn++;  // false negative
                }
            } else if (y_pred[i] == cls) {
                fp++;      // false positive
            }
        }

        // Precision calculation
        int denom_p = tp + fp;
        T precision_i = T(0);
        if (denom_p == 0) {
            if (zero_division == 2) {
                continue;  // skip this class
            }
            precision_i = (zero_division == 1 ? T(1) : T(0));
        } else {
            precision_i = static_cast<T>(tp) / static_cast<T>(denom_p);
        }

        // Recall calculation
        int denom_r = tp + fn;
        T recall_i = (denom_r == 0)
            ? (zero_division == 2 ? T(-1) : (zero_division == 1 ? T(1) : T(0)))
            : static_cast<T>(tp) / static_cast<T>(denom_r);
        if (denom_r == 0 && zero_division == 2) {
            continue;  // skip class when both precision and recall undefined
        }

        // F1 calculation
        T sum_pr = precision_i + recall_i;
        T f1_i = T(0);
        if (sum_pr == T(0)) {
            // precision_i and recall_i are both zero (or undefined)
            if (zero_division == 2) {
                continue;  // skip this class
            }
            f1_i = (zero_division == 1 ? T(1) : T(0));
        } else {
            f1_i = T(2) * precision_i * recall_i / sum_pr;
        }

        int sup = support[cls];
        weighted_sum += f1_i * static_cast<T>(sup);
        total_support += sup;
    }

    // If no classes contributed, return zero
    if (total_support == 0) {
        return T(0);
    }
    return weighted_sum / static_cast<T>(total_support);
}

template <typename T>
T get_hamming_loss(const std::vector<T>& y_true,
                   const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Sizes of y_true and y_pred must match");
    }

    size_t n = y_true.size();
    if (n == 0) {
        return static_cast<T>(0.0f);
    }

    size_t mismatches = 0;
    for (size_t i = 0; i < n; ++i) {
        if (y_true[i] != y_pred[i]) {
            mismatches++;
        }
    }

    return static_cast<T>(mismatches) / static_cast<T>(n);
}

template <typename T>
T get_mean_squared_error(const std::vector<T>& y_true,
                         const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Sizes of y_true and y_pred must match");
    }
    size_t n = y_true.size();
    if (n == 0) {
        return static_cast<T>(0.0f);
    }

    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff = y_true[i] - y_pred[i];
        sum_sq += diff * diff;
    }
    return sum_sq / n;
}

template <typename T>
T get_mean_absolute_error(const std::vector<T>& y_true,
                          const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Sizes of y_true and y_pred must match");
    }
    size_t n = y_true.size();
    if (n == 0) {
        return static_cast<T>(0.0f);
    }

    float sum_abs = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum_abs += std::abs(y_true[i] - y_pred[i]);
    }
    return sum_abs / n;
}

template <typename T>
T get_r2_score(const std::vector<T>& y_true,
               const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Sizes of y_true and y_pred must match");
    }
    size_t n = y_true.size();
    if (n == 0) {
        return static_cast<T>(0.0f);
    }

    float mean_y = std::accumulate(y_true.begin(), y_true.end(), 0.0f) / n;

    float ss_res = 0.0f;
    float ss_tot = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float diff_res = y_true[i] - y_pred[i];
        ss_res += diff_res * diff_res;
        float diff_tot = y_true[i] - mean_y;
        ss_tot += diff_tot * diff_tot;
    }

    if (ss_tot == 0.0f) {
        return static_cast<T>(0.0f);
    }
    return static_cast<T>(1.0f - ss_res / ss_tot);
}

template <typename T>
T get_mape(const std::vector<T>& y_true,
           const std::vector<T>& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::invalid_argument("Sizes of y_true and y_pred must match");
    }

    size_t n = y_true.size();
    if (n == 0) {
        return static_cast<T>(0.0f);
    }

    float sum_ape = 0.0f;
    size_t count = 0;

    for (size_t i = 0; i < n; ++i) {
        if (y_true[i] != T(0)) {
            float ape = std::abs((y_true[i] - y_pred[i]) / y_true[i]);
            sum_ape += ape;
            count++;
        }
    }

    if (count == 0) {
        return static_cast<T>(0.0f);
    }
    return static_cast<T>((sum_ape / count) * 100.0f);
}

} // namespace mlmetrics

#endif // MLMETRICS_H

