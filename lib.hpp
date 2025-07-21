#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <ostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace dscilib {

// ---- Function Prototypes ----

// Calculates sum of squared errors for a given model and dataset.
template <typename InputType, typename FuncType>
double sum_squared_error(std::vector<double>& coefficients,
                        std::vector<InputType>& inputs,
                        std::vector<double>& outputs,
                        FuncType function);

// Rotates a matrix 90 degrees clockwise.
std::vector<std::vector<double>> rotate_90_cw(std::vector<std::vector<double>> matrix);

// Inverts a matrix by rotating and reversing rows.
std::vector<std::vector<double>> transpose_matrix(std::vector<std::vector<double>>matrix);

// Estimates the partial derivative of a function with respect to a variable.
template <typename FuncType, typename ArgsType>
double partial_derivative(FuncType function, double& wrt, ArgsType& args);

// Performs coordinate descent optimization on coefficients.
template <typename CoefficientType, typename InputType, typename OutputType, typename FuncType>
void coordinate_descent(std::vector<CoefficientType>& coefficients,
                       std::vector<InputType>& inputs,
                       std::vector<OutputType>& outputs,
                       FuncType& func);

// Finds a root using Newton's method for a single variable.
template <typename FuncType>
double newton_root(double& input, FuncType function);

// Reads a CSV file into a 2D vector of strings.
std::vector<std::vector<std::string>> read_csv_string(std::string file_name);

// Reads a CSV file into a 2D vector of doubles.
std::vector<std::vector<double>> read_csv_double(std::string file_name);

// ---- Internal Implementation Wrappers ----
namespace detail {
// Up max training speed when working with large coefficients
// Lower min training speed for precision
double MAX_TRAINING_SPEED = 100.0;
double MIN_TRAINING_SPEED = 0.001;

// Wrapper for sum_squared_error to use in optimization routines.
template <typename CoefficientType, typename InputType, typename OutputType, typename FuncType>
struct SumSquaredErrorWrapper {
    std::vector<CoefficientType> coefficients;
    std::vector<InputType> inputs;
    std::vector<OutputType> outputs;
    FuncType& function;
    double operator()(std::vector<CoefficientType> coeffs) {
        return sum_squared_error(coeffs, inputs, outputs, function);
    }
};
} // namespace detail

// Rotates a matrix 90 degrees clockwise.
inline std::vector<std::vector<double>> rotate_90_cw(std::vector<std::vector<double>> matrix) {
    size_t rows = matrix.size();
    if (rows == 0)
        return {};
    size_t cols = matrix[0].size();

    std::vector<std::vector<double>> rotated(cols, std::vector<double>(rows));
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            rotated[j][rows - 1 - i] = matrix[i][j];

    return rotated;
}

inline std::vector<std::vector<double>>
transpose_matrix(std::vector<std::vector<double>> matrix) {
  size_t rows = matrix.size();
  if (rows == 0)
    return {};
  size_t cols = matrix[0].size();
  std::vector<std::vector<double>> tranposed_matrix(cols,
                                                    std::vector<double>(rows));
  for (size_t i = 0; i < rows; ++i)
    for (size_t j = 0; j < cols; ++j)
      tranposed_matrix[j][i] = matrix[i][j];
  return tranposed_matrix;
}

template <typename type>
inline std::vector<type> batch(std::vector<type> matrix,
                               const double sampleSize) {
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(matrix.begin(), matrix.end(), g);
  std::vector<type> newMatrix(matrix.begin(), matrix.begin() + sampleSize);
  return newMatrix;
}

// Calculates sum of squared errors for a given model and dataset.
template <typename InputType, typename FuncType>
inline double sum_squared_error(std::vector<double>& coefficients,
                               std::vector<InputType>& inputs,
                               std::vector<double>& outputs,
                               FuncType function) {
    long double sum = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        double pred = function(inputs[i], coefficients);
        double diff = pred - outputs[i];
        sum += diff * diff;
    }
    return sum;
}

// Estimates the partial derivative of a function with respect to a variable.
template <typename FuncType, typename ArgsType>
inline double partial_derivative(FuncType function, double& wrt, ArgsType& args) {
    constexpr double epsilon = 1e-6;
    wrt += epsilon;
    double loss_plus = function(args);
    wrt -= 2 * epsilon;
    double loss_minus = function(args);
    wrt += epsilon;
    return (loss_plus - loss_minus) / (2.0 * epsilon);
}

// Performs coordinate descent optimization on coefficients.
template <typename CoefficientType, typename InputType, typename OutputType, typename FuncType>
inline void coordinate_descent(std::vector<CoefficientType>& coefficients,
                              std::vector<InputType>& inputs,
                              std::vector<OutputType>& outputs,
                              FuncType& func) {
    double epsilon = 1e-6;
    for (double& coefficient : coefficients) {
        detail::SumSquaredErrorWrapper<CoefficientType, InputType, OutputType, FuncType>
            loss_func{coefficients, inputs, outputs, func};
        double gradient = partial_derivative(loss_func, coefficient, coefficients);
        double speed = detail::MAX_TRAINING_SPEED;
        double base_loss = sum_squared_error(coefficients, inputs, outputs, func);
        double old_coefficient = coefficient;
        coefficient -= gradient * speed;
        while (sum_squared_error(coefficients, inputs, outputs, func) > base_loss &&
               speed >= detail::MIN_TRAINING_SPEED) {
            speed *= 0.1;
            coefficient = old_coefficient - gradient * speed;
        }
        if (sum_squared_error(coefficients, inputs, outputs, func) > base_loss) {
            coefficient = old_coefficient - gradient * speed;
        }
    }
}

// Finds a root using Newton's method for a single variable.
template <typename FuncType>
inline double newton_root(double& input, FuncType function) {
    double derivative_estimate = partial_derivative(function, input, input);
    double x_1 = input;
    double y_1 = function(input);
    double root = (derivative_estimate * x_1 - y_1) / derivative_estimate;
    input = root;
    return input;
}

// Reads a CSV file into a 2D vector of strings.
inline std::vector<std::vector<std::string>> read_csv_string(std::string file_name) {
    std::fstream fin;
    fin.open(file_name, std::ios::in);
    std::string row, temp, line, col;
    std::vector<std::vector<std::string>> data;
    std::vector<std::string> curr;
    while (getline(fin, line, '\n')) {
        std::stringstream s(line);
        while (getline(s, col, ',')) {
            if (!col.empty()) {
                curr.push_back(col);
            } else {
                curr.push_back("0");
            }
        }
        data.push_back(curr);
        curr.clear();
    }
    return data;
}

// Reads a CSV file into a 2D vector of doubles.
inline std::vector<std::vector<double>> read_csv_double(std::string file_name) {
    std::fstream fin;
    fin.open(file_name, std::ios::in);
    std::string row, temp, line, col;
    std::vector<std::vector<double>> data;
    std::vector<double> curr;
    while (getline(fin, line, '\n')) {
        std::stringstream s(line);
        while (getline(s, col, ',')) {
            if (!col.empty()) {
                curr.push_back(stod(col));
            } else {
                curr.push_back(0);
            }
        }
        data.push_back(curr);
        curr.clear();
    }
    return data;
}

} // namespace dscilib
