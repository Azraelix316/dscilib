#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dscilib {

// ---- Function Prototypes ----

// Calculates sum of squared errors for a given model and dataset.
template <typename InputType, typename FuncType>
double sum_squared_error(std::vector<double> &coefficients,
                         std::vector<InputType> &inputs,
                         std::vector<double> &outputs, FuncType function);

// Rotates a matrix 90 degrees clockwise.
std::vector<std::vector<double>>
rotate_90_cw(std::vector<std::vector<double>> matrix);

// Inverts a matrix by rotating and reversing rows.
std::vector<std::vector<double>>
transpose_matrix(std::vector<std::vector<double>> matrix);

// Estimates the partial derivative of a function with respect to a variable.
template <typename FuncType, typename ArgsType>
double partial_derivative(FuncType function, double &wrt, ArgsType &args);

// Performs coordinate descent optimization on coefficients.
template <typename CoefficientType, typename InputType, typename OutputType,
          typename FuncType>
void coordinate_descent(std::vector<CoefficientType> &coefficients,
                        std::vector<InputType> &inputs,
                        std::vector<OutputType> &outputs, FuncType &func);

// Finds a root using Newton's method for a single variable.
template <typename FuncType>
double newton_root(double &input, FuncType function);

// multiplies an mxn matrix and a nxp matrix.
std::vector<std::vector<double>>
matrix_mult(const std::vector<std::vector<double>> &m1,
            const std::vector<std::vector<double>> &m2);
// Finds principle components of a 2D dataset
std::vector<std::vector<double>>
principle_component_analysis(std::vector<std::vector<double>> dataset,
                             const double n_principle_components);

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
template <typename CoefficientType, typename InputType, typename OutputType,
          typename FuncType>
struct SumSquaredErrorWrapper {
  std::vector<CoefficientType> coefficients;
  std::vector<InputType> inputs;
  std::vector<OutputType> outputs;
  FuncType &function;
  double operator()(std::vector<CoefficientType> coeffs) {
    return sum_squared_error(coeffs, inputs, outputs, function);
  }
};
} // namespace detail

// Rotates a matrix 90 degrees clockwise.
inline std::vector<std::vector<double>>
rotate_90_cw(std::vector<std::vector<double>> matrix) {
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
inline double sum_squared_error(std::vector<double> &coefficients,
                                std::vector<InputType> &inputs,
                                std::vector<double> &outputs,
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
inline double partial_derivative(FuncType function, double &wrt,
                                 ArgsType &args) {
  constexpr double epsilon = 1e-6;
  wrt += epsilon;
  double loss_plus = function(args);
  wrt -= 2 * epsilon;
  double loss_minus = function(args);
  wrt += epsilon;
  return (loss_plus - loss_minus) / (2.0 * epsilon);
}

// Performs coordinate descent optimization on coefficients.
template <typename CoefficientType, typename InputType, typename OutputType,
          typename FuncType>
inline void coordinate_descent(std::vector<CoefficientType> &coefficients,
                               std::vector<InputType> &inputs,
                               std::vector<OutputType> &outputs,
                               FuncType &func) {
  double epsilon = 1e-6;
  for (double &coefficient : coefficients) {
    detail::SumSquaredErrorWrapper<CoefficientType, InputType, OutputType,
                                   FuncType>
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
inline double newton_root(double &input, FuncType function) {
  double derivative_estimate = partial_derivative(function, input, input);
  double x_1 = input;
  double y_1 = function(input);
  double root = (derivative_estimate * x_1 - y_1) / derivative_estimate;
  input = root;
  return input;
}
inline std::vector<std::vector<double>>
matrix_mult(const std::vector<std::vector<double>> &m1,
            const std::vector<std::vector<double>> &m2) {
  size_t n = m1[0].size();
  // also equal to m2.size();
  size_t m = m1.size();
  size_t p = m2[0].size();
  std::vector<std::vector<double>> newMatrix(m, std::vector<double>(p, 0.0));
  if (m1[0].size() != m2.size()) {
    throw std::invalid_argument(
        "Incompatible matrix dimensions: the columns of matrix 1 are not equal "
        "to the rows of matrix 2");
  }
  // m1[0].size() is the number of columns in the first matrix
  // m2.size() is the number of rows in the second matrix
  // newMatrix[i][j]=all elements in one row of m1 * all elements in the column
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < n; k++) {
        newMatrix[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
  return newMatrix;
}
inline double L2_Norm(std::vector<double> vector) {
  double norm = 0;
  for (const double &element : vector) {
    norm += element * element;
  }
  return sqrt(norm);
}
inline std::vector<double>
power_iteration(std::vector<std::vector<double>> matrix,
                std::vector<double> guess) {
  std::vector<double> newGuess =
      transpose_matrix(matrix_mult(matrix, transpose_matrix({guess})))[0];
  double inverseNorm = 1.0 / L2_Norm(newGuess);
  for (int i = 0; i < newGuess.size(); i++) {
    newGuess[i] *= inverseNorm;
  }
  return newGuess;
}

inline double eigenvalue(std::vector<std::vector<double>> matrix,
                         std::vector<double> eigenvector) {
  double vTv =
      matrix_mult({eigenvector}, transpose_matrix({eigenvector}))[0][0];
  std::cout << vTv << std::endl;
  std::vector<std::vector<double>> Av = matrix_mult(
      {eigenvector}, matrix_mult(matrix, transpose_matrix({eigenvector})));
  for (std::vector<double> row : Av) {
    for (double ele : row) {
      std::cout << ele << "  ";
    }
    std::cout << std::endl;
  }
  return matrix_mult(
      {eigenvector},
      matrix_mult(matrix, transpose_matrix({eigenvector})))[0][0];
}
inline std::vector<std::vector<double>>
PCA(std::vector<std::vector<double>> dataset,
    const double &n_principle_components) {
  // Center data
  std::vector<std::vector<double>> centered_data = transpose_matrix(dataset);
  for (int i = 0; i < centered_data.size(); i++) {
    double mean = 0.0;
    for (int j = 0; j < centered_data[i].size(); j++) {
      mean += centered_data[i][j] / centered_data[i].size();
    }
    for (int j = 0; j < centered_data[i].size(); j++) {
      centered_data[i][j] -= mean;
    }
  }

  for (std::vector<double> row : centered_data) {
    for (double ele : row) {
      std::cout << ele << "  ";
    }
    std::cout << std::endl;
  }
  // flipped transpose operation because of centering strat
  std::vector<std::vector<double>> covariance_matrix =
      matrix_mult(centered_data, transpose_matrix(centered_data));
  for (int i = 0; i < covariance_matrix.size(); i++) {
    for (int j = 0; j < covariance_matrix.size(); j++) {
      covariance_matrix[i][j] *= 1.0 / (centered_data[0].size() - 1);
    }
  }

  // Calculate the covariance matrix
  /*
   *Search for greatest eigenvalues to get greatest eigenvectors of the matrix
   * dataset transpose* dataset
   * Each eigenvalue corresponds to one principle component
   */
  std::vector<std::vector<double>> principle_components;
  for (int i = 0; i < n_principle_components; i++) {
    std::vector<double> guess(covariance_matrix.size(), 1);
    for (int i = 0; i < 10; i++) {
      guess = power_iteration(covariance_matrix, {guess});
      std::cout << guess[0] << " " << guess[1] << std::endl;
    }
    principle_components.push_back(guess);
    // deflate data eventually
    std::cout << (eigenvalue(covariance_matrix, guess));
  }

  return principle_components;
}

// Reads a CSV file into a 2D vector of strings.
inline std::vector<std::vector<std::string>>
read_csv_string(std::string file_name) {
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
