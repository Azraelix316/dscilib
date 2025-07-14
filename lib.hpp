#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
namespace dscilib {
// Up max training speed when working with large coefficients
// Lower min training speed for precision;
double MAX_TRAINING_SPEED = 100.0;
double MIN_TRAINING_SPEED = 0.001;

inline std::vector<std::vector<double>>
rotate90CW(std::vector<std::vector<double>> &matrix) {
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
invert(std::vector<std::vector<double>> &matrix) {
  matrix = rotate90CW(matrix);
  for (std::vector<double> row : matrix) {
    std::reverse(row.begin(), row.end());
  }
  return matrix;
}
template <typename inputType, typename func>
inline double ssqr(std::vector<double> &coefficients,
                   std::vector<inputType> &inputs, std::vector<double> &outputs,
                   func function) {
  long double sum = 0.0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    double pred = function(inputs[i], coefficients);
    double diff = pred - outputs[i];
    sum += diff * diff;
  }
  return sum;
}

/*
 * Coordinate descent algorithm for optimization
 * Optimizes one coefficient at a time
 * Input in the coefficients to modify, the test inputs and outputs, and the
 * predictive function. (Note that the predictive function should only input one
 * at a time.)
 */
template <typename coefficientType, typename inputType, typename outputType,
          typename function>
inline void coord_descent(std::vector<coefficientType> &coefficients,
                          std::vector<inputType> &inputs,
                          std::vector<outputType> &outputs, function &func) {

  double epsilon = 1e-6;

  for (double &coefficient : coefficients) {
    const double epsilon = 1e-6;
    coefficient += epsilon;
    double loss_plus = ssqr(coefficients, inputs, outputs, func);
    coefficient -= 2 * epsilon;
    double loss_minus = ssqr(coefficients, inputs, outputs, func);
    coefficient += epsilon;
    double gradient = (loss_plus - loss_minus) / (2.0 * epsilon);
    double speed = MAX_TRAINING_SPEED;
    double base_loss = ssqr(coefficients, inputs, outputs, func);
    double old_coefficient = coefficient;
    coefficient -= gradient * speed;
    while (ssqr(coefficients, inputs, outputs, func) > base_loss &&
           speed >= MIN_TRAINING_SPEED) {
      speed *= 0.1;
      coefficient = old_coefficient - gradient * speed;
    }
    if (ssqr(coefficients, inputs, outputs, func) > base_loss) {
      coefficient = old_coefficient - gradient * speed;
    }
  }
}

template <typename coefficientType, typename inputType, typename outputType,
          typename function>
inline void single_newton_opt(std::vector<coefficientType> &coefficients,
                              std::vector<inputType> &inputs,
                              std::vector<outputType> &outputs, function func) {
  for (double &coefficient : coefficients) {
    const double epsilon = 1e-6;
    coefficient += epsilon;
    double loss_plus = ssqr(coefficients, inputs, outputs, func);
    coefficient -= 2 * epsilon;
    double loss_minus = ssqr(coefficients, inputs, outputs, func);
    coefficient += epsilon;
    double derivative_estimate = (loss_plus - loss_minus) / (2.0 * epsilon);
    double x_1 = coefficient;
    double y_1 = ssqr(coefficients, inputs, outputs, func);
    double root = (derivative_estimate * x_1 - y_1) / derivative_estimate;
    coefficient = root;
  }
  /*
   * Estimate derivative
   * Calculate derivative at the point f'(coefficient)
   * Use point slope form to create equation y=mx-mx1+y1
   * Substitute x1,y1
   * Solve for x when y=0.0
   * Change coefficient to x
   */
}
/*
 * Data reading function from csv files
 */
inline std::vector<std::vector<std::string>>
read_from_csv_string(std::string fileName) {
  std::fstream fin;
  fin.open(fileName, std::ios::in);
  // data collection
  std::string row, temp, line, col;
  std::vector<std::vector<std::string>> data;
  std::vector<std::string> curr;
  while (getline(fin, line, '\n')) {
    std::stringstream s(line);
    // read every column and store it into col
    while (getline(s, col, ',')) {
      // add all the column data into a vector
      if (!col.empty()) {
        curr.push_back(col);
      } else {
        curr.push_back("0");
      }
    }
    data.push_back(curr);
    // pushes the vector into a 2d array data
    curr.clear();
  }
  return data;
}

inline std::vector<std::vector<double>>
read_from_csv_double(std::string fileName) {
  std::fstream fin;
  fin.open(fileName, std::ios::in);
  // data collection
  std::string row, temp, line, col;
  std::vector<std::vector<double>> data;
  std::vector<double> curr;
  while (getline(fin, line, '\n')) {
    std::stringstream s(line);
    // read every column and store it into col
    while (getline(s, col, ',')) {
      // add all the column data into a vector
      if (!col.empty()) {
        curr.push_back(stod(col));
      } else {
        curr.push_back(0);
      }
    }
    data.push_back(curr);
    // pushes the vector into a 2d array data
    curr.clear();
  }
  return data;
}

} // namespace dscilib
