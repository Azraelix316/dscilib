#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>
namespace dscilib {
// Up max training speed when working with large coefficients
// Lower min training speed for precision;
const double MAX_TRAINING_SPEED = 100.0;
const double MIN_TRAINING_SPEED = 0.01;

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

inline double ssqr(const std::vector<double> &coefficients,
                   const std::vector<double> &inputs,
                   const std::vector<double> &outputs,
                   double (*func)(double, const std::vector<double>)) {
  long double sum = 0.0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    double pred = func(inputs[i], coefficients);
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

inline void adaptive_coord_descent(
    std::vector<double> &coefficients, std::vector<double> &inputs,
    std::vector<double> &outputs, double (*func)(double, std::vector<double>)) {
  for (double &coefficient : coefficients) {
    double speed = MAX_TRAINING_SPEED;
    while (speed >= MIN_TRAINING_SPEED) {
      double least = ssqr(coefficients, inputs, outputs, func);
      while (least >= ssqr(coefficients, inputs, outputs, func)) {
        coefficient -= speed;
        least = std::min(ssqr(coefficients, inputs, outputs, func), least);
      }
      least = ssqr(coefficients, inputs, outputs, func);
      while (least >= ssqr(coefficients, inputs, outputs, func)) {
        coefficient += speed;
        least = std::min(ssqr(coefficients, inputs, outputs, func), least);
      }
      speed *= 0.1;
    }
  }
}

inline void adaptive_coord_descent(std::vector<double> &coefficients,
                                   std::vector<double> &inputs,
                                   std::vector<double> &outputs,
                                   double (*func)(std::vector<double>,
                                                  std::vector<double>)) {
  coefficients.push_back(0);
}

inline void adaptive_coord_descent(
    std::vector<double> &coefficients, std::vector<double> &inputs,
    std::vector<double> &outputs,
    std::vector<double> (*func)(double, std::vector<double>)) {
  coefficients.push_back(0);
}
inline void adaptive_coord_descent(
    std::vector<double> &coefficients, std::vector<double> &inputs,
    std::vector<double> &outputs,
    std::vector<double> (*func)(std::vector<double>, std::vector<double>)) {
  coefficients.push_back(0);
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
