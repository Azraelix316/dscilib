#include "lib.hpp"
#include <chrono>
#include <iostream>
#include <ostream>
#include <vector>
double function(std::vector<double> coefficients, std::vector<double> inputs) {
  return coefficients[0] * inputs[0] + coefficients[1] * inputs[1] +
         coefficients[2];
}
int main() {
  std::vector<double> coefficients = {1.0, 1.0, 1.0};
  std::vector<std::vector<double>> inputs =
      dscilib::read_csv_double("tests.csv");
  std::vector<double> outputs = {1, 2, 4, 6, 8, 11};
  dscilib::detail::printArr(inputs);
  dscilib::detail::printVec(outputs);
  dscilib::coordinate_descent(coefficients, inputs, outputs, function, 0.5);
  dscilib::detail::printVec(coefficients);
  for (int i = 0; i < inputs.size(); i++) {
    std::cout << "\n" << function(coefficients, inputs[i]) << "  ";
  }
}
