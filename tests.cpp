#include "lib.hpp"
#include <chrono>
#include <iostream>
#include <ostream>
#include <vector>
double function(std::vector<double> coefficients, double inputs) {
  return coefficients[0] * inputs + coefficients[1];
}
int main() {
  std::vector<double> coefficients = {1.0, 1.0};
  std::vector<double> inputs =
      dscilib::to_transposed(dscilib::read_csv_double("tests.csv"))[0];
  std::vector<double> outputs = {1, 2, 4, 6, 8, 11};
  /*for (int i = 0; i < 1000; i++) {*/
  /*  dscilib::coordinate_descent_iter(coefficients, inputs, outputs,
   * function);*/
  /*}*/
  dscilib::coordinate_descent(coefficients, inputs, outputs, function, 0.49);
  std::cout << coefficients[0] << "  " << coefficients[1];
  for (int i = 0; i < inputs.size(); i++) {
    std::cout << "\n" << function(coefficients, inputs[i]) << "  ";
  }
}
