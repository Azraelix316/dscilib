#include "lib.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iterator>
#include <ostream>
#include <vector>

double linReg(double input, std::vector<double> coefficients) {
  return coefficients[0] * input + coefficients[1];
}

int main() {
  std::vector<std::vector<double>> data;
  std::vector<double> coefficients{0, 0};
  srand(time(0));
  std::vector<double> initial_coefficients{static_cast<double>(rand()),
                                           static_cast<double>(rand())};
  coefficients = initial_coefficients;
  data = dscilib::read_from_csv_double("tests.csv");
  data = dscilib::transpose(data);
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10000; i++) {
    /*dscilib::coord_descent(coefficients, data[0], data[1], linReg);*/
  }
  std::vector<std::vector<double>> batched = dscilib::batch(data, 2);
  for (std::vector<double> batch : batched) {
    for (double batc : batch) {
      std::cout << batc << " ";
    }
    std::cout << std::endl;
  }
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  coefficients = initial_coefficients;
  std::cout << duration.count() << " ns" << std::endl;
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 10000; i++) {
    /*dscilib::new_coord_descent(coefficients, data[0], data[1], linReg);*/
  }
  duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  std::cout << duration.count() << " ns" << std::endl;
}
