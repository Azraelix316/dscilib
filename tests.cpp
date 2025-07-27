#include "lib.hpp"
#include <chrono>
#include <iostream>
#include <ostream>
#include <vector>

/*double linReg(double input, std::vector<double> coefficients) {*/
/*  return coefficients[0] * input + coefficients[1];*/
/*}*/
/**/
/*int main() {*/
/*  std::vector<std::vector<double>> data;*/
/*  std::vector<double> coefficients{0, 0};*/
/*  srand(time(0));*/
/*  std::vector<double> initial_coefficients{static_cast<double>(rand()),*/
/*                                           static_cast<double>(rand())};*/
/*  coefficients = initial_coefficients;*/
/*  data = dscilib::read_from_csv_double("tests.csv");*/
/*  data = dscilib::rotate90CW(data);*/
/*  auto start = std::chrono::high_resolution_clock::now();*/
/*  for (int i = 0; i < 10000; i++) {*/
/*    dscilib::coord_descent(coefficients, data[0], data[1], linReg);*/
/*  }*/
/*  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(*/
/*      std::chrono::high_resolution_clock::now() - start);*/
/*  std::cout << coefficients[0] << "  " << coefficients[1] << std::endl;*/
/*}*/
double linReg(double input, std::vector<double> coefficients) {
  return coefficients[0] * input + coefficients[1];
}
double quadratic(double input) {
  return input * input * input - 3 * input + 2 * input * input;
}
int main() {

  std::vector<std::vector<double>> m1{
      {2.5, 2.4}, {0.5, 0.7}, {2.2, 2.9}, {1.9, 2.2}, {3.1, 3}};

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100000; i++)
    std::vector<double> PC1 = dscilib::PCA(m1, 2)[0];
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  std::cout << duration.count() << "  ns" << std::endl;

  auto start2 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 100000; i++)
    std::vector<double> PC1fast = dscilib::NEW_PCA(m1, 2)[0];
  auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start2);
  std::cout << duration2.count() << "  ns" << std::endl;
}
