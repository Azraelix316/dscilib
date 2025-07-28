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

  std::vector<std::vector<double>> m1 = dscilib::read_csv_double("tests3.csv");
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<double> PC1;
  std::vector<double> PC2;
  std::vector<double> PC3;
  PC1 = dscilib::PCA(m1, 3)[0];
  PC2 = dscilib::PCA(m1, 3)[1];
  PC3 = dscilib::PCA(m1, 3)[2];
  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::high_resolution_clock::now() - start);
  std::cout << duration.count() << "  ns" << std::endl;
  std::cout << PC1[0] << "  " << PC1[1] << "  " << PC1[2] << std::endl;
  std::cout << PC2[0] << "  " << PC2[1] << "  " << PC2[2] << std::endl;
  std::cout << PC3[0] << "  " << PC3[1] << "  " << PC3[2] << std::endl;
}
