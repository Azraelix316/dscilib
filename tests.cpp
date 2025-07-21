#include "lib.hpp"
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
  std::vector<std::vector<double>> data =
      dscilib::read_csv_double("tests.csv");
  data = dscilib::transpose_matrix(data);
  std::vector<double> coefficients{0, 0};
  /*for (int i = 0; i < data.size(); i++) {*/
  /*  inputs.push_back({data[i][0], data[i][1]});*/
  /*}*/
  /*std::vector<double> outputs;*/
  /*for (int i = 0; i < data.size(); i++) {*/
  /*  outputs.push_back(data[i][2]);*/
  /*}*/
  for (int i = 0; i < 100; i++) {
    dscilib::coordinate_descent(coefficients, data[0], data[1], linReg);
  }
  double x=30.0;
  for (int i=0;i<10;i++) {
    dscilib::newton_root(x, quadratic);
  }
  std::cout << "Root found: " << x << std::endl;
  std::cout << coefficients[0] << "  " << coefficients[1];
}
