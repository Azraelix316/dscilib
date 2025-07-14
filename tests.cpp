#include "lib.hpp"
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <iterator>
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
int main() {
  std::vector<double> inputs;
  std::vector<double> coefficients{3, 0};
  std::vector<std::vector<double>> data =
      dscilib::read_from_csv_double("tests_2.csv");
  dscilib::invert(data);
  /*for (int i = 0; i < data.size(); i++) {*/
  /*  inputs.push_back({data[i][0], data[i][1]});*/
  /*}*/
  /*std::vector<double> outputs;*/
  /*for (int i = 0; i < data.size(); i++) {*/
  /*  outputs.push_back(data[i][2]);*/
  /*}*/

  for (int i = 0; i < 100; i++) {
    dscilib::single_newton_opt(coefficients, data[0], data[1], linReg);
  }
  std::cout << coefficients[0] << "  " << coefficients[1] << std::endl;
}
