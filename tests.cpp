#include "lib.hpp"
#include <chrono>
#include <iostream>
#include <ostream>
#include <vector>
int main() {
  std::vector<std::vector<double>> m1 = dscilib::read_csv_double("tests3.csv");
  std::vector<double> PC1;
  PC1 = dscilib::PCA(m1, 1)[0];
  std::cout << PC1[0] << "  " << PC1[1];
}
