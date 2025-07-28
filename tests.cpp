#include "lib.hpp"
#include <chrono>
#include <iostream>
#include <ostream>
#include <vector>
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
}
