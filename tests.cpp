#include "lib.hpp"
#include <chrono>
#include <ios>
#include <iostream>
#include <ostream>
#include <vector>
int main() {
  std::vector<std::vector<double>> results =
      dscilib::SIR_model(999, 1, 0, 0.3, 0.1, 300);
  std::fstream fout;
  fout.open("result.csv", std::ios::out);
  int i = 0;
  for (std::vector<double> result : results) {
    for (double val : result) {
      std::cout << val << "  ";
      fout << val << ",";
    }
    std::cout << "\n";
    fout << i << ",";
    i++;
    fout << "\n";
  }
}
