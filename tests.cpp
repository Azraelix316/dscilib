#include "lib.hpp"
#include <chrono>
#include <iostream>
#include <numeric>
#include <ostream>
#include <vector>
int main() {
  std::vector<std::vector<double>> transition_matrix;
  std::vector<double> state = {50, 50, 0};
  std::vector<double> capacity = {30, 20, 50};
  double alpha = 0.5;
  int n = capacity.size();
  std::vector<double> vec(n);
  std::vector<std::vector<double>> ones;
  for (int i = 0; i < n; i++) {
    ones.push_back({1});
  }
  // Step 1: normalize capacity to get π
  double sum = std::accumulate(capacity.begin(), capacity.end(), 0.0);
  std::vector<std::vector<double>> pi;
  for (int i = 0; i < n; ++i)
    capacity[i] = capacity[i] / sum;
  pi.push_back(capacity);
  auto I = dscilib::identity_matrix(n);
  dscilib::transpose(pi);
  dscilib::transpose(ones);
  std::vector<std::vector<double>> input1 =
      dscilib::matrix_scalar_mult(I, alpha);
  std::vector<std::vector<double>> input2 =
      dscilib::matrix_scalar_mult(dscilib::matrix_mult(pi, ones), 1 - alpha);
  transition_matrix = dscilib::add_matrices(input1, input2);
  dscilib::detail::printArr(pi);
  dscilib::detail::printArr(transition_matrix);
  for (int i = 0; i < 100; i++) {
    state = dscilib::matrix_vec_mult(transition_matrix, state);
    dscilib::detail::printVec(state);
  }
}
