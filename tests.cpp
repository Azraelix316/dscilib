#include "lib.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
std::vector<double> SIR_model_recursive(double susceptible, double infected,
                                        double recovered, double beta,
                                        double gamma, double reSus,
                                        const int &iterations) {
  std::vector<double> result(iterations);
  const double N = susceptible + infected + recovered;
  for (int i = 0; i < iterations; i++) {
    double currS = susceptible;
    double currI = infected;
    susceptible += (-beta / N * currI * currS);
    infected += (beta / N * currI * currS) - (gamma * currI);
    susceptible += reSus * recovered;
    recovered += (gamma * currI) - (reSus * recovered);
    result[i] = {infected};
  }
  return result;
}
double SIR_wrapper(std::vector<double> coefficients, double inputs) {
  return SIR_model_recursive(coefficients[0], coefficients[1], coefficients[2],
                             coefficients[3], coefficients[4], coefficients[5],
                             inputs + 1)[inputs];
}
int main() {
  srand(time(NULL));
  std::vector<double> coefficients = {1570, 1, 0, 1.821, 1.2, 0.115};
  std::vector<std::vector<double>> data = dscilib::read_csv_double("data.csv");
  dscilib::transpose(data);
  std::cout << "transposed." << std::endl;
  std::vector<double> bestCoefficients = coefficients;
  double best =
      dscilib::sum_squared_error(coefficients, data[0], data[1], SIR_wrapper);
  for (int i = 0; i < 100; i++) {
    dscilib::coordinate_descent_iter(coefficients, data[0], data[1],
                                     SIR_wrapper);
  }
  best =
      dscilib::sum_squared_error(coefficients, data[0], data[1], SIR_wrapper);
  /*for (int i = 0; i < 10000000; i++) {*/
  /*  coefficients = {(double)rand() / RAND_MAX * 2000,*/
  /*                  1,*/
  /*                  0,*/
  /*                  (double)rand() / RAND_MAX * 5,*/
  /*                  (double)rand() / RAND_MAX * 5,*/
  /*                  (double)rand() / RAND_MAX * 5};*/
  /*  if (best > dscilib::sum_squared_error(coefficients, data[0], data[1],*/
  /*                                        SIR_wrapper)) {*/
  /*    best = dscilib::sum_squared_error(coefficients, data[0], data[1],*/
  /*                                      SIR_wrapper);*/
  /*    bestCoefficients = coefficients;*/
  /*    std::cout << best << " error, iterations: " << i << std::endl;*/
  /*    dscilib::detail::printVec(bestCoefficients);*/
  /*    std::cout << std::endl;*/
  /*  }*/
  /*}*/
  bestCoefficients = coefficients;
  dscilib::detail::printVec(bestCoefficients);
  std::cout << std::endl << best << std::endl;
  for (int i = 0; i < data[0].size(); i++) {
    std::cout << SIR_wrapper(bestCoefficients, i) << "  " << data[1][i] << "  "
              << pow((SIR_wrapper(bestCoefficients, i) - data[1][i]), 2)
              << std::endl;
  }
  std::cout << dscilib::R_squared(coefficients, data[0], data[1], SIR_wrapper);
  /*for (int i = 0; i < 1; i++) {*/
  /*  dscilib::coordinate_descent_iter(coefficients, data[0], data[1],*/
  /*                                   SIR_wrapper);*/
  /*}*/
  std::fstream fout;
  fout.open("results.csv", std::ios::out);
  for (int i = 0; i < data[0].size(); i++) {
    fout << SIR_wrapper(bestCoefficients, i) << std::endl;
  }

  std::cout << "Printed";
}
