#include <vector>
/*
 * Coordinate descent algorithm for optimization
 * Optimizes one coefficient at a time
 * Input in the coefficients to modify, the test inputs and outputs, and the
 * predictive function. (Note that the predictive function should only input one
 * at a time.)
 */
namespace dscilib {
inline double ssqr(std::vector<double> &coefficients,
                   std::vector<double> &inputs,
                   double (*func)(double, std::vector<double>)) {
  long double sum = 0.0;
  for (double &input : inputs) {
    sum += func(input, coefficients);
  }
  return sum;
}

inline void adaptive_coord_descent(
    std::vector<double> &coefficients, std::vector<double> &inputs,
    std::vector<double> &outputs, double (*func)(double, std::vector<double>)) {
  for (double &coefficient : coefficients) {
    double speed = 100.0;
    while (speed > 0.01) {
      double least = ssqr(coefficients, inputs, func);
      while (least >= ssqr(coefficients, inputs, func)) {
        coefficient += speed;
        least = ssqr(coefficients, inputs, func);
      }
      least = ssqr(coefficients, inputs, func);
      while (least >= ssqr(coefficients, inputs, func)) {
        coefficient -= speed;
        least = ssqr(coefficients, inputs, func);
      }
      speed *= 0.1;
    }
  }
}

inline void adaptive_coord_descent(std::vector<double> &coefficients,
                                   std::vector<double> &inputs,
                                   std::vector<double> &outputs,
                                   double (*func)(std::vector<double>,
                                                  std::vector<double>)) {
  coefficients.push_back(0);
}

inline void adaptive_coord_descent(
    std::vector<double> &coefficients, std::vector<double> &inputs,
    std::vector<double> &outputs,
    std::vector<double> (*func)(double, std::vector<double>)) {
  coefficients.push_back(0);
}
inline void adaptive_coord_descent(
    std::vector<double> &coefficients, std::vector<double> &inputs,
    std::vector<double> &outputs,
    std::vector<double> (*func)(std::vector<double>, std::vector<double>)) {
  coefficients.push_back(0);
}

} // namespace dscilib
