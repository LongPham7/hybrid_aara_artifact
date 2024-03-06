#include <iostream>

#include "hit_and_run.h"
#include "logconcave_hmc.h"
#include "probabilistic_programming.h"
#include "probability_distributions.h"

double uniform_distribution_neg_log_prob(const double *current_state) {
  return 0;
}

double *uniform_distribution_gradient(const double *current_state) {
  double *result = new double[2];
  result[0] = 0;
  result[1] = 0;
  return result;
}

double *create_A() {
  unsigned int num_rows = 6;
  unsigned int num_cols = 2;
  double *A = new double[num_rows * num_cols];

  A[0] = -1;
  A[1] = 0;  // 0 <= x
  A[2] = 1;
  A[3] = 0;  // x <= 5
  A[4] = 0;
  A[5] = -1;  // 0 <= y
  A[6] = 0;
  A[7] = 1;  // y <= 5
  A[8] = 1;
  A[9] = -1;  // x y <= 0.01
  A[10] = -1;
  A[11] = 1;  // -x + y <= 0.01
  return A;
}

double *create_b() {
  unsigned int num_rows = 6;
  double *b = new double[num_rows];
  b[0] = 0;
  b[1] = 10;
  b[2] = 0;
  b[3] = 10;
  b[4] = 0.001;
  b[5] = 0.001;
  return b;
}

void hmc_for_testing() {
  unsigned int num_rows = 6;
  unsigned int num_cols = 2;
  double *A = create_A();
  double *b = create_b();

  double L = 4;
  double m = 4;
  unsigned int num_samples = 200;
  unsigned int walk_length = 150;
  double step_size = 1;

  double *starting_point = new double[num_cols];
  starting_point[0] = 0.9;
  starting_point[1] = 0.9;

  double *array_samples = hmc_function_pointer_interface(
      num_rows, num_cols, A, b, L, m, num_samples, walk_length, step_size,
      starting_point, uniform_distribution_neg_log_prob,
      uniform_distribution_gradient);

  unsigned int num_burns = num_samples / 2;
  unsigned int num_samples_after_burns = num_samples - num_burns;

  // Print the samples stored in array_samples
  std::cout << "Result of reflective HMC" << std::endl;
  for (auto i = 0; i != num_samples_after_burns; i++) {
    std::cout << "Sample " << i << ": ";
    for (auto j = 0; j != num_cols; j++) {
      std::cout << array_samples[i * num_cols + j] << " ";
    }
    std::cout << "\n";
  }

  // Clean up the memory
  delete[] A;
  delete[] b;
  delete[] starting_point;
}

void gaussian_rdhr_for_testing() {
  unsigned int num_rows = 6;
  unsigned int num_cols = 2;
  double *A = create_A();
  double *b = create_b();

  double variance = 36;
  unsigned int num_samples = 200;
  unsigned int walk_length = 150;

  double *array_samples = gaussian_rdhr(num_rows, num_cols, A, b, variance,
                                        num_samples, walk_length);

  // Print the samples stored in array_samples
  std::cout << "Result of Gaussian RDHR" << std::endl;
  for (auto i = 0; i != num_samples; i++) {
    std::cout << "Sample " << i << ": ";
    for (auto j = 0; j != num_cols; j++) {
      std::cout << array_samples[i * num_cols + j] << " ";
    }
    std::cout << "\n";
  }

  // Clean up the memory
  delete[] A;
  delete[] b;
}

void uniform_rdhr_for_testing() {
  unsigned int num_rows = 6;
  unsigned int num_cols = 2;
  double *A = create_A();
  double *b = create_b();

  unsigned int num_samples = 200;
  unsigned int walk_length = 150;

  double *array_samples =
      uniform_rdhr(num_rows, num_cols, A, b, num_samples, walk_length);

  // Print the samples stored in array_samples
  std::cout << "Result of uniform RDHR" << std::endl;
  for (auto i = 0; i != num_samples; i++) {
    std::cout << "Sample " << i << ": ";
    for (auto j = 0; j != num_cols; j++) {
      std::cout << array_samples[i * num_cols + j] << " ";
    }
    std::cout << "\n";
  }

  // Clean up the memory
  delete[] A;
  delete[] b;
}

int main() {
  // hmc_for_testing();
  // gaussian_rdhr_for_testing();
  // uniform_rdhr_for_testing();

  // test_gumbel();
  test_automatic_differentiation();

  return 0;
}
