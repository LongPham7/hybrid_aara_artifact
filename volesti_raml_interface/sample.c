#include <stdio.h>
#include <stdlib.h>

#include "hit_and_run.h"
#include "logconcave_hmc.h"

double uniform_distribution_neg_log_prob(const double *current_state) {
  return 0;
}

double *uniform_distribution_gradient(const double *current_state) {
  double *result = malloc(sizeof(double) * 2);
  result[0] = 0;
  result[1] = 0;
  return result;
}

double *create_A() {
  unsigned int num_rows = 6;
  unsigned int num_cols = 2;
  double *A = malloc(sizeof(double) * num_rows * num_cols);
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
  double *b = malloc(sizeof(double) * num_rows);
  b[0] = 0;
  b[1] = 10;
  b[2] = 0;
  b[3] = 10;
  b[4] = 0.001;
  b[5] = 0.001;
  return b;
}

void hmc() {
  unsigned int num_rows = 6;
  unsigned int num_cols = 2;
  double *A = create_A();
  double *b = create_b();

  double L = 4;
  double m = 4;
  unsigned int num_samples = 200;
  unsigned int walk_length = 150;
  double step_size = 1;

  double *starting_point = malloc(sizeof(double) * num_cols);
  starting_point[0] = 0.9;
  starting_point[1] = 0.9;

  double *array_samples =
      run_hmc(num_rows, num_cols, A, b, L, m, num_samples, walk_length,
              step_size, starting_point, uniform_distribution_neg_log_prob,
              uniform_distribution_gradient);

  unsigned int num_burns = num_samples / 2;
  unsigned int num_samples_after_burns = num_samples - num_burns;

  // Print the samples stored in array_samples
  printf("Result of reflective HMC\n");
  for (int i = 0; i != num_samples_after_burns; i++) {
    printf("Sample %i: ", i);
    for (int j = 0; j != num_cols; j++) {
      printf("%f ", array_samples[i * num_cols + j]);
    }
    printf("\n");
  }

  // Clean up the memory
  free(A);
  free(b);
  free(starting_point);
}

void gaussian_rdhr() {
  unsigned int num_rows = 6;
  unsigned int num_cols = 2;
  double *A = create_A();
  double *b = create_b();

  double variance = 36;
  unsigned int num_samples = 200;
  unsigned int walk_length = 150;

  double *array_samples = run_gaussian_rdhr(num_rows, num_cols, A, b, variance,
                                            num_samples, walk_length);

  // Print the samples stored in array_samples
  printf("Result of Gaussian RDHR\n");
  for (int i = 0; i != num_samples; i++) {
    printf("Sample %i: ", i);
    for (int j = 0; j != num_cols; j++) {
      printf("%f ", array_samples[i * num_cols + j]);
    }
    printf("\n");
  }

  // Clean up the memory
  free(A);
  free(b);
}

void uniform_rdhr() {
  unsigned int num_rows = 6;
  unsigned int num_cols = 2;
  double *A = create_A();
  double *b = create_b();

  unsigned int num_samples = 200;
  unsigned int walk_length = 150;

  double *array_samples =
      run_uniform_rdhr(num_rows, num_cols, A, b, num_samples, walk_length);

  // Print the samples stored in array_samples
  printf("Result of uniform RDHR\n");
  for (int i = 0; i != num_samples; i++) {
    printf("Sample %i: ", i);
    for (int j = 0; j != num_cols; j++) {
      printf("%f ", array_samples[i * num_cols + j]);
    }
    printf("\n");
  }

  // Clean up the memory
  free(A);
  free(b);
}

int main() {
  hmc();
  // gaussian_rdhr();
  // uniform_rdhr();
  return 0;
}
