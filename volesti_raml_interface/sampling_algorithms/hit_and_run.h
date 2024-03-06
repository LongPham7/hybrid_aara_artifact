#ifndef HIT_AND_RUN
#define HIT_AND_RUN

#ifdef __cplusplus
extern "C" {
#endif

double *gaussian_rdhr(int const num_rows, int const num_cols,
                      double *coefficients_A, double *coefficients_b,
                      double const variance, int const num_samples,
                      int const walk_length);

double *uniform_rdhr(int const num_rows, int const num_cols,
                     double *coefficients_A, double *coefficients_b,
                     int const num_samples, int const walk_length);

double *gaussian_cdhr(int const num_rows, int const num_cols,
                      double *coefficients_A, double *coefficients_b,
                      double const variance, int const num_samples,
                      int const walk_length);

double *uniform_cdhr(int const num_rows, int const num_cols,
                     double *coefficients_A, double *coefficients_b,
                     int const num_samples, int const walk_length);

double *uniform_billiard(int const num_rows, int const num_cols,
                         double *coefficients_A, double *coefficients_b,
                         int const num_samples, int const walk_length);

#ifdef __cplusplus
}
#endif

#endif
