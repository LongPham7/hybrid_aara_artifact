#ifndef PROBABILITY_DISTRIBUTIONS
#define PROBABILITY_DISTRIBUTIONS

#include <autodiff/reverse/var.hpp>
using namespace autodiff;

#ifdef __cplusplus
extern "C" {
#endif

enum distribution_name { Weibull, Gumbel, Gaussian };

enum distribution_target_type {
  Individual_coefficients,
  Average_of_coefficients
};

struct coefficient_distribution_target_type {
  int* selected_coefficients;
  int num_selected_coefficients;
  distribution_target_type target_type;
};

struct distribution_type {
  // We support Weibull, Gumbel, and Gaussian distributions.
  distribution_name distribution_name;

  // Weibull distributions have parameters (alpha, sigma), Gumbel distributions
  // have parameters (mu, beta), and Gaussian distributions have parameters (mu,
  // sigma).
  double first_parameter;
  double second_parameter;
};

var weibull_log_pdf(const var& alpha, const var& sigma, const var& x);
var gumbel_log_pdf(const var& mu, const var& beta, const var& x);
var gaussian_log_pdf(const var& mu, const var& sigma, const var& x);

var log_pdf_of_given_distribution(const distribution_type& distribution,
                                  const var& x);
void print_distribution(distribution_type distribution);

void test_weibull(void);
void test_gumbel(void);
void test_gaussian(void);

#ifdef __cplusplus
}
#endif

#endif
