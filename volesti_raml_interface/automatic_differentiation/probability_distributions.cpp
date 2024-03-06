#include "probability_distributions.h"

#include <string.h>

#include <autodiff/reverse/var.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace autodiff;

// Constant log (sqrt (2 * pi)). It is used in the calculation of the Gaussian
// log pdf.
#define LOG_SQRT_TWO_PI 0.91893853320467267

var weibull_log_pdf(const var& alpha, const var& sigma, const var& x) {
  var negative_infinity = -std::numeric_limits<double>::infinity();
  var x_normalized = x / sigma;
  return condition(
      x < 0, negative_infinity,
      condition(x == 0, condition(alpha == 1, -log(sigma), negative_infinity),
                condition(alpha == 1, -(x / sigma) - log(sigma),
                          log(alpha) - log(sigma) - pow(x_normalized, alpha) +
                              (alpha - 1) * log(x_normalized))));
}

var gumbel_log_pdf(const var& mu, const var& beta, const var& x) {
  var x_normalized = (x - mu) / beta;
  return -log(beta) - x_normalized - exp(-x_normalized);
}

var gaussian_log_pdf(const var& mu, const var& sigma, const var& x) {
  var x_normalized = (x - mu) / sigma;
  return -(x_normalized * x_normalized / 2) - LOG_SQRT_TWO_PI - log(sigma);
}

var log_pdf_of_given_distribution(const distribution_type& distribution,
                                  const var& x) {
  distribution_name distribution_name = distribution.distribution_name;
  double first_parameter = distribution.first_parameter;
  double second_parameter = distribution.second_parameter;

  if (distribution_name == Weibull) {
    return weibull_log_pdf(first_parameter, second_parameter, x);
  } else if (distribution_name == Gumbel) {
    return gumbel_log_pdf(first_parameter, second_parameter, x);
  } else if (distribution_name == Gaussian) {
    return gaussian_log_pdf(first_parameter, second_parameter, x);
  } else {
    throw std::invalid_argument("The given distribution type is invalid.");
  }
}

void print_distribution(distribution_type distribution) {
  distribution_name distribution_name = distribution.distribution_name;
  double first_parameter = distribution.first_parameter;
  double second_parameter = distribution.second_parameter;
  if (distribution_name == Weibull) {
    std::cout << "Weibull(alpha = " << first_parameter
              << ", sigma = " << second_parameter << ")";
  } else if (distribution_name == Gumbel) {
    std::cout << "Gumbel(mu = " << first_parameter
              << ", beta = " << second_parameter << ")";
  } else if (distribution_name == Gaussian) {
    std::cout << "Gaussian(mu = " << first_parameter
              << ", sigma = " << second_parameter << ")";
  } else {
    throw std::invalid_argument("The given distribution type is invalid.");
  }
}

// Sanity check of a Weibull distribution
void test_weibull() {
  var alpha = 1;
  var sigma = 6;
  var x = 1.1;
  var y = weibull_log_pdf(alpha, sigma, x);
  auto [pdf_gradient] = derivatives(y, wrt(x));

  std::cout << "Log pdf of Weibull(alpha = " << alpha << ", sigma = " << sigma
            << ") at x = " << x << ": " << y << std::endl;
  std::cout << "Log pdf gradient of Weibull(alpha = " << alpha
            << ", sigma = " << sigma << ") at x = " << x << ": " << pdf_gradient
            << std::endl;
}

// Sanity check of a Gumbel distribution
void test_gumbel() {
  var mu = 0;
  var beta = 1;
  var x = 1.1;
  var y = gumbel_log_pdf(mu, beta, x);
  auto [pdf_gradient] = derivatives(y, wrt(x));

  std::cout << "Log pdf of Gumbel(mu = " << mu << ", beta = " << beta
            << ") at x = " << x << ": " << y << std::endl;
  std::cout << "Log pdf gradient of Gumbel(mu = " << mu << ", beta = " << beta
            << ") at x = " << x << ": " << pdf_gradient << std::endl;
}

// Sanity check of a Gaussian distribution
void test_gaussian() {
  var mu = 0;
  var sigma = 2;
  var x = 1.1;
  var y = gaussian_log_pdf(mu, sigma, x);
  auto [pdf_gradient] = derivatives(y, wrt(x));

  std::cout << "Log pdf of Gaussian(mu = " << mu << ", sigma = " << sigma
            << ") at x = " << x << ": " << y << std::endl;
  std::cout << "Log pdf gradient of Gaussian(mu = " << mu
            << ", sigma = " << sigma << ") at x = " << x << ": " << pdf_gradient
            << std::endl;
}
