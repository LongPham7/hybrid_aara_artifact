#include <iostream>

// Volesti include
#include "random_walks/random_walks.hpp"

// autodiff include
#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>
using namespace autodiff;

#include "probabilistic_programming.h"
#include "probability_distributions.h"
#include "runtime_data.h"

probability_distribution_statistical_aara::
    probability_distribution_statistical_aara(
        runtime_data_sample *runtime_data_, int num_samples_, int dim_,
        distribution_type coefficient_distribution_,
        distribution_type cost_model_,
        coefficient_distribution_target_type coefficient_distribution_target_,
        distribution_target_type cost_model_target_)
    : runtime_data(runtime_data_),
      num_samples(num_samples_),
      dim(dim_),
      coefficient_distribution(coefficient_distribution_),
      cost_model(cost_model_),
      coefficient_distribution_target(coefficient_distribution_target_),
      cost_model_target(cost_model_target_) {}

/* Calculate the log pdf of coefficients where coefficient_distribution applies
to individual coefficients. That is, each coefficient is associated with
coefficient_distribution. The total log pdf is given by the sum of individual
coefficients' log pdf. */
var probability_distribution_statistical_aara::coefficient_log_pdf_individual(
    const ArrayXvar &x) {
  var cumulative_log_pdf = 0;

  if (coefficient_distribution_target.num_selected_coefficients == -1) {
    for (auto i = 0; i != dim; i++) {
      cumulative_log_pdf +=
          log_pdf_of_given_distribution(coefficient_distribution, x[i]);
    }
  } else {
    for (auto i = 0;
         i != coefficient_distribution_target.num_selected_coefficients; i++) {
      int coefficient_index =
          coefficient_distribution_target.selected_coefficients[i];
      cumulative_log_pdf += log_pdf_of_given_distribution(
          coefficient_distribution, x[coefficient_index]);
    }
  }
  return cumulative_log_pdf;
}

/* Calculate the log pdf of coefficients where coefficient_distribution applies
to the average of all coefficients. That is, we first calculate the average
coefficient and then calculate its log pdf with respect to
coefficient_distribution. */
var probability_distribution_statistical_aara::coefficient_log_pdf_average(
    const ArrayXvar &x) {
  return log_pdf_of_given_distribution(coefficient_distribution, x.sum() / dim);
}

/* Calculate the log pdf of cost gaps where cost_model applies to individual
cost gaps. */
var probability_distribution_statistical_aara::cost_gap_log_pdf_individual(
    const ArrayXvar &x) {
  var cumulative_log_pdf = 0;

  for (auto i = 0; i != num_samples; i++) {
    const runtime_data_sample &current_sample{runtime_data[i]};
    const int *array_cindices = current_sample.array_cindices;
    const double *potential_of_cindices = current_sample.potential_of_cindices;
    const int num_cindices = current_sample.num_cindices;

    const int *array_indices = current_sample.array_indices;
    const double *potential_of_indices = current_sample.potential_of_indices;
    const int num_indices = current_sample.num_indices;

    const double cost = current_sample.cost;

    var cumulative_input_potential = 0;
    for (auto j = 0; j != num_cindices; j++) {
      int index = array_cindices[j];
      double potential = potential_of_cindices[j];
      cumulative_input_potential += x[index] * potential;
    }

    var cumulative_output_potential = 0;
    for (auto j = 0; j != num_indices; j++) {
      int index = array_indices[j];
      double potential = potential_of_indices[j];
      cumulative_output_potential += x[index] * potential;
    }

    cumulative_log_pdf += log_pdf_of_given_distribution(
        cost_model,
        cumulative_input_potential - cumulative_output_potential - cost);
  }

  return cumulative_log_pdf;
}

/* Calculate the log pdf of cost gaps where cost_model applies to the average
cost gap. cost_gap_log_pdf_average yields significantly faster automatic
differentiation than cost_gap_log_pdf_individual. cost_gap_log_pdf_average
calculates log_pdf_of_given_distribution(cost model, ...) only once at the end
of the function body, while cost_gap_log_pdf_individual needs to calculate
log_pdf_of_given_distribution(cost model, ...) for each cost gap. Because
log_pdf_of_given_distribution(cost model, ...) is (most likely) computationally
expensive, the difference between cost_gap_log_pdf_individual and
cost_gap_log_pdf_average in their numbers of calculating
log_pdf_of_given_distribution leads to a significant difference in their running
time. */
var probability_distribution_statistical_aara::cost_gap_log_pdf_average(
    const ArrayXvar &x) {
  var cumulative_cost_gaps = 0;

  for (auto i = 0; i != num_samples; i++) {
    const runtime_data_sample &current_sample{runtime_data[i]};
    const int *array_cindices = current_sample.array_cindices;
    const double *potential_of_cindices = current_sample.potential_of_cindices;
    const int num_cindices = current_sample.num_cindices;

    const int *array_indices = current_sample.array_indices;
    const double *potential_of_indices = current_sample.potential_of_indices;
    const int num_indices = current_sample.num_indices;

    const double cost = current_sample.cost;

    var cumulative_input_potential = 0;
    for (auto j = 0; j != num_cindices; j++) {
      int index = array_cindices[j];
      double potential = potential_of_cindices[j];
      cumulative_input_potential += x[index] * potential;
    }

    var cumulative_output_potential = 0;
    for (auto j = 0; j != num_indices; j++) {
      int index = array_indices[j];
      double potential = potential_of_indices[j];
      cumulative_output_potential += x[index] * potential;
    }

    cumulative_cost_gaps +=
        cumulative_input_potential - cumulative_output_potential - cost;
  }
  return log_pdf_of_given_distribution(cost_model,
                                       cumulative_cost_gaps / num_samples);
}

var probability_distribution_statistical_aara::log_pdf(const ArrayXvar &x) {
  // assert(x.rows() == dim);
  var coefficient_log_pdf;
  if (coefficient_distribution_target.target_type == Individual_coefficients) {
    coefficient_log_pdf = coefficient_log_pdf_individual(x);
  } else if (coefficient_distribution_target.target_type ==
             Average_of_coefficients) {
    coefficient_log_pdf = coefficient_log_pdf_average(x);
  } else {
    std::invalid_argument(
        "The given coefficient distribution target is invalid.");
  }

  var cost_gap_log_pdf;
  if (cost_model_target == Individual_coefficients) {
    cost_gap_log_pdf = cost_gap_log_pdf_individual(x);
  } else if (cost_model_target == Average_of_coefficients) {
    cost_gap_log_pdf = cost_gap_log_pdf_average(x);
  } else {
    std::invalid_argument(
        "The given coefficient distribution target is invalid.");
  }
  return coefficient_log_pdf + cost_gap_log_pdf;
}

double probability_distribution_statistical_aara::log_pdf_point_interface(
    const Point &point_x) {
  ArrayXvar x{point_x.getCoefficients()};
  var y = log_pdf(x);
  return (double)y;
}

Eigen::VectorXd probability_distribution_statistical_aara::gradient_log_pdf(
    ArrayXvar x) {
  // assert(x.rows() == dim);
  var y = log_pdf(x);
  return gradient(y, x);
}

Point probability_distribution_statistical_aara::
    gradient_log_pdf_point_interface(Point point_x) {
  ArrayXvar x{point_x.getCoefficients()};
  Eigen::VectorXd dydx = gradient_log_pdf(x);
  Point point_dydx{dydx};
  return point_dydx;
}

void test_automatic_differentiation() {
  runtime_data_sample *runtime_data_for_testing =
      create_runtime_data_for_testing();
  distribution_type coefficient_distribution{Gaussian, 0, 0.2};
  distribution_type cost_model{Weibull, 1, 6};
  coefficient_distribution_target_type coefficient_distribution_target{
      nullptr, -1, Individual_coefficients};
  probability_distribution_statistical_aara
      probability_distribution_for_testing{
          runtime_data_for_testing, 5,          3,
          coefficient_distribution, cost_model, coefficient_distribution_target,
          Individual_coefficients};

  ArrayXvar x(3);
  x << 0.01, 1.1, 0.1;
  var y = probability_distribution_for_testing.log_pdf(x);
  Eigen::VectorXd dydx =
      probability_distribution_for_testing.gradient_log_pdf(x);
  std::cout << "Log pdf for testing: " << y << std::endl;
  std::cout << "Log pdf's gradient for testing:\n" << dydx << std::endl;
}

probability_distribution_cost_data_categorized_by_sizes::
    probability_distribution_cost_data_categorized_by_sizes(
        int num_size_categories_, int *array_sizes_of_categories_,
        double *array_costs_, distribution_type cost_model_)
    : num_size_categories(num_size_categories_),
      array_sizes_of_categories(array_sizes_of_categories_),
      array_costs(array_costs_),
      cost_model(cost_model_) {}

var probability_distribution_cost_data_categorized_by_sizes::log_pdf(
    const ArrayXvar &x) {
  var cumulative_log_pdf = 0;
  int cumulative_index = 0;
  for (auto i = 0; i < num_size_categories; i++) {
    for (auto j = 0; j < array_sizes_of_categories[i]; j++) {
      double cost = array_costs[cumulative_index];
      cumulative_index++;
      cumulative_log_pdf +=
          log_pdf_of_given_distribution(cost_model, x[i] - cost);
    }
  }

  return cumulative_log_pdf;
}

double probability_distribution_cost_data_categorized_by_sizes::
    log_pdf_point_interface(const Point &point_x) {
  ArrayXvar x{point_x.getCoefficients()};
  var y = log_pdf(x);
  return (double)y;
}

Eigen::VectorXd
probability_distribution_cost_data_categorized_by_sizes::gradient_log_pdf(
    ArrayXvar x) {
  var y = log_pdf(x);
  return gradient(y, x);
}

Point probability_distribution_cost_data_categorized_by_sizes::
    gradient_log_pdf_point_interface(Point point_x) {
  ArrayXvar x{point_x.getCoefficients()};
  Eigen::VectorXd dydx = gradient_log_pdf(x);
  Point point_dydx{dydx};
  return point_dydx;
}
