// VolEsti (volume computation and sampling library)

// Copyright (c) 2012-2020 Vissarion Fisikopoulos
// Copyright (c) 2018-2020 Apostolos Chalkis
// Copyright (c) 2020-2020 Marios Papachristou

// Modified by Long Pham

// Licensed under GNU LGPL.3, see LICENCE file

#include "logconcave_hmc.h"

#include <iostream>
#include <vector>

#include "create_polytope.h"
#include "ode_solvers/ode_solvers.hpp"
#include "probabilistic_programming.h"
#include "probability_distributions.h"
#include "random_walks/random_walks.hpp"
#include "runtime_data.h"

// #define DEBUG

// HMCFunctor_function_pointer_interface's code is attributed to the following
// source files in volesti's project directory: (i)
// volesti/examples/logconcave/simple_hmc.cpp and (ii)
// volesti/R-proj/src/oracle_functors_rcpp.h.
struct HMCFunctor_function_pointer_interface {
  template <typename NT>
  struct parameters {
    int dim;    // Dimension
    int order;  // For HMC, the order is 2
    NT L;       // Lipschitz constant for gradient
    NT m;       // Strong convexity constant
    NT kappa;   // Condition number

    parameters(int dim_, NT L_, NT m_)
        // The value of kappa is set to L / m in
        // R-proj/src/oracle_functors_rcpp.h, so I adopt the same definition.
        : dim(dim_), order(2), L(L_), m(m_), kappa(L_ / m_) {}
  };

  // Functor representing the negative log pdf of the prior probability
  // distribution in Bayesian inference. Note that it is "negative" log pdf,
  // rather than the ordinary, positive log pdf. This can be seen from how
  // FunctionFunctor is implemented in
  // volesti/R-proj/src/oracle_functors_rcpp.h, which serves as the R-interface
  // of volesti.
  template <typename Point>
  struct FunctionFunctor {
    typedef typename Point::FT NT;

    parameters<NT> &params;

    // Function pointer to the negative log pdf
    NT(*neg_log_pdf)
    (const NT *);

    FunctionFunctor(parameters<NT> &params_, NT (*neg_log_pdf_)(const NT *))
        : params(params_), neg_log_pdf(neg_log_pdf_) {}

    // The index i represents the state vector index
    NT operator()(Point const &x) const {
      const NT *current_point_array = x.getCoefficients().data();
      return neg_log_pdf(current_point_array);
    }
  };

  // Functor representing the gradient of the (positive) log pdf of the prior
  // distribution. Bizarrely, this is the gradient of the "positive" log pdf,
  // while FunctionFunctor captures the "negative" log pdf. The fact that
  // GradientFunctor captures the positive log pdf's gradient can be seen from
  // volesti/R-proj/src/oracle_functors_rcpp.h, where GradientFunctor negates
  // the negative log pdf's gradient.
  template <typename Point>
  struct GradientFunctor {
    typedef typename Point::FT NT;
    typedef std::vector<Point> pts;

    parameters<NT> &params;

    // Function pointer for the gradient of the (positive) log pdf
    NT *(*gradient_log_pdf)(const NT *);

    GradientFunctor(parameters<NT> &params_,
                    NT *(*gradient_log_pdf_)(const NT *))
        : params(params_), gradient_log_pdf(gradient_log_pdf_) {}

    // The index i represents the state vector index
    Point operator()(int const &i, pts const &xs, NT const &t) const {
      if (i == params.order - 1) {
        const NT *current_point_array = xs[0].getCoefficients().data();
        const NT *gradient_array = gradient_log_pdf(current_point_array);
        Point gradient_point(params.dim);
        for (auto i = 0; i != params.dim; i++) {
          gradient_point.set_coord(i, gradient_array[i]);
        }
        // If we run this file from OCaml and comment out the following line,
        // the double-free error happens. So the output (i.e. a pointer to an
        // array) of neg_gradient_log_pdf should not be deleted here by us, but
        // by the caller.
        // delete[] neg_gradient_array;
        return gradient_point;
      } else {
        return xs[i + 1];  // returns velocity (i.e. the derivative of x)
      }
    }
  };
};

template <typename probability_distribution_type>
struct HMCFunctor_probability_distribution_interface {
  template <typename NT>
  struct parameters {
    int dim;    // Dimension
    int order;  // For HMC, the order is 2
    NT L;       // Lipschitz constant for gradient
    NT m;       // Strong convexity constant
    NT kappa;   // Condition number

    parameters(int dim_, NT L_, NT m_)
        // The value of kappa is set to L / m in
        // R-proj/src/oracle_functors_rcpp.h, so I adopt the same definition.
        : dim(dim_), order(2), L(L_), m(m_), kappa(L_ / m_) {}
  };

  // Functor representing the negative log pdf of the prior probability
  // distribution in Bayesian inference.
  template <typename Point>
  struct FunctionFunctor {
    typedef typename Point::FT NT;

    parameters<NT> &params;

    // A struct capturing the runtime data
    probability_distribution_type &probability_distribution;

    FunctionFunctor(parameters<NT> &params_,
                    probability_distribution_type &probability_distribution_)
        : params(params_),
          probability_distribution(probability_distribution_) {}

    // The index i represents the state vector index
    NT operator()(Point const &x) const {
      // Note that we must negate the log pdf because Volesti requires the
      // "negative" log pdf, instead of the positive one.
      return (-1) * probability_distribution.log_pdf_point_interface(x);
    }
  };

  // Functor representing the gradient of the (positive) log pdf of the prior
  // distribution.
  template <typename Point>
  struct GradientFunctor {
    typedef typename Point::FT NT;
    typedef std::vector<Point> pts;

    parameters<NT> &params;

    // A struct capturing the runtime data
    probability_distribution_type &probability_distribution;

    GradientFunctor(parameters<NT> &params_,
                    probability_distribution_type &probability_distribution_)
        : params(params_),
          probability_distribution(probability_distribution_) {}

    // The index i represents the state vector index
    Point operator()(int const &i, pts const &xs, NT const &t) const {
      if (i == params.order - 1) {
        return probability_distribution.gradient_log_pdf_point_interface(xs[0]);
      } else {
        return xs[i + 1];  // returns velocity (i.e. the derivative of x)
      }
    }
  };
};

template <typename NegativeLogPDFunctor, typename GradientFunctor>
double *hmc_core(int const num_rows, int const num_cols, double *coefficients_A,
                 double *coefficients_b, double const L, double const m,
                 int const num_samples, int const walk_length,
                 double const step_size, double *starting_point,
                 NegativeLogPDFunctor f, GradientFunctor F) {
  typedef double NT;
  typedef Cartesian<NT> Kernel;
  typedef typename Kernel::Point Point;
  typedef HPolytope<Point> Hpolytope;
  typedef typename Hpolytope::MT MT;
  typedef typename Hpolytope::VT VT;
  typedef BoostRandomNumberGenerator<boost::mt19937, NT> RandomNumberGenerator;

  int dim = num_cols;
  RandomNumberGenerator rng(dim);

  // Define a polytope
  Hpolytope P =
      create_polytope(num_rows, num_cols, coefficients_A, coefficients_b);

#ifdef DEBUG
  P.print();
#endif

  // It is important to compute the inner ball of P. It normalizes P via the
  // normalize() function. Without this normalization, HMC does work properly.
  std::pair<Point, NT> InnerBall = P.ComputeInnerBall();
  if (InnerBall.second < 0.0) {
    throw std::invalid_argument("The linear program is infeasible in hmc_core");
  }

#ifdef DEBUG
  std::cerr << "Chebyshev center of polytope P in reflective HMC: "
            << InnerBall.first.getCoefficients().transpose() << std::endl;
#endif

  // Define a starting point
  Point x0(dim);
  for (auto i = 0; i != dim; i++) {
    x0.set_coord(i, starting_point[i]);
  }

  if (P.is_in(x0) == 0) {
    std::cerr << "Starting point of reflective HMC: "
              << x0.getCoefficients().transpose() << std::endl;
    throw std::invalid_argument(
        "The starting point is not inside the polytope.");
  }

  int num_burns = num_samples / 2;  // Half will be burned
  int num_samples_after_burns = num_samples - num_burns;

  // If the Chebyshev ball's radius is too small (basically zero), it means P's
  // feasible region has a strictly lower dimension than the full state space.
  // Geometrically, the feasible region is flat. One example is a linear program
  // with an equality constraint between two LP variables. In such a case,
  // random-walk-based sampling algorithms (e.g. RDHR and reflective HMC) don't
  // work well, because it is very easy to accidentally exit P. Instead of
  // running RDHR, we simply return the Chebyshev center.
  if (InnerBall.second < CHEBYSHEV_RADIUS_EPSILON) {
    std::cerr << "The Chebyshev ball's radius is too small: "
              << InnerBall.second << std::endl;
    std::cerr
        << "So we return the starting point instead of running reflective HMC"
        << std::endl;
    double *array_samples = new double[num_samples * dim];
    for (auto i = 0; i < num_samples_after_burns; i++) {
      for (auto j = 0; j < dim; j++) {
        array_samples[i * dim + j] = x0.getCoefficients()(j);
      }
    }
    return array_samples;
  }

  // Define HMC walk
  HamiltonianMonteCarloWalk::parameters<NT, GradientFunctor> hmc_params(F, dim);
  hmc_params.eta = step_size;
  typedef LeapfrogODESolver<Point, NT, Hpolytope, GradientFunctor> Solver;
  HamiltonianMonteCarloWalk::Walk<Point, Hpolytope, RandomNumberGenerator,
                                  GradientFunctor, NegativeLogPDFunctor, Solver>
      hmc_walk(&P, x0, F, f, hmc_params);

#ifdef DEBUG
  hmc_walk.disable_adaptive();
#endif

  // Samples drawn from the HMC sampler are stored in array_samples.
  double *array_samples = new double[num_samples_after_burns * dim];

  int num_of_blocks_for_reports = 10;
  int block_size_for_reports = (num_samples >= num_of_blocks_for_reports)
                                   ? (num_samples / num_of_blocks_for_reports)
                                   : 1;

  // Perform HMC
  for (auto i = 0; i < num_samples; i++) {
    hmc_walk.apply(rng, walk_length);

    // Report how many blocks of HMC iterations have been finished
    if (i % block_size_for_reports == 0) {
      std::cout << (i / block_size_for_reports) << " blocks out of "
                << num_of_blocks_for_reports << " in HMC are finished"
                << std::endl;
    }

    if (i >= num_burns) {
      const typename Point::Coeff sample = hmc_walk.x.getCoefficients();
      for (auto j = 0; j != dim; j++) {
        array_samples[(i - num_burns) * dim + j] = sample(j);
      }
    }

    // print_sample(num_cols, hmc_walk.x.getCoefficients());
  }

  std::cerr << "Reflective HMC statistics:" << std::endl;
  std::cerr << "Step size (final): " << hmc_walk.solver->eta << std::endl;
  std::cerr << "Discard Ratio: " << hmc_walk.discard_ratio << std::endl;
  std::cerr << "Average Acceptance Log-prob: "
            << exp(hmc_walk.average_acceptance_log_prob) << std::endl;

  return array_samples;
}

double *hmc_function_pointer_interface(
    int const num_rows, int const num_cols, double *coefficients_A,
    double *coefficients_b, double const L, double const m,
    int const num_samples, int const walk_length, double const step_size,
    double *starting_point, double (*neg_log_pdf)(const double *),
    double *(*gradient_log_pdf)(const double *)) {
  typedef double NT;
  typedef Cartesian<NT> Kernel;
  typedef typename Kernel::Point Point;

  int dim = num_cols;

  // Define functors for the negative log pdf and the gradient of the (positive)
  // log pdf
  typedef HMCFunctor_function_pointer_interface::FunctionFunctor<Point>
      NegativeLogPDFFunctor;
  typedef HMCFunctor_function_pointer_interface::GradientFunctor<Point>
      GradientFunctor;
  HMCFunctor_function_pointer_interface::parameters<NT> params(dim, L, m);
  NegativeLogPDFFunctor f(params, neg_log_pdf);
  GradientFunctor F(params, gradient_log_pdf);

#ifdef DEBUG
  // Test neg_log_pdf and gradient_log_pdf at the starting point
  double neg_log_pdf_result = neg_log_pdf(starting_point);
  double *gradient_log_pdf_result = gradient_log_pdf(starting_point);
  Point gradient_log_pdf_result_point(dim);
  for (auto i = 0; i != dim; i++) {
    gradient_log_pdf_result_point.set_coord(i, gradient_log_pdf_result[i]);
  }
  std::cout << "Neg log pdf at the starting point: " << neg_log_pdf_result
            << std::endl;
  std::cout << "Gradient at the starting point: "
            << gradient_log_pdf_result_point.getCoefficients().transpose()
            << std::endl;
#endif

  return hmc_core<NegativeLogPDFFunctor, GradientFunctor>(
      num_rows, num_cols, coefficients_A, coefficients_b, L, m, num_samples,
      walk_length, step_size, starting_point, f, F);
}

double *hmc_runtime_data_interface(
    int const num_rows, int const num_cols, double *coefficients_A,
    double *coefficients_b, double const L, double const m,
    int const num_samples_drawn, int const walk_length, double const step_size,
    double *starting_point, runtime_data_sample *runtime_data,
    int const num_samples_in_runtime_data,
    distribution_type coefficient_distribution, distribution_type cost_model,
    coefficient_distribution_target_type coefficient_distribution_target,
    distribution_target_type cost_model_target) {
  typedef double NT;
  typedef Cartesian<NT> Kernel;
  typedef typename Kernel::Point Point;

  int dim = num_cols;

#ifdef DEBUG
  std::cout << "Coefficient distribution: ";
  print_distribution(coefficient_distribution);
  std::cout << std::endl;
  std::cout << "Cost model: ";
  print_distribution(cost_model);
  std::cout << std::endl;
#endif

  probability_distribution_statistical_aara probability_distribution{
      runtime_data,
      num_samples_in_runtime_data,
      dim,
      coefficient_distribution,
      cost_model,
      coefficient_distribution_target,
      cost_model_target};

  // Define functors for the negative log pdf and the gradient of the (positive)
  // log pdf
  typedef HMCFunctor_probability_distribution_interface<
      probability_distribution_statistical_aara>::FunctionFunctor<Point>
      NegativeLogPDFFunctor;
  typedef HMCFunctor_probability_distribution_interface<
      probability_distribution_statistical_aara>::GradientFunctor<Point>
      GradientFunctor;
  HMCFunctor_probability_distribution_interface<
      probability_distribution_statistical_aara>::parameters<NT>
      params(dim, L, m);
  NegativeLogPDFFunctor f(params, probability_distribution);
  GradientFunctor F(params, probability_distribution);

#ifdef DEBUG
  // Test neg log pdf and positive log pdf's gradient at the starting point
  Point x0(dim);
  for (auto i = 0; i != dim; i++) {
    x0.set_coord(i, starting_point[i]);
  }

  double log_pdf = probability_distribution.log_pdf_point_interface(x0);
  Point gradient_log_pdf =
      probability_distribution.gradient_log_pdf_point_interface(x0);
  std::cout << "Log pdf at the starting point: " << log_pdf << std::endl;
  std::cout << "Gradient of the log pdf at the starting point: "
            << gradient_log_pdf.getCoefficients().transpose() << std::endl;
#endif

  return hmc_core<NegativeLogPDFFunctor, GradientFunctor>(
      num_rows, num_cols, coefficients_A, coefficients_b, L, m,
      num_samples_drawn, walk_length, step_size, starting_point, f, F);
}

double *hmc_cost_data_categorized_by_sizes(
    int const num_rows, int const num_cols, double *coefficients_A,
    double *coefficients_b, double const L, double const m,
    int const num_samples_drawn, int const walk_length, double const step_size,
    double *starting_point, int *array_sizes_of_categories, double *array_costs,
    distribution_type cost_model) {
  typedef double NT;
  typedef Cartesian<NT> Kernel;
  typedef typename Kernel::Point Point;

  int dim = num_cols;

#ifdef DEBUG
  std::cout << "Coefficient distribution: ";
  print_distribution(coefficient_distribution);
  std::cout << std::endl;
  std::cout << "Cost model: ";
  print_distribution(cost_model);
  std::cout << std::endl;
#endif

  // The number of size categories in cost data is given by the number of
  // columns in matrix A.
  probability_distribution_cost_data_categorized_by_sizes
      probability_distribution{num_cols, array_sizes_of_categories, array_costs,
                               cost_model};

  // Define functors for the negative log pdf and the gradient of the (positive)
  // log pdf
  typedef HMCFunctor_probability_distribution_interface<
      probability_distribution_cost_data_categorized_by_sizes>::
      FunctionFunctor<Point>
          NegativeLogPDFFunctor;
  typedef HMCFunctor_probability_distribution_interface<
      probability_distribution_cost_data_categorized_by_sizes>::
      GradientFunctor<Point>
          GradientFunctor;
  HMCFunctor_probability_distribution_interface<
      probability_distribution_cost_data_categorized_by_sizes>::parameters<NT>
      params(dim, L, m);
  NegativeLogPDFFunctor f(params, probability_distribution);
  GradientFunctor F(params, probability_distribution);

#ifdef DEBUG
  // Test neg log pdf and positive log pdf's gradient at the starting point
  Point x0(dim);
  for (auto i = 0; i != dim; i++) {
    x0.set_coord(i, starting_point[i]);
  }

  double log_pdf = probability_distribution.log_pdf_point_interface(x0);
  Point gradient_log_pdf =
      probability_distribution.gradient_log_pdf_point_interface(x0);
  std::cout << "Log pdf at the starting point: " << log_pdf << std::endl;
  std::cout << "Gradient of the log pdf at the starting point: "
            << gradient_log_pdf.getCoefficients().transpose() << std::endl;
#endif

  return hmc_core<NegativeLogPDFFunctor, GradientFunctor>(
      num_rows, num_cols, coefficients_A, coefficients_b, L, m,
      num_samples_drawn, walk_length, step_size, starting_point, f, F);
}
