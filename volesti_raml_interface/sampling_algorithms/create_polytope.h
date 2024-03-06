#ifndef CREATE_POLYTOPE
#define CREATE_POLYTOPE

#include "generators/known_polytope_generators.h"
#include "random_walks/random_walks.hpp"

#define CHEBYSHEV_RADIUS_EPSILON 0.0001

typename HPolytope<typename Cartesian<double>::Point>::MT create_matrix_A(
    int const &num_rows, int const &num_cols, double *coefficients);

typename HPolytope<typename Cartesian<double>::Point>::VT create_vector_b(
    int const &size, double *coefficients);

HPolytope<typename Cartesian<double>::Point> create_polytope(
    int const &num_rows, int const &num_cols, double *coefficients_A,
    double *coefficients_b);

#ifdef __cplusplus
extern "C" {
#endif

int get_feasibility_status_of_lp(int const num_rows, int const num_cols,
                                 double *coefficients_A,
                                 double *coefficients_b);

double compute_chebyshev_radius(int const num_rows, int const num_cols,
                                double *coefficients_A, double *coefficients_b);

int identify_first_implicit_equality_row_index_in_matrix(
    int const num_rows, int const num_cols, double *coefficients_A,
    double *coefficients_b);

int *iteratively_perturb_vector_b(int const num_rows, int const num_cols,
                                  double *coefficients_A,
                                  double *coefficients_b);

#ifdef __cplusplus
}
#endif

#endif
