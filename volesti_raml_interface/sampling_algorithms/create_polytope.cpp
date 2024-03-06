#include "create_polytope.h"

#include <iostream>
#include <vector>

#include "generators/known_polytope_generators.h"
#include "lp_lib.h"

// #define DEBUG

typedef double NT;
typedef Cartesian<NT> Kernel;
typedef typename Kernel::Point Point;
typedef HPolytope<Point> Hpolytope;
typedef typename Hpolytope::MT MT;
typedef typename Hpolytope::VT VT;

MT create_matrix_A(int const &num_rows, int const &num_cols,
                   double *coefficients) {
  MT A;
  A.resize(num_rows, num_cols);

  for (auto i = 0; i != num_rows; i++) {
    for (auto j = 0; j != num_cols; j++) {
      A(i, j) = coefficients[i * num_cols + j];
    }
  }
  return A;
}

VT create_vector_b(int const &size, double *coefficients) {
  VT b;
  b.resize(size);
  for (auto i = 0; i != size; i++) {
    b(i) = coefficients[i];
  }
  return b;
}

Hpolytope create_polytope(int const &num_rows, int const &num_cols,
                          double *coefficients_A, double *coefficients_b) {
  MT A = create_matrix_A(num_rows, num_cols, coefficients_A);
  VT b = create_vector_b(num_rows, coefficients_b);
  Hpolytope P(num_cols, A, b);
  return P;
}

int get_feasibility_status_of_lp(int const num_rows, int const num_cols,
                                 double *coefficients_A,
                                 double *coefficients_b) {
  // Pointer to the linear program under construction
  lprec *lp;

  /* Consider a linear program Ax <= b. row is an array storing (not necessarily
  all) coefficients of a single row of matrix A. If row only stores some
  coefficients (e.g. only non-zero coefficients), their locations are specified
  by the array column_numbers. This will speed up the construction of a linear
  program if it contains a large number of zeros. */
  int *column_numbers = NULL;
  REAL *row = NULL;
  int ret = 0;

  /* We will build the model row by row. So we start with creating a model with
  0 rows and num_cols many columns. */
  lp = make_lp(0, num_cols);
  if (lp == NULL) {
    ret = 1;  // couldn't construct a new model...
  }

  if (ret == 0) {
    // Create space large enough for one row
    column_numbers = (int *)malloc(num_cols * sizeof(*column_numbers));
    row = (REAL *)malloc(num_cols * sizeof(*row));
    if ((column_numbers == NULL) || (row == NULL)) ret = 2;
  }

  for (auto j = 0; j != num_cols; j++) {
    column_numbers[j] = j + 1;
  }

  // set_add_rowmode makes building the model faster if it is done rows by row.
  set_add_rowmode(lp, TRUE);

  // Define matrix A and vector b of the LP instance under construction
  for (auto i = 0; i != num_rows; i++) {
    for (auto j = 0; j != num_cols; j++) {
      row[j] = coefficients_A[num_cols * i + j];
    }
    if (!add_constraintex(lp, num_cols, row, column_numbers, LE,
                          coefficients_b[i])) {
      ret = 3;
    }
  }

  if (ret == 0) {
    // rowmode should be turned off again when done building the model
    set_add_rowmode(lp, FALSE);

    /* Because we just want to check the feasibility of a linear program, it
    does not matter what specific objective function we use. */
    for (auto j = 0; j != num_cols; j++) {
      row[j] = 0;
    }

    // Set the objective in lpsolve
    if (!set_obj_fnex(lp, num_cols, row, column_numbers)) ret = 4;
  }

  if (ret == 0) {
    // Set the object direction to minimize
    set_minim(lp);

    // I only want to see important messages on screen while solving
    set_verbose(lp, IMPORTANT);

    /* Now let lpsolve calculate a solution */
    ret = solve(lp);
    if (ret == OPTIMAL) {
#ifdef DEBUG
      std::cout << "The given LP is feasible" << std::endl;
#endif
      ret = 0;
    } else if (ret == INFEASIBLE) {
#ifdef DEBUG
      std::cout << "The given LP is infeasible" << std::endl;
#endif
      ret = 5;
    } else {
      std::cerr << "The given LP is neither feasible nor infeasible: "
                << get_statustext(lp, ret) << std::endl;
      ret = 6;
    }
  }

  // Free up the allocated memory
  if (row != NULL) free(row);
  if (column_numbers != NULL) free(column_numbers);

  if (lp != NULL) {
    // Clean up such that all used memory by lpsolve is freed
    delete_lp(lp);
  }

  return ret;
}

double compute_chebyshev_radius(int const num_rows, int const num_cols,
                                double *coefficients_A,
                                double *coefficients_b) {
  Hpolytope P =
      create_polytope(num_rows, num_cols, coefficients_A, coefficients_b);
  std::pair<Point, NT> InnerBall = P.ComputeInnerBall();
  if (InnerBall.second < 0.0) {
    throw std::invalid_argument(
        "The linear program is infeasible in compute_chebysehv_radius");
  }
#ifdef DEBUG
  std::cerr << "Polytope P's Chebyshev radius: " << InnerBall.second
            << std::endl;
#endif
  return InnerBall.second;
}

int identify_first_implicit_equality_row_index_in_matrix(
    int const num_rows, int const num_cols, double *coefficients_A,
    double *coefficients_b) {
  int left_index = 0;
  int right_index = num_rows;
  int mid_index;

  int is_feasible = get_feasibility_status_of_lp(
      num_rows, num_cols, coefficients_A, coefficients_b);

  if (is_feasible == 0) {
    std::cout << "The given LP is feasible" << std::endl;
  } else if (is_feasible == 5) {
    std::cout << "The given LP is infeasible" << std::endl;
  } else {
    std::cerr << "The given LP is neither feasible nor infeasible" << std::endl;
  }

  double chebyshev_epsilon = 0.0001;
  double radius;

  while (left_index < right_index) {
    mid_index = (left_index + right_index) / 2;
    radius = compute_chebyshev_radius(mid_index + 1, num_cols, coefficients_A,
                                      coefficients_b);
    std::cout << "Binary search: left_index = " << left_index
              << " right_index = " << right_index << " radius = " << radius
              << std::endl;

    if (radius < chebyshev_epsilon) {
      right_index = mid_index;
    } else {
      left_index = mid_index + 1;
    }
  }
  return left_index;
}

void print_row(int const i, int const num_rows, int const num_cols,
               double *coefficients_A, double *coefficients_b) {
  std::cout << "Row " << i << ": ";
  for (auto j = 0; j < num_cols; j++) {
    double current_element = coefficients_A[i * num_cols + j];
    if (current_element != 0) {
      std::cout << "(v, c) = (" << j << ", " << current_element << ") ";
    }
  }
  std::cout << "upper bound = " << coefficients_b[i] << std::endl;
}

int *iteratively_perturb_vector_b(int const num_rows, int const num_cols,
                                  double *coefficients_A,
                                  double *coefficients_b) {
  lprec *lp;
  int *column_numbers = NULL;
  REAL *row = NULL;
  int ret = 0;

  std::vector<int> implicit_equality_indices_vector;

  lp = make_lp(0, num_cols);
  if (lp == NULL) {
    ret = 1; /* couldn't construct a new model... */
  }

  if (ret == 0) {
    /* create space large enough for one row */
    column_numbers = (int *)malloc(num_cols * sizeof(*column_numbers));
    row = (REAL *)malloc(num_cols * sizeof(*row));
    if ((column_numbers == NULL) || (row == NULL)) ret = 2;
  }

  for (auto j = 0; j != num_cols; j++) {
    column_numbers[j] = j + 1;
  }

  set_add_rowmode(
      lp, TRUE); /* makes building the model faster if it is done rows by row */

  for (auto i = 0; i != num_rows; i++) {
    for (auto j = 0; j != num_cols; j++) {
      row[j] = coefficients_A[num_cols * i + j];
    }
    if (!add_constraintex(lp, num_cols, row, column_numbers, LE,
                          coefficients_b[i])) {
      ret = 3;
    }
  }

  if (ret == 0) {
    /* rowmode should be turned off again when done building the model */
    set_add_rowmode(lp, FALSE);

    for (auto j = 0; j != num_cols; j++) {
      row[j] = 0;
    }

    /* set the objective in lpsolve */
    if (!set_obj_fnex(lp, num_cols, row, column_numbers)) ret = 4;
  }

  if (ret == 0) {
    /* set the object direction to minimize */
    set_minim(lp);

    /* I only want to see important messages on screen while solving */
    set_verbose(lp, IMPORTANT);

    /* Now let lpsolve calculate a solution */
    ret = solve(lp);
    if (ret == OPTIMAL) {
      std::cout << "The given LP with all rows is feasible" << std::endl;
      ret = 0;
    } else if (ret == INFEASIBLE) {
      std::cout << "The given LP with all rows is infeasible" << std::endl;
      ret = 5;
    } else {
      std::cerr
          << "The given LP with all rows is neither feasible nor infeasible: "
          << get_statustext(lp, ret) << std::endl;
      ret = 6;
    }
  }

  if (ret == 0) {
    double perturbation_epsilon = 0.0001;  // Amount of perturbation
    int current_row_index;
    double current_b_value;
    for (auto i = 0; i < num_rows; i++) {
      current_row_index = i + 1;
      current_b_value = get_rh(lp, current_row_index);
      set_rh(lp, current_row_index, current_b_value - perturbation_epsilon);

      ret = solve(lp);
      // int foo_iter = 0;
      // while (ret != OPTIMAL && foo_iter < 5) {
      //   ret = solve(lp);
      //   foo_iter++;
      // }

      if (ret == OPTIMAL) {
        set_rh(lp, current_row_index, current_b_value);
      } else if (ret == INFEASIBLE) {
        set_rh(lp, current_row_index, current_b_value);
        implicit_equality_indices_vector.push_back(i);

#ifdef DEBUG
        print_row(i, num_rows, num_cols, coefficients_A, coefficients_b);
#endif

      } else {
        std::cerr << "The given LP with a perturbation is neither feasible nor "
                     "infeasible: "
                  << get_statustext(lp, ret) << std::endl;
        ret = 6;
        break;
      }
    }
  }

  /* free allocated memory */
  if (row != NULL) free(row);
  if (column_numbers != NULL) free(column_numbers);

  if (lp != NULL) {
    /* clean up such that all used memory by lpsolve is freed */
    delete_lp(lp);
  }

  /* Allocate an array to store all indices of implicit equalities that we have
  discovered. */
  int *implicit_equality_indices_array =
      new int[implicit_equality_indices_vector.size() + 1];

  /* We store the number of implicit indices in the first (i.e. 0-th) array
  element. This is why the total number of array elements is
  implicit_equality_indices_vector.size() + 1. */
  implicit_equality_indices_array[0] = implicit_equality_indices_vector.size();
  for (auto i = 0; i < implicit_equality_indices_vector.size(); i++) {
    implicit_equality_indices_array[i + 1] =
        implicit_equality_indices_vector[i];
  }

  std::cout << "We have found " << implicit_equality_indices_vector.size()
            << " many implicit equalities by perturbations" << std::endl;

  return implicit_equality_indices_array;
}
