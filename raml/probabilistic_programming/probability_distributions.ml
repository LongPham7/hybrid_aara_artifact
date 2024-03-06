open Core

(* This module offers data types for representing probability distributions and
functions for manipulating them. These data types and functions are used in
hybrid AARA. As of now, we support Weibull, Gumbel, and Gaussian distributions.
*)

(* Alpha is the shape parameter, and sigma, which must be non-zero, is the scale
parameter. *)
type weibull_distribution_type = { alpha : float; sigma : float }

(* Mu is the location parameter, and beta, which must be non-zero, is the scale
parameter. *)
type gumbel_distribution_type = { mu : float; beta : float }

(* Mu is the location parameter, and sigma, which must be non-zero is the scale
parameter. *)
type gaussian_distribution_type = { mu : float; sigma : float }

type probability_distribution_type =
  | Weibull of weibull_distribution_type
  | Gumbel of gumbel_distribution_type
  | Gaussian of gaussian_distribution_type

(* Individual_coefficient means the distribution applies to each individual
resource coefficient in hybrid AARA. Average_coefficient means the
distribution applies to the average of all coefficients. *)
type probability_distribution_target_type =
  | Individual_coefficients
  | Average_of_coefficients

type probability_distribution_with_target_type = {
  distribution : probability_distribution_type;
  target : probability_distribution_target_type;
}

(* Return whether a given probability distribution is truncated: its support
(i.e. the region of positive density) is limited to a proper subset of the
entire space. In Weibull distributions, the density is positive only when the
random variable is non-negative. So Weibull distributions are truncated. *)
let is_cost_model_truncated (cost_model : probability_distribution_type) : bool
    =
  match cost_model with Weibull _ -> true | Gumbel _ | Gaussian _ -> false

(* Return the name of the given probability distribution in Stan *)
let distribution_name_in_stan probability_distribution =
  match probability_distribution with
  | Weibull _ -> "weibull"
  | Gumbel _ -> "gumbel"
  | Gaussian _ -> "normal"

(* Return the names of probability distribution parameters in Stan *)
let parameter_names_in_stan probability_distribution =
  match probability_distribution with
  | Weibull _ -> [ "alpha"; "sigma" ]
  | Gumbel _ -> [ "mu"; "beta" ]
  | Gaussian _ -> [ "mu"; "sigma" ]
