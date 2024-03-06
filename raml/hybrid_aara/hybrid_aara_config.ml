open Core
open Probability_distributions
open Runtime_cost_data

(* This module parses a configuration file (in JSON) for hybrid AARA.

We have three modes for hybrid AARA: Opt, BayesWC, and BayesPC. They are each
explained below. In addition to these three modes for hybrid AARA, we have one
more mode, called Disabled, that disables data-driven analysis. So the mode
Disabled only performs static analysis; that is, it amounts to conventional
AARA.

To set the stage, let P be the program that we want to analyze. Suppose program
P contains an expression Raml.stat(e), indicating that expression e should be
analyzed by data-drive analysis.

1. Pure and Hybrid Opt: In Pure Opt, we create a set of linear constraints
   stating that predicted costs are larger than or equal to observed costs in a
   runtime cost dataset. We then optimize the resource annotations of e, subject
   to the linear constraints.

   In Hybrid Opt, we combine two sets of linear constraints, one set induced by
   the type system of AARA and another set induced by the runtime cost data of
   expression e. We then optimize the resource annotations of the entire program
   P, subject to the aggregate linear constraints.

2. Pure and Hybrid BayesWC: In Pure BayesWC, we perform Bayesian inference
   (specifically Bayesian polynomial regression) to approximate a posterior
   distribution of the theoretically worst-case cost for each input size s
   present in e's runtime cost data. The theoretically worst-case costs should
   be larger than or equal to the maximum observed costs. We then optimize the
   resource annotations of expression e, subject to the linear constraints that
   the predicted costs are larger than or equal to the theoretically worst-case
   costs.

   In Hybrid BayesWC, we combine two sets of linear constraints, one from AARA's
   type system and another from BayesWC applied to expression e. We then
   optimize the resource annotations of the entire program P, subject to the
   aggregate linear constraints.

3. Pure and Hybrid BayesPC: In Pure BayesPC, we perform Bayesian inference
   (specifically Weibull survival analysis) to infer not only theoretically
   worst-case costs but also polynomial coefficients of cost bounds (i.e.,
   resource annotations of expression e).

   In Hybrid BayesPC, we run a sampling-based probabilistic inference algorithm
   within the feasible region of a linear program induced by AARA's type system.
   We then draw a posterior sample and extract the resource annotations of
   expression e. We then optimize the resource annotations of the program P,
   subject to the constraints that the resource annotations of expression e are
   fixed to be the posterior sample we have just drawn.
   
For configuration files, ideally, I should use TOML rather than JSON. The former
was designed for configuration files specifically, while the latter was designed
for data-transfer files. Due to this difference in the original design goals,
TOML supports comments, while JSON does not. Nonetheless, I chose the JSON
format for configuration files because OCaml seems to have better support for
JSON than TOML. The OCaml library Yojson for handling JSON files is
well-documented and is explained well in the book "Real World OCaml" by
Anil Madhavapeddy and Yaron Minsky. *)

type optimization_objective_weights = Equal_weights | Lexicographic_weights

type opt_objective_params = {
  cost_gaps_optimization : bool;
  coefficients_optimization : optimization_objective_weights;
}

(* output_lp_vars is the list of LP variables whose inference results are
written to a JSON file. output_file is the name of the JSON file to be written
to. This type is used in Opt and BayesPC. *)
type output_lp_vars_params = { output_lp_vars : int list; output_file : string }

(* lp_vars is the list of LP variables whose posterior distributions are printed
out. This is useful when we want to figure out the optimal resource coefficients
of internal nodes (e.g., those corresponding to an auxiliary function used
inside the function under analysis) of a typing tree. *)
type opt_params = {
  lp_objective : opt_objective_params;
  output_params : output_lp_vars_params option;
  dataset : Location.t dataset_with_potential_categorized_by_indices;
}

(* upper_bound is the upper bound of LP variables' box constraints. The lower
bound is automatically set to zero. Additionally, users are allowed to a list
(upper_unbounded_vars) of LP variables that should have no upper bounds (i.e.
the upper bound is Float.max_value). This feature is used in, for example, the
differential Bayesian resource analysis of quickselect where we need a very
large bound for a few LP variables. *)
type bayespc_box_constraint_params = {
  upper_bound : float;
  upper_unbounded_vars : int list;
}

type bayespc_lp_params = {
  box_constraint : bayespc_box_constraint_params;
  implicit_equality_removal : bool;
  output_potential_set_to_zero : bool;
}

type hit_and_run_algorithm =
  | Gaussian_rdhr
  | Uniform_rdhr
  | Gaussian_cdhr
  | Uniform_cdhr
  | Uniform_billiard

type hit_and_run_params = {
  algorithm : hit_and_run_algorithm;
  variance : float;
  num_samples : int;
  walk_length : int;
}

type bayespc_hmc_params = {
  coefficient_distribution_with_target :
    probability_distribution_with_target_type;
  cost_model_with_target : probability_distribution_with_target_type;
  num_samples : int;
  walk_length : int;
  step_size : float;
}

type bayespc_params = {
  lp_params : bayespc_lp_params;
  warmup_params : hit_and_run_params;
  hmc_params : bayespc_hmc_params;
  output_params : output_lp_vars_params option;
  dataset : Location.t dataset_with_potential_categorized_by_indices;
}

(* This is only used in the old version of BayesWC before we incorporate Weibull
survival analysis. *)
type bayeswc_hmc_params = {
  cost_model : probability_distribution_type;
  num_samples : int;
  walk_length : int;
  step_size : float;
}

type bayeswc_stan_params = {
  scale_beta : float;
  scale_s : float;
  num_chains : int;
  num_stan_samples : int;
}

type bayeswc_params = {
  input_file : string option;
  stan_params : bayeswc_stan_params;
  lp_objective : opt_objective_params;
  dataset : Location.t dataset_with_potential;
}

type hybrid_aara_params =
  | Disabled
  | Opt of opt_params
  | BayesWC of bayeswc_params
  | BayesPC of bayespc_params

(* Parsing functions for configurations in JSON *)

let parse_opt_lp_objective lp_objective_json =
  let open Yojson.Basic.Util in
  let cost_gaps_optimization =
    lp_objective_json |> member "cost_gaps_optimization" |> to_bool
  in
  let coefficients_string =
    lp_objective_json |> member "coefficients_optimization" |> to_string
  in
  let coefficients_optimization =
    match coefficients_string with
    | "Equal_weights" -> Equal_weights
    | "Lexicographic_weights" -> Lexicographic_weights
    | _ -> failwith "The given LP objective type for coefficients is invalid"
  in
  { cost_gaps_optimization; coefficients_optimization }

let parse_probability_distribution distribution_json =
  let open Yojson.Basic.Util in
  let extract_distribution_parameter parameter_name =
    (* The function Yojson.Basic.Util.to_float does not automatically cast JSON
    integers to OCaml floats. Hence, to write down floats in JSON, we must
    explicitly put decimal points, as in OCaml. *)
    distribution_json |> member parameter_name |> to_float
  in
  let distribution_type =
    distribution_json |> member "distribution_type" |> to_string
  in
  match distribution_type with
  | "Weibull" ->
      let alpha = extract_distribution_parameter "alpha" in
      let sigma = extract_distribution_parameter "sigma" in
      Weibull { alpha; sigma }
  | "Gumbel" ->
      let mu = extract_distribution_parameter "mu" in
      let beta = extract_distribution_parameter "beta" in
      Gumbel { mu; beta }
  | "Gaussian" ->
      let mu = extract_distribution_parameter "mu" in
      let sigma = extract_distribution_parameter "sigma" in
      Gaussian { mu; sigma }
  | _ ->
      failwith "The given JSON cannot be parsed as a probability distribution."

let parse_probability_distribution_with_target distribution_with_target_json =
  let open Yojson.Basic.Util in
  let distribution =
    distribution_with_target_json |> member "distribution"
    |> parse_probability_distribution
  in
  let target_string =
    distribution_with_target_json |> member "target" |> to_string
  in
  let target =
    match target_string with
    | "Individual_coefficients" -> Individual_coefficients
    | "Average_of_coefficients" -> Average_of_coefficients
    | _ ->
        failwith
          "The given JSON cannot be parsed as a probability distribution \
           target."
  in
  { distribution; target }

let parse_bayespc_constraint_params box_constraint_params_json =
  let open Yojson.Basic.Util in
  let upper_bound =
    box_constraint_params_json |> member "upper_bound" |> to_float
  in
  let upper_unbounded_vars_json =
    box_constraint_params_json |> member "upper_unbounded_vars"
  in
  let upper_unbounded_vars =
    (* The default is the empty list; that is, all LP variables have the same
    upper bound in their box constraints. *)
    match upper_unbounded_vars_json with
    | `Null -> []
    | _ -> upper_unbounded_vars_json |> to_list |> filter_int
  in
  { upper_bound; upper_unbounded_vars }

let parse_bayespc_lp_params lp_params_json =
  let open Yojson.Basic.Util in
  let box_constraint =
    lp_params_json |> member "box_constraint" |> parse_bayespc_constraint_params
  in
  let implicit_equality_removal_json =
    lp_params_json |> member "implicit_equality_removal"
  in
  let implicit_equality_removal =
    match implicit_equality_removal_json with
    | `Null -> true (* The default is to remove all implicit equalities *)
    | _ -> implicit_equality_removal_json |> to_bool
  in
  let output_potential_set_to_zero_json =
    lp_params_json |> member "output_potential_set_to_zero"
  in
  let output_potential_set_to_zero =
    match output_potential_set_to_zero_json with
    (* The default is to let the return type's coefficients be arbitrary *)
    | `Null -> false
    | _ -> output_potential_set_to_zero_json |> to_bool
  in
  { box_constraint; implicit_equality_removal; output_potential_set_to_zero }

let parse_hit_and_run_params hit_and_run_params_json =
  let open Yojson.Basic.Util in
  let algorithm_json =
    hit_and_run_params_json |> member "algorithm" |> to_string
  in
  let algorithm =
    match algorithm_json with
    | "Gaussian_rdhr" -> Gaussian_rdhr
    | "Uniform_rdhr" -> Uniform_rdhr
    | "Gaussian_cdhr" -> Gaussian_cdhr
    | "Uniform_cdhr" -> Uniform_cdhr
    | "Uniform_billiard" -> Uniform_billiard
    | _ -> failwith "The given algorithm is invalid"
  in
  (* Users must specify variance even if the algorithm of their choice does not
  need the variance. *)
  let variance = hit_and_run_params_json |> member "variance" |> to_float in
  let num_samples = hit_and_run_params_json |> member "num_samples" |> to_int in
  let walk_length = hit_and_run_params_json |> member "walk_length" |> to_int in
  { algorithm; variance; num_samples; walk_length }

let parse_bayespc_hmc_params hmc_params_json =
  let open Yojson.Basic.Util in
  let coefficient_distribution_with_target =
    hmc_params_json
    |> member "coefficient_distribution_with_target"
    |> parse_probability_distribution_with_target
  in
  let cost_model_with_target =
    hmc_params_json
    |> member "cost_model_with_target"
    |> parse_probability_distribution_with_target
  in
  let num_samples = hmc_params_json |> member "num_samples" |> to_int in
  let walk_length = hmc_params_json |> member "walk_length" |> to_int in
  let step_size = hmc_params_json |> member "step_size" |> to_float in
  {
    coefficient_distribution_with_target;
    cost_model_with_target;
    num_samples;
    walk_length;
    step_size;
  }

let parse_output_lp_vars_params output_params_json =
  let open Yojson.Basic.Util in
  match output_params_json with
  | `Null -> None
  | _ ->
      let output_lp_vars =
        output_params_json |> member "output_lp_vars" |> to_list |> filter_int
      in
      let output_file =
        output_params_json |> member "output_file" |> to_string
      in
      Some { output_lp_vars; output_file }

let parse_bayeswc_stan_params stan_params_json =
  let open Yojson.Basic.Util in
  let scale_beta = stan_params_json |> member "scale_beta" |> to_float in
  let scale_s = stan_params_json |> member "scale_s" |> to_float in
  let num_chains = stan_params_json |> member "num_chains" |> to_int in
  let num_stan_samples =
    stan_params_json |> member "num_stan_samples" |> to_int
  in
  { scale_beta; scale_s; num_chains; num_stan_samples }

(* Extract worst-case samples from runtime cost data and categorize them by
indices *)
let retain_worst_case_samples_and_categorize_by_indices dataset =
  let dataset_with_only_worst_case_samples =
    Runtime_cost_data.retain_worst_case_samples dataset
  in
  let dataset_with_potential_categorized_by_indices =
    Runtime_cost_data.categorize_dataset_with_potential_by_indices
      dataset_with_only_worst_case_samples
  in
  (* For debugging *)
  (* let () =
    Hybrid_aara_pprint
    .print_dataset_with_potential_categorized_by_indices
      dataset_with_potential_categorized_by_indices
  in *)
  dataset_with_potential_categorized_by_indices

let parse_hybrid_aara_config_json config_json dataset =
  let open Yojson.Basic.Util in
  let hybrid_aara_mode = config_json |> member "mode" |> to_string in
  match hybrid_aara_mode with
  | "disabled" -> Disabled
  | "opt" ->
      let lp_objective =
        config_json |> member "lp_objective" |> parse_opt_lp_objective
      in
      let output_params =
        config_json |> member "output_params" |> parse_output_lp_vars_params
      in
      let dataset_processed =
        retain_worst_case_samples_and_categorize_by_indices dataset
      in
      Opt { lp_objective; output_params; dataset = dataset_processed }
  | "bayespc" ->
      let lp_params =
        config_json |> member "lp_params" |> parse_bayespc_lp_params
      in
      let warmup_params =
        config_json |> member "warmup_params" |> parse_hit_and_run_params
      in
      let hmc_params =
        config_json |> member "hmc_params" |> parse_bayespc_hmc_params
      in
      let output_params =
        config_json |> member "output_params" |> parse_output_lp_vars_params
      in
      let dataset_processed =
        retain_worst_case_samples_and_categorize_by_indices dataset
      in
      BayesPC
        {
          lp_params;
          warmup_params;
          hmc_params;
          output_params;
          dataset = dataset_processed;
        }
  | "bayeswc" ->
      let input_file_json = config_json |> member "input_file" in
      let input_file =
        match input_file_json with
        | `Null -> None (* The default is None. *)
        | _ -> Some (input_file_json |> to_string)
      in
      let stan_params =
        config_json |> member "stan_params" |> parse_bayeswc_stan_params
      in
      let lp_objective =
        config_json |> member "lp_objective" |> parse_opt_lp_objective
      in
      BayesWC { input_file; stan_params; lp_objective; dataset }
  | _ ->
      failwith
        "The given JSON cannot be parsed as a hybrid AARA configuration file."
