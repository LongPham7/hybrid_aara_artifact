open Core
open Python_interface

(* This module implements Weibull survival analysis. It is used in BayesWC,
where we perform Weibull survival analysis to infer theoretically worst-case
costs. The probabilistic model for Weibull survival analysis comes from a
tutorial of PyMC:
https://www.pymc.io/projects/examples/en/stable/survival_analysis/bayes_param_survival_pymc3.html

This probabilistic model for Weibull survival analysis takes the logarithm of
cost. So the model only works when the costs are strictly positive. The cost of
zero will result in an error. *)

let _ =
  Py.initialize ();
  print_endline "The Python library for py.ml is loaded"

(* The following Stan code uses a truncated normal distribution for the latent
variable s, which should come from a half-normal distribution according to the
PyMC tutorial. This is fine because, as described in the Wikipedia article on
half-normal distributions, half-normal distributions and truncated zero-mean
normal distributions coincide. *)

let stan_model =
  "\n\
   data {\n\
  \    int<lower=0> num_runtime_samples;\n\
  \    int<lower=0> num_sizes;\n\
  \    matrix<lower=0>[num_runtime_samples, num_sizes] size_matrix;\n\
  \    vector[num_runtime_samples] log_cost_vector;\n\
  \    \n\
  \    real<lower=0> sigma_beta; // scale parameter of a normal distribution \
   for beta\n\
  \    real<lower=0> sigma_s; // scale parameter of a half-normal distribution \
   for s\n\
   }\n\
   parameters {\n\
  \    vector[num_sizes] beta_vector;\n\
  \    real<lower=0> s;\n\
   }\n\
   model {\n\
  \    beta_vector ~ normal(0, sigma_beta);\n\
  \    s ~ normal(0, sigma_s) T[0, ];\n\
  \    \n\
  \    for (n in 1:num_runtime_samples) {\n\
  \        log_cost_vector[n] ~ gumbel(size_matrix[n] * beta_vector, s);\n\
  \    }\n\
   }\n"

let bayesian_inference stan_params size_matrix_ocaml cost_vector_ocaml
    num_runtime_samples num_sizes =
  let ({ scale_beta; scale_s; num_chains; num_stan_samples }
        : Hybrid_aara_config.bayeswc_stan_params) =
    stan_params
  in
  let size_matrix_python =
    convert_ocaml_nested_int_list_to_python_list size_matrix_ocaml
  in
  let log_cost_vector_ocaml = List.map cost_vector_ocaml ~f:Float.log in
  let log_cost_vector_python =
    convert_ocaml_float_list_to_python_list log_cost_vector_ocaml
  in
  let stan_data =
    Py.Dict.of_bindings_string
      [
        ("num_runtime_samples", Py.Int.of_int num_runtime_samples);
        ("num_sizes", Py.Int.of_int num_sizes);
        ("size_matrix", size_matrix_python);
        ("log_cost_vector", log_cost_vector_python);
        ("sigma_beta", Py.Float.of_float scale_beta);
        ("sigma_s", Py.Float.of_float scale_s);
      ]
  in
  let posterior =
    Py.Module.get_function_with_keywords (Py.import "stan") "build"
      [| Py.String.of_string stan_model |]
      [ ("data", stan_data); ("random_seed", Py.Int.of_int 1) ]
  in
  let sample_method = Py.Object.find_attr_string posterior "sample" in
  let raw_inference_result =
    Py.Callable.to_function_with_keywords sample_method [||]
      [
        ("num_chains", Py.Int.of_int num_chains);
        ("num_samples", Py.Int.of_int num_stan_samples);
      ]
  in
  raw_inference_result

let extract_inferred_model_parameters inference_result field_name =
  let get_field_from_fit field_name =
    Py.Callable.to_function
      (Py.Object.find_attr_string inference_result "__getitem__")
      [| Py.String.of_string field_name |]
  in
  let numpy_module = Py.import "numpy" in
  let transpose python_matrix =
    Py.Module.get_function numpy_module "transpose" [| python_matrix |]
  in
  let convert_numpy_array_to_python_list array =
    Py.Callable.to_function (Py.Object.find_attr_string array "tolist") [||]
  in
  field_name |> get_field_from_fit |> transpose
  |> convert_numpy_array_to_python_list
  |> convert_python_nested_float_list_to_ocaml_list

(* Conditionally sample an inferred worst-case cost from a Gumbel distribution
(for the logarithm of costs). This sampling is done conditionally on that the
worst-case cost is larger than or equal to the maximum observed cost.

Given a Gumbel distribution (specified by its location and scale parameters), we
first calculate gumbel_cdf_right_edge, which denotes the CDF of the Gumbel
distribution at the log maximum observed cost. We then draw a sample uniformly
randomly from the interval [gumbel_cdf_right_edge, 1]. From this sample, we
compute the corresponding log cost. In effect, we draw a sample from the Gumbel
distribution truncated to the interval [log_max_observed_cost, infinite). *)
let gumbel_conditional_sampling loc scale log_max_observed_cost =
  let open Owl_stats in
  assert (scale > 0.);
  let gumbel_cdf_right_edge =
    gumbel1_cdf (log_max_observed_cost -. loc) (1. /. scale) 1.
  in
  assert (gumbel_cdf_right_edge < 1.0);
  let gumbel_cdf_sample = uniform_rvs gumbel_cdf_right_edge 1.0 in
  loc +. gumbel1_ppf gumbel_cdf_sample (1. /. scale) 1.

let evaluate_inferred_worst_case_cost_single_sample beta_vector s size_vector
    max_observed_cost =
  let beta_size_vector_zipped = List.zip_exn beta_vector size_vector in
  let loc =
    List.fold beta_size_vector_zipped ~init:0. ~f:(fun acc (x, y) ->
        acc +. (x *. Float.of_int y))
  in
  let log_max_observed_cost = Float.log max_observed_cost in
  let log_inferred_cost =
    gumbel_conditional_sampling loc s log_max_observed_cost
  in
  assert (log_inferred_cost >= log_max_observed_cost);
  Float.exp log_inferred_cost

let evaluate_inferred_worst_case_costs_all_samples beta_vector s
    size_matrix_and_cost_vector =
  List.map size_matrix_and_cost_vector ~f:(fun (size_vector, log_max_cost) ->
      evaluate_inferred_worst_case_cost_single_sample beta_vector s size_vector
        log_max_cost)

let evaluate_inferred_worst_case_costs beta_vector_distribution s_distribution
    size_matrix_and_cost_vector =
  let beta_vector_s_zipped =
    List.zip_exn beta_vector_distribution s_distribution
  in
  List.map beta_vector_s_zipped ~f:(fun (beta_vector, s) ->
      evaluate_inferred_worst_case_costs_all_samples beta_vector s
        size_matrix_and_cost_vector)

(* Extract all cindices of degree at most one. So this includes the degree-zero
cindices (i.e., constant potential in the typing context). For simplicity, we
only use degree-one cindices (e.g., the outer length of a nested list) as
covariates in Weibull survival analysis. *)
let extract_degree_one_cindices_dataset_of_e dataset_of_e =
  let list_cindex_potential, _, list_costs = dataset_of_e in
  let cindex_is_degree_one cindex_list_potential =
    let cindex, _ = cindex_list_potential in
    Indices.cdegree cindex <= 1
  in
  let list_cindex_potential_degree_one =
    List.filter list_cindex_potential ~f:cindex_is_degree_one
  in
  (list_cindex_potential_degree_one, list_costs)

let rec create_size_matrix list_cindex_potential =
  match list_cindex_potential with
  | [] -> failwith "The list of cindices and potential is empty"
  | [ (_, list_potential) ] -> List.map list_potential ~f:(fun x -> [ x ])
  | (_, head_list_potential) :: tail_list ->
      let recursive_size_marix = create_size_matrix tail_list in
      let head_potential_recursive_matrix_zipped =
        List.zip_exn head_list_potential recursive_size_marix
      in
      List.map head_potential_recursive_matrix_zipped
        ~f:(fun (p, size_vector) -> p :: size_vector)

let extract_size_matrix_and_cost_vector dataset_of_e_categorized_by_indices =
  let list_cindex_potential_degree_one, list_costs =
    extract_degree_one_cindices_dataset_of_e dataset_of_e_categorized_by_indices
  in
  let size_matrix = create_size_matrix list_cindex_potential_degree_one in
  List.zip_exn size_matrix list_costs

let survival_analysis_dataset_of_e stan_params dataset_of_e =
  (* To conduct Bayesian survival analysis, we use all samples in the runtime
  data. However, to evaluate predicted worst-case costs, we compute a predicted
  worst-case cost for each size category in the runtime cost data, instead of
  one cost for each sample. *)
  let list_cindex_potential_degree_one, list_costs =
    dataset_of_e |> Runtime_cost_data.categorize_list_samples_by_indices
    |> extract_degree_one_cindices_dataset_of_e
  in
  (* For debugging *)
  (* let () =
    print_endline "List of degree-zero and degree-one cindices";
    List.iter list_cindex_potential_degree_one ~f:(fun (cindex, _) ->
        Indices.print_cindex cindex;
        print_newline ())
  in *)
  let num_sizes = List.length list_cindex_potential_degree_one in
  let num_runtime_samples = List.length list_costs in
  let size_matrix = create_size_matrix list_cindex_potential_degree_one in
  let raw_inference_result =
    bayesian_inference stan_params size_matrix list_costs num_runtime_samples
      num_sizes
  in
  let beta_vector_distribution =
    extract_inferred_model_parameters raw_inference_result "beta_vector"
  in
  let s_distribution =
    extract_inferred_model_parameters raw_inference_result "s" |> List.concat
  in
  (* Extract worst-case samples of size categories in the runtime cost data *)
  let dataset_of_e_worst_case_only_categorized_by_indices =
    dataset_of_e |> Runtime_cost_data.retain_worst_case_samples_dataset_of_e
    |> Runtime_cost_data.categorize_list_samples_by_indices
  in
  (* Create a size matrix and cost vector for worst-case samples in the runtime
  data. They are used to compute predicted worst-case costs. *)
  let size_matrix_and_cost_vector_worst_case_only =
    extract_size_matrix_and_cost_vector
      dataset_of_e_worst_case_only_categorized_by_indices
  in
  let inference_result =
    evaluate_inferred_worst_case_costs beta_vector_distribution s_distribution
      size_matrix_and_cost_vector_worst_case_only
  in
  let list_cindex_potential, list_index_potential, _ =
    dataset_of_e_worst_case_only_categorized_by_indices
  in
  List.map inference_result ~f:(fun list_costs ->
      (list_cindex_potential, list_index_potential, list_costs))

let rec survival_analysis stan_params dataset =
  match dataset with
  | [] -> failwith "The dataset is empty"
  | [ (head_e, dataset_of_e) ] ->
      let survival_analysis_result =
        survival_analysis_dataset_of_e stan_params dataset_of_e
      in
      List.map survival_analysis_result ~f:(fun x -> [ (head_e, x) ])
  | (head_e, head_dataset_of_e) :: tail_dataset ->
      let recursive_survival_analysis_result =
        survival_analysis stan_params tail_dataset
      in
      let head_survival_analysis_result =
        survival_analysis_dataset_of_e stan_params head_dataset_of_e
      in
      (* Because the survival analysis results for all expressions have the same
      number of posterior samples, they can be safely zipped together. *)
      let zipped_list =
        List.zip_exn head_survival_analysis_result
          recursive_survival_analysis_result
      in
      List.map zipped_list
        ~f:(fun (head_list_samples, list_samples_other_expressions) ->
          (head_e, head_list_samples) :: list_samples_other_expressions)

(* Testing *)

let test () =
  let input_data_append =
    let index_list = List.range 2 20 in
    List.map index_list ~f:(fun x -> (x - 1, x))
  in
  let size_matrix_ocaml =
    List.map input_data_append ~f:(fun (input_size1, input_size2) ->
        [ 1; input_size1; input_size2 ])
  in
  let cost_vector_ocaml =
    List.map input_data_append ~f:(fun (input_size1, input_size2) ->
        Float.of_int input_size1)
  in
  let num_runtime_samples = List.length input_data_append in
  let num_sizes = 3 in
  let stan_params : Hybrid_aara_config.bayeswc_stan_params =
    { scale_beta = 5.; scale_s = 5.; num_chains = 2; num_stan_samples = 500 }
  in
  let raw_inference_result =
    bayesian_inference stan_params size_matrix_ocaml cost_vector_ocaml
      num_runtime_samples num_sizes
  in
  let beta_vector_distribution =
    extract_inferred_model_parameters raw_inference_result "beta_vector"
  in
  let s_distribution =
    extract_inferred_model_parameters raw_inference_result "s" |> List.concat
  in
  let inference_result =
    evaluate_inferred_worst_case_costs beta_vector_distribution s_distribution
      (List.zip_exn size_matrix_ocaml cost_vector_ocaml)
  in
  let () =
    print_endline "Cost vector";
    List.iter cost_vector_ocaml ~f:(fun x -> printf "%f " x);
    print_newline ()
  in
  let print_inferred_cost_vector cost_vector =
    print_endline "Inferred cost vector:";
    List.iter cost_vector ~f:(fun x -> printf "%f " x);
    print_newline ()
  in
  List.iter inference_result ~f:print_inferred_cost_vector
