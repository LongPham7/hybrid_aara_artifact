open Core
open Probability_distributions
open Hybrid_aara_config

(* This module implements the RaML-volesti interface. It lets us run the
hit-and-runt algorithms and reflective Hamiltonian Monte Carlo (HMC). *)

let warm_up_by_hit_and_run list_A list_b dim warmup_params =
  (* We must choose a sufficiently large value for Gaussian RDHR's variance.
  For example, if we use the variance of 1.0 for the pure Bayesian resource
  analysis of the append function, Gaussian RDHR does not terminate, because
  it breaks the infinite loop in
  volesti/include/random_walks/gaussian_rdhr_walk.hpp with an extremely low
  probability. *)
  let { algorithm; variance; num_samples; walk_length } = warmup_params in
  let warm_up_flattened_result =
    match algorithm with
    | Gaussian_rdhr ->
        Volesti_hit_and_run.gaussian_rdhr list_A list_b variance num_samples
          walk_length
    | Uniform_rdhr ->
        Volesti_hit_and_run.uniform_rdhr list_A list_b num_samples walk_length
    | Gaussian_cdhr ->
        Volesti_hit_and_run.gaussian_cdhr list_A list_b variance num_samples
          walk_length
    | Uniform_cdhr ->
        Volesti_hit_and_run.uniform_cdhr list_A list_b num_samples walk_length
    | Uniform_billiard ->
        Volesti_hit_and_run.uniform_billiard list_A list_b num_samples
          walk_length
  in
  let warmup_result =
    Volesti_common.partition_into_blocks warm_up_flattened_result dim
  in
  let algorithm_name =
    match algorithm with
    | Gaussian_rdhr -> "Gaussian RDHR"
    | Uniform_rdhr -> "Uniform RDHR"
    | Gaussian_cdhr -> "Gaussian CDHR"
    | Uniform_cdhr -> "Uniform CDHR"
    | Uniform_billiard -> "Uniform billiard walk"
  in
  let () =
    print_endline ("Result of warm-up by " ^ algorithm_name ^ ":");
    Volesti_common.print_volesti_result warmup_result
  in
  warmup_result

let bayesian_inference_with_linear_constraints ?(selected_coefficients = None)
    matrix_A list_b list_vars equality_constraints runtime_data warmup_params
    hmc_params qs atarg =
  let dim = List.length list_vars in
  let list_A = List.concat matrix_A in
  let warmup_result = warm_up_by_hit_and_run list_A list_b dim warmup_params in
  let lipschitz = 1. in
  let m = 1. in
  let {
    coefficient_distribution_with_target;
    cost_model_with_target;
    num_samples;
    walk_length;
    step_size;
  } =
    hmc_params
  in
  let {
    distribution = coefficient_distribution;
    target = coefficient_distribution_target;
  } =
    coefficient_distribution_with_target
  in
  let { distribution = cost_model; target = cost_model_target } =
    cost_model_with_target
  in
  let coefficient_distribution_target_c =
    match selected_coefficients with
    | None ->
        Volesti_hmc
        .convert_coefficient_distribution_target_without_selection_ocaml_to_c
          coefficient_distribution_target
    | Some list_selected_coefficients ->
        Volesti_hmc
        .convert_coefficient_distribution_target_with_selection_ocaml_to_c
          list_selected_coefficients coefficient_distribution_target
  in
  let cost_model_target_c =
    Volesti_hmc.convert_distribution_target_ocaml_to_c cost_model_target
  in
  let starting_point = List.last_exn warmup_result in
  let () =
    print_endline "Starting point of reflective HMC:";
    Volesti_common.print_volesti_vector starting_point
  in
  let hmc_flattened_result =
    Volesti_hmc.hmc list_A list_b lipschitz m num_samples walk_length step_size
      starting_point list_vars equality_constraints runtime_data
      coefficient_distribution cost_model coefficient_distribution_target_c
      cost_model_target_c
  in
  let hmc_result =
    Volesti_common.partition_into_blocks hmc_flattened_result dim
  in
  let () =
    print_endline "Result of reflective HMC:";
    Volesti_common.print_volesti_result hmc_result
  in
  hmc_result

(* Bayesian inference on each size category present in runtime cost data. This
is only used in the old version of LP_with_runtime_data_adjusted_by_BI before we
incorporate Weibull survival analysis. *)

let extract_n_worst_case_samples ?(n = 10) list_list_costs =
  let extract_n_largest_values list_costs =
    let list_costs_sorted =
      List.sort ~compare:(fun x y -> -Float.compare x y) list_costs
    in
    List.take list_costs_sorted n
  in
  List.map list_list_costs ~f:extract_n_largest_values

let extract_worst_case_samples list_list_costs =
  let extract_largest_value list_costs =
    List.fold list_costs ~init:0. ~f:(fun acc x -> Float.max acc x)
  in
  List.map list_list_costs extract_largest_value

let bayesian_inference_on_cost_data_of_e_categorized_by_sizes warmup_params
    hmc_params dataset =
  let dataset_categorized_by_sizes =
    Runtime_cost_data.categorize_cost_data_by_sizes_dataset_of_e dataset
  in
  let list_list_costs =
    dataset_categorized_by_sizes
    |> List.map ~f:(fun (_, _, list_cost) -> list_cost)
    |> extract_n_worst_case_samples
  in
  (* For debugging *)
  let () =
    print_endline "Categorization of dataset by sizes:";
    printf "Number of size categories = %i\n" (List.length list_list_costs);
    print_string "Number of samples in each size category: ";
    List.iter list_list_costs ~f:(fun xs -> printf "%i " (List.length xs));
    print_newline ();
    print_string "Worst-case cost of each size category: ";
    List.iter (extract_worst_case_samples list_list_costs) ~f:(fun x ->
        printf "%.1f " x);
    print_newline ()
  in
  let list_box_constraint_rows =
    Linear_programs_processing
    .create_box_constraints_for_cost_data_categorized_by_sizes list_list_costs
  in
  let dim = List.length list_list_costs in
  let list_vars = List.range 0 dim in
  let matrix_A, list_b =
    Linear_programs_processing.transform_rows_to_matrix list_box_constraint_rows
      list_vars
  in
  let list_A = List.concat matrix_A in
  let warmup_result = warm_up_by_hit_and_run list_A list_b dim warmup_params in
  let lipschitz = 1. in
  let m = 1. in
  let { cost_model; num_samples; walk_length; step_size } = hmc_params in
  let starting_point = List.last_exn warmup_result in
  let () =
    print_endline "Starting point of reflective HMC:";
    Volesti_common.print_volesti_vector starting_point
  in
  let hmc_flattened_result =
    Volesti_hmc.hmc_on_cost_data list_A list_b list_list_costs lipschitz m
      num_samples walk_length step_size starting_point cost_model
  in
  let hmc_result =
    Volesti_common.partition_into_blocks hmc_flattened_result dim
  in
  let () =
    print_endline "Result of reflective HMC:";
    Volesti_common.print_volesti_result hmc_result
  in
  (* In the dataset categorized by sizes, for each combination of sizes, insert
  the inference result of HMC. *)
  let rec zip_dataset_categorized_by_sizes_with_list_cost dataset list_cost =
    match (dataset, list_cost) with
    | [], _ -> []
    | _, [] -> failwith "The dataset and list of costs have different lengths"
    | head_sample :: tail_samples, head_cost :: tail_costs ->
        let list_cindex_potential, list_index_potential, _ = head_sample in
        (list_cindex_potential, list_index_potential, head_cost)
        :: zip_dataset_categorized_by_sizes_with_list_cost tail_samples
             tail_costs
  in
  List.map hmc_result
    ~f:
      (zip_dataset_categorized_by_sizes_with_list_cost
         dataset_categorized_by_sizes)

let rec bayesian_inference_on_cost_data_categorized_by_sizes warmup_params
    hmc_params dataset =
  match dataset with
  | [] ->
      failwith
        "The dataset is empty in the Bayesian inference for the cost data"
  | [ (head_e, head_dataset_of_e) ] ->
      let head_dataset_of_e_hmc_result =
        bayesian_inference_on_cost_data_of_e_categorized_by_sizes warmup_params
          hmc_params head_dataset_of_e
      in
      List.map head_dataset_of_e_hmc_result ~f:(fun x -> [ (head_e, x) ])
  | (head_e, head_dataset_of_e) :: tail_dataset ->
      let recursive_hmc_result =
        bayesian_inference_on_cost_data_categorized_by_sizes warmup_params
          hmc_params tail_dataset
      in
      let head_dataset_of_e_hmc_result =
        bayesian_inference_on_cost_data_of_e_categorized_by_sizes warmup_params
          hmc_params head_dataset_of_e
      in
      (* Because the HMC inference results for all expressions have the same
      number of posterior samples, they can be safely zipped together. *)
      let zipped_list =
        List.zip_exn head_dataset_of_e_hmc_result recursive_hmc_result
      in
      List.map zipped_list ~f:(fun (head_sample, sample_recursive_result) ->
          (head_e, head_sample) :: sample_recursive_result)
