open Core
open Rconfig
open Hybrid_aara_config
open Toolbox
open Rtypes
open Expressions
open Solver
open Indices
open Annotations
open Metric
open Probability_distributions
open Analysis

(* This module is for the resource analysis of OCaml expressions and functions.
The analysis of expressions is done by the function analyze_expression, and the
resource analysis of functions is performed by the function analyze_function.
This module builds on the module Analysis, which provides the functions forward
and backward to walk through an expression and collect linear constraints.

For the resource analysis of functions, we can perform hybrid AARA, where we
combine inference results of both static analysis (i.e., conventional AARA) and
data-driven analysis (i.e., Bayesian inference). Hybrid AARA is only available
in the resource analysis of functions - it does not support the resource
analysis of expressions. So the function analyze_expression only performs
conventional AARA. *)

module Make (Solver : SOLVER) (Amode : AMODE) = struct
  (* This module Analysis_with_data builds on the module Analysis. *)
  module Analysis = Analysis.Make (Solver) (Amode)
  module Anno = Analysis.Anno

  let amode_zero_out_tanno = Analysis.amode_zero_out_tanno

  let init_type_collection = Analysis.init_type_collection

  let forward = Analysis.forward

  let backward = Analysis.backward

  let zero_fanno_type = Analysis.zero_fanno_type

  let apply_fanno = Analysis.apply_fanno

  type analysis_arg = Analysis.analysis_arg

  let recorded_fun_types = Analysis.recorded_fun_types

  exception Analysis_error = Analysis.Analysis_error

  exception Anno_exn = Analysis.Anno_exn

  (* This function only performs conventional AARA - it does not support hybrid
  AARA. *)
  let analyze_expression ~degree ~metric ?(collect_fun_types = Pnone) exp =
    let rtanno = Anno.fresh_tanno degree exp.exp_type in
    let () = amode_zero_out_tanno rtanno in
    let arg : analysis_arg =
      {
        anl_exp = exp;
        anl_metric = metric;
        anl_deg = degree;
        anl_rtanno = rtanno;
        anl_sigma = String.Map.empty;
        anl_tstack = [];
        anl_level = 0;
        (* This function analyze_expression only supports conventional AARA. So
        the runtime cost data in the field anl_hybrid_aara_data is set to None.
        *)
        anl_hybrid_aara_data = None;
      }
    in
    init_type_collection collect_fun_types;
    let canno = backward arg in
    let q0 = can_find canno ([], String.Map.empty) in
    let _ = Anno.S.add_objective q0 1.0 in
    match Anno.S.first_solve () with
    | Feasible ->
        let get_solution = Anno.S.get_solution in
        let update_tanno tanno = tanno_map get_solution tanno in
        let fun_types =
          List.map !recorded_fun_types (fun (fid, atanno, rtanno) ->
              (fid, update_tanno atanno, update_tanno rtanno))
        in
        Some (get_solution q0, fun_types)
    | Infeasible -> None

  exception LP_infeasible

  let get_prioritized_vars raml_type annotated_type degree =
    let iss = prioritized_indices raml_type degree in
    (* For debugging *)
    (* let () =
      print_endline
        "Grouping of indices in the descending order of priorities/degrees: ";
      List.iter iss ~f:(fun is ->
          List.iter is ~f:Indices.print_index;
          print_newline ())
    in
    let () =
      let f_print_information indices =
        List.iter
          ~f:(fun index ->
            Indices.print_index index;
            let q = tan_find annotated_type index in
            printf "; corresponding lp_var: %s\n"
              (Sexp.to_string (Anno.S.sexp_of_var q)))
          indices
      in
      print_endline "Indices and their corresponding LP variables: ";
      List.iter ~f:f_print_information iss
    in *)
    let f is =
      List.map
        ~f:(fun i ->
          let q = tan_find annotated_type i in
          q)
        is
    in
    List.map ~f iss

  let get_prioritized_list_vars list_raml_types annotated_type degree =
    get_prioritized_vars (Ttuple list_raml_types) annotated_type degree

  (* This type captures the output of analyzing a function where, in the output,
  each index is mapped to a single coefficient. The output of the
  conventional, non-hybrid AARA has this type. *)
  type function_analysis_output_coefficient =
    float Annotations.type_anno
    * float Annotations.type_anno
    * (string * float Annotations.type_anno * float Annotations.type_anno) list

  type list_coefficients = float list

  (* This type captures the output of analyzing a function where, in the output,
  each index is mapped to a list of coefficients, as opposed to a single
  coefficient. This list represents/approximates a probability distribution of
  coefficients. The output of hybrid AARA has this type. *)
  type function_analysis_output_distribution =
    list_coefficients Annotations.type_anno
    * list_coefficients Annotations.type_anno
    * ( string
      * list_coefficients Annotations.type_anno
      * list_coefficients Annotations.type_anno )
      list

  (* Output type of analyzing a function. In the output, each index is mapped to
  either a single coefficient or a list of coefficients (i.e. a probability
  distribution of coefficients). *)
  type function_analysis_output =
    | Output_coefficient of function_analysis_output_coefficient
    | Output_distribution of function_analysis_output_distribution

  (* Solve a linear program with the objective function given by qs. If it is
  feasible, we add the objective function's value as a constraint to the linear
  program. *)
  let solve_with_obj ?(minimization = true) solver_method qs =
    let module VMap = Anno.S.VMap in
    let objective =
      let f vmap q =
        if minimization then VMap.set vmap q 1.0 else VMap.set vmap q (-1.0)
      in
      List.fold qs ~init:VMap.empty ~f
    in
    let () = Anno.S.set_objective objective in
    match solver_method () with
    | Infeasible -> raise LP_infeasible
    | Feasible ->
        let obj_val = Anno.S.get_objective_val () in
        let obj_list = Anno.S.VMap.to_alist objective in
        (* The following linear constraint is added to the original linear
        program in order to fix the optimal values of resource coefficients.
        We traverse the resource coefficients in the descending order of
        degrees/priorities, compute their optimal values, and then fix them by
        adding linear constrains to the original linear program. In effect, we
        solve a multi-objective linear program where the objectives are given
        by the resource coefficients lexicographically ordered by degrees. *)
        Anno.S.add_constr_list ~lower:0.0 ~upper:obj_val obj_list

  (* Given a mapping from indices to LP variables, extract an annotated type
  from the LP solution stored in a solver. This function is used whenever we want
  to store the current LP solution before resetting the solver for the next
  instance of LP. *)
  let extract_annotated_type_from_lp_solution annotated_type =
    let { tan_type = t; tan_deg = deg; tan_map = f } = annotated_type in
    let empty = [] and add_ind index acc = index :: acc in
    let all_indices = indices_max_deg ~empty ~add_ind t deg in
    let list_index_coefficient =
      List.map all_indices ~f:(fun index ->
          (index, Anno.S.get_solution (f index)))
    in
    let equal_indices x y = Indices.compare_index x y = 0 in
    let index_map index =
      List.Assoc.find_exn list_index_coefficient ~equal:equal_indices index
    in
    { annotated_type with tan_map = index_map }

  (* Solve a linear program with (lexicographically ordered) multiple objective
  functions. The lexicographic order is determined by the degrees of indices.
  Higher-degree coefficients have higher priorities than lower-degree
  coefficients. Input atarg is the annotated type for arguments, and atres is
  the annotated type for results.

  In addition to minimizing the input resource coefficients, users can opt to
  maximize the output resource coefficients. But the resulting linear program
  may be unbounded. For example, if a function always returns the empty list,
  the function's output resource coefficients can be anything due to the empty
  list. *)
  let solve_with_multi_obj qs_max qss ?(maybe_qss_return_type = None) atarg
      atres =
    try
      let () = solve_with_obj Anno.S.first_solve qs_max in
      let () = List.iter ~f:(solve_with_obj Anno.S.resolve) qss in
      let () =
        match maybe_qss_return_type with
        | None -> ()
        | Some qss_return_type ->
            List.iter
              ~f:(solve_with_obj ~minimization:false Anno.S.resolve)
              qss_return_type
      in
      (* In the original RaML code base, the annotated argument type and
      annotated return type are created using the update function defined below.
      However, because the update function is partially applied, at runtime, it
      does not evaluate the closure Anno.S.get_solution. It is only evaluated
      when the closure is actually applied to some argument. Furthermore,
      Anno.S.get_solution inside update's definition accesses a ref value
      solution. This ref is initialized to the empty value whenever the solver
      is reset. Consequently, if we use the update function to later access the
      previous LP solution, we will unexpectedly get an LP solution of the next
      linear program. To avoid this pitfall, we use the helper function
      extract_annotated_type_from_lp_solution to retrieve the ref solution right
      away. *)
      (* let update_tanno tanno = tanno_map Anno.S.get_solution tanno in
      let fun_types =
        List.map !recorded_fun_types (fun (fid, atanno, rtanno) ->
            (fid, update_tanno atanno, update_tanno rtanno))
      in *)
      let fun_types =
        List.map !recorded_fun_types (fun (fid, atanno, rtanno) ->
            ( fid,
              extract_annotated_type_from_lp_solution atanno,
              extract_annotated_type_from_lp_solution rtanno ))
      in
      Some
        ( extract_annotated_type_from_lp_solution atarg,
          extract_annotated_type_from_lp_solution atres,
          fun_types )
    with LP_infeasible -> None

  (* Analyze a function by solving linear programs, possibly augmented with
  hybrid AARA's data. The data is either None (in the mode Disabled) or
  runtime cost data (in the modes Opt, BayesWC, and BayesPC). *)
  let analyze_function_by_lp_with_data ~degree ~metric
      ?(collect_fun_types = Pnone) exp f_name hybrid_aara_data =
    let sigma = String.Map.empty in
    match (forward sigma exp hybrid_aara_data, exp.exp_type) with
    | Fopen _, _ -> raise (Analysis_error "Expecting closed forward result.")
    | Fclosed (Farrow fanno), Tarrow (targs, tres, _) -> (
        assert (fanno.fan_targs = targs);
        assert (fanno.fan_tres = tres);
        let () = init_type_collection collect_fun_types in
        let ftargs = List.map targs zero_fanno_type in
        let _, fa = apply_fanno fanno ftargs in
        let atres = Anno.fresh_tanno degree tres in
        (* For debugging *)
        (* Anno.print_tanno atres; *)
        let () = amode_zero_out_tanno atres in
        let atarg = fa atres metric in
        let qss = get_prioritized_list_vars targs atarg degree in
        (* If we want to maximize the output resource coefficients in addition
        to minimizing the input resource coefficients, we pass qss_return_type
        to solve_with_multi_obj. *)
        (* let qss_return_type = get_prioritized_vars tres atres degree in *)
        (* Write out the linear program in an MPS file before any objective
        functions are added as linear constraints to the original linear
        program *)
        (* let () =
          Anno.S.write_to_file
            ("/home/longpham/Desktop/raml/lp_files/lp_file_" ^ f_name ^ ".mps")
        in *)
        match qss with
        | [] -> raise (Analysis_error "Empty index set.")
        | qs_max :: qss -> solve_with_multi_obj qs_max qss atarg atres )
    | Fclosed _, _ -> raise (Analysis_error "Expecting function type.")

  (* Given a list of annotated types where indices are mapped to single resource
  coefficients, merge them into a single annotated type where indices are mapped
  to lists of coefficients. *)
  let combine_list_annotated_types list_annotated_type =
    assert (not (List.is_empty list_annotated_type));
    let { tan_type; tan_deg } = List.hd_exn list_annotated_type in
    (* Check that all annotated types in the input list have the same tan_type
    and tan_deg. *)
    assert (
      List.for_all list_annotated_type ~f:(fun x ->
          let { tan_deg = current_tan_deg } = x in
          current_tan_deg = tan_deg) );
    assert (
      List.for_all list_annotated_type ~f:(fun x ->
          let { tan_type = current_tan_type } = x in
          Rtypes.compare_rtype (fun () () -> 0) current_tan_type tan_type = 0)
    );
    let index_map index =
      List.map list_annotated_type ~f:(fun x ->
          let { tan_map } = x in
          tan_map index)
    in
    { tan_type; tan_deg; tan_map = index_map }

  let combine_list_annotated_recorded_function_types list_annotated_types =
    let insert_list_types_into_acc acc current_list =
      match acc with
      | [] ->
          List.map current_list ~f:(fun (function_name, atarg, atres) ->
              (function_name, [ atarg ], [ atres ]))
      | _ ->
          let rec insert_single_type_into_acc acc current_type =
            let current_fname, current_atarg, current_atres = current_type in
            match acc with
            | [] -> [ (current_fname, [ current_atarg ], [ current_atres ]) ]
            | (head_fname, head_atarg, head_atres) :: acc_tail ->
                if String.equal current_fname head_fname then
                  ( current_fname,
                    current_atarg :: head_atarg,
                    current_atres :: head_atres )
                  :: acc_tail
                else
                  (head_fname, head_atarg, head_atres)
                  :: insert_single_type_into_acc acc_tail current_type
          in
          List.fold current_list ~init:acc ~f:insert_single_type_into_acc
    in
    let list_fname_atarg_atres =
      List.fold list_annotated_types ~init:[] ~f:insert_list_types_into_acc
    in
    List.map list_fname_atarg_atres ~f:(fun (fname, list_atarg, list_atres) ->
        ( fname,
          combine_list_annotated_types list_atarg,
          combine_list_annotated_types list_atres ))

  (* Analyze a function in the mode Disabled (i.e. without using hybrid
  AARA's data). This basically amounts to the conventional AARA. *)
  let analyze_function_hybrid_aara_disabled ~degree ~metric
      ?(collect_fun_types = Pnone) exp f_name =
    let solution =
      analyze_function_by_lp_with_data ~degree ~metric ~collect_fun_types exp
        f_name None
    in
    match solution with
    | None -> None
    | Some feasible_solution -> Some (Output_coefficient feasible_solution)

  let merge_two_lists_var_coefficient list_var_coefficient1
      list_var_coefficient2 =
    let rec insert_into_acc acc (var, coefficient) =
      match acc with
      | [] -> [ (var, coefficient) ]
      | (head_v, head_c) :: acc_tail ->
          if var = head_v then (var, coefficient +. head_c) :: acc_tail
          else (head_v, head_c) :: insert_into_acc acc_tail (var, coefficient)
    in
    List.fold list_var_coefficient2 ~init:list_var_coefficient1
      ~f:insert_into_acc

  let remove_duplicate_var list_var_coefficient =
    merge_two_lists_var_coefficient [] list_var_coefficient

  let create_cost_gap_objective_dataset_of_e dataset_of_e =
    let sum_list_potential list_potential =
      List.fold list_potential ~init:0. ~f:(fun x y -> x +. y)
    in
    let squash_list_potential list_var_potential =
      List.map list_var_potential ~f:(fun (v, list_p) ->
          (v, sum_list_potential list_p))
    in
    let squash_and_negate_list_potential list_var_potential =
      List.map list_var_potential ~f:(fun (v, list_p) ->
          (v, -.sum_list_potential list_p))
    in
    let list_cindex_var_potential, list_index_var_potential, _ = dataset_of_e in
    let list_cindex_var_sum_potential =
      list_cindex_var_potential |> squash_list_potential |> remove_duplicate_var
    in
    (* We negate the coefficients of LP variables in the return type before
    merging them with the typing context. *)
    let list_index_var_sum_potential =
      list_index_var_potential |> squash_and_negate_list_potential
      |> remove_duplicate_var
    in
    merge_two_lists_var_coefficient list_cindex_var_sum_potential
      list_index_var_sum_potential

  let create_cost_gap_objective_runtime_data runtime_data =
    let list_list_var_coefficient =
      List.map runtime_data ~f:create_cost_gap_objective_dataset_of_e
    in
    List.fold list_list_var_coefficient ~init:[]
      ~f:merge_two_lists_var_coefficient

  (* Evaluate the cost gap after optimization. It is used for a sanity check. *)
  let evaluate_cost_gap_runtime_data runtime_data =
    let sum_list_potential list_potential =
      List.fold list_potential ~init:0. ~f:(fun x y -> x +. y)
    in
    let evaluate_potential_var (var_int, list_potential) =
      let index_value = var_int |> Anno.S.int_to_var |> Anno.S.get_solution in
      let total_potential = sum_list_potential list_potential in
      index_value *. total_potential
    in
    let evaluate_potential_list_vars list_var_int_potential =
      let list_var_total_potential =
        List.map list_var_int_potential ~f:evaluate_potential_var
      in
      sum_list_potential list_var_total_potential
    in
    let evaluate_cost_gap_dataset_of_e dataset_of_e =
      let list_cindex_var_potential, list_index_var_potential, list_costs =
        dataset_of_e
      in
      let input_potential =
        evaluate_potential_list_vars list_cindex_var_potential
      in
      let output_potential =
        evaluate_potential_list_vars list_index_var_potential
      in
      let total_costs = sum_list_potential list_costs in
      input_potential -. output_potential -. total_costs
    in
    let list_total_cost_gaps =
      List.map runtime_data ~f:evaluate_cost_gap_dataset_of_e
    in
    sum_list_potential list_total_cost_gaps

  let solve_with_runtime_data qs_max qss atarg atres runtime_data
      lp_objective_params =
    let module VMap = Anno.S.VMap in
    let cost_gap_objective_list_var_int =
      create_cost_gap_objective_runtime_data runtime_data
    in
    let cost_gap_objective_list_var =
      List.map cost_gap_objective_list_var_int ~f:(fun (v, c) ->
          (Anno.S.int_to_var v, c))
    in
    let cost_gap_objective_map =
      let f vmap (var, coefficient) = VMap.set vmap var coefficient in
      List.fold cost_gap_objective_list_var ~init:VMap.empty ~f
    in
    let { cost_gaps_optimization; coefficients_optimization } =
      lp_objective_params
    in
    (* For debugging *)
    (* let () =
      let list_rows =
        Anno.S.get_list_rows ()
        |> Linear_programs_processing.convert_clp_rows_to_rows_in_lists
      in
      print_endline
        "This is the list of rows in the function solve_with_runtime_data";
      Linear_programs_processing.print_list_rows list_rows
    in *)
    try
      (* First optimize cost gaps if cost_gaps = true. *)
      let () =
        if cost_gaps_optimization then
          let () = Anno.S.set_objective cost_gap_objective_map in
          match Anno.S.first_solve () with
          | Infeasible -> raise LP_infeasible
          | Feasible ->
              let obj_val = Anno.S.get_objective_val () in
              Anno.S.add_constr_list ~lower:0.0 ~upper:obj_val
                cost_gap_objective_list_var
        else ()
      in
      (* Next, optimize coefficients, either with equal weights or lexicographic
      weights. *)
      let () =
        match coefficients_optimization with
        | Equal_weights ->
            solve_with_obj Anno.S.resolve (List.concat (qs_max :: qss))
        | Lexicographic_weights ->
            let () = solve_with_obj Anno.S.resolve qs_max in
            List.iter ~f:(solve_with_obj Anno.S.resolve) qss
      in
      (* For debugging *)
      (* let () =
        let cost_gap = evaluate_cost_gap_runtime_data runtime_data in
        printf "Total cost gaps = %f\n" cost_gap
      in *)
      let fun_types =
        List.map !recorded_fun_types (fun (fid, atanno, rtanno) ->
            ( fid,
              extract_annotated_type_from_lp_solution atanno,
              extract_annotated_type_from_lp_solution rtanno ))
      in
      Some
        ( extract_annotated_type_from_lp_solution atarg,
          extract_annotated_type_from_lp_solution atres,
          fun_types )
    with LP_infeasible -> None

  let analyze_function_opt_and_custom_objective ~degree ~metric
      ?(collect_fun_types = Pnone) lp_objective runtime_data exp =
    let sigma = String.Map.empty in
    match (forward sigma exp (Some runtime_data), exp.exp_type) with
    | Fopen _, _ -> raise (Analysis_error "Expecting closed forward result.")
    | Fclosed (Farrow fanno), Tarrow (targs, tres, _) -> (
        assert (fanno.fan_targs = targs);
        assert (fanno.fan_tres = tres);
        let () = init_type_collection collect_fun_types in
        let ftargs = List.map targs zero_fanno_type in
        let _, fa = apply_fanno fanno ftargs in
        let atres = Anno.fresh_tanno degree tres in
        (* For debugging *)
        (* Anno.print_tanno atres; *)
        let () = amode_zero_out_tanno atres in
        let atarg = fa atres metric in
        let qss = get_prioritized_list_vars targs atarg degree in
        match qss with
        | [] -> raise (Analysis_error "Empty index set.")
        | qs_max :: qss ->
            let runtime_data_with_lp_var =
              Anno.S.get_runtime_data_tick_metric ()
            in
            solve_with_runtime_data qs_max qss atarg atres
              runtime_data_with_lp_var lp_objective )
    | Fclosed _, _ -> raise (Analysis_error "Expecting function type.")

  let write_lp_vars_inferred_coefficients output_params =
    match output_params with
    | None -> ()
    | Some output_params ->
        let { output_lp_vars; output_file } = output_params in
        let list_lp_vars_coefficients =
          List.map output_lp_vars ~f:(fun v ->
              (v, Anno.S.get_solution (Anno.S.int_to_var v)))
        in
        Hybrid_aara_pprint.write_lp_vars_inferred_coefficients_to_json_file
          list_lp_vars_coefficients output_file

  (* Analyze a function in the mode Opt *)
  let analyze_function_opt ~degree ~metric ?(collect_fun_types = Pnone)
      analysis_params exp f_name =
    let ({ lp_objective; output_params; dataset = runtime_data } : opt_params) =
      analysis_params
    in
    let solution =
      analyze_function_opt_and_custom_objective ~degree ~metric
        ~collect_fun_types lp_objective runtime_data exp
    in
    match solution with
    | None -> None
    | Some feasible_solution ->
        let () =
          write_lp_vars_inferred_coefficients output_params;
          let maximum_value_soution =
            match Anno.S.get_largest_value_in_solution () with
            | None -> failwith "A solution to LP is unavailable"
            | Some x -> x
          in
          printf "Largest value in the optimal solution = %f\n"
            maximum_value_soution
        in
        Some (Output_coefficient feasible_solution)

  let create_box_constraints_bayespc num_vars box_constraint_params rows =
    let { upper_bound; upper_unbounded_vars } = box_constraint_params in
    Linear_programs_processing.create_box_constraints ~upper_bound
      ~upper_unbounded_vars num_vars
    @ rows

  let repeatedly_remove_implicit_equalities_bayespc num_vars
      implicit_equality_removal_params inequality_rows_and_equality_constraints
      =
    if implicit_equality_removal_params then
      Implicit_equalities_perturbations
      .repeatedly_remove_implicit_equalities_by_perturbations num_vars
        inequality_rows_and_equality_constraints
    else inequality_rows_and_equality_constraints

  let set_return_type_vars_to_zero_bayespc output_potential_set_to_zero_params
      list_vars list_rows =
    if output_potential_set_to_zero_params then
      Linear_programs_processing.set_return_type_vars_to_zero list_vars
        list_rows
    else list_rows

  let process_linear_program_bayespc num_vars list_rows lp_params qs_return_type
      =
    let {
      box_constraint;
      implicit_equality_removal = implicit_equality_removal_params;
      output_potential_set_to_zero = output_potential_set_to_zero_params;
    } =
      lp_params
    in
    let list_return_type_var_ints =
      List.map qs_return_type ~f:Anno.S.var_to_int
    in
    let list_inequality_rows, list_equality_constraints =
      list_rows |> Linear_programs_processing.convert_clp_rows_to_rows_in_lists
      |> Linear_programs_processing.preprocess_rows
      |> create_box_constraints_bayespc num_vars box_constraint
      |> set_return_type_vars_to_zero_bayespc
           output_potential_set_to_zero_params list_return_type_var_ints
      |> Linear_programs_processing.remove_equality_constraints
      |> Linear_programs_processing.repeatedly_merge_box_constraints
      |> repeatedly_remove_implicit_equalities_bayespc num_vars
           implicit_equality_removal_params
    in
    let remaining_vars, _ =
      Linear_programs_processing.categorize_LP_variables num_vars
        list_equality_constraints
    in
    let matrix_A, list_b =
      Linear_programs_processing.transform_rows_to_matrix list_inequality_rows
        remaining_vars
    in
    let () =
      printf
        "Before equality elimination: number of vars: %i; number of \
         constraints: %i\n"
        num_vars
        (Anno.S.get_num_constraints ());
      printf
        "After equality elimination: number of vars: %i; number of inequality \
         constraints: %i; number of equality constraints: %i\n"
        (List.length remaining_vars)
        (List.length list_inequality_rows)
        (List.length list_equality_constraints)
    in
    (* For debugging *)
    (* let () =
      printf "List of %i remaining vars: " (List.length remaining_vars);
      List.iter remaining_vars ~f:(fun v -> printf "%i " v);
      print_newline ();
      print_endline "List of rows after removing equality constraints:";
      Linear_programs_processing.print_list_rows list_inequality_rows;
      print_endline "List of equality constraints:";
      Linear_programs_processing.print_list_equality_constraints
        list_equality_constraints
    in *)
    (* For debugging *)
    (* let () =
      print_endline "Matrix A and vector b:";
      printf "Number of rows: %i; number of cols: %i\n" (List.length matrix_A)
        (List.length remaining_vars);
      Hybrid_aara_pprint.print_linear_program matrix_A list_b
    in *)
    (matrix_A, list_b, remaining_vars, list_equality_constraints)

  let write_lp_vars_distributions_with_adjustment
      list_samples_from_posterior_distribution remaining_vars
      list_equality_constraints output_params =
    let { output_lp_vars; output_file } = output_params in
    let retrieve_coefficient var_int sample =
      Linear_programs_processing.get_value_of_LP_variable remaining_vars
        list_equality_constraints var_int sample
    in
    let retrieve_distribution var_int =
      List.map list_samples_from_posterior_distribution
        ~f:(retrieve_coefficient var_int)
    in
    let list_lp_vars_distribution =
      List.map output_lp_vars ~f:(fun v -> (v, retrieve_distribution v))
    in
    Hybrid_aara_pprint.write_lp_vars_distributions_to_json_file
      list_lp_vars_distribution output_file

  let extract_coefficients_from_runtime_data_with_lp_vars
      runtime_data_with_lp_vars =
    let extract_coefficients_from_dataset_of_e dataset_of_e =
      let list_cindex_var_potential, list_index_var_potential, _ =
        dataset_of_e
      in
      let list_cindex_vars =
        List.map list_cindex_var_potential ~f:(fun (v, _) -> v)
      in
      let list_index_vars =
        List.map list_index_var_potential ~f:(fun (v, _) -> v)
      in
      list_cindex_vars @ list_index_vars
    in
    let list_vars =
      List.concat
        (List.map runtime_data_with_lp_vars
           ~f:extract_coefficients_from_dataset_of_e)
    in
    List.dedup_and_sort list_vars ~compare:Int.compare

  let adjust_vars_in_list remaining_vars list_equality_constraints list_vars =
    let adjust_var var =
      let list_index_after_adjustment =
        Runtime_data_for_automatic_differentiation.map_var_to_list_index
          remaining_vars list_equality_constraints var
      in
      let list_index_coefficient, _ = list_index_after_adjustment in
      List.map list_index_coefficient ~f:(fun (v, _) -> v)
    in
    let list_adjusted_vars = List.concat (List.map list_vars ~f:adjust_var) in
    List.dedup_and_sort list_adjusted_vars ~compare:Int.compare

  let add_lp_constraint_to_fix_variable (var_int, value) =
    let constraint_list = [ (Anno.S.int_to_var var_int, 1.0) ] in
    (* let epsilon = 0.0001 in
    Anno.S.add_constr_list ~lower:(value -. epsilon) ~upper:(value +. epsilon)
      constraint_list *)
    Anno.S.add_constr_list ~lower:value ~upper:value constraint_list

  let analyze_function_by_lp_with_list_fixed_lp_vars ~degree ~metric
      ?(collect_fun_types = Pnone) exp f_name runtime_data list_fixed_lp_vars =
    let sigma = String.Map.empty in
    match (forward sigma exp (Some runtime_data), exp.exp_type) with
    | Fopen _, _ -> raise (Analysis_error "Expecting closed forward result.")
    | Fclosed (Farrow fanno), Tarrow (targs, tres, _) -> (
        assert (fanno.fan_targs = targs);
        assert (fanno.fan_tres = tres);
        let () = init_type_collection collect_fun_types in
        let ftargs = List.map targs zero_fanno_type in
        let _, fa = apply_fanno fanno ftargs in
        let atres = Anno.fresh_tanno degree tres in
        (* For debugging *)
        (* Anno.print_tanno atres; *)
        let () = amode_zero_out_tanno atres in
        let atarg = fa atres metric in
        let () =
          List.iter list_fixed_lp_vars ~f:add_lp_constraint_to_fix_variable
        in
        (* For debugging *)
        (* let () =
          let list_rows =
            Anno.S.get_list_rows ()
            |> Linear_programs_processing.convert_clp_rows_to_rows_in_lists
          in
          print_endline
            "This is the list of rows in the function \
             analyze_function_by_lp_with_list_fixed_lp_vars";
          Linear_programs_processing.print_list_rows list_rows
        in *)
        let qss = get_prioritized_list_vars targs atarg degree in
        match qss with
        | [] -> raise (Analysis_error "Empty index set.")
        | qs_max :: qss -> solve_with_multi_obj qs_max qss atarg atres )
    | Fclosed _, _ -> raise (Analysis_error "Expecting function type.")

  let analyze_all_lists_fixed_lp_vars ~degree ~metric
      ?(collect_fun_types = Pnone) exp f_name runtime_data
      list_list_fixed_lp_vars =
    let analyze list_fixed_lp_vars =
      (* Remember to reset the LP solver - it is stateful. *)
      Anno.S.reset_everything ();
      analyze_function_by_lp_with_list_fixed_lp_vars ~degree ~metric
        ~collect_fun_types exp f_name runtime_data list_fixed_lp_vars
    in
    let merge_list_lp_feasible_solutions list_lp_feasible_solutions =
      let list_atarg, list_atres, list_fun_types =
        List.fold list_lp_feasible_solutions ~init:([], [], [])
          ~f:(fun (xs, ys, zs) (x, y, z) -> (x :: xs, y :: ys, z :: zs))
      in
      match list_lp_feasible_solutions with
      | [] -> None
      | _ ->
          let combined_atarg = combine_list_annotated_types list_atarg in
          let combined_atres = combine_list_annotated_types list_atres in
          let combined_fun_types =
            combine_list_annotated_recorded_function_types list_fun_types
          in
          Some
            (Output_distribution
               (combined_atarg, combined_atres, combined_fun_types))
    in
    let list_lp_feasible_solutions =
      List.filter_map list_list_fixed_lp_vars ~f:analyze
    in
    let () =
      printf "We found %i feasible solutions out of %i samples\n"
        (List.length list_lp_feasible_solutions)
        (List.length list_list_fixed_lp_vars)
    in
    merge_list_lp_feasible_solutions list_lp_feasible_solutions

  (* Analyze a function in the mode BayesPC *)
  let analyze_function_bayespc ~degree ~metric ?(collect_fun_types = Pnone)
      analysis_params exp f_name =
    let {
      lp_params;
      warmup_params;
      hmc_params;
      output_params;
      dataset = runtime_data;
    } =
      analysis_params
    in
    let sigma = String.Map.empty in
    match (forward sigma exp (Some runtime_data), exp.exp_type) with
    | Fopen _, _ -> raise (Analysis_error "Expecting closed forward result.")
    | Fclosed (Farrow fanno), Tarrow (targs, tres, _) -> (
        assert (fanno.fan_targs = targs);
        assert (fanno.fan_tres = tres);
        let () = init_type_collection collect_fun_types in
        let ftargs = List.map targs zero_fanno_type in
        let _, fa = apply_fanno fanno ftargs in
        let atres = Anno.fresh_tanno degree tres in
        (* For debugging *)
        (* Anno.print_tanno atres; *)
        let () = amode_zero_out_tanno atres in
        let atarg = fa atres metric in
        let qss = get_prioritized_list_vars targs atarg degree in
        (* Write out the linear program in an MPS file before any objective
        functions are added as linear contraints to the original linear
        program *)
        (* let () =
          Anno.S.write_to_file
            ("/home/longpham/Desktop/raml/lp_files/lp_file_" ^ f_name ^ ".mps")
        in *)
        let runtime_data_with_lp_vars =
          Anno.S.get_runtime_data_tick_metric ()
        in
        match (qss, runtime_data_with_lp_vars) with
        | [], _ -> raise (Analysis_error "Empty index set.")
        (* If runtime_data_with_lp_vars is empty, it means the function's typing
        tree contains no nodes for Bayesian inference.anl_rtanno So we simply
        perform conventional AARA. *)
        | qs_max :: qss, [] -> (
            let solution = solve_with_multi_obj qs_max qss atarg atres in
            match solution with
            | None -> None
            | Some feasible_solution ->
                Some (Output_coefficient feasible_solution) )
        | _, _ ->
            let qs_return_type =
              List.concat (get_prioritized_vars tres atres degree)
            in
            let num_vars = Anno.S.get_num_vars () in
            let list_rows = Anno.S.get_list_rows () in
            let matrix_A, list_b, remaining_vars, list_equality_constraints =
              process_linear_program_bayespc num_vars list_rows lp_params
                qs_return_type
            in
            let selected_coefficients_before_adjustment =
              extract_coefficients_from_runtime_data_with_lp_vars
                runtime_data_with_lp_vars
            in
            let selected_coefficients_after_adjustment =
              adjust_vars_in_list remaining_vars list_equality_constraints
                selected_coefficients_before_adjustment
            in
            (* For debugging *)
            let () =
              print_endline "List of selected variables before adjusgtment:";
              List.iter selected_coefficients_before_adjustment ~f:(fun v ->
                  printf "%i " v);
              print_newline ();
              print_endline "List of selected variables after adjusgtment:";
              List.iter selected_coefficients_after_adjustment ~f:(fun v ->
                  printf "%i " v);
              print_newline ()
            in
            let list_samples_from_posterior_distribution =
              Volesti_bayesian_inference
              .bayesian_inference_with_linear_constraints
                ~selected_coefficients:
                  (Some selected_coefficients_after_adjustment) matrix_A list_b
                remaining_vars list_equality_constraints
                runtime_data_with_lp_vars warmup_params hmc_params
                (List.concat qss) atarg
            in
            (* Write out the posterior distributions of user-specified LP
            variables in a file. This is used when we want to check the
            posterior distributions of internal nodes of a typing tree. *)
            let () =
              match output_params with
              | None -> ()
              | Some output_params_content ->
                  write_lp_vars_distributions_with_adjustment
                    list_samples_from_posterior_distribution remaining_vars
                    list_equality_constraints output_params_content
            in
            let list_list_fixed_lp_vars =
              let retrieve_var_from_sample var sample =
                Linear_programs_processing.get_value_of_LP_variable
                  remaining_vars list_equality_constraints var sample
              in
              let retrieve_all_selected_vars sample =
                List.map selected_coefficients_before_adjustment ~f:(fun v ->
                    (v, retrieve_var_from_sample v sample))
              in
              List.map list_samples_from_posterior_distribution
                ~f:retrieve_all_selected_vars
            in
            analyze_all_lists_fixed_lp_vars ~degree ~metric ~collect_fun_types
              exp f_name runtime_data list_list_fixed_lp_vars )
    | Fclosed _, _ -> raise (Analysis_error "Expecting function type.")

  (* Analyze a function the mode BayesWC *)
  let analyze_function_bayeswc ~degree ~metric ?(collect_fun_types = Pnone)
      analysis_params exp f_name =
    let { input_file; stan_params; lp_objective; dataset } = analysis_params in
    (* let () = Runtime_cost_data.import_predicted_worst_case_samples dataset input_file in *)
    let list_dataset_categorized_by_indices =
      match input_file with
      | None ->
          let dataset_nonzero_costs =
            Runtime_cost_data.extract_samples_with_nonzero_costs_dataset dataset
          in
          Stan_survival_analysis.survival_analysis stan_params
            dataset_nonzero_costs
          (* Volesti_bayesian_inference
          .bayesian_inference_on_cost_data_categorized_by_sizes warmup_params
            hmc_params dataset *)
      | Some filename ->
          let () =
            print_endline "Predicted worst-case costs will be imported."
          in
          let predicted_worst_case_costs_imported =
            Runtime_cost_data.import_predicted_worst_case_samples filename
          in
          let representative_runtime_sample =
            Runtime_cost_data.get_representative_runtime_sample dataset
          in
          let predicted_worst_case_costs =
            Runtime_cost_data.match_index_string_with_index
              predicted_worst_case_costs_imported representative_runtime_sample
          in
          let predicted_worst_case_costs_with_expressions =
            Runtime_cost_data.attach_expression_to_predicted_worst_case_costs
              predicted_worst_case_costs dataset
          in
          List.map predicted_worst_case_costs_with_expressions
            ~f:Runtime_cost_data.categorize_dataset_with_potential_by_indices
    in
    (* For debugging *)
    (* let () =
      print_endline "Dataset adjusted by Bayesian inference";
      List.iter list_dataset_categorized_by_indices
        ~f:
          Hybrid_aara_pprint
          .print_dataset_with_potential_categorized_by_indices
    in *)
    let analyze dataset_adjusted_by_hmc =
      Anno.S.reset_everything ();
      analyze_function_opt_and_custom_objective ~degree ~metric
        ~collect_fun_types lp_objective dataset_adjusted_by_hmc exp
    in
    let list_lp_feasible_solutions =
      List.filter_map list_dataset_categorized_by_indices ~f:analyze
    in
    let merge_list_lp_feasible_solutions list_lp_feasible_solutions =
      let list_atarg, list_atres, list_fun_types =
        List.fold list_lp_feasible_solutions ~init:([], [], [])
          ~f:(fun (xs, ys, zs) (x, y, z) -> (x :: xs, y :: ys, z :: zs))
      in
      match list_lp_feasible_solutions with
      | [] -> None
      | _ ->
          let combined_atarg = combine_list_annotated_types list_atarg in
          let combined_atres = combine_list_annotated_types list_atres in
          let combined_fun_types =
            combine_list_annotated_recorded_function_types list_fun_types
          in
          Some
            (Output_distribution
               (combined_atarg, combined_atres, combined_fun_types))
    in
    merge_list_lp_feasible_solutions list_lp_feasible_solutions

  let analyze_function ~degree ~metric ?(collect_fun_types = Pnone)
      ~hybrid_aara_params exp f_name =
    match hybrid_aara_params with
    | Disabled ->
        analyze_function_hybrid_aara_disabled ~degree ~metric ~collect_fun_types
          exp f_name
    | Opt analysis_params ->
        analyze_function_opt ~degree ~metric ~collect_fun_types analysis_params
          exp f_name
    | BayesPC analysis_params ->
        analyze_function_bayespc ~degree ~metric ~collect_fun_types
          analysis_params exp f_name
    | BayesWC analysis_params ->
        analyze_function_bayeswc ~degree ~metric ~collect_fun_types
          analysis_params exp f_name
end
