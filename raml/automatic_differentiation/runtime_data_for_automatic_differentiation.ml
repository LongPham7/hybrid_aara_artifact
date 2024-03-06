open Core

type runtime_data_categorized_by_vars =
  ((int * float list) list * (int * float list) list * float list) list

type runtime_data_sample_ocaml = (int * float) list * (int * float) list * float

type runtime_data_vars_are_expanded = runtime_data_sample_ocaml list

let print_runtime_data_with_expanded_vars
    (runtime_data : runtime_data_vars_are_expanded) : unit =
  let print_sample sample =
    let list_cindex_var_potential, list_index_var_potential, cost = sample in
    print_string "Input potential: (index, potential) = ";
    List.iter list_cindex_var_potential ~f:(fun (v, p) ->
        printf "(%i, %f) " v p);
    print_newline ();
    print_string "Output potential: (index, potential) = ";
    List.iter list_cindex_var_potential ~f:(fun (v, p) ->
        printf "(%i, %f) " v p);
    print_newline ();
    printf "Cost: %f\n" cost
  in
  List.iter runtime_data ~f:print_sample

(* Given runtime cost data of type runtime_data_categorized_by_vars, we
transform it such that it has type runtime_data_vars_are_expanded. In the input
runtime cost data before the transformation, each (c)index comes with a list of
potential. On the other hand, in the output of the transformation, the runtime
cost data is a list of samples where each (c)index is annotated with exactly one
potential. *)

let expand_var_in_var_list_potential var_and_list_potential =
  let var, list_potential = var_and_list_potential in
  List.map list_potential ~f:(fun potential -> (var, potential))

let change_categorization_of_nested_list (nested_list : 'a list list) :
    'a list list =
  let insert_into_acc acc inner_list =
    match acc with
    | [] -> List.map inner_list ~f:(fun x -> [ x ])
    | _ ->
        let zipped_list = List.zip_exn acc inner_list in
        List.map zipped_list ~f:(fun (inner_acc, element) ->
            element :: inner_acc)
  in
  List.fold nested_list ~init:[] ~f:insert_into_acc

let expand_indices_in_runtime_data_of_e runtime_data_of_e =
  let list_cindex_var_list_potential, list_index_var_list_potential, list_costs
      =
    runtime_data_of_e
  in
  let list_list_cindex_potential =
    list_cindex_var_list_potential
    |> List.map ~f:expand_var_in_var_list_potential
    |> change_categorization_of_nested_list
  in
  let list_list_index_potential =
    list_index_var_list_potential
    |> List.map ~f:expand_var_in_var_list_potential
    |> change_categorization_of_nested_list
  in
  let zipped_list =
    List.zip_exn list_list_cindex_potential list_list_index_potential
  in
  let zipped_list_with_costs = List.zip_exn zipped_list list_costs in
  List.map zipped_list_with_costs ~f:(fun ((x, y), z) -> (x, y, z))

let expand_vars_in_runtime_data
    (runtime_data : runtime_data_categorized_by_vars) :
    runtime_data_vars_are_expanded =
  List.concat (List.map runtime_data ~f:expand_indices_in_runtime_data_of_e)

(* Adjust LP variables in runtime cost data according to the result of
equality-constraint elimination *)

(* Map an LP variable before equality elimination to the one after the
elimination. If the given LP variable was not eliminated, then the output is
just the corresponding index after all remaining variables have been shifted.
Otherwise, if the variable has been eliminated, a linear combination of the
remaining variables equivalent to the eliminated variable is returned. *)
let map_var_to_list_index (list_vars : int list)
    (equality_constraints : Linear_programs_processing.equality_constraint list)
    (var : int) : (int * float) list * float =
  match List.exists list_vars ~f:(fun v -> v = var) with
  | true ->
      let index =
        Linear_programs_processing.get_index_from_list_vars list_vars var
      in
      ([ (index, 1.) ], 0.)
  | false ->
      Linear_programs_processing.get_list_index_from_equality_constraint
        list_vars equality_constraints var

let adjust_vars_in_list_var_potential_after_equality_elimination
    (list_vars : int list)
    (equality_constraints : Linear_programs_processing.equality_constraint list)
    (list_var_potential : (int * float) list) : (int * float) list * float =
  let insert_into_acc (acc_list_coefficients, acc_constant) (var, potential) =
    let list_coefficients, constant =
      map_var_to_list_index list_vars equality_constraints var
    in
    let updated_acc_list_coefficients =
      Linear_programs_processing.multiply_and_add_coefficient_lists
        acc_list_coefficients potential list_coefficients
    in
    (updated_acc_list_coefficients, (acc_constant *. potential) +. constant)
  in
  List.fold list_var_potential ~init:([], 0.) ~f:insert_into_acc

let adjust_vars_in_runtime_data_sample_after_equality_elimination
    (list_vars : int list)
    (equality_constraints : Linear_programs_processing.equality_constraint list)
    (sample : runtime_data_sample_ocaml) =
  let list_cindex_var_potential, list_index_var_potential, cost = sample in
  let list_cindex_var_potential_updated, cindex_constant =
    adjust_vars_in_list_var_potential_after_equality_elimination list_vars
      equality_constraints list_cindex_var_potential
  in
  let list_index_var_potential_updated, index_constant =
    adjust_vars_in_list_var_potential_after_equality_elimination list_vars
      equality_constraints list_index_var_potential
  in
  ( list_cindex_var_potential_updated,
    list_index_var_potential_updated,
    cost +. index_constant -. cindex_constant )

let adjust_vars_in_runtime_data_after_equality_elimination
    (list_vars : int list)
    (equality_constraints : Linear_programs_processing.equality_constraint list)
    (runtime_data : runtime_data_vars_are_expanded) =
  List.map runtime_data
    ~f:
      (adjust_vars_in_runtime_data_sample_after_equality_elimination list_vars
         equality_constraints)
