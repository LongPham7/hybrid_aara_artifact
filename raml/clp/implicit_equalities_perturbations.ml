open Core
open Linear_programs_processing

(* Printing functions *)

let print_row_with_vars_adjusted list_vars row =
  let { row_lower; row_upper; row_elements } = row in
  print_string "Row: ";
  List.iter row_elements ~f:(fun (v, c) ->
      printf "(v, c) = (%i, %.1f) " (get_index_from_list_vars list_vars v) c);
  printf "lower: %.1f upper: %.1f" row_lower row_upper;
  print_newline ()

(* Check the feasibility of a linear program *)

let check_feasibility_of_lp list_vars list_inequality_rows =
  let matrix_A, list_b =
    transform_rows_to_matrix list_inequality_rows list_vars
  in
  let list_A = List.concat matrix_A in
  let status = Volesti_common.get_feasibility_status_of_lp list_A list_b in
  if status = 0 then true
  else if status = 5 then false
  else failwith "An error in LP has occurred"

(* Iteratively perturb the vector b in the LP Ax <= b in C++ *)

let convert_row_to_equality_row_with_index index num_bounds row =
  assert (0 <= index && index < num_bounds && num_bounds <= 2);
  let { row_lower; row_upper; row_elements } = row in
  (* If a row has both a lower and upper bounds, the upper-bounded constraint
  comes first in the matrix. *)
  if
    (num_bounds = 1 && row_lower <> Float.min_value)
    || (num_bounds = 2 && index = 1)
  then { row with row_upper = row_lower }
  else { row with row_lower = row_upper }

let rec get_rows_in_list_row_num_bounds
    (acc_inequality_rows, acc_equality_rows, acc_num_bounds) list_row_num_bounds
    list_indices =
  match list_indices with
  | [] -> (acc_inequality_rows, acc_equality_rows)
  | head_index :: tail_indices -> (
      match list_row_num_bounds with
      | [] ->
          failwith
            "The list of rows is empty, while the list of indices is non-empty"
      | (head_row, num_bounds) :: tail_rows ->
          let relative_index = head_index - acc_num_bounds in
          if relative_index < num_bounds then
            let implicit_equality_row =
              convert_row_to_equality_row_with_index relative_index num_bounds
                head_row
            in
            let acc_equalities_updated =
              implicit_equality_row :: acc_equality_rows
            in
            get_rows_in_list_row_num_bounds
              ( acc_inequality_rows,
                acc_equalities_updated,
                acc_num_bounds + num_bounds )
              tail_rows tail_indices
          else
            let acc_inequalities_updated = head_row :: acc_inequality_rows in
            get_rows_in_list_row_num_bounds
              ( acc_inequalities_updated,
                acc_equality_rows,
                acc_num_bounds + num_bounds )
              tail_rows
              (head_index :: tail_indices) )

let get_rows_in_matrix list_inequality_rows list_indices =
  let num_of_bounds row =
    let { row_lower; row_upper } = row in
    match (row_lower = Float.min_value, row_upper = Float.max_value) with
    | true, true -> failwith "Both bounds are infinite"
    | false, true -> 1
    | true, false -> 1
    | false, false -> 2
  in
  let list_row_with_num_of_bounds =
    List.map list_inequality_rows ~f:(fun r -> (r, num_of_bounds r))
  in
  let list_indices_sorted = List.sort ~compare:Int.compare list_indices in
  get_rows_in_list_row_num_bounds ([], [], 0) list_row_with_num_of_bounds
    list_indices_sorted

let identify_implicit_equalities_by_perturbations list_vars list_inequality_rows
    =
  let matrix_A, list_b =
    transform_rows_to_matrix list_inequality_rows list_vars
  in
  let list_A = List.concat matrix_A in
  let list_implicit_equality_indices =
    Volesti_common.iteratively_perturb_vector_b list_A list_b
  in
  get_rows_in_matrix list_inequality_rows list_implicit_equality_indices

let sanity_check_row solution row =
  let { row_lower; row_upper; row_elements } = row in
  let row_elements_values =
    List.map row_elements ~f:(fun (v, c) -> c *. solution v)
  in
  let row_elements_sum =
    List.fold row_elements_values ~init:0. ~f:(fun x y -> x +. y)
  in
  if approximately_bounded row_elements_sum row_lower row_upper then true
  else (
    print_endline "Inconsistent row with the given solution";
    List.iter row_elements ~f:(fun (v, c) ->
        printf "(var, value, coeff) = (%i, %.4f, %.4f " v (solution v) c);
    printf "row_lower = %.4f row_upper = %.4f " row_lower row_upper;
    print_newline ();
    false )

let sanity_check_equality_constraint solution equality =
  let { var; coefficient_list; constant } = equality in
  let var_value = solution var in
  let coefficient_list_value =
    List.map coefficient_list ~f:(fun (v, c) -> c *. solution v)
  in
  let coefficient_list_sum =
    List.fold coefficient_list_value ~init:0. ~f:(fun x y -> x +. y)
  in
  if approximately_equal var_value (coefficient_list_sum +. constant) then true
  else (
    print_endline "Inconsistent equality constraint with the given solution";
    printf "(var, value) = (%i, %.4f) " var (solution var);
    List.iter coefficient_list ~f:(fun (v, c) ->
        printf "(var, value, coeff) = (%i, %.4f, %.4f " v (solution v) c);
    printf "constant = %.4f" constant;
    print_newline ();
    false )

let rec substitute_implicit_equalities
    (list_inequality_rows, list_equality_constraints) list_equality_rows =
  match list_equality_rows with
  | [] -> (list_inequality_rows, list_equality_constraints)
  | head_equality_row :: tail_equalities ->
      let head_equality_constraint =
        convert_row_to_equality_constraint head_equality_row
      in
      let list_inequality_rows_substituted =
        List.filter_map list_inequality_rows ~f:(fun r ->
            substitute_equality_into_row r head_equality_constraint)
      in
      let list_equality_constraints_substituted =
        List.map list_equality_constraints ~f:(fun c ->
            substitute_equality_into_equality c head_equality_constraint)
      in
      let remaining_equality_rows_substituted =
        List.filter_map list_equality_rows ~f:(fun r ->
            substitute_equality_into_row r head_equality_constraint)
      in
      substitute_implicit_equalities
        ( list_inequality_rows_substituted,
          head_equality_constraint :: list_equality_constraints_substituted )
        remaining_equality_rows_substituted

let rec repeatedly_remove_implicit_equalities_by_perturbations num_vars
    (list_inequality_rows, list_equality_constraints) =
  let remaining_vars, _ =
    categorize_LP_variables num_vars list_equality_constraints
  in
  let list_inequalities, list_implicit_equalities =
    identify_implicit_equalities_by_perturbations remaining_vars
      list_inequality_rows
  in
  (* For debugging *)
  let () =
    let is_feasible =
      check_feasibility_of_lp remaining_vars list_inequality_rows
    in
    if is_feasible then
      print_endline "The LP before identifying implicit equalities is feasible"
    else
      print_endline
        "The LP before identifying implicit equalities is infeasible"
  in
  (* For debugging *)
  (* let () =
    print_endline "List of implicit equalities discovered by perturbations";
    List.iter list_implicit_equalities ~f:print_row
  in *)
  match list_implicit_equalities with
  | [] ->
      let () =
        print_endline "We cannot find any implicit equalities by perturbations"
      in
      (list_inequality_rows, list_equality_constraints)
  | _ ->
      let list_inequality_rows_updated, list_equality_constraints_updated =
        substitute_implicit_equalities
          (list_inequality_rows, list_equality_constraints)
          list_implicit_equalities
        |> repeatedly_merge_box_constraints
      in
      let () =
        printf "We have found %i many implicit equalities by perturbations"
          (List.length list_implicit_equalities);
        print_newline ()
      in
      repeatedly_remove_implicit_equalities_by_perturbations num_vars
        (list_inequality_rows_updated, list_equality_constraints_updated)
