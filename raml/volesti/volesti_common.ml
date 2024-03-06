open Core
open Ctypes
open Foreign

(* This module offers utility functions for working with the C++ library
volesti. *)

let rec partition_into_blocks (flattened_list : float list) (block_size : int) :
    float list list =
  match flattened_list with
  | [] -> []
  | _ ->
      let first_block, tail = List.split_n flattened_list block_size in
      first_block :: partition_into_blocks tail block_size

let print_volesti_vector ?(verbose = false)
    ?(num_first_and_last_vector_components_to_display = 15) sample =
  let num_vector_components = List.length sample in
  let () =
    if
      num_vector_components
      <= num_first_and_last_vector_components_to_display * 2
      || verbose
    then List.iter sample ~f:(fun x -> printf "%f " x)
    else
      let first_sublist =
        List.take sample num_first_and_last_vector_components_to_display
      in
      let second_sublist =
        List.drop sample
          ( num_vector_components
          - num_first_and_last_vector_components_to_display )
      in
      List.iter first_sublist ~f:(fun x -> printf "%f " x);
      printf "... %i more vector components ... "
        ( num_vector_components
        - (num_first_and_last_vector_components_to_display * 2) );
      List.iter second_sublist ~f:(fun x -> printf "%f " x)
  in
  print_newline ()

let print_volesti_result ?(verbose = false)
    ?(num_first_and_last_lines_to_display = 5)
    ?(num_first_and_last_vector_components_to_display = 15)
    (list_samples : float list list) =
  let num_samples = List.length list_samples in
  let print_sample =
    print_volesti_vector ~verbose
      ~num_first_and_last_vector_components_to_display
  in
  if num_samples <= num_first_and_last_lines_to_display * 2 || verbose then
    List.iter list_samples ~f:print_sample
  else (
    List.iter
      (List.take list_samples num_first_and_last_lines_to_display)
      ~f:print_sample;
    printf "... %i more samples ...\n"
      (num_samples - (2 * num_first_and_last_lines_to_display));
    List.iter
      (List.drop list_samples
         (num_samples - num_first_and_last_lines_to_display))
      ~f:print_sample )

(* Compute the Chebyshev radius (i.e. the radius of the largest inner ball that
fits inside a linear program's feasible region). *)

let compute_chebyshev_radius_c =
  foreign "compute_chebyshev_radius"
    (int @-> int @-> ptr double @-> ptr double @-> returning double)

let compute_chebyshev_radius list_A list_b =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  compute_chebyshev_radius_c num_rows num_cols (CArray.start array_A)
    (CArray.start array_b)

(* Determine the feasibility of linear programs *)

let get_feasibility_status_of_lp_c =
  foreign "get_feasibility_status_of_lp"
    (int @-> int @-> ptr double @-> ptr double @-> returning int)

let get_feasibility_status_of_lp list_A list_b =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  get_feasibility_status_of_lp_c num_rows num_cols (CArray.start array_A)
    (CArray.start array_b)

(* Identify the first implicit equality in a linear program in C++ by computing
Chebyshev balls repeatedly. Here, a linear program is expressed as A x <= b,
rather than a list of rows defined in the module Linear_programs_processing. *)

let identify_first_implicit_equality_row_index_in_matrix_c =
  foreign "identify_first_implicit_equality_row_index_in_matrix"
    (int @-> int @-> ptr double @-> ptr double @-> returning int)

let identify_first_implicit_equality_row_index_in_matrix list_A list_b =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  identify_first_implicit_equality_row_index_in_matrix_c num_rows num_cols
    (CArray.start array_A) (CArray.start array_b)

(* Identify implicit equalities by iteratively perturbing vector b in the LP A x
<= b. *)

let iteratively_perturb_vector_b_c =
  foreign "iteratively_perturb_vector_b"
    (int @-> int @-> ptr double @-> ptr double @-> returning (ptr int))

let iteratively_perturb_vector_b list_A list_b =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  let pointer_to_array_result =
    iteratively_perturb_vector_b_c num_rows num_cols (CArray.start array_A)
      (CArray.start array_b)
  in
  (* The first array element stores the number of implicit equalities. *)
  let num_implicit_equalities = !@pointer_to_array_result in
  let () =
    printf "We have successfully received %i many implicit equalities\n"
      num_implicit_equalities
  in
  let num_elements_and_list_indices =
    CArray.to_list
      (CArray.from_ptr pointer_to_array_result (num_implicit_equalities + 1))
  in
  List.tl_exn num_elements_and_list_indices
