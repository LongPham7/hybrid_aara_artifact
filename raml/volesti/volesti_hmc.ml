open Core
open Probability_distributions
open Ctypes
open Foreign
open Hybrid_aara_config

(* This module offers OCaml functions for running reflective Hamiltonian Monte
Carlo (HMC) implemented in the C++ library volesti. *)

(* Define the Ctype binding for the C struct runtime_data_sample. It will be
called runtime_data_sample_c in this source file. *)

type runtime_data_sample_c

let runtime_data_sample_c : runtime_data_sample_c structure typ =
  structure "runtime_data_sample"

let array_cindices_field =
  field runtime_data_sample_c "array_cindices" (ptr int)

let potential_of_cindices_field =
  field runtime_data_sample_c "potential_of_cindices" (ptr double)

let num_cindices_field = field runtime_data_sample_c "num_cindices" int

let array_indices_field = field runtime_data_sample_c "array_indices" (ptr int)

let potential_of_indices_field =
  field runtime_data_sample_c "potential_of_indices" (ptr double)

let num_indices_field = field runtime_data_sample_c "num_indices" int

let cost_field = field runtime_data_sample_c "cost" double

let () = seal runtime_data_sample_c

(* Define the Ctype binding for the C struct
coefficient_distribution_target_type. *)

type coefficient_distribution_target_type_c

let coefficient_distribution_target_type_c :
    coefficient_distribution_target_type_c structure typ =
  structure "coefficient_distribution_target_type"

let selected_coefficients_field =
  field coefficient_distribution_target_type_c "selected_coefficients" (ptr int)

let num_selected_coefficients_field =
  field coefficient_distribution_target_type_c "num_selected_coefficients" int

let target_type_field =
  field coefficient_distribution_target_type_c "target_type" int

let () = seal coefficient_distribution_target_type_c

(* Printing functions for debugging *)

let print_runtime_data_c =
  foreign "print_runtime_data"
    (ptr runtime_data_sample_c @-> int @-> returning void)

let print_pointer_int_array_ocaml pointer_int_array num_samples =
  let int_array = CArray.from_ptr pointer_int_array num_samples in
  print_string "Int array content: ";
  CArray.iter (fun x -> printf "%i " x) int_array

let print_pointer_double_array_ocaml pointer_double_array num_samples =
  let double_array = CArray.from_ptr pointer_double_array num_samples in
  print_string "Double array content: ";
  CArray.iter (fun x -> printf "%.1f " x) double_array

let print_runtime_data_sample_ocaml sample =
  let pointer_array_cindices = getf sample array_cindices_field in
  let pointer_potential_of_cindices = getf sample potential_of_cindices_field in
  let num_cindices = getf sample num_cindices_field in
  let pointer_array_indices = getf sample array_indices_field in
  let pointer_potential_of_indices = getf sample potential_of_indices_field in
  let num_indices = getf sample num_indices_field in
  let cost = getf sample cost_field in
  print_endline "Content of a runtime sample:";
  print_string "Cindices: ";
  print_pointer_int_array_ocaml pointer_array_cindices num_cindices;
  print_pointer_double_array_ocaml pointer_potential_of_cindices num_cindices;
  printf "Number of cindices: %i\n" num_cindices;
  print_string "Indices: ";
  print_pointer_int_array_ocaml pointer_array_indices num_indices;
  print_pointer_double_array_ocaml pointer_potential_of_indices num_indices;
  printf "Number of indices: %i\n" num_indices;
  printf "Cost: %.1f\n" cost

let print_pointer_array_of_samples pointer_array_samples num_samples =
  let array_samples = CArray.from_ptr pointer_array_samples num_samples in
  (* CArray.iter print_runtime_data_sample_ocaml array_samples *)
  print_runtime_data_sample_ocaml (CArray.get array_samples 0);
  print_runtime_data_sample_ocaml (CArray.get array_samples 1);
  print_runtime_data_sample_ocaml (CArray.get array_samples 2)

(* Convert runtime cost data from the OCaml format (i.e. has type
runtime_data_vars_are_expanded) to the C format (i.e. an array of the C struct
runtime_data_sample) *)

(* Create a Ctype array from an OCaml list and return the pointer to the array's
first element. CArray already provides the of_list function for building a new
Ctype array from a given OCaml list. So an obvious implementation of this
function is CArray.of_list typ list_ocaml |> CArray.start. However, for some
reason, once we create a large number of the struct sample, a garbage collector
kicks in, collecting the underlying output of this implementation. 

This behavior doesn't make sense to me. The above implementation returns a
pointer to the output array. So even after we exit this function and return to
its caller, we can still reach the underlying array via the pointer.
Nonetheless, the garbage collector still collects the underlying array. As a
workaround, we must implement CArray.of_list from scratch. The workaround below
is basically identical to CArray.of_list's implementation in Ctype. This
workaround, for some reason, prevents the garbage collection of the underlying
arrays, even though it is identical to CArray.of_list. *)
let create_pointer_array_from_ocaml_list typ list_ocaml =
  (* CArray.of_list typ list_ocaml |> CArray.start *)
  let array_c =
    CArray.make (* For debugging *)
      ~finalise:(fun _ -> ())
      (* ~finalise:(fun _ -> print_endline "Ctype array is garbage-collected.") *)
      typ (List.length list_ocaml)
  in
  let () = List.iteri list_ocaml ~f:(fun i x -> CArray.set array_c i x) in
  CArray.start array_c

let convert_ocaml_runtime_data_to_c_runtime_data
    (runtime_data_ocaml :
      Runtime_data_for_automatic_differentiation.runtime_data_vars_are_expanded)
    =
  let get_arrays_indices_and_potential list_index_potential =
    let list_indices, list_potential = List.unzip list_index_potential in
    let array_indices = create_pointer_array_from_ocaml_list int list_indices in
    let array_potential =
      create_pointer_array_from_ocaml_list double list_potential
    in
    (array_indices, array_potential)
  in
  let convert_ocaml_sample_to_c_sample sample_ocaml =
    let list_cindex_potential, list_index_potential, cost = sample_ocaml in
    let num_cindices = List.length list_cindex_potential in
    let num_indices = List.length list_index_potential in
    let array_cindices, potential_of_cindices =
      get_arrays_indices_and_potential list_cindex_potential
    in
    let array_indices, potential_of_indices =
      get_arrays_indices_and_potential list_index_potential
    in
    let sample_c =
      make (* For debugging *)
        ~finalise:(fun _ -> ())
        (* ~finalise:(fun _ ->
          printf "Ctype struct sample is garbage-collected.\n") *)
        runtime_data_sample_c
    in
    let () =
      setf sample_c array_cindices_field array_cindices;
      setf sample_c potential_of_cindices_field potential_of_cindices;
      setf sample_c num_cindices_field num_cindices;
      setf sample_c array_indices_field array_indices;
      setf sample_c potential_of_indices_field potential_of_indices;
      setf sample_c num_indices_field num_indices;
      setf sample_c cost_field cost
    in
    sample_c
  in
  let num_samples_in_runtime_data = List.length runtime_data_ocaml in
  let runtime_data_c =
    runtime_data_ocaml
    |> List.map ~f:convert_ocaml_sample_to_c_sample
    |> CArray.of_list runtime_data_sample_c
    |> CArray.start
  in
  (* For debugging *)
  (* print_endline "Runtime samples printed by an OCaml function";
  print_pointer_array_of_samples runtime_data_c num_samples_in_runtime_data;
  print_endline "Runtime samples printed by a C function";
  print_runtime_data_c runtime_data_c num_samples_in_runtime_data; *)
  (runtime_data_c, num_samples_in_runtime_data)

(* Test if we can pass an array of runtime_data_sample_c's from OCaml to C++
(via C) and print them out *)

let runtime_data_for_testing =
  let create_runtime_sample first_input_size =
    let new_sample = make runtime_data_sample_c in
    let array_cindices = CArray.start (CArray.of_list int [ 1; 2 ]) in
    let potential_of_cindices =
      CArray.start (CArray.of_list double [ Float.of_int first_input_size; 5. ])
    in
    let array_indices = CArray.start (CArray.of_list int []) in
    let potential_of_indices = CArray.start (CArray.of_list double []) in
    let () =
      setf new_sample array_cindices_field array_cindices;
      setf new_sample potential_of_cindices_field potential_of_cindices;
      setf new_sample num_cindices_field 2;
      setf new_sample array_indices_field array_indices;
      setf new_sample potential_of_indices_field potential_of_indices;
      setf new_sample num_indices_field 0;
      setf new_sample cost_field (Float.of_int first_input_size)
    in
    new_sample
  in
  let list_input_sizes = List.range 0 5 in
  let list_samples = List.map list_input_sizes ~f:create_runtime_sample in
  CArray.start (CArray.of_list runtime_data_sample_c list_samples)

let print_runtime_data_for_testing () =
  print_runtime_data_c runtime_data_for_testing 5

(* Define the Ctype binding for the C struct distribution_type *)

type probability_distribution_type_c

let probability_distribution_type_c :
    probability_distribution_type_c structure typ =
  structure "distribution_type"

(* The field distribution_name uses C enumerations. Number 0 means Weibull, 1
means Gumbel, and 2 means Gaussian distributions. *)
let distribution_name_field =
  field probability_distribution_type_c "distribution_name" int

let first_parameter_field =
  field probability_distribution_type_c "first_parameter" double

let second_parameter_field =
  field probability_distribution_type_c "second_parameter" double

let () = seal probability_distribution_type_c

(* Convert a distribution from the OCaml format (i.e. has type
probability_distribution_type defined in the Probability_distributions module)
to the C format (i.e. the C struct distribution_type *)
let convert_distribution_ocaml_to_c
    (distribution : probability_distribution_type) =
  let new_distribution = make probability_distribution_type_c in
  let () =
    match distribution with
    | Weibull { alpha; sigma } ->
        setf new_distribution distribution_name_field 0;
        setf new_distribution first_parameter_field alpha;
        setf new_distribution second_parameter_field sigma
    | Gumbel { mu; beta } ->
        setf new_distribution distribution_name_field 1;
        setf new_distribution first_parameter_field mu;
        setf new_distribution second_parameter_field beta
    | Gaussian { mu; sigma } ->
        setf new_distribution distribution_name_field 2;
        setf new_distribution first_parameter_field mu;
        setf new_distribution second_parameter_field sigma
  in
  new_distribution

let print_distribution_c =
  foreign "print_distribution"
    (probability_distribution_type_c @-> returning void)

let print_distribution_c_for_testing () =
  let distribution = Weibull { alpha = 1.; sigma = 6. } in
  distribution |> convert_distribution_ocaml_to_c |> print_distribution_c

let convert_distribution_target_ocaml_to_c distribution_target =
  match distribution_target with
  | Individual_coefficients -> 0
  | Average_of_coefficients -> 1

let convert_coefficient_distribution_target_without_selection_ocaml_to_c
    distribution_target =
  let coefficient_distribution_target =
    make coefficient_distribution_target_type_c
  in
  let () =
    setf coefficient_distribution_target selected_coefficients_field
      (from_voidp int null);
    setf coefficient_distribution_target num_selected_coefficients_field (-1);
    setf coefficient_distribution_target target_type_field
      (convert_distribution_target_ocaml_to_c distribution_target)
  in
  coefficient_distribution_target

let convert_coefficient_distribution_target_with_selection_ocaml_to_c
    selected_coefficients distribution_target =
  let selected_coefficients_array_pointer =
    create_pointer_array_from_ocaml_list int selected_coefficients
    (* The following causes a segmentation fault because, for some reason, the
    newly created array is garbage-collected halfway through execution. *)
    (* CArray.start (CArray.of_list int selected_coefficients) *)
  in
  let num_selected_coefficients = List.length selected_coefficients in
  let coefficient_distribution_target =
    make coefficient_distribution_target_type_c
  in
  let () =
    setf coefficient_distribution_target selected_coefficients_field
      selected_coefficients_array_pointer;
    setf coefficient_distribution_target num_selected_coefficients_field
      num_selected_coefficients;
    setf coefficient_distribution_target target_type_field
      (convert_distribution_target_ocaml_to_c distribution_target)
  in
  coefficient_distribution_target

(* Runtime-data interface of Volesti's reflective HMC *)

let hmc_runtime_data_interface =
  foreign "hmc_runtime_data_interface"
    ( int @-> int @-> ptr double @-> ptr double @-> double @-> double @-> int
    @-> int @-> double @-> ptr double @-> ptr runtime_data_sample_c @-> int
    @-> probability_distribution_type_c @-> probability_distribution_type_c
    @-> coefficient_distribution_target_type_c @-> int
    @-> returning (ptr double) )

let hmc list_A list_b lipschitz m num_samples walk_length step_size
    list_starting_point list_vars equality_constraints runtime_data_ocaml
    coefficient_distribution cost_model coefficient_distribution_target
    cost_model_target =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let array_starting_point = CArray.of_list double list_starting_point in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  let num_samples_after_burns = num_samples - (num_samples / 2) in
  (* For debugging *)
  (* Hybrid_aara_pprint.print_dataset_with_lp_vars runtime_data_ocaml; *)
  let runtime_data_c, num_samples_in_runtime_data =
    runtime_data_ocaml
    |> Runtime_data_for_automatic_differentiation.expand_vars_in_runtime_data
    |> Runtime_data_for_automatic_differentiation
       .adjust_vars_in_runtime_data_after_equality_elimination list_vars
         equality_constraints
    |> convert_ocaml_runtime_data_to_c_runtime_data
  in
  (* For debugging *)
  (* print_runtime_data_ocaml_interface runtime_data_c num_samples_in_runtime_data; *)
  let coefficient_distribution_c =
    convert_distribution_ocaml_to_c coefficient_distribution
  in
  let cost_model_c = convert_distribution_ocaml_to_c cost_model in
  let result =
    hmc_runtime_data_interface num_rows num_cols (CArray.start array_A)
      (CArray.start array_b) lipschitz m num_samples walk_length step_size
      (CArray.start array_starting_point)
      runtime_data_c num_samples_in_runtime_data coefficient_distribution_c
      cost_model_c coefficient_distribution_target cost_model_target
  in
  CArray.to_list (CArray.from_ptr result (num_cols * num_samples_after_burns))

let hmc_cost_data_categorized_by_sizes =
  foreign "hmc_cost_data_categorized_by_sizes"
    ( int @-> int @-> ptr double @-> ptr double @-> double @-> double @-> int
    @-> int @-> double @-> ptr double @-> ptr int @-> ptr double
    @-> probability_distribution_type_c
    @-> returning (ptr double) )

let hmc_on_cost_data list_A list_b runtime_data lipschitz m num_samples
    walk_length step_size list_starting_point cost_model =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let array_starting_point = CArray.of_list double list_starting_point in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  let num_samples_after_burns = num_samples - (num_samples / 2) in
  assert (num_cols = List.length runtime_data);
  let list_sizes_of_categories = List.map runtime_data ~f:List.length in
  let array_sizes_of_categories =
    CArray.start (CArray.of_list int list_sizes_of_categories)
  in
  let array_costs =
    runtime_data |> List.concat |> CArray.of_list double |> CArray.start
  in
  let cost_model_c = convert_distribution_ocaml_to_c cost_model in
  let result =
    hmc_cost_data_categorized_by_sizes num_rows num_cols (CArray.start array_A)
      (CArray.start array_b) lipschitz m num_samples walk_length step_size
      (CArray.start array_starting_point)
      array_sizes_of_categories array_costs cost_model_c
  in
  CArray.to_list (CArray.from_ptr result (num_cols * num_samples_after_burns))
