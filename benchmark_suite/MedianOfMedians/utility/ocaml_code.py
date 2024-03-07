import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join("/home", "hybrid_aara", "benchmark_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_integer_list_pair_python_to_ocaml

# Create OCaml code

linear_select_data_driven_ocaml_code = \
    """
exception Invalid_input

let incur_cost (hd : int) =
  if (hd mod 10) = 0 then Raml.tick 1.0 else Raml.tick 0.5

let rec append (xs : int list) (ys : int list) =
  match xs with [] -> ys | hd :: tl -> hd :: append tl ys

let rec insert (x : int) (list : int list) =
  match list with
  | [] -> [ x ]
  | y :: ys -> if x <= y then x :: y :: ys else y :: insert x ys

let rec insertion_sort (list : int list) =
  match list with [] -> [] | x :: xs -> insert x (insertion_sort xs)

let median_of_list_of_five (xs : int list) =
  let sorted_xs = insertion_sort xs in
  match sorted_xs with
  | [ x1; x2; x3; x4; x5 ] -> (x3, [ x1; x2; x4; x5 ])
  | _ -> raise Invalid_input

let rec partition_into_blocks (xs : int list) =
  match xs with
  | [] -> ([], [])
  | x1 :: x2 :: x3 :: x4 :: x5 :: tl ->
      let median, leftover = median_of_list_of_five [ x1; x2; x3; x4; x5 ] in
      let list_medians, list_leftover = partition_into_blocks tl in
      (median :: list_medians, append leftover list_leftover)
  | _ -> raise Invalid_input

let rec partition (pivot : int) (xs : int list) =
  match xs with
  | [] -> ([], [])
  | hd :: tl ->
      let lower_list, upper_list = partition pivot tl in
      let _ = incur_cost hd in
      if hd <= pivot then (hd :: lower_list, upper_list)
      else (lower_list, hd :: upper_list)

let rec lower_list_length_after_partition (pivot : int) (xs : int list) =
  match xs with
  | [] -> 0
  | hd :: tl ->
      let lower_list_length = lower_list_length_after_partition pivot tl in
      if hd <= pivot then lower_list_length + 1 else lower_list_length

let rec list_length (xs : int list) =
  match xs with [] -> 0 | hd :: tl -> 1 + list_length tl

let rec find_minimum_acc (acc : int list) (candidate : int) (xs : int list) =
  match xs with
  | [] -> (candidate, acc)
  | hd :: tl ->
      if hd < candidate then find_minimum_acc (candidate :: acc) hd tl
      else find_minimum_acc (hd :: acc) candidate tl

let find_minimum (xs : int list) =
  match xs with
  | [] -> raise Invalid_input
  | hd :: tl -> find_minimum_acc [] hd tl

let rec preprocess_list_acc (minima_acc : int list) (xs : int list) =
  let xs_length = list_length xs in
  if xs_length mod 5 = 0 then (minima_acc, xs)
  else
    let minimum, leftover = find_minimum xs in
    preprocess_list_acc (minimum :: minima_acc) leftover

let rec get_nth_element (index : int) (xs : int list) =
  match xs with
  | [] -> raise Invalid_input
  | hd :: tl -> if index = 0 then hd else get_nth_element (index - 1) tl

let rec linear_select (index : int) (xs : int list) =
  match xs with
  | [] -> raise Invalid_input
  | _ ->
      let minima, xs_trimmed = preprocess_list_acc [] xs in
      let mod_five = list_length minima in
      if index < mod_five then get_nth_element (mod_five - index - 1) minima
      else
        let index_trimmed = index - mod_five in
        let list_medians, _ = partition_into_blocks xs_trimmed in
        let num_medians = list_length list_medians in
        let index_median = num_medians / 2 in
        let median_of_medians =
          Raml.stat (linear_select index_median list_medians)
        in
        let lower_list_length =
          lower_list_length_after_partition median_of_medians xs_trimmed
        in
        if index_trimmed = lower_list_length - 1 then
          let _, _ = partition median_of_medians xs_trimmed in
          median_of_medians
        else if index_trimmed < lower_list_length - 1 then
          let lower_list, _ = partition median_of_medians xs_trimmed in
          Raml.stat (linear_select index_trimmed lower_list)
        else
          let new_index = index_trimmed - lower_list_length in
          let _, upper_list = partition median_of_medians xs_trimmed in
          Raml.stat (linear_select new_index upper_list)

let linear_select2 (index : int) (xs : int list) =
  Raml.stat (linear_select index xs)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""

linear_select_hybrid_ocaml_code = \
    """
exception Invalid_input

let incur_cost (hd : int) =
  if (hd mod 10) = 0 then Raml.tick 1.0 else Raml.tick 0.5

let rec append (xs : int list) (ys : int list) =
  match xs with [] -> ys | hd :: tl -> hd :: append tl ys

let rec insert (x : int) (list : int list) =
  match list with
  | [] -> [ x ]
  | y :: ys -> if x <= y then x :: y :: ys else y :: insert x ys

let rec insertion_sort (list : int list) =
  match list with [] -> [] | x :: xs -> insert x (insertion_sort xs)

let median_of_list_of_five (xs : int list) =
  let sorted_xs = insertion_sort xs in
  match sorted_xs with
  | [ x1; x2; x3; x4; x5 ] -> x3
  | _ -> raise Invalid_input

let rec partition_into_blocks (xs : int list) =
  match xs with
  | [] -> []
  | x1 :: x2 :: x3 :: x4 :: x5 :: tl ->
      let median = median_of_list_of_five [ x1; x2; x3; x4; x5 ] in
      let list_medians = partition_into_blocks tl in
      median :: list_medians
  | _ -> raise Invalid_input

let rec partition (pivot : int) (xs : int list) =
  match xs with
  | [] -> ([], [])
  | hd :: tl ->
      let lower_list, upper_list = partition pivot tl in
      let _ = incur_cost hd in
      if hd <= pivot then (hd :: lower_list, upper_list)
      else (lower_list, hd :: upper_list)

let rec lower_list_length_after_partition (pivot : int) (xs : int list) =
  match xs with
  | [] -> 0
  | hd :: tl ->
      let lower_list_length = lower_list_length_after_partition pivot tl in
      if hd <= pivot then lower_list_length + 1 else lower_list_length

let rec list_length (xs : int list) =
  match xs with [] -> 0 | hd :: tl -> 1 + list_length tl

let rec find_minimum_acc (acc : int list) (candidate : int) (xs : int list) =
  match xs with
  | [] -> (candidate, acc)
  | hd :: tl ->
      if hd < candidate then find_minimum_acc (candidate :: acc) hd tl
      else find_minimum_acc (hd :: acc) candidate tl

let find_minimum (xs : int list) =
  match xs with
  | [] -> raise Invalid_input
  | hd :: tl -> find_minimum_acc [] hd tl

let rec preprocess_list_acc (minima_acc : int list) (xs : int list) =
  let xs_length = list_length xs in
  if xs_length mod 5 = 0 then (minima_acc, xs)
  else
    let minimum, leftover = find_minimum xs in
    preprocess_list_acc (minimum :: minima_acc) leftover

let rec get_nth_element (index : int) (xs : int list) =
  match xs with
  | [] -> raise Invalid_input
  | hd :: tl -> if index = 0 then hd else get_nth_element (index - 1) tl

let rec linear_select (index : int) (xs : int list) =
  match xs with
  | [] -> raise Invalid_input
  | _ ->
      let minima, xs_trimmed = preprocess_list_acc [] xs in
      let mod_five = list_length minima in
      if index < mod_five then get_nth_element (mod_five - index - 1) minima
      else
        let index_trimmed = index - mod_five in
        let list_medians = partition_into_blocks xs_trimmed in
        let num_medians = list_length list_medians in
        let index_median = num_medians / 2 in
        let median_of_medians = linear_select index_median list_medians in
        let lower_list_length =
          lower_list_length_after_partition median_of_medians xs_trimmed
        in
        if index_trimmed = lower_list_length - 1 then
          let _, _ = Raml.stat (partition median_of_medians xs_trimmed) in
          median_of_medians
        else if index_trimmed < lower_list_length - 1 then
          let lower_list, _ =
            Raml.stat (partition median_of_medians xs_trimmed)
          in
          linear_select index_trimmed lower_list
        else
          let new_index = index_trimmed - lower_list_length in
          let _, upper_list =
            Raml.stat (partition median_of_medians xs_trimmed)
          in
          linear_select new_index upper_list

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_integer_list_pair_python_to_ocaml(
        input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    if hybrid_mode == "data_driven":
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset (fun (index, xs) -> linear_select2 index xs)\n"
        return linear_select_data_driven_ocaml_code + input_generation_code
    else:
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset (fun (index, xs) -> linear_select index xs)\n"
        return linear_select_hybrid_ocaml_code + input_generation_code
