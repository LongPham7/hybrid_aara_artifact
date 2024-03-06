import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join("/home", "hybrid_aara", "benchmark_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_lists_python_to_ocaml


# Create OCaml code


z_algorithm_data_driven_ocaml_code = \
    """
exception Invalid_input

let incur_cost (hd : int) =
  let modulo = 5 in
  if (hd mod 100) = 0 then Raml.tick 1.0 
  else (if (hd mod modulo) = 1 then Raml.tick 0.85
        else if (hd mod modulo) = 2 then Raml.tick 0.65 else Raml.tick 0.5)

let rec list_length (xs : int list) =
  match xs with [] -> 0 | hd :: tl -> 1 + list_length tl

let hd_exn (xs : int list) =
  match xs with [] -> raise Invalid_input | hd :: _ -> hd

let min (x1 : int) (x2 : int) = if x1 < x2 then x1 else x2

let rec drop_n_elements (xs : int list) (n : int) =
  match xs with
  | [] -> []
  | hd :: tl -> if n = 0 then hd :: tl else drop_n_elements tl (n - 1)

let rec longest_common_prefix (xs1 : int list) (xs2 : int list) =
  match xs1 with
  | [] -> 0
  | hd1 :: tl1 -> (
      match xs2 with
      | [] -> 0
      | hd2 :: tl2 ->
          if hd1 = hd2 then
            let _ = incur_cost (hd1 + hd2) in
            1 + longest_common_prefix tl1 tl2
          else 0 )

let rec z_algorithm_acc (acc : int list) (original_string : int list)
    (current_string : int list) (left : int) (right : int) =
  match current_string with
  | [] -> acc
  | hd :: tl ->
      let _ = incur_cost hd in
      let current_index = list_length acc in
      let old_result =
        if left = 0 then 0 else hd_exn (drop_n_elements acc (left - 1))
      in
      let current_result_initial =
        if current_index < right then min (right - current_index) old_result
        else 0
      in
      let first_sublist =
        drop_n_elements original_string current_result_initial
      in
      let second_sublist =
        drop_n_elements current_string current_result_initial
      in
      let common_prefix_size =
        longest_common_prefix first_sublist second_sublist
      in
      let current_result = current_result_initial + common_prefix_size in
      let cumulative_result_updated = current_result :: acc in
      if current_index + current_result > right then
        z_algorithm_acc cumulative_result_updated original_string tl
          current_index
          (current_index + current_result)
      else
        z_algorithm_acc cumulative_result_updated original_string tl left right

let rec reverse_acc (acc : int list) (xs : int list) =
  match xs with [] -> acc | hd :: tl -> reverse_acc (hd :: acc) tl

let z_algorithm (xs : int list) =
  match xs with
  | [] -> []
  | hd :: tl ->
      let result = z_algorithm_acc [ 0 ] xs tl 0 0 in
      reverse_acc [] result

let z_algorithm2 (xs : int list) = Raml.stat (z_algorithm xs)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""

z_algorithm_hybrid_ocaml_code = \
    """
exception Invalid_input

let incur_cost (hd : int) =
  let modulo = 5 in
  if (hd mod 100) = 0 then Raml.tick 1.0 
  else (if (hd mod modulo) = 1 then Raml.tick 0.85
        else if (hd mod modulo) = 2 then Raml.tick 0.65 else Raml.tick 0.5)

let rec list_length (xs : int list) =
  match xs with [] -> 0 | hd :: tl -> 1 + list_length tl

let hd_exn (xs : int list) =
  match xs with [] -> raise Invalid_input | hd :: _ -> hd

let min (x1 : int) (x2 : int) = if x1 < x2 then x1 else x2

let rec drop_n_elements (xs : int list) (n : int) =
  match xs with
  | [] -> []
  | hd :: tl -> if n = 0 then hd :: tl else drop_n_elements tl (n - 1)

let rec longest_common_prefix (xs1 : int list) (xs2 : int list) =
  match xs1 with
  | [] -> 0
  | hd1 :: tl1 -> (
      match xs2 with
      | [] -> 0
      | hd2 :: tl2 ->
          if hd1 = hd2 then
            let _ = incur_cost (hd1 + hd2) in
            1 + longest_common_prefix tl1 tl2
          else 0 )

let rec z_algorithm_acc (acc : int list) (original_string : int list)
    (current_string : int list) (left : int) (right : int) =
  match current_string with
  | [] -> acc
  | hd :: tl ->
      let _ = incur_cost hd in
      let current_index = list_length acc in
      let old_result =
        if left = 0 then 0 else hd_exn (drop_n_elements acc (left - 1))
      in
      let current_result_initial =
        if current_index < right then min (right - current_index) old_result
        else 0
      in
      let first_sublist =
        drop_n_elements original_string current_result_initial
      in
      let second_sublist =
        drop_n_elements current_string current_result_initial
      in
      let common_prefix_size =
        Raml.stat (longest_common_prefix first_sublist second_sublist)
      in
      let current_result = current_result_initial + common_prefix_size in
      let cumulative_result_updated = current_result :: acc in
      if current_index + current_result > right then
        z_algorithm_acc cumulative_result_updated original_string tl
          current_index
          (current_index + current_result)
      else
        z_algorithm_acc cumulative_result_updated original_string tl left right

let rec reverse_acc (acc : int list) (xs : int list) =
  match xs with [] -> acc | hd :: tl -> reverse_acc (hd :: acc) tl

let z_algorithm (xs : int list) =
  match xs with
  | [] -> []
  | hd :: tl ->
      let result = z_algorithm_acc [ 0 ] xs tl 0 0 in
      reverse_acc [] result

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_lists_python_to_ocaml(input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    if hybrid_mode == "data_driven":
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset z_algorithm2\n"
        return z_algorithm_data_driven_ocaml_code + input_generation_code
    else:
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset z_algorithm\n"
        return z_algorithm_hybrid_ocaml_code + input_generation_code
