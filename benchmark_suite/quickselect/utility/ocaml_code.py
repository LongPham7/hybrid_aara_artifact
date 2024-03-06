import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join("/home", "hybrid_aara", "statistical_aara_test_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_integer_list_pair_python_to_ocaml

# Create OCaml code


quickselect_data_driven_ocaml_code = \
"""
exception Invalid_input

let incur_cost (hd : int) =
  if (hd mod 10) = 0 then Raml.tick 1.0 else Raml.tick 0.5

let rec append (xs : int list) (ys : int list) =
  match xs with [] -> ys | hd :: tl -> hd :: append tl ys

let rec partition (pivot : int) (xs : int list) =
  match xs with
  | [] -> ([], [])
  | hd :: tl ->
      let lower_list, upper_list = partition pivot tl in
      let _ = incur_cost hd in
      if hd <= pivot then (hd :: lower_list, upper_list)
      else (lower_list, hd :: upper_list)

let rec list_length (xs : int list) =
  match xs with [] -> 0 | hd :: tl -> 1 + list_length tl

let rec quickselect (index : int) (xs : int list) =
  match xs with
  | [] -> raise Invalid_input
  | [ x ] -> if index = 0 then x else raise Invalid_input
  | hd :: tl ->
      let lower_list, upper_list = partition hd tl in
      let lower_list_length = list_length lower_list in
      if index < lower_list_length then quickselect index lower_list
      else if index = lower_list_length then hd
      else
        let new_index = index - lower_list_length - 1 in
        quickselect new_index upper_list

let quickselect2 (index : int) (xs : int list) =
  Raml.stat (quickselect index xs)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""

quickselect_hybrid_ocaml_code = \
"""
exception Invalid_input

let incur_cost (hd : int) =
  if (hd mod 200) = 0 then Raml.tick 1.0 else Raml.tick 0.5

let rec append (xs : int list) (ys : int list) =
  match xs with [] -> ys | hd :: tl -> hd :: append tl ys

let rec partition (pivot : int) (xs : int list) =
  match xs with
  | [] -> ([], [])
  | hd :: tl ->
      (* let lower_list, upper_list = Raml.stat (partition pivot tl) in *)
      let lower_list, upper_list = partition pivot tl in
      let _ = incur_cost hd in
      if hd <= pivot then (hd :: lower_list, upper_list)
      else (lower_list, hd :: upper_list)

let rec partition_cost_free (pivot : int) (xs : int list) =
  match xs with
  | [] -> ([], [])
  | hd :: tl ->
      let lower_list, upper_list = partition_cost_free pivot tl in
      if hd <= pivot then (hd :: lower_list, upper_list)
      else (lower_list, hd :: upper_list)

let rec list_length (xs : int list) =
  match xs with [] -> 0 | hd :: tl -> 1 + list_length tl

let rec quickselect (index : int) (xs : int list) =
  match xs with
  | [] -> raise Invalid_input
  | [ x ] -> if index = 0 then x else raise Invalid_input
  | hd :: tl ->
      (* This is a workaround for an issue with the let-normal form inside
      Raml.stat(...) *)
      let tl = tl in
      let lower_list, _ = partition_cost_free hd tl in
      let lower_list_length = list_length lower_list in
      if index < lower_list_length then
        let lower_list, _ = Raml.stat (partition hd tl) in
        quickselect index lower_list
      else if index = lower_list_length then 
        let _, _ = Raml.stat (partition hd tl) in
        hd
      else
        let _, upper_list = Raml.stat (partition hd tl) in
        quickselect (index - lower_list_length - 1) upper_list

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_integer_list_pair_python_to_ocaml(input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    if hybrid_mode == "data_driven":
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset (fun (index, xs) -> quickselect2 index xs)\n"
        return quickselect_data_driven_ocaml_code + input_generation_code
    else:
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset (fun (index, xs) -> quickselect index xs)\n"
        return quickselect_hybrid_ocaml_code + input_generation_code
