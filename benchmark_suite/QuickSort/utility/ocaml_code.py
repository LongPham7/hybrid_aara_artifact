import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join("/home", "hybrid_aara", "benchmark_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_lists_python_to_ocaml

# Create OCaml code

quicksort_data_driven_ocaml_code = \
    """
let incur_cost (hd : int) =
  if (hd mod 5) = 0 then Raml.tick 1.0 else Raml.tick 0.5

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

let rec quicksort (xs : int list) =
  match xs with
  | [] -> []
  | hd :: tl ->
      let lower_list, upper_list = partition hd tl in
      let lower_list_sorted = quicksort lower_list in
      let upper_list_sorted = quicksort upper_list in
      append lower_list_sorted (hd :: upper_list_sorted)

let quicksort2 (xs : int list) = Raml.stat (quicksort xs)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""

quicksort_hybrid_ocaml_code = \
    """
let incur_cost (hd : int) =
  if (hd mod 5) = 0 then Raml.tick 1.0 else Raml.tick 0.5

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

let rec quicksort (xs : int list) =
  match xs with
  | [] -> []
  | hd :: tl ->
      let lower_list, upper_list = Raml.stat (partition hd tl) in
      let lower_list_sorted = quicksort lower_list in
      let upper_list_sorted = quicksort upper_list in
      append lower_list_sorted (hd :: upper_list_sorted)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_lists_python_to_ocaml(input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    if hybrid_mode == "data_driven":
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset quicksort2\n"
        return quicksort_data_driven_ocaml_code + input_generation_code
    else:
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset quicksort\n"
        return quicksort_hybrid_ocaml_code + input_generation_code
