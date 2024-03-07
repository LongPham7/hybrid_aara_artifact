import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join("/home", "hybrid_aara", "benchmark_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_lists_python_to_ocaml

# Create OCaml code

insertion_sort_data_driven_ocaml_code = \
    """
let incur_cost (hd : int) =
  let modulo = 5 in
  if (hd mod 200) = 0 then Raml.tick 1.0 
  else (if (hd mod modulo) = 1 then Raml.tick 0.85
        else if (hd mod modulo) = 2 then Raml.tick 0.65 else Raml.tick 0.5)

let rec insert (x : int) (xs : int list) =
  match xs with
  | [] -> [ x ]
  | hd :: tl ->
      let _ = incur_cost hd in
      if x <= hd then x :: hd :: tl else hd :: insert x tl

let rec insertion_sort (xs : int list) =
  match xs with [] -> [] | hd :: tl -> insert hd (insertion_sort tl)

let rec insertion_sort_second_time (xs : int list) =
  match xs with
  | [] -> []
  | hd :: tl -> insert hd (insertion_sort_second_time tl)

let insertion_sort_second_time2 (xs : int list) = 
  Raml.stat (insertion_sort_second_time xs)

let double_insertion_sort (xs : int list) =
  let sorted_xs = insertion_sort xs in
  Raml.stat (insertion_sort_second_time sorted_xs)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""

insertion_sort_hybrid_ocaml_code = \
    """
let incur_cost (hd : int) =
  let modulo = 5 in
  if (hd mod 200) = 0 then Raml.tick 1.0 
  else (if (hd mod modulo) = 1 then Raml.tick 0.85
        else if (hd mod modulo) = 2 then Raml.tick 0.65 else Raml.tick 0.5)

let rec insert (x : int) (xs : int list) =
  match xs with
  | [] -> [ x ]
  | hd :: tl ->
      let _ = incur_cost hd in
      if x <= hd then x :: hd :: tl else hd :: insert x tl

let rec insertion_sort (xs : int list) =
  match xs with [] -> [] | hd :: tl -> insert hd (insertion_sort tl)

let rec insert_second_time (x : int) (xs : int list) =
  match xs with
  | [] -> [ x ]
  | hd :: tl ->
      let _ = incur_cost hd in
      if x <= hd then x :: hd :: tl
      (* else hd :: Raml.stat (insert_second_time x tl) *)
      else hd :: (insert_second_time x tl)

let rec insertion_sort_second_time (xs : int list) =
  match xs with
  | [] -> []
  | hd :: tl ->
      let rec_result = insertion_sort_second_time tl in
      Raml.stat (insert_second_time hd rec_result)

let double_insertion_sort (xs : int list) =
  let sorted_xs = insertion_sort xs in
  insertion_sort_second_time sorted_xs

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_lists_python_to_ocaml(input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    if hybrid_mode == "data_driven":
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset double_insertion_sort\n"
        return insertion_sort_data_driven_ocaml_code + input_generation_code
    else:
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset double_insertion_sort\n"
        return insertion_sort_hybrid_ocaml_code + input_generation_code
