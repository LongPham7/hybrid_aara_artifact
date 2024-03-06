import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join("/home", "hybrid_aara", "statistical_aara_test_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_lists_python_to_ocaml

# Create OCaml code


round_data_driven_ocaml_code = \
    """
let incur_cost (hd : int) =
  if (hd mod 10) = 0 then Raml.tick 1.0 else Raml.tick 0.5

let rec double (xs : int list) =
  match xs with [] -> [] | hd :: tl -> hd :: hd :: double tl

let rec half (xs : int list) =
  match xs with [] -> [] | [ x ] -> [] | x1 :: x2 :: tl -> x1 :: half tl

let rec round (xs : int list) =
  match xs with
  | [] -> []
  | hd :: tl ->
      let half_result = half tl in
      let recursive_result = round half_result in
      hd :: double recursive_result

let rec linear_traversal (xs : int list) =
  match xs with
  | [] -> []
  | hd :: tl ->
      let _ = incur_cost hd in
      hd :: linear_traversal tl

let round_followed_by_linear_traversal (xs : int list) =
  let round_result = round xs in
  linear_traversal round_result

let round2 (xs : int list) = 
  Raml.stat (round_followed_by_linear_traversal xs)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_lists_python_to_ocaml(input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    assert (hybrid_mode == "data_driven")
    input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
        "\nin map input_dataset round2\n"
    return round_data_driven_ocaml_code + input_generation_code
