import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join("/home", "hybrid_aara", "benchmark_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_lists_python_to_ocaml


# Create OCaml code

even_split_odd_tail_data_driven_ocaml_code = \
    """
exception Invalid_input

let incur_cost (hd : int) =
  if (hd mod 10) = 0 then Raml.tick 1.0 else Raml.tick 0.5

let rec linear_traversal (xs : int list) =
  match xs with
  | [] -> []
  | hd :: tl ->
      let _ = incur_cost hd in
      hd :: linear_traversal tl

let rec is_even (xs : int list) =
  match xs with [] -> true | [ x ] -> false | x1 :: x2 :: tl -> is_even tl

let tail (xs : int list) =
  match xs with [] -> raise Invalid_input | hd :: tl -> tl

let rec split (xs : int list) =
  match xs with
  | [] -> []
  | [ x ] -> raise Invalid_input
  | x1 :: x2 :: tl -> x1 :: split tl

let rec even_split_odd_tail (xs : int list) : int list =
  let xs_traversed = linear_traversal xs in
  match xs_traversed with
  | [] -> []
  | hd :: tl ->
      let xs_is_even = is_even xs_traversed in
      if xs_is_even then
        let split_result = split xs_traversed in
        even_split_odd_tail split_result
      else
        let tail_result = tail xs_traversed in
        even_split_odd_tail tail_result

let even_split_odd_tail2 (xs : int list) : int list =
  Raml.stat (even_split_odd_tail xs)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_lists_python_to_ocaml(input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    assert (hybrid_mode == "data_driven")
    input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
        "\nin map input_dataset even_split_odd_tail2\n"
    return even_split_odd_tail_data_driven_ocaml_code + input_generation_code
