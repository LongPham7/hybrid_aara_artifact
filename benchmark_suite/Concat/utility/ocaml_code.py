import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join("/home", "hybrid_aara", "benchmark_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_nested_lists_python_to_ocaml

# Create OCaml code

concat_data_driven_ocaml_code = \
    """
let incur_cost (hd : int) =
  if (hd mod 5) = 0 then Raml.tick 1.0 else Raml.tick 0.5

let rec append (xs : int list) (ys : int list) =
  match xs with
  | [] -> ys
  | hd :: tl ->
      let _ = incur_cost hd in
      hd :: append tl ys

let rec concat (xss : int list list) =
  match xss with [] -> [] | hd :: tl -> append hd (Raml.stat (concat tl))
  
let concat2 (xss : int list list) = Raml.stat (concat xss)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""

concat_hybrid_ocaml_code = \
    """
let incur_cost (hd : int) =
  if (hd mod 5) = 0 then Raml.tick 1.0 else Raml.tick 0.5

let rec append (xs : int list) (ys : int list) =
  match xs with
  | [] -> ys
  | hd :: tl ->
      let _ = incur_cost hd in
      hd :: append tl ys

let rec concat (xss : int list list) =
  match xss with
  | [] -> []
  | hd :: tl ->
      let rec_tl = concat tl in
      Raml.stat (append hd rec_tl)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_nested_lists_python_to_ocaml(input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    if hybrid_mode == "data_driven":
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset concat2\n"
        return concat_data_driven_ocaml_code + input_generation_code
    else:
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset concat\n"
        return concat_hybrid_ocaml_code + input_generation_code
