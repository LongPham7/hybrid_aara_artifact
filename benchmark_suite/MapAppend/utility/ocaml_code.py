import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join("/home", "hybrid_aara", "benchmark_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_pairs_lists_python_to_ocaml

# Create OCaml code

append_data_driven_ocaml_code = \
    """
let incur_cost (hd : int) =
  let modulo = 5 in
  if (hd mod 100) = 0 then Raml.tick 1.0 
  else (if (hd mod modulo) = 1 then Raml.tick 0.85 
        else (if (hd mod modulo) = 2 then Raml.tick 0.65 else Raml.tick 0.5))

let rec append (xs : int list) (ys : int list) =
    match xs with
    | [] -> ys
    | hd :: tl ->
        let _ = incur_cost hd in
        hd :: (append tl ys)

let append2 (xs : int list) (ys : int list) = Raml.stat (append xs ys)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""

append_hybrid_ocaml_code = \
    """
let incur_cost (hd : int) =
  let modulo = 5 in
  if (hd mod 100) = 0 then Raml.tick 1.0 
  else (if (hd mod modulo) = 1 then Raml.tick 0.85 
        else (if (hd mod modulo) = 2 then Raml.tick 0.65 else Raml.tick 0.5))

let step_function (x : int) (xs : int list) (ys : int list) =
  let _ = incur_cost x in (xs, ys)

let rec append (xs : int list) (ys : int list) =
  match xs with
  | [] -> ys
  | hd :: tl ->
    let rec_xs, rec_ys = Raml.stat (step_function hd tl ys) in
    hd :: append rec_xs rec_ys

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_pairs_lists_python_to_ocaml(input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    if hybrid_mode == "data_driven":
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset (fun (x, y) -> append2 x y)\n"
        return append_data_driven_ocaml_code + input_generation_code
    else:
        input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
            "\nin map input_dataset (fun (x, y) -> append x y)\n"
        return append_hybrid_ocaml_code + input_generation_code
