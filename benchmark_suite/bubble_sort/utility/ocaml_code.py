import os
import sys
module_path = os.path.abspath(os.path.expanduser(os.path.join(
    "/home", "hybrid_aara", "statistical_aara_test_suite", "toolbox")))
if module_path not in sys.path:
    sys.path.append(module_path)

from input_data_generation import convert_lists_python_to_ocaml

# Create OCaml code

bubble_sort_data_driven_ocaml_code = \
    """
let incur_cost (hd : int) =
  if (hd mod 10) = 0 then Raml.tick 1.0 else Raml.tick 0.5

let rec scan_and_swap (xs : int list) =
  match xs with
  | [] -> ([], false)
  | [ x ] -> ([ x ], false)
  | x1 :: x2 :: tl ->
      let _ = incur_cost x1 in
      if x1 <= x2 then
        let recursive_result, is_swapped = scan_and_swap (x2 :: tl) in
        (x1 :: recursive_result, is_swapped)
      else
        let recursive_result, _ = scan_and_swap (x1 :: tl) in
        (x2 :: recursive_result, true)

let rec bubble_sort (xs : int list) =
  let xs_scanned, is_swapped = scan_and_swap xs in
  if is_swapped then bubble_sort xs_scanned else xs_scanned

let bubble_sort2 (xs : int list) = Raml.stat (bubble_sort xs)

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

"""


def create_ocaml_code(input_data_python, analysis_info):
    input_data_ocaml = convert_lists_python_to_ocaml(input_data_python)
    hybrid_mode = analysis_info["hybrid_mode"]
    assert (hybrid_mode == "data_driven")
    input_generation_code = ";;\n\nlet input_dataset = " + input_data_ocaml + \
        "\nin map input_dataset bubble_sort2\n"
    return bubble_sort_data_driven_ocaml_code + input_generation_code
