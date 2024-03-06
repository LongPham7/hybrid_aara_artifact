open Core

(* This module offers functions related to the OCaml-Python interface. *)

(* List conversion from OCaml to Python *)

let convert_ocaml_int_list_to_python_list ls =
  Py.List.of_list (List.map ls ~f:Py.Int.of_int)

let convert_ocaml_float_list_to_python_list ls =
  Py.List.of_list (List.map ls ~f:Py.Float.of_float)

(* Nested list conversion from OCaml to Python *)

let convert_ocaml_nested_int_list_to_python_list nested_list =
  Py.List.of_list
    (List.map nested_list ~f:convert_ocaml_int_list_to_python_list)

(* List conversion from Python to OCaml *)

let convert_python_float_list_to_ocaml_list ls =
  List.map (Py.List.to_list ls) ~f:(fun x -> Py.Float.to_float x)

(* Nested list conversion from Python to OCaml *)

let convert_python_nested_float_list_to_ocaml_list nested_list =
  List.map (Py.List.to_list nested_list) ~f:(fun ls ->
      convert_python_float_list_to_ocaml_list ls)
