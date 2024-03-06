open Core

(* This module offers commonly used functions for manipulating JSON-related data
structures. *)

let convert_int_list_to_json list_coefficients =
  `List (List.map list_coefficients ~f:(fun c -> `Int c))

let convert_float_list_to_json list_coefficients =
  `List (List.map list_coefficients ~f:(fun c -> `Float c))
