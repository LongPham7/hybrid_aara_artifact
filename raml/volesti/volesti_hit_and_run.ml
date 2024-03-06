open Core
open Ctypes
open Foreign
open Volesti_common
open Hybrid_aara_config

(* This module offers OCaml functions for running the hit-and-run algorithms
implemented in the C++ library volesti. *)

let gaussian_rdhr_ocaml_interface =
  foreign "gaussian_rdhr"
    ( int @-> int @-> ptr double @-> ptr double @-> double @-> int @-> int
    @-> returning (ptr double) )

let uniform_rdhr_ocaml_interface =
  foreign "uniform_rdhr"
    ( int @-> int @-> ptr double @-> ptr double @-> int @-> int
    @-> returning (ptr double) )

let gaussian_cdhr_ocaml_interface =
  foreign "gaussian_cdhr"
    ( int @-> int @-> ptr double @-> ptr double @-> double @-> int @-> int
    @-> returning (ptr double) )

let uniform_cdhr_ocaml_interface =
  foreign "uniform_cdhr"
    ( int @-> int @-> ptr double @-> ptr double @-> int @-> int
    @-> returning (ptr double) )

let uniform_billiard_ocaml_interface =
  foreign "uniform_billiard"
    ( int @-> int @-> ptr double @-> ptr double @-> int @-> int
    @-> returning (ptr double) )

let gaussian_rdhr list_A list_b variance num_samples walk_length =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  let result =
    gaussian_rdhr_ocaml_interface num_rows num_cols (CArray.start array_A)
      (CArray.start array_b) variance num_samples walk_length
  in
  CArray.to_list (CArray.from_ptr result (num_cols * num_samples))

let uniform_rdhr list_A list_b num_samples walk_length =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  let result =
    uniform_rdhr_ocaml_interface num_rows num_cols (CArray.start array_A)
      (CArray.start array_b) num_samples walk_length
  in
  CArray.to_list (CArray.from_ptr result (num_cols * num_samples))

let gaussian_cdhr list_A list_b variance num_samples walk_length =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  let result =
    gaussian_cdhr_ocaml_interface num_rows num_cols (CArray.start array_A)
      (CArray.start array_b) variance num_samples walk_length
  in
  CArray.to_list (CArray.from_ptr result (num_cols * num_samples))

let uniform_cdhr list_A list_b num_samples walk_length =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  let result =
    uniform_cdhr_ocaml_interface num_rows num_cols (CArray.start array_A)
      (CArray.start array_b) num_samples walk_length
  in
  CArray.to_list (CArray.from_ptr result (num_cols * num_samples))

let uniform_billiard list_A list_b num_samples walk_length =
  let array_A = CArray.of_list double list_A in
  let array_b = CArray.of_list double list_b in
  let num_rows_times_cols = CArray.length array_A in
  let num_rows = CArray.length array_b in
  let num_cols = num_rows_times_cols / num_rows in
  let result =
    uniform_billiard_ocaml_interface num_rows num_cols (CArray.start array_A)
      (CArray.start array_b) num_samples walk_length
  in
  CArray.to_list (CArray.from_ptr result (num_cols * num_samples))

let test_gaussian_rdhr () =
  let list_A = [ -1.; 0.; 1.; 0.; 0.; -1.; 0.; 1.; 1.; -1.; -1.; 1. ] in
  let list_b = [ 0.; 10.; 0.; 10.; 0.001; 0.001 ] in
  let dim = 2 in
  let variance = 36. in
  let num_samples = 200 in
  let walk_length = 150 in
  let flattened_list_samples =
    gaussian_rdhr list_A list_b variance num_samples walk_length
  in
  let list_samples = partition_into_blocks flattened_list_samples dim in
  print_volesti_result list_samples

let test_uniform_rdhr () =
  let list_A = [ -1.; 0.; 1.; 0.; 0.; -1.; 0.; 1.; 1.; -1.; -1.; 1. ] in
  let list_b = [ 0.; 10.; 0.; 10.; 0.001; 0.001 ] in
  let dim = 2 in
  let num_samples = 200 in
  let walk_length = 150 in
  let flattened_list_samples =
    uniform_rdhr list_A list_b num_samples walk_length
  in
  let list_samples = partition_into_blocks flattened_list_samples dim in
  print_volesti_result list_samples
