open Core
open Format
open Rtypes
open Annotations
open Pprint

(* This module offers pretty-printing functions for data structures used in
hybrid AARA (e.g., runtime cost data). *)

let print_costs costs =
  match costs with
  | [ eval_cost; tick_cost; heap_cost; flip_cost ] ->
      printf
        ( "Costs of the sample:\n" ^^ "  Evaluation steps: %.2f\n"
        ^^ "  Ticks:            %.2f\n" ^^ "  Heap space:       %.2f\n"
        ^^ "  Flips:            %.2f\n@." )
        eval_cost tick_cost heap_cost flip_cost;
      pp_print_flush std_formatter ()
  | _ -> Misc.fatal_error "This is dead code."

(* Printing functions related to runtime cost dataset *)

let print_tick_cost (tick_cost : float) =
  printf "Cost: %.3f\n" tick_cost;
  pp_print_flush std_formatter ()

let print_raw_dataset (dataset : 'a Eval.raw_dataset) (heap : 'a Eval.heap) =
  let print_input ((variable_name, variable_type), loc) =
    printf "%s: %s = " variable_name (rtype_to_string variable_type);
    print_value (loc, heap)
  in
  let print_inputs evaluation_context =
    print_endline "Inputs in the evaluation context:";
    List.iter evaluation_context ~f:print_input
  in
  let print_output loc =
    print_endline "Output:";
    print_value (loc, heap);
    print_newline ()
  in
  let print_sample (e, evaluation_context, output, costs) =
    print_endline "Expression:";
    print_expression e;
    print_inputs evaluation_context;
    print_output output;
    print_costs costs
  in
  print_endline "Raw dataset:";
  List.iter dataset ~f:print_sample

let print_list_samples
    (list_samples : Runtime_cost_data.sample_without_expression list)
    (heap : 'a Eval.heap) =
  let print_input ((variable_name, variable_type), loc) =
    printf "%s: %s = " variable_name (rtype_to_string variable_type);
    print_value (loc, heap)
  in
  let print_inputs evaluation_context =
    print_endline "Inputs in the evaluation context:";
    List.iter evaluation_context ~f:print_input
  in
  let print_output loc =
    print_endline "Output:";
    print_value (loc, heap)
  in
  let print_sample (evaluation_context, output, cost) =
    print_inputs evaluation_context;
    print_output output;
    print_tick_cost cost;
    print_newline ()
  in
  List.iter list_samples ~f:print_sample

let print_dataset_categorized_by_expression
    (dataset : 'a Runtime_cost_data.dataset_categorized_by_expressions)
    (heap : 'a Eval.heap) =
  let print_expression_list_samples (e, list_samples) =
    print_endline "Expression:";
    print_expression e;
    print_list_samples list_samples heap
  in
  print_endline "Dataset categorized by expressions:";
  List.iter dataset ~f:print_expression_list_samples

let print_dataset_with_potential
    (dataset : 'a Runtime_cost_data.dataset_with_potential) =
  let print_potential_context potential_context =
    print_endline "Potential in the context:";
    List.iter potential_context ~f:(fun (cindex, potential) ->
        print_string "Cindex: ";
        print_cindex cindex;
        printf " Potential: %i\n" potential);
    pp_print_flush std_formatter ()
  in
  let print_potential_output potential_output =
    print_endline "Potential in the output:";
    List.iter potential_output ~f:(fun (index, potential) ->
        print_string "Index: ";
        print_index index;
        printf " Potential: %i\n" potential);
    pp_print_flush std_formatter ()
  in
  let print_sample (potential_context, potential_output, cost) =
    print_potential_context potential_context;
    print_potential_output potential_output;
    print_tick_cost cost
  in
  let print_list_samples (e, list_samples) =
    print_endline "Expression:";
    print_expression e;
    List.iter list_samples ~f:print_sample
  in
  print_endline "Dataset with potential:";
  List.iter dataset print_list_samples

let print_list_potential_categorized_by_indices
    (dataset : Runtime_cost_data.list_potential_categorized_by_indices) =
  let print_cindex_potentials (cindex, list_potentials) =
    print_string "Cindex: ";
    print_cindex cindex;
    print_string " List of potential: [";
    print_list_sep list_potentials print_int "; ";
    print_string "]\n"
  in
  let print_list_cindex_potentials list_cindex_potentials =
    List.iter list_cindex_potentials ~f:print_cindex_potentials;
    pp_print_flush std_formatter ()
  in
  let print_index_potentials (index, list_potentials) =
    print_string "Index: ";
    print_index index;
    print_string " List of potential: [";
    print_list_sep list_potentials print_int "; ";
    print_string "]\n"
  in
  let print_list_index_potentials list_index_potentials =
    List.iter list_index_potentials ~f:print_index_potentials;
    pp_print_flush std_formatter ()
  in
  let print_list_costs list_costs =
    print_string "List of costs:\n[";
    print_list_sep list_costs print_tick_cost "; ";
    print_string "]\n\n";
    pp_print_flush std_formatter ()
  in
  let list_cindex_potentials, list_index_potentials, list_costs = dataset in
  print_endline "Indices and potentials in the evaluation context: ";
  print_list_cindex_potentials list_cindex_potentials;
  print_endline "Indices and potentials in the output: ";
  print_list_index_potentials list_index_potentials;
  print_list_costs list_costs

let print_dataset_with_potential_categorized_by_indices
    (dataset :
      'a Runtime_cost_data.dataset_with_potential_categorized_by_indices) =
  let print_dataset_of_an_expression (e, dataset_of_e) =
    print_endline "Expression:";
    print_expression e;
    print_list_potential_categorized_by_indices dataset_of_e
  in
  List.iter dataset ~f:print_dataset_of_an_expression

(* Printing functions related to linear programs *)

let print_linear_program (list_A : float list list) (list_b : float list) =
  let num_constraints = List.length list_A in
  let () = assert (num_constraints = List.length list_b) in
  let print_row row_number row =
    printf "Row %i: " row_number;
    List.iter row ~f:(fun x -> printf "%.1f " x);
    print_newline ()
  in
  print_endline "Matrix A:";
  List.iteri list_A ~f:print_row;
  print_endline "Vector b:";
  List.iter list_b ~f:(fun x -> printf "%.1f " x);
  print_newline ()

(* Printing functions for dataset with integer-valued LP variables *)

let print_dataset_with_lp_vars
    (dataset :
      ((int * float list) list * (int * float list) list * float list) list) =
  let print_var_list_potential list_var_potential =
    let var, list_potential = list_var_potential in
    printf "LP var: %i; list of potential: " var;
    List.iter list_potential ~f:(fun p -> printf "%.1f " p);
    print_newline ()
  in
  let print_dataset_of_e dataset_of_e =
    let list_cindex_var_potential, list_index_var_potential, list_costs =
      dataset_of_e
    in
    print_endline "List of cindex vars and their lists of potential:";
    List.iter list_cindex_var_potential ~f:print_var_list_potential;
    print_endline "List of index vars and their lists of potential:";
    List.iter list_index_var_potential ~f:print_var_list_potential;
    print_endline "List of costs:";
    List.iter list_costs ~f:(fun x -> printf "%.1f " x);
    print_newline ()
  in
  List.iter dataset ~f:print_dataset_of_e

let print_dataset_lp_vars_are_expanded
    (list_samples : ((int * float) list * (int * float) list * float) list) =
  let print_list_var_potential list_var_potential =
    print_string "(var, potential) = ";
    List.iter list_var_potential ~f:(fun (var, potential) ->
        printf "(%i, %.1f) " var potential)
  in
  let print_sample sample =
    let list_cindex_var_potential, list_index_var_potential, cost = sample in
    print_string "Typing context: ";
    print_list_var_potential list_cindex_var_potential;
    print_newline ();
    print_string "Return type: ";
    print_list_var_potential list_index_var_potential;
    print_newline ();
    printf "Cost: %.1f" cost;
    print_newline ()
  in
  List.iter list_samples ~f:print_sample

(* Printing functions for reporting resource analysis results *)

(* The version of fprint_type_anno that handles an annotated type where each
index is mapped to a list of coefficients, rather than a single coefficient. The
list of coefficients represents a probability distribution, and is used in
hybrid AARA. For each list of coefficients, we print out its mean and
standard deviation. *)
let fprint_type_anno_distribution f tanno =
  pp_open_vbox f 2;
  let print_annos f tanno =
    let add_ind ind () =
      let list_q = tan_find tanno ind in
      assert (not (List.is_empty list_q));
      let array_q = Array.of_list list_q in
      let avg_q = Owl_stats.mean array_q in
      let std_q = Owl_stats.std array_q in
      fprintf f "@[(avg, std) = (%10g, %10g)  <--  %a@]@ " avg_q std_q
        fprint_index ind
    in
    Indices.indices_max_deg ~empty:() ~add_ind tanno.tan_type tanno.tan_deg
  in
  print_annos f tanno;
  pp_close_box f ();
  pp_print_flush f ()

(* The version of fprint_anno_funtype that handles an annotated type where each
index is mapped to a list of coefficients, rather than a single coefficient. The
list of coefficients represents a probability distribution, and is used in
hybrid AARA. *)
let fprint_anno_funtype_distribution ?(indent = "") ?(simple_name = false) f
    (fid, atanno, rtanno) =
  let arrow_type =
    match atanno.tan_type with
    | Ttuple ts -> Tarrow (ts, rtanno.tan_type, ())
    | _ -> raise (Invalid_argument "Expecting tuple type.")
  in
  let fid =
    if simple_name then
      match String.lsplit2 fid ~on:'#' with None -> fid | Some (s1, s2) -> s1
    else fid
  in
  let fprint_raml_type f t = fprint_raml_type ~indent:2 f t in
  fprintf f "@.== %s :\n\n%s%a@." fid indent fprint_raml_type arrow_type;
  fprintf f "\n%sNon-zero annotations of the argument:\n%a" indent
    fprint_type_anno_distribution atanno;
  fprintf f "\n%sNon-zero annotations of result:\n%a" indent
    fprint_type_anno_distribution rtanno;
  let pol, descs = Polynomials.describe_pol_distribution atanno in
  let _ = fprintf f "\n%sSimplified bound:\n   %s" indent indent in
  let _ = fprint_polynomial f pol in
  if List.length descs > 0 then (
    let _ = fprintf f "\n %swhere" indent in
    fprint_pol_desc f descs;
    fprintf f "\n" )
  else ();
  fprintf f "@."

let print_anno_funtype_distribution ?(indent = "") ?(simple_name = false)
    ?output:(formatter = std_formatter) atyp =
  fprint_anno_funtype_distribution formatter ~indent ~simple_name atyp

(* Printing functions related to posterior distributions *)

(* Write out (i) the coefficients in a function's resource annotation and (ii)
the analysis time to a JSON file. This function is used when the analysis
returns a single inferred cost bound, rather than a posterior distribution of
inferred cost bounds. *)
let write_coefficients_and_analysis_time_to_json_file output_file
    (fid, annotated_arg_type, annotated_return_type) analysis_time =
  let index_and_coefficient_json annotated_type =
    let list_indices =
      Indices.indices_max_deg ~empty:[]
        ~add_ind:(fun index acc -> index :: acc)
        annotated_type.tan_type annotated_type.tan_deg
    in
    let coefficient_json index =
      let coefficient = tan_find annotated_type index in
      `Float coefficient
    in
    let index_string_coefficient_json index =
      let () = Pprint.fprint_index str_formatter index in
      let string_of_index = flush_str_formatter () in
      (string_of_index, coefficient_json index)
    in
    `Assoc (List.map list_indices index_string_coefficient_json)
  in
  let inference_result_json =
    `Assoc
      [
        ("fid", `String fid);
        ("typing_context", index_and_coefficient_json annotated_arg_type);
        ("return_type", index_and_coefficient_json annotated_return_type);
        ("analysis_time", `Float analysis_time);
      ]
  in
  Yojson.Basic.to_file output_file inference_result_json

(* Write out (i) the posterior distribution of a function's resource annotation
and (ii) the analysis time to a JSON file.*)
let write_distribution_and_analysis_time_to_json_file output_file
    (fid, annotated_arg_type, annotated_return_type) analysis_time =
  let index_and_distribution_json annotated_type =
    let list_indices =
      Indices.indices_max_deg ~empty:[]
        ~add_ind:(fun index acc -> index :: acc)
        annotated_type.tan_type annotated_type.tan_deg
    in
    let distribution_json index =
      let list_coefficients = tan_find annotated_type index in
      assert (not (List.is_empty list_coefficients));
      Json_toolbox.convert_float_list_to_json list_coefficients
    in
    let index_string_distribution_json index =
      let () = Pprint.fprint_index str_formatter index in
      let string_of_index = flush_str_formatter () in
      (string_of_index, distribution_json index)
    in
    `Assoc (List.map list_indices index_string_distribution_json)
  in
  let inference_result_json =
    `Assoc
      [
        ("fid", `String fid);
        ("typing_context", index_and_distribution_json annotated_arg_type);
        ("return_type", index_and_distribution_json annotated_return_type);
        ("analysis_time", `Float analysis_time);
      ]
  in
  Yojson.Basic.to_file output_file inference_result_json

(* Write out the inferred coefficients of given LP variables to a JSON file.
Here, each LP variable is associated with a single inferred coefficient, rather
than a collection (i.e., posterior distribution) of coefficients. This function
is used when we want to look at the resource coefficients around Raml.stat(e) in
hybrid Opt. *)
let write_lp_vars_inferred_coefficients_to_json_file list_vars_coefficients
    output_file =
  let convert_lp_var_coefficient_to_json (var_int, coefficient) =
    (Int.to_string var_int, `Float coefficient)
  in
  let lp_vars_coefficients_json =
    `Assoc
      (List.map list_vars_coefficients ~f:convert_lp_var_coefficient_to_json)
  in
  Yojson.Basic.to_file output_file lp_vars_coefficients_json

(* Write out the posterior distribution of given LP variables to a JSON file.
This is used when we want to look at the posterior distributions of Bayesian
inference on a helper function that corresponds to an internal node of a typing
tree. Before using function, we must figure out what LP variables are used in
this internal node. *)
let write_lp_vars_distributions_to_json_file list_vars_distributions output_file
    =
  let convert_lp_var_distribution_to_json (var_int, distribution) =
    (Int.to_string var_int, Json_toolbox.convert_float_list_to_json distribution)
  in
  let lp_vars_distributions_json =
    `Assoc
      (List.map list_vars_distributions ~f:convert_lp_var_distribution_to_json)
  in
  Yojson.Basic.to_file output_file lp_vars_distributions_json
