open Core
open Rtypes
open Expressions

(* This module checks whether two given function applications are identical,
modulo variable renaming. If they are identical modulo variable renaming, the
module computes the correspondence (i.e., mapping) between the free variables in
the two function applications.

This module is used when we categorize a runtime cost dataset according to
expressions. In this step, we want to group together two runtime cost samples if
they have the identical function application.

It is non-trivial to check if two function applications are identical. If we
have functions applications of the same function f in two different call sites
in the source code, they create two syntactically different string-encodings of
the function applications. Concretely, these two function applications have
slightly different input-variable names (e.g., f x#1 and f x#2). *)

let is_equal_function_names ?(prefix = '#') f1 f2 =
  let original_function_name f_name =
    match String.lsplit2 f_name ~on:prefix with
    | None -> f1
    | Some (first_substring, _) -> first_substring
  in
  let original_function_name1 = original_function_name f1 in
  let original_function_name2 = original_function_name f2 in
  String.( = ) original_function_name1 original_function_name2

let rec is_equal_function_applications exp1 exp2 =
  let { exp_desc = exp_desc1; exp_type = exp_type1 } = exp1 in
  let { exp_desc = exp_desc2; exp_type = exp_type2 } = exp2 in
  match (exp_desc1, exp_desc2) with
  | Ebase_const constant1, Ebase_const constant2 ->
      compare_constant constant1 constant2 = 0
  | Ebase_fun builtin_fun1, Ebase_fun builtin_fun2 ->
      compare_builtin_fun builtin_fun1 builtin_fun2 = 0
  | Ebase_op builtin_op1, Ebase_op builtin_op2 ->
      compare_builtin_op builtin_op1 builtin_op2 = 0
  | Evar var_id1, Evar var_id2 -> String.( = ) var_id1 var_id2
  | Eapp (call_name1, f1, args1), Eapp (call_name2, f2, args2) -> (
      let f_result = is_equal_function_applications f1 f2 in
      match List.zip args1 args2 with
      | None -> false
      | Some zipped_list ->
          let list_recursive_results =
            List.for_all zipped_list ~f:(fun (arg1, arg2) ->
                is_equal_function_applications arg1 arg2)
          in
          f_result && list_recursive_results )
  | _, _ -> false

let rec free_variable_correspondence_function_applications
    (exp1 : ('a, unit) expression) (exp2 : ('b, unit) expression) =
  let { exp_desc = exp_desc1; exp_type = exp_type1 } = exp1 in
  let { exp_desc = exp_desc2; exp_type = exp_type2 } = exp2 in
  match (exp_desc1, exp_desc2) with
  | Ebase_const constant1, Ebase_const constant2 ->
      if compare_constant constant1 constant2 = 0 then Some [] else None
  | Ebase_fun builtin_fun1, Ebase_fun builtin_fun2 ->
      if compare_builtin_fun builtin_fun1 builtin_fun2 = 0 then Some []
      else None
  | Ebase_op builtin_op1, Ebase_op builtin_op2 ->
      if compare_builtin_op builtin_op1 builtin_op2 = 0 then Some [] else None
  | Evar var_id1, Evar var_id2 ->
      if compare_raml_type exp_type1 exp_type2 = 0 then
        match exp_type1 with
        | Tarrow _ ->
            if String.( = ) var_id1 var_id2 then Some []
            else if is_equal_function_names var_id1 var_id2 then
              Some [ (var_id1, var_id2) ]
            else None
        | _ -> Some [ (var_id1, var_id2) ]
      else None
  | Eapp (call_name1, f1, args1), Eapp (call_name2, f2, args2) -> (
      let f_result = free_variable_correspondence_function_applications f1 f2 in
      match List.zip args1 args2 with
      | None -> None
      | Some zipped_list ->
          let list_recursive_results =
            List.map zipped_list ~f:(fun (arg1, arg2) ->
                free_variable_correspondence_function_applications arg1 arg2)
          in
          let folding_function acc x =
            match (acc, x) with
            | None, _ -> None
            | _, None -> None
            | Some renaming1, Some renaming2 -> Some (renaming1 @ renaming2)
          in
          List.fold
            (f_result :: list_recursive_results)
            ~init:(Some []) ~f:folding_function )
  | _, _ -> None

(* It is the second version of
free_variable_correspondence_function_applications. Unlike the first version,
this version requires the second argument to have type (raml_type list, unit)
expression, which is the output of the function Typecheck.typecheck_stack. Given
an expression e of type A1 -> A2, suppose we run Typecheck.typecheck_stack on e.
Then in the output, confusingly, e has exp_type = A2 and exp_info = [A1].
Therefore, to recover the original arrow type of e, we must look at not only
e.exp_type but also e.exp_info. *)
let rec free_variable_correspondence_function_applications_typecheck_tstack
    (exp1 : ('a, unit) expression) (exp2 : (raml_type list, unit) expression) =
  let { exp_desc = exp_desc1; exp_type = exp_type1 } = exp1 in
  let { exp_desc = exp_desc2; exp_type = exp_type2; exp_info = exp_info2 } =
    exp2
  in
  match (exp_desc1, exp_desc2) with
  | Ebase_const constant1, Ebase_const constant2 ->
      if compare_constant constant1 constant2 = 0 then Some [] else None
  | Ebase_fun builtin_fun1, Ebase_fun builtin_fun2 ->
      if compare_builtin_fun builtin_fun1 builtin_fun2 = 0 then Some []
      else None
  | Ebase_op builtin_op1, Ebase_op builtin_op2 ->
      if compare_builtin_op builtin_op1 builtin_op2 = 0 then Some [] else None
  | Evar var_id1, Evar var_id2 ->
      let exp_type2_original =
        if List.is_empty exp_info2 then exp_type2
        else Tarrow (exp_info2, exp_type2, ())
      in
      if compare_raml_type exp_type1 exp_type2_original = 0 then
        match exp_type1 with
        | Tarrow _ ->
            if String.( = ) var_id1 var_id2 then Some []
            else if is_equal_function_names var_id1 var_id2 then
              Some [ (var_id1, var_id2) ]
            else None
        | _ -> Some [ (var_id1, var_id2) ]
      else None
  | Eapp (call_name1, f1, args1), Eapp (call_name2, f2, args2) -> (
      let f_result =
        free_variable_correspondence_function_applications_typecheck_tstack f1
          f2
      in
      match List.zip args1 args2 with
      | None -> None
      | Some zipped_list ->
          let list_recursive_results =
            List.map zipped_list ~f:(fun (arg1, arg2) ->
                free_variable_correspondence_function_applications_typecheck_tstack
                  arg1 arg2)
          in
          let folding_function acc x =
            match (acc, x) with
            | None, _ -> None
            | _, None -> None
            | Some renaming1, Some renaming2 -> Some (renaming1 @ renaming2)
          in
          List.fold
            (f_result :: list_recursive_results)
            ~init:(Some []) ~f:folding_function )
  | _, _ -> None

(* It returns the list of argument variables of an input expression, provided
that it is a function application where all arguments are variables (as opposed
to non-variable expressions). This function is used to extract argument
variables from append's step function in the experiment of coefficient
perturbation. *)
let free_variable_correspondence_append_stepping_function exp =
  let return_var_string exp =
    let { exp_desc } = exp in
    match exp_desc with
    | Evar var_string -> var_string
    | _ -> failwith "Given expression is not a variable"
  in
  let { exp_desc } = exp in
  let args =
    match exp_desc with
    | Eapp (call_name, f, args) -> args
    | _ -> failwith "Pattern matching for append's stepping function fails"
  in
  List.map args ~f:return_var_string
