open Core
open Eval

(* This module offers data types and functions for runtime cost data in hybrid
AARA. 

Suppose we perform hybrid AARA on a program P, whose source code contains
Raml.stat(e). This means we perform data-driven analysis on code fragment e. Its
runtime cost dataset contains (i) expression e, (ii) information about the input
(i.e., evaluation context) to expression e, (iii) information about the output
of expression e, and (iv) the cost of evaluating expression e. We need to record
expression e in the dataset because the program P may contain several instances
of Raml.stat(e). 

To collect runtime cost measurements of expression e, we run it on many inputs,
and records the expression, inputs, outputs, and costs. Thus, the raw result of
runtime-cost-data collection is simply a list of runtime cost measurements, each
of which contains expression e, its input, output, and cost. But this format of
the dataset is difficult to work with. For example, we want to group together
all measurements that belong to the same expression. Another example is that we
want to group together all input values of the same free variable appearing in
expression e (i.e., a variable appearing in e's typing context).

The process of converting a raw dataset to a desirable format is split into
stages. The functions in this module are laid out in the chronological order of
the stages. *)

(* Stage 0: extract tick costs from a raw dataset of type 'a Eval.raw_dataset.
In hybrid AARA, we focus on tick costs. *)

type 'a sample_with_only_tick_cost =
  ('a, unit) Expressions.expression
  * (Expressions.var_bind * location) list
  * location
  * float

(* Raw dataset of runtime costs *)
type 'a raw_dataset_with_only_tick_costs = 'a sample_with_only_tick_cost list

let extract_tick_metric_from_raw_dataset (raw_dataset : 'a raw_dataset) :
    'a raw_dataset_with_only_tick_costs =
  let extract_tick_metric sample =
    let expression, context, output, list_costs = sample in
    let tick_cost =
      match list_costs with
      | [ _; tick_cost; _; _ ] -> tick_cost
      | _ -> failwith "The list of costs has a wrong form"
    in
    (expression, context, output, tick_cost)
  in
  List.map raw_dataset ~f:extract_tick_metric

(* Stage 1: Eval.evaluate returns a raw dataset of runtime cost data. It is a
list of samples, each of which stores an expression, an evaluation context, an
output, and runtime costs. In the first stage of processing, we group these
samples together according to their associated expressions. *)

(* Individual sample in the categorized dataset. It is identical to
Eval.sample_with_expression, except that sample_without_expression does not
contain an expression. *)
type sample_without_expression =
  (Expressions.var_bind * location) list * location * float

(* Dataset of runtime costs categorized by expressions. That is, each expression
comes with a list of samples. *)
type 'a dataset_categorized_by_expressions =
  (('a, unit) Expressions.expression * sample_without_expression list) list

let categorize_dataset_by_expressions
    (dataset : 'a raw_dataset_with_only_tick_costs) :
    'a dataset_categorized_by_expressions =
  let insert_sample_to_list_samples (context, output, cost)
      (expression, list_var_samples) =
    (expression, (context, output, cost) :: list_var_samples)
  in
  let rec insert_sample_categorized_dataset categorized_dataset sample =
    let expression, context, output, cost = sample in
    match categorized_dataset with
    | [] -> [ (expression, [ (context, output, cost) ]) ]
    | (head_expression, list_samples) :: categorized_dataset_tail ->
        if
          Free_variable_correspondence.is_equal_function_applications expression
            head_expression
        then
          insert_sample_to_list_samples (context, output, cost)
            (expression, list_samples)
          :: categorized_dataset_tail
        else
          (head_expression, list_samples)
          :: insert_sample_categorized_dataset categorized_dataset_tail sample
  in
  List.fold dataset ~init:[] ~f:insert_sample_categorized_dataset

let categorize_dataset_by_expressions_modulo_free_variables
    (dataset : 'a dataset_categorized_by_expressions) :
    'a dataset_categorized_by_expressions =
  let rename_var_bind free_variable_correspondence (var_id, rtype) =
    match
      List.Assoc.find free_variable_correspondence ~equal:String.( = ) var_id
    with
    | None -> (var_id, rtype)
    | Some renamed_var_id -> (renamed_var_id, rtype)
  in
  let rename_context free_variable_correspondence context =
    List.map context ~f:(fun (var_bind, location) ->
        (rename_var_bind free_variable_correspondence var_bind, location))
  in
  let rename_sample free_variable_correspondence sample =
    let context, output, cost = sample in
    (rename_context free_variable_correspondence context, output, cost)
  in
  let rename_list_samples free_variable_correspondence list_samples =
    List.map list_samples ~f:(rename_sample free_variable_correspondence)
  in
  let rec accumulate_list_samples acc (expression, list_samples) =
    match acc with
    | [] -> [ (expression, list_samples) ]
    | ((representative_expression, head_list_samples) as head) :: acc_tail -> (
        match
          Free_variable_correspondence
          .free_variable_correspondence_function_applications expression
            representative_expression
        with
        | None ->
            head :: accumulate_list_samples acc_tail (expression, list_samples)
        | Some free_variable_correspondence ->
            let augmented_list_samples =
              rename_list_samples free_variable_correspondence list_samples
              @ head_list_samples
            in
            (representative_expression, augmented_list_samples) :: acc_tail )
  in
  List.fold dataset ~init:[] ~f:accumulate_list_samples

(* Stage 2: Suppose we are given a dataset of type 'a
dataset_categorized_by_expressions. Here, each sample records an evaluation
context and an output. For each base resource polynomial up to a certain degree,
we compute its potential with respect to a given sample.

See Section 5 of the paper "Towards Automatic Resource Bound Analysis for OCaml"
for how to compute base resource polynomials' potential. *)

(* Individual sample with indices and potential. A sample indicates which cindex
(in the typing context) and index (in the expression's type) is associated with
how much potential. Here, the amount of potential refers to the amount given by
the cindex/index's base resource polynomial. A sample has three components. The
first component is for the evaluation context. The second component is for the
return type. The third component stores computational costs. *)
type sample_with_potential =
  (Indices.cindex * int) list * (Indices.index * int) list * float

(* Dataset of runtime costs and potential for each cindices (in the typing
context) and indices (in the expression's type) *)
type 'a dataset_with_potential =
  (('a, unit) Expressions.expression * sample_with_potential list) list

(* Split an input index list according to an input weight list. A weight list is
a list of natural numbers, and their sum must be equal to the length of the
index list. *)
let rec distribute_indices index_list weight_list =
  match weight_list with
  | [] ->
      if List.is_empty index_list then []
      else failwith "Mismatch between index_list and weight_list"
  | w :: ws ->
      let first_index_list, second_index_list = List.split_n index_list w in
      first_index_list :: distribute_indices second_index_list ws

(* Given an index, evaluate its potential with respect to a given value. The
formula for calculating the amount of a base resource polynomial with respect to
a semantic value is detailed in Section 5 of the paper "Towards Automatic
Resource Bound Analysis for OCaml". *)
let rec evaluate_base_polynomial (index : Indices.index) (loc : location)
    (heap : 'a heap) : int =
  match index with
  | Iunit -> 1
  | Inat n -> failwith "Hybrid AARA doesn't support Inat"
  | Ituple index_list -> evaluate_base_polynomial_tuple index_list loc heap
  | Iind index_list -> evaluate_base_polynomial_ind index_list loc heap
  | Iprob _ -> failwith "Hybrid AARA doesn't support Iprob"

and evaluate_base_polynomial_tuple (index_list : Indices.index list)
    (loc : location) (heap : 'a heap) : int =
  match Map.find heap loc with
  | Some (Vtuple loc_list) -> (
      match List.zip index_list loc_list with
      | Some index_location_list ->
          List.fold index_location_list ~init:1 ~f:(fun acc (index, loc) ->
              acc * evaluate_base_polynomial index loc heap)
      | None ->
          failwith
            "Index list and location list for a semantic value of a tuple have \
             different lengths" )
  | _ -> failwith "Location is not a tuple"

and evaluate_base_polynomial_ind
    (index_list : (Indices.index * Rtypes.constr_id) list) (loc : location)
    (heap : 'a heap) : int =
  match index_list with
  | [] ->
      (* The potential of the empty list of indices is one, which is the
      multiplicative unit. *)
      1
  | (index_first_element, index_first_constr_id) :: index_tail -> (
      match Map.find heap loc with
      | Some (Vconst (constr_id_node, loc_node)) -> (
          (* printf "index_first_constr_id: %s; constr_id_node: %s\n" index_first_constr_id constr_id_node; *)
          match Map.find heap loc_node with
          | Some (Vtuple []) -> failwith "Vtuple [] shouldn't arise"
          | Some (Vtuple (loc_node_data :: loc_list)) ->
              let subtrees_potential_without_current_node =
                evaluate_base_polynomial_ind_subtrees index_list loc_list heap
              in
              if index_first_constr_id <> constr_id_node then
                subtrees_potential_without_current_node
              else
                let node_data_potential =
                  evaluate_base_polynomial index_first_element loc_node_data
                    heap
                in
                let subtrees_potential_with_current_node =
                  evaluate_base_polynomial_ind_subtrees index_tail loc_list heap
                in
                (node_data_potential * subtrees_potential_with_current_node)
                + subtrees_potential_without_current_node
          | Some _ ->
              if index_first_constr_id = constr_id_node then
                evaluate_base_polynomial index_first_element loc_node heap
              else 0
          | None -> failwith "loc_node is not found in the heap" )
      | _ ->
          failwith "Value is not of the form Vconst (constr_id_node, loc_node)"
      )

and evaluate_base_polynomial_ind_subtrees
    (index_list : (Indices.index * Rtypes.constr_id) list) (subtrees : int list)
    (heap : 'a heap) : int =
  let all_possible_distribution_weights =
    Indices.distr_weight (List.length index_list) (List.length subtrees)
  in
  let all_possible_distribution_indices =
    List.map all_possible_distribution_weights ~f:(fun ws ->
        distribute_indices index_list ws)
  in
  let evaluate_base_polynomial_ind_subtree index_distribution =
    let zipped_list_indices_subtrees = List.zip index_distribution subtrees in
    match zipped_list_indices_subtrees with
    | Some ls ->
        List.map ls ~f:(fun (index_sublist, subtree_loc) ->
            evaluate_base_polynomial_ind index_sublist subtree_loc heap)
    | None -> failwith "Mismatch in lengths of index_distribution and subtrees"
  in
  let distribution_subtree_potential =
    List.map all_possible_distribution_indices
      ~f:evaluate_base_polynomial_ind_subtree
  in
  (* distribution_potential is given by calculating the product of each inner
	integer list's elements. *)
  let distribution_potential =
    List.map distribution_subtree_potential ~f:(fun ls ->
        List.fold ls ~init:1 ~f:(fun acc x -> acc * x))
  in
  (* The final output is given by calculating the sum of the list elements. *)
  List.fold distribution_potential ~init:0 ~f:(fun acc x -> acc + x)

let evaluate_potential_list_samples
    (expression : ('a, 'b) Expressions.expression)
    (list_samples : sample_without_expression list) (heap : 'a heap)
    (degree : int) : sample_with_potential list =
  let typing_context_list =
    match List.hd list_samples with
    | None -> failwith "The list of samples is empty"
    | Some (evaluation_context, _, _) ->
        List.map evaluation_context ~f:(fun (var_bind, _) -> var_bind)
  in
  let typing_context = String.Map.of_alist_exn typing_context_list in
  let all_indices_typing_context =
    Indices.cindices_max_deg ~empty:[]
      ~add_ind:(fun cindex acc -> cindex :: acc)
      [] typing_context degree
  in
  let return_type = expression.Expressions.exp_type in
  let all_indices_return_type =
    Indices.indices_max_deg ~empty:[]
      ~add_ind:(fun index acc -> index :: acc)
      return_type degree
  in
  let rec find_loc_var_id var_id evaluation_context =
    match evaluation_context with
    | [] -> failwith "Evaluation context is empty"
    | ((head_var_id, _), loc) :: evaluation_context_tail ->
        if String.( = ) var_id head_var_id then loc
        else find_loc_var_id var_id evaluation_context_tail
  in
  let evaluate_potential_index var_id index evaluation_context =
    let loc = find_loc_var_id var_id evaluation_context in
    evaluate_base_polynomial index loc heap
  in
  let evaluate_potential_cindex (_, cindex) evaluation_context =
    let cindex_list = Map.to_alist cindex in
    let list_potential =
      List.map cindex_list ~f:(fun (var_id, index) ->
          evaluate_potential_index var_id index evaluation_context)
    in
    List.fold list_potential ~init:1 ~f:(fun acc x -> acc * x)
  in
  let evaluate_potential_all_cindices (evaluation_context, _, _) =
    List.map all_indices_typing_context ~f:(fun cindex ->
        (cindex, evaluate_potential_cindex cindex evaluation_context))
  in
  let evaluate_potential_index_return_type (_, output, _) =
    List.map all_indices_return_type ~f:(fun index ->
        (index, evaluate_base_polynomial index output heap))
  in
  let return_cost (_, _, cost) = cost in
  List.map list_samples ~f:(fun sample ->
      ( evaluate_potential_all_cindices sample,
        evaluate_potential_index_return_type sample,
        return_cost sample ))

let evaluate_potential_dataset (dataset : 'a dataset_categorized_by_expressions)
    (heap : 'a heap) (degree : int) : 'a dataset_with_potential =
  List.map dataset ~f:(fun (representative_e, list_samples) ->
      ( representative_e,
        evaluate_potential_list_samples representative_e list_samples heap
          degree ))

(* Stage 3: only retain worst-case samples for each combination of cindices'
potential and indices' potential. *)

let list_cindex_potential_equality list1 list2 =
  (* For debugging *)
  (* let () =
    print_endline "Equality checking for cindices: list of cindices";
    List.iter list1 ~f:(fun (cindex, potential) ->
        Indices.print_cindex cindex;
        printf " potential = %d\n" potential)
  in *)
  let find_coefficient_of_corresponding_cindex cindex =
    List.Assoc.find_exn list2
      ~equal:(fun x y -> Indices.compare_cindex x y = 0)
      cindex
  in
  let equal_potential (cindex, potential1) =
    let potential2 = find_coefficient_of_corresponding_cindex cindex in
    Int.equal potential1 potential2
  in
  List.for_all list1 ~f:equal_potential

let list_index_potential_equality list1 list2 =
  (* For debugging *)
  (* let () =
    print_endline "Equality checking for indices: list of indices";
    List.iter list1 ~f:(fun (index, potential) ->
        Indices.print_index index;
        printf " potential = %d\n" potential)
  in *)
  let find_coefficient_of_index_in_list2 index =
    List.Assoc.find_exn list2
      ~equal:(fun x y -> Indices.compare_index x y = 0)
      index
  in
  let equal_potential_with_list2 (index, potential1) =
    let potential2 = find_coefficient_of_index_in_list2 index in
    potential1 = potential2
  in
  List.for_all list1 ~f:equal_potential_with_list2

let list_cindex_and_index_potential_equality list1 list2 =
  let list_cindex_potential1, list_index_potential1 = list1 in
  let list_cindex_potential2, list_index_potential2 = list2 in
  list_cindex_potential_equality list_cindex_potential1 list_cindex_potential2
  && list_index_potential_equality list_index_potential1 list_index_potential2

let rec insert_sample_into_list_if_worst_case list_samples target_sample =
  match list_samples with
  | [] -> [ target_sample ]
  | head_sample :: tail_samples ->
      let list_cindex_potential1, list_index_potential1, target_cost =
        target_sample
      in
      let list_cindex_potential2, list_index_potential2, head_cost =
        head_sample
      in
      let is_equal =
        list_cindex_and_index_potential_equality
          (list_cindex_potential1, list_index_potential1)
          (list_cindex_potential2, list_index_potential2)
      in
      (* We assume that list_samples only contains at most one sample for each
      combination of cindices' potential and indices' potential. *)
      if is_equal then
        if target_cost <= head_cost then head_sample :: tail_samples
        else target_sample :: tail_samples
      else
        head_sample
        :: insert_sample_into_list_if_worst_case tail_samples target_sample

let retain_worst_case_samples_dataset_of_e
    (dataset_of_e : sample_with_potential list) : sample_with_potential list =
  let dataset_of_e_worst_case_samples =
    List.fold dataset_of_e ~init:[] ~f:insert_sample_into_list_if_worst_case
  in
  (* For debugging *)
  let () =
    printf "Original dataset: %i samples\n" (List.length dataset_of_e);
    printf "New dataset with worst-case samples only: %i samples\n"
      (List.length dataset_of_e_worst_case_samples)
  in
  dataset_of_e_worst_case_samples

let retain_worst_case_samples dataset =
  List.map dataset ~f:(fun (e, dataset_of_e) ->
      (e, retain_worst_case_samples_dataset_of_e dataset_of_e))

(* Categorize runtime cost data by input and output sizes (more precisely, the
coefficients of cindices and indices). *)

let rec insert_sample_into_list_categorized_by_sizes list_samples sample =
  let list_cindex_potential1, list_index_potential1, cost1 = sample in
  match list_samples with
  | [] -> [ (list_cindex_potential1, list_index_potential1, [ cost1 ]) ]
  | head_sample :: tail_samples ->
      let list_cindex_potential2, list_index_potential2, list_cost2 =
        head_sample
      in
      let is_equal =
        list_cindex_and_index_potential_equality
          (list_cindex_potential1, list_index_potential1)
          (list_cindex_potential2, list_index_potential2)
      in
      if is_equal then
        (list_cindex_potential1, list_index_potential1, cost1 :: list_cost2)
        :: tail_samples
      else
        head_sample
        :: insert_sample_into_list_categorized_by_sizes tail_samples sample

let categorize_cost_data_by_sizes_dataset_of_e dataset_of_e =
  List.fold dataset_of_e ~init:[]
    ~f:insert_sample_into_list_categorized_by_sizes

(* Stage 4: For each cindex (in a typing context) and index (in the return
type) in a dataset, we compute their list of potential. *)

(* We categorize an entire dataset according to cindices (in the typing context)
and indices (in the expression's type). Hence, each cindex/index comes with a
list of potential. The first component is for all cindices (up to a certain
degree) in the typing context, and the second component is for all indices (up
to a certain degree) in the expression's type. *)
type list_potential_categorized_by_indices =
  (Indices.cindex * int list) list
  * (Indices.index * int list) list
  * float list

(* Dataset of runtime costs and potential categorized by cindices (in the typing
context) and indices (in the expression's type) *)
type 'a dataset_with_potential_categorized_by_indices =
  (('a, unit) Expressions.expression * list_potential_categorized_by_indices)
  list

let insert_sample_to_categorize_by_indices acc
    (list_cindex_potential, list_index_potential, cost) =
  let rec insert_cindex_potential acc (cindex, potential) =
    match acc with
    | [] -> [ (cindex, [ potential ]) ]
    | (head_cindex, list_potential) :: acc_tail ->
        if Indices.compare_cindex cindex head_cindex = 0 then
          (head_cindex, potential :: list_potential) :: acc_tail
        else
          (head_cindex, list_potential)
          :: insert_cindex_potential acc_tail (cindex, potential)
  in
  let rec insert_index_potential acc (index, potential) =
    match acc with
    | [] -> [ (index, [ potential ]) ]
    | (head_index, list_potential) :: acc_tail ->
        if Indices.compare_index index head_index = 0 then
          (head_index, potential :: list_potential) :: acc_tail
        else
          (head_index, list_potential)
          :: insert_index_potential acc_tail (index, potential)
  in
  let acc_cindex_potential, acc_index_potential, acc_costs = acc in
  let updated_acc_cindex =
    List.fold list_cindex_potential ~init:acc_cindex_potential
      ~f:insert_cindex_potential
  in
  let updated_acc_index =
    List.fold list_index_potential ~init:acc_index_potential
      ~f:insert_index_potential
  in
  let updated_acc_costs = cost :: acc_costs in
  (updated_acc_cindex, updated_acc_index, updated_acc_costs)

let categorize_list_samples_by_indices
    (list_samples : sample_with_potential list) :
    list_potential_categorized_by_indices =
  List.fold list_samples ~init:([], [], [])
    ~f:insert_sample_to_categorize_by_indices

let categorize_dataset_with_potential_by_indices
    (dataset : 'a dataset_with_potential) :
    'a dataset_with_potential_categorized_by_indices =
  List.map dataset ~f:(fun (expressions, list_samples) ->
      (expressions, categorize_list_samples_by_indices list_samples))

let rec rename_dataset (free_variable_correspondence : (string * string) list)
    (dataset : list_potential_categorized_by_indices) :
    list_potential_categorized_by_indices =
  let list_cindex_potential, list_index_potential, list_costs = dataset in
  let renamed_list_cindex_potential =
    List.map list_cindex_potential ~f:(fun (cindex, list_potential) ->
        ( Indices.rename_cindex free_variable_correspondence cindex,
          list_potential ))
  in
  (renamed_list_cindex_potential, list_index_potential, list_costs)

let rec find_dataset_associated_with_expression
    (dataset : 'a dataset_with_potential_categorized_by_indices)
    (expression : (Rtypes.raml_type list, unit) Expressions.expression) :
    list_potential_categorized_by_indices option =
  match dataset with
  | [] -> None
  | (head_e, head_dataset) :: dataset_tail -> (
      match
        Free_variable_correspondence
        .free_variable_correspondence_function_applications_typecheck_tstack
          head_e expression
      with
      | None -> find_dataset_associated_with_expression dataset_tail expression
      | Some free_variable_correspondence ->
          Some (rename_dataset free_variable_correspondence head_dataset) )

(* Import predicted worst-case costs from an external source *)

let import_predicted_worst_case_samples filename =
  let open Yojson.Basic.Util in
  let extract_list_index_potential list_index_potential_json =
    list_index_potential_json |> to_assoc
    |> List.map ~f:(fun (index, potential_json) ->
           (index, to_int potential_json))
  in
  let extract_runtime_sample runtime_sample_json =
    let list_cindex_string_potential =
      runtime_sample_json |> member "cindex" |> extract_list_index_potential
    in
    let list_index_string_potential =
      runtime_sample_json |> member "index" |> extract_list_index_potential
    in
    let cost = runtime_sample_json |> member "cost" |> to_float in
    (list_cindex_string_potential, list_index_string_potential, cost)
  in
  let extract_list_runtime_samples list_runtime_samples_json =
    list_runtime_samples_json |> to_list |> List.map ~f:extract_runtime_sample
  in
  let extract_list_dataset_samples list_dataset_samples_json =
    list_dataset_samples_json |> to_list
    |> List.map ~f:extract_list_runtime_samples
  in
  let predicted_worst_case_costs_json = Yojson.Basic.from_file filename in
  predicted_worst_case_costs_json |> extract_list_dataset_samples

let match_index_string_with_index predicted_worst_case_costs
    representative_runtime_sample =
  let string_of_cindex cindex =
    Indices.print_cindex ~output:Format.str_formatter cindex;
    Format.flush_str_formatter ()
  in
  let string_of_index index =
    Indices.print_index ~output:Format.str_formatter index;
    Format.flush_str_formatter ()
  in
  let list_cindex_potential, list_index_potential, _ =
    representative_runtime_sample
  in
  let list_string_and_cindex =
    (* For debugging *)
    List.iter list_cindex_potential ~f:(fun (cindex, _) ->
        print_endline (string_of_cindex cindex));
    List.map list_cindex_potential ~f:(fun (cindex, _) ->
        (string_of_cindex cindex, cindex))
  in
  let list_string_and_index =
    List.map list_index_potential ~f:(fun (index, _) ->
        (string_of_index index, index))
  in
  let find_potential_cindex cindex_string =
    List.Assoc.find_exn list_string_and_cindex ~equal:String.equal cindex_string
  in
  let find_potential_index index_string =
    List.Assoc.find_exn list_string_and_index ~equal:String.equal index_string
  in
  let replace_strings_with_indices_runtime_sample runtime_sample =
    let list_cindex_string_potential, list_index_string_potential, cost =
      runtime_sample
    in
    let list_cindex_potential =
      List.map list_cindex_string_potential
        ~f:(fun (cindex_string, potential) ->
          (find_potential_cindex cindex_string, potential))
    in
    let list_index_potential =
      List.map list_index_string_potential ~f:(fun (index_string, potential) ->
          (find_potential_index index_string, potential))
    in
    (list_cindex_potential, list_index_potential, cost)
  in
  let replace_strings_with_indices_runtime_dataset runtime_dataset =
    List.map runtime_dataset ~f:replace_strings_with_indices_runtime_sample
  in
  List.map predicted_worst_case_costs
    ~f:replace_strings_with_indices_runtime_dataset

(* Get the first runtime sample in a dataset, assuming that the dataset contains
exactly one expression *)
let get_representative_runtime_sample dataset =
  match dataset with
  | [] -> failwith "The dataset is empty"
  | [ dataset_of_e ] -> (
      let e, list_samples = dataset_of_e in
      match list_samples with
      | [] -> failwith "List of samples is empty"
      | hd_runtime_sample :: _ -> hd_runtime_sample )
  | _ -> failwith "The dataset contains more than one expressions"

let attach_expression_to_predicted_worst_case_costs predicted_worst_case_costs
    dataset =
  match dataset with
  | [] -> failwith "The dataset is empty"
  | [ dataset_of_e ] ->
      let e, _ = dataset_of_e in
      List.map predicted_worst_case_costs ~f:(fun list_samples ->
          [ (e, list_samples) ])
  | _ -> failwith "The dataset contains more than one expressions"

(* Only retain samples with non-zero costs. This is used in Weibull survival
analysis where we need to take logarithm of costs. If the cost is zero, its
logarithm is undefined, causing an error. *)
let extract_samples_with_nonzero_costs_dataset dataset =
  let contains_nonzero_costs runtime_sample =
    let _, _, cost = runtime_sample in
    cost > 0.
  in
  let extract_samples_with_nonzero_costs_dataset_of_e dataset_of_e =
    List.filter dataset_of_e ~f:contains_nonzero_costs
  in
  List.map dataset ~f:(fun (e, dataset_of_e) ->
      (e, extract_samples_with_nonzero_costs_dataset_of_e dataset_of_e))
