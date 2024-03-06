open Core

(* The type row is almost identical to the type Clp.row, except that
row_elements is encoded as a list here, while it uses an array in Clp.row. *)
type row = {
  row_lower : float;
  row_upper : float;
  row_elements : (int * float) list;
}

(* An equality constraint has the form x = a_1 * x_1 + ... + a_n * x_n + c,
where a_1, ..., a_n, c are constants, and x, x_1, ..., x_n are LP variables. The
var field stores variable x, the coefficient_list field stores the list [(x_1,
a_1); ...; (x_n, a_n)], and the constant field stores c. *)
type equality_constraint = {
  var : int;
  coefficient_list : (int * float) list;
  constant : float;
}

(* Equality of floating-point numbers*)

let approximately_equal ?(epsilon = Float.int_pow 10. (-6)) x y =
  if -.epsilon <= x -. y && x -. y <= epsilon then true else false

let approximately_bounded ?(epsilon = Float.int_pow 10. (-6)) x lower upper =
  if lower -. epsilon <= x && x <= upper +. epsilon then true else false

(* Printing functions *)

let print_row row =
  let { row_lower; row_upper; row_elements } = row in
  print_string "Row: ";
  List.iter row_elements ~f:(fun (v, c) -> printf "(v, c) = (%i, %.3f) " v c);
  printf "lower: %.3f upper: %.3f" row_lower row_upper;
  print_newline ()

let print_equality_constraint equality_constraint =
  let { var; coefficient_list; constant } = equality_constraint in
  printf "Equality: var = %i " var;
  List.iter coefficient_list ~f:(fun (v, c) ->
      printf "(v, c) = (%i, %.3f) " v c);
  printf "constant: %.3f" constant;
  print_newline ()

let print_list_rows (list_rows : row list) =
  let print_row index row =
    let { row_lower; row_upper; row_elements } = row in
    printf "Row %i: " index;
    List.iter row_elements ~f:(fun (v, c) -> printf "(v, c) = (%i, %.3f) " v c);
    printf "lower: %.3f upper: %.3f" row_lower row_upper;
    print_newline ()
  in
  List.iteri list_rows ~f:print_row

let print_list_equality_constraints
    (list_equality_constraints : equality_constraint list) =
  let print_equality_constraint index equality_constraint =
    let { var; coefficient_list; constant } = equality_constraint in
    printf "Equality %i: var = %i " index var;
    List.iter coefficient_list ~f:(fun (v, c) ->
        printf "(v, c) = (%i, %.3f) " v c);
    printf "constant: %.3f" constant;
    print_newline ()
  in
  List.iteri list_equality_constraints ~f:print_equality_constraint

(* We perform the following preprocessing: (i) convert a row from the type
Clp.row to row, (ii) remove duplicate LP variables, (iii) remove zero
coefficients, (iv) discard an entire row if it is empty, and (v) perturb an
equality constraint if a user wishes. *)

let convert_clp_rows_to_rows_in_lists (list_rows : Clp.row list) : row list =
  let convert_clp_row row =
    let { Clp.row_lower; Clp.row_upper; Clp.row_elements } = row in
    { row_lower; row_upper; row_elements = Array.to_list row_elements }
  in
  List.map list_rows convert_clp_row

(* This function returns an option value (i.e. either None or Some row). If the
input row is reduced to the empty row as a result of preprocessing, None is
returned. *)
let preprocess_row (row : row) : row option =
  let { row_lower; row_upper; row_elements } = row in
  (* Check that row_lower and row_upper are sensible *)
  assert (
    row_lower <> Float.infinity
    && row_lower <= row_upper
    && row_upper <> Float.neg Float.infinity );
  (* Remove duplicate LP variables from a row. So if an LP variable appears
  more than once in a row, its coefficients are summed. *)
  let rec accumulate_LP_vars acc (current_var, current_coefficient) =
    match acc with
    | [] -> [ (current_var, current_coefficient) ]
    | (head_var, head_coefficient) :: acc_tail ->
        if head_var = current_var then
          (head_var, head_coefficient +. current_coefficient) :: acc_tail
        else
          (head_var, head_coefficient)
          :: accumulate_LP_vars acc_tail (current_var, current_coefficient)
  in
  let row_elements_no_duplicates =
    List.fold row_elements ~init:[] ~f:accumulate_LP_vars
  in
  (* Remove zero coefficients *)
  let row_elements_no_zero_coefficients =
    List.filter row_elements_no_duplicates ~f:(fun (_, c) ->
        not (approximately_equal c 0.))
  in
  match row_elements_no_zero_coefficients with
  | [] ->
      (* If the row becomes empty as a result of preprocessing, discard it. In
      such a case, we check that row_lower and row_upper are consistent with the
      empty row, which represents zero. *)
      assert (row_lower <= 0. && 0. <= row_upper);
      None
  | _ ->
      Some
        {
          row_lower;
          row_upper;
          row_elements = row_elements_no_zero_coefficients;
        }

let preprocess_rows (list_rows : row list) : row list =
  List.filter_map list_rows ~f:preprocess_row

(* Create box constraints (i.e. constant lower and upper bounds of individual LP
variables) *)

let create_box_constraints ?(lower_bound = 0.) ?(upper_bound = 10.)
    ?(upper_unbounded_vars = []) (num_vars : int) : row list =
  let create_box_constraint var =
    if
      List.exists upper_unbounded_vars ~f:(fun unbounded_var ->
          var = unbounded_var)
    then
      {
        row_lower = lower_bound;
        row_upper = Float.max_value;
        row_elements = [ (var, 1.) ];
      }
    else
      {
        row_lower = lower_bound;
        row_upper = upper_bound;
        row_elements = [ (var, 1.) ];
      }
  in
  List.map (List.range 0 num_vars) ~f:create_box_constraint

(* Remove equality constraints from a linear program and store them separately.
In the presence of equality constraints, the dimension of a linear program's
feasible region is strictly lower than the number of LP variables. Consequently,
it causes trouble when we run, for example, hit-and-run samplers from volesti.
One way to fix this issue is to remove equality constraints from a linear
program and store them separately. Another approach is to perturb equality
constraints. *)

let convert_row_to_equality_constraint (equality_row : row) :
    equality_constraint =
  let { row_lower; row_upper; row_elements } = equality_row in
  assert (row_lower = row_upper);
  assert (List.length row_elements > 0);
  let var, coefficient = List.hd_exn row_elements in
  assert (not (approximately_equal coefficient 0.));
  let row_elements_tail = List.tl_exn row_elements in
  let coefficient_list =
    List.map row_elements_tail ~f:(fun (v, x) ->
        (v, Float.neg (x /. coefficient)))
  in
  { var; coefficient_list; constant = row_lower /. coefficient }

(* Given three inputs (target, multiplier, and coefficient_list), this function
returns the result of target + multiplier * coefficient_list. This operation is
useful when we want to substitute a row for some variable in another row. *)
let multiply_and_add_coefficient_lists (target : ('a * float) list)
    (multiplier : float) (coefficient_list : ('a * float) list) :
    ('a * float) list =
  let coefficient_list_multiplied =
    List.map coefficient_list ~f:(fun (v, c) -> (v, c *. multiplier))
  in
  let rec insert_into_acc acc (var, coefficient) =
    match acc with
    | [] -> [ (var, coefficient) ]
    | (head_var, head_coefficient) :: acc_tail ->
        if head_var = var then
          (head_var, head_coefficient +. coefficient) :: acc_tail
        else
          (head_var, head_coefficient)
          :: insert_into_acc acc_tail (var, coefficient)
  in
  let result_of_substitution =
    List.fold coefficient_list_multiplied ~init:target ~f:insert_into_acc
  in
  (* Remove zero coefficients *)
  List.filter result_of_substitution ~f:(fun (_, c) ->
      not (approximately_equal c 0.))

(* Substitute an equality constraint of the form x = a_1 * x_1 + ... + a_n * x_n
for x in a row. If the row becomes empty as a result of substitution, we return
None. *)
let substitute_equality_into_row (row : row)
    (equality_constraint : equality_constraint) : row option =
  let { row_lower; row_upper; row_elements } = row in
  let { var; coefficient_list; constant } = equality_constraint in
  match List.Assoc.find row_elements ~equal:Int.equal var with
  | None -> Some row
  | Some multiplier -> (
      let constant_multiplied = constant *. multiplier in
      let row_elements_var_removed =
        List.Assoc.remove row_elements ~equal:Int.equal var
      in
      let substituted_row_elements =
        multiply_and_add_coefficient_lists row_elements_var_removed multiplier
          coefficient_list
      in
      match substituted_row_elements with
      | [] ->
          (* If the row becomes empty as a result of substitution, return None.
          For a sanity check, we check that row_lower and row_upper after
          substitution are consistent with the empty row (which represents
          zero). *)
          if
            row_lower -. constant_multiplied > 0.
            || 0. > row_upper -. constant_multiplied
          then (
            print_endline "Assertion in a substitution failed:";
            print_row row;
            print_equality_constraint equality_constraint )
          else ();
          assert (
            row_lower -. constant_multiplied <= 0.
            && 0. <= row_upper -. constant_multiplied );
          None
      | _ ->
          Some
            {
              row_lower = row_lower -. constant_multiplied;
              row_upper = row_upper -. constant_multiplied;
              row_elements = substituted_row_elements;
            } )

(* Substitute an equality constraint of the form x = a_1 * x_1 + ... + a_n * x_n
for x in another equality constraint *)
let substitute_equality_into_equality
    (substitution_target : equality_constraint) (equality : equality_constraint)
    : equality_constraint =
  let {
    var = var_target;
    coefficient_list = coefficient_list_target;
    constant = constant_target;
  } =
    substitution_target
  in
  let { var; coefficient_list; constant } = equality in
  assert (var_target <> var);
  match List.Assoc.find coefficient_list_target ~equal:Int.equal var with
  | None -> substitution_target
  | Some multiplier ->
      let constant_multiplied = constant *. multiplier in
      let coefficient_list_target_var_removed =
        List.Assoc.remove coefficient_list_target ~equal:Int.equal var
      in
      let substituted_coefficient_list =
        multiply_and_add_coefficient_lists coefficient_list_target_var_removed
          multiplier coefficient_list
      in
      {
        var = var_target;
        coefficient_list = substituted_coefficient_list;
        constant = constant_target +. constant_multiplied;
      }

(* Categorize a list of rows into two lists: (i) a list of inequality rows (i.e.
row_lower <> row_upper) and (ii) a list of equality rows *)
let categorize_rows_into_inequality_and_equality (list_rows : row list) :
    row list * row list =
  List.partition_tf list_rows ~f:(fun row ->
      let { row_lower; row_upper } = row in
      not (approximately_equal row_lower row_upper))

let remove_equality_constraints (list_rows : row list) :
    row list * equality_constraint list =
  let inequality_rows, equality_rows =
    categorize_rows_into_inequality_and_equality list_rows
  in
  let rec repeatedly_substitute_equality_constraints inequality_rows
      equality_constraints equality_rows =
    match equality_rows with
    | [] -> (inequality_rows, equality_constraints)
    | head_equality_row :: equality_rows_tail ->
        let equality_constraint =
          convert_row_to_equality_constraint head_equality_row
        in
        let inequality_rows_substituted =
          List.filter_map inequality_rows ~f:(fun r ->
              substitute_equality_into_row r equality_constraint)
        in
        let equality_constraints_substituted =
          List.map equality_constraints ~f:(fun c ->
              substitute_equality_into_equality c equality_constraint)
        in
        let equality_rows_substituted =
          List.filter_map equality_rows_tail ~f:(fun r ->
              substitute_equality_into_row r equality_constraint)
        in
        repeatedly_substitute_equality_constraints inequality_rows_substituted
          (equality_constraint :: equality_constraints_substituted)
          equality_rows_substituted
  in
  repeatedly_substitute_equality_constraints inequality_rows [] equality_rows

(* Remove redundant box constraints from a list of rows. This removal is helpful
because the following situation commonly arises. Suppose we have three LP
variables: x, y, and z. They each have a box constraint of the form 0 <= x, y,
z <= 10. Additionally, assume we have equality constraints x = y and y = z. If
we eliminate the equality constraints, we substitute x for y and z.
Consequently, we will end up with three copies of the same box constraint 0 <=
x <= 10. Therefore, for storage efficiency, we should eliminate redundant box
constraints after equality constraints are removed. 

Suppose we have two box constraints that involve the same LP variable
First box: { row_lower = first_row_lower; 
             row_upper = first_row_upper; 
             row_elements = [(first_var, 1.0)] }
Second box: { row_lower = second_row_lower; 
             row_upper = second_row_upper; 
             row_elements = [(second_var, 1.0)] },
where first_var = second_var. Then their intersection is defined as
Intersection: { row_lower = Float.max first_row_lower second_row_lower; 
                row_upper = Float.min first_row_upper second_row_upper; 
                row_elements = [(first_var, 1.0)] }. 
The more general case where first_var's and second_var's coefficients are not
1.0 can be handled similarly. *)

let is_box_constraint (box : row) : bool =
  let { row_elements } = box in
  match row_elements with [ (_, _) ] -> true | _ -> false

type intersection_rows =
  | Row_intersection of row
  | Equality_intersection of equality_constraint
  | No_intersection

(* Compute the intersection of two box constraints. It returns either
Row_intersection (if the intersection is an inequality box constraint),
Equality_intersection (if the intersection is an equality (box) constraint), or
None (if the two input box constraints involve different LP variables). *)
let intersection_of_two_box_constraints (first_box : row) (second_box : row) :
    intersection_rows =
  let normalize_box_constraint lower_bound upper_bound coefficient =
    assert (not (approximately_equal coefficient 0.));
    if coefficient > 0. then
      (lower_bound /. coefficient, upper_bound /. coefficient)
    else (upper_bound /. coefficient, lower_bound /. coefficient)
  in
  let {
    row_lower = first_row_lower;
    row_upper = first_row_upper;
    row_elements = first_row_elements;
  } =
    first_box
  in
  let {
    row_lower = second_row_lower;
    row_upper = second_row_upper;
    row_elements = second_row_elements;
  } =
    second_box
  in
  match (first_row_elements, second_row_elements) with
  | [ (first_var, first_coefficient) ], [ (second_var, second_coefficient) ]
    when first_var = second_var ->
      let first_normalized_lower, first_normalized_upper =
        normalize_box_constraint first_row_lower first_row_upper
          first_coefficient
      in
      let second_normalized_lower, second_normalized_upper =
        normalize_box_constraint second_row_lower second_row_upper
          second_coefficient
      in
      let combined_normalized_lower =
        Float.max first_normalized_lower second_normalized_lower
      in
      let combined_normalized_upper =
        Float.min first_normalized_upper second_normalized_upper
      in
      if combined_normalized_lower > combined_normalized_upper then
        failwith
          "We detect infeasibility while computing the intersection of box \
           constraints"
      else if combined_normalized_lower = combined_normalized_upper then
        Equality_intersection
          {
            var = first_var;
            coefficient_list = [];
            constant = combined_normalized_lower;
          }
      else
        Row_intersection
          {
            row_lower = combined_normalized_lower;
            row_upper = combined_normalized_upper;
            row_elements = [ (first_var, 1.) ];
          }
  | _ -> No_intersection

(* Compute the intersection of a box (target_box) and a list of boxes
(list_boxes). We assume that the box constraints in list_box all contain
different LP variables. Hence, at most one element from list_boxes has an
intersection with target_box. Suppose we have found a box constraint (say, box2)
that intersects with target_box. If the intersection is an inequality box
constraint, we replace box2 in list_boxes with the intersection and return the
updated list. Otherwise, if the intersection is an equality constraint, we
simply remove box2 from the list, and return the updated list together with the
equality constraint. *)
let rec intersection_of_box_and_list_boxes (target_box : row)
    (list_boxes : row list) : row list * equality_constraint option =
  match list_boxes with
  | [] -> ([ target_box ], None)
  | head_box :: box_tail -> (
      match intersection_of_two_box_constraints target_box head_box with
      | Row_intersection intersection_box -> (intersection_box :: box_tail, None)
      (* We assume that each element of list_boxes is a distinct box constraint.
      So list_boxes can contain at most one identical box to target_box. *)
      | Equality_intersection equality -> (box_tail, Some equality)
      | No_intersection ->
          let recursive_list_box, possible_equality =
            intersection_of_box_and_list_boxes target_box box_tail
          in
          (head_box :: recursive_list_box, possible_equality) )

let substitute_into_inequalities_and_equality_constraints
    (list_rows, equality_constraints) equality =
  let list_rows_substituted =
    List.filter_map list_rows ~f:(fun r ->
        substitute_equality_into_row r equality)
  in
  let equality_constraints_substituted =
    List.map equality_constraints ~f:(fun c ->
        substitute_equality_into_equality c equality)
  in
  (list_rows_substituted, equality_constraints_substituted)

(* Check whether a given box constraint's LP variable also appears in the list
of equality constraints. *)
let is_box_in_list_equality_constraints box list_equality_constraints =
  let { row_elements } = box in
  let var_of_box =
    match row_elements with
    | [ (var, _) ] -> var
    | _ -> failwith "The given row is not a box constraint"
  in
  let list_vars_in_equalities =
    List.map list_equality_constraints ~f:(fun c -> c.var)
  in
  List.exists list_vars_in_equalities ~f:(fun v -> v = var_of_box)

(* Insert a row into an accumulator. This function is used inside the recursive
function remove_duplicated_box_constraints. *)
let insert_row_into_acc_of_constraints
    (box_constraints_acc, non_box_constraints_acc, equality_acc) row =
  if is_box_constraint row then
    (* Check whether row's LP appears in equality_acc. If it does, some other
    box constraint of the same LP variable must have been turned into an
    equality constraint. So we simply discard row, keeping the accumulator
    unmodified. *)
    if is_box_in_list_equality_constraints row equality_acc then
      (box_constraints_acc, non_box_constraints_acc, equality_acc)
    else
      let box_constraints_acc_updated, possible_equality =
        intersection_of_box_and_list_boxes row box_constraints_acc
      in
      (* If the intersection of box constraints turns out to be an equality
      constraint, we add it to the accumulator equality_acc. *)
      match possible_equality with
      | Some equality ->
          ( box_constraints_acc_updated,
            non_box_constraints_acc,
            equality :: equality_acc )
      | None ->
          (box_constraints_acc_updated, non_box_constraints_acc, equality_acc)
  else (box_constraints_acc, row :: non_box_constraints_acc, equality_acc)

(* Make a single traversal over the list of inequality constraints (list_rows).
For each box constraint in the list, we check if it intersects with other box
constraints. If so, we compute their intersections, thereby making the list more
compact. Finally, if this procedure results in equality box constraints, we
substitute them into the lists of inequalities and equalities. *)
let merge_box_constraints
    ((list_rows, equality_constraints) : row list * equality_constraint list) :
    row list * equality_constraint list * bool =
  let box_constraints, non_box_constraints, additional_equalities =
    List.fold list_rows ~init:([], [], []) ~f:insert_row_into_acc_of_constraints
  in
  (* Substitute newly created equalities in additional_equalities into
  non_box_constraints (i.e. the remaining inequalities) and equality_constraints
  (i.e. the existing equalities). *)
  let non_box_constraints_substituted, equality_constraints_substitued =
    List.fold additional_equalities
      ~init:(non_box_constraints, equality_constraints)
      ~f:substitute_into_inequalities_and_equality_constraints
  in
  ( box_constraints @ non_box_constraints_substituted,
    additional_equalities @ equality_constraints_substitued,
    not (List.is_empty additional_equalities) )

(* Repeatedly remove redundant box constraints. A single pass of box-constraint
removal by the function remove_duplicated_box_constraints may produce new box
constraints as a result of substitution. Therefore, we need to repeatedly run
remove_duplicated_box_constraints until the list of inequalities saturates. *)
let rec repeatedly_merge_box_constraints (list_rows, equality_constraints) =
  let ( list_rows_updated,
        equality_constraints_updated,
        any_additional_equalities_for_substitution ) =
    merge_box_constraints (list_rows, equality_constraints)
  in
  if any_additional_equalities_for_substitution then
    repeatedly_merge_box_constraints
      (list_rows_updated, equality_constraints_updated)
  else (list_rows_updated, equality_constraints_updated)

(* Remove redundant non-box inequality rows in addition to redundant box
constraints. I initially implemented this feature for the differential Bayesian
resource analysis of quickselect with degree two. However, later, as I decided
to only retain worst-case samples in the runtime cost data, the number of
duplicated rows became marginal. So this feature is no longer important.

Two rows are deemed equal if they have the identical LP variables and
coefficients in row_elements. So if row1's row_elements is a scalar product of
row2's row_elements (where the scalar is different from one), we do not detect
their equality. *)

(* Compute the intersection of two rows. *)

let row_elements_equality row_elements1 row_elements2 =
  let sort_row_elements row_elements =
    List.sort (fun (v1, _) (v2, _) -> Int.compare v1 v2) row_elements
  in
  let row_elements1_sorted = sort_row_elements row_elements1 in
  let row_elements2_sorted = sort_row_elements row_elements2 in
  let zipped_row_elements =
    List.zip row_elements1_sorted row_elements2_sorted
  in
  match zipped_row_elements with
  | None -> false
  | Some zipped_list ->
      let any_mismatch_in_row_elements =
        List.exists zipped_list ~f:(fun ((v1, c1), (v2, c2)) ->
            v1 <> v2 || not (approximately_equal c1 c2))
      in
      not any_mismatch_in_row_elements

let intersection_of_two_rows row1 row2 =
  let {
    row_lower = row_lower1;
    row_upper = row_upper1;
    row_elements = row_elements1;
  } =
    row1
  in
  let {
    row_lower = row_lower2;
    row_upper = row_upper2;
    row_elements = row_elements2;
  } =
    row2
  in
  let row_elements_are_equal =
    row_elements_equality row_elements1 row_elements2
  in
  if row_elements_are_equal then
    (* TODO: If row_lower = row_upper, what should we do? How often does it
    happen? *)
    Some
      {
        row_lower = Float.max row_lower1 row_lower2;
        row_upper = Float.min row_upper1 row_upper2;
        row_elements = row_elements1;
      }
  else None

let rec intersection_of_row_and_list_rows (target_row : row)
    (list_rows : row list) : row list =
  match list_rows with
  | [] -> [ target_row ]
  | head_row :: tail_rows -> (
      (* We assume that list_rows contains no duplicates. So list_rows can
      contain at most one identical row to target_row. *)
      let intersection_row = intersection_of_two_rows target_row head_row in
      match intersection_row with
      | Some intersection -> intersection :: tail_rows
      | None ->
          head_row :: intersection_of_row_and_list_rows target_row tail_rows )

let remove_identical_rows (list_rows : row list) : row list =
  List.fold list_rows ~init:[] ~f:(fun acc r ->
      intersection_of_row_and_list_rows r acc)

(* Split LP variables into two groups: (i) those that still appear in the linear
program after equality constraints have been removed and (ii) those that are
tracked by equality constraints *)

let categorize_LP_variables (num_vars : int)
    (equality_constraints : equality_constraint list) : int list * int list =
  let equality_vars =
    List.map equality_constraints ~f:(fun equality_constraint ->
        equality_constraint.var)
  in
  let all_vars = List.range 0 num_vars in
  let remaining_vars =
    List.filter all_vars ~f:(fun v1 ->
        not (List.exists equality_vars ~f:(fun v2 -> v1 = v2)))
  in
  (remaining_vars, equality_vars)

(* Let LP1 be the linear program before equality constraint elimination and LP2
be the linear program after the elimination. All variables of LP2 are squeezed
to a smaller, contiguous range of natural numbers from zero to (n-1), where n is
the number of variables in LP2 after equality constraint elimination. In the
following functions, we map LP1's variables to LP2's. *)

let get_index_from_list_vars (list_vars : int list) (var : int) : int =
  let result_findi = List.findi list_vars ~f:(fun _ v -> var = v) in
  match result_findi with
  | None ->
      (* For debugging *)
      let () =
        print_endline "List of vars in get_index_from_list_vars:";
        List.iter list_vars ~f:(fun v -> printf "%i " v);
        print_newline ();
        printf "Target var in get_index_from_list_vars: %i\n" var
      in
      failwith "The list doesn't store all LP variables"
  | Some (index, _) -> index

let get_list_index_from_equality_constraint (list_vars : int list)
    (equality_constraints : equality_constraint list) (var : int) :
    (int * float) list * float =
  let equality_constraint =
    List.find_exn equality_constraints ~f:(fun equality ->
        let { var = v } = equality in
        v = var)
  in
  let { coefficient_list; constant } = equality_constraint in
  let coefficient_list_with_mapped_indices =
    List.map coefficient_list ~f:(fun (v, c) ->
        (get_index_from_list_vars list_vars v, c))
  in
  (coefficient_list_with_mapped_indices, constant)

(* Convert the row-encoding of linear programs (i.e. linear programs encoded as
lists of rows) to the matrix-encoding (i.e. linear programs encoded as A x <= b,
where A is a matrix and b is a vector) *)

let transform_row_elements_to_vector (row_elements : (int * float) list)
    (list_vars : int list) : float list =
  let num_vars = List.length list_vars in
  let expanded_row = Array.init num_vars (fun _ -> 0.) in
  let () =
    List.iter row_elements ~f:(fun (var, coefficient) ->
        expanded_row.(get_index_from_list_vars list_vars var) <- coefficient)
  in
  Array.to_list expanded_row

let transform_row_to_vector (row : row) (list_vars : int list) :
    float list list * float list =
  let { row_lower; row_upper; row_elements } = row in
  let expanded_row = transform_row_elements_to_vector row_elements list_vars in
  match
    (row_upper = Float.max_value, row_lower = Float.neg Float.max_value)
  with
  | true, true -> ([], [])
  | false, true -> ([ expanded_row ], [ row_upper ])
  | true, false ->
      ( [ List.map expanded_row ~f:(fun x -> Float.neg x) ],
        [ Float.neg row_lower ] )
  | false, false ->
      (* If a row has both a lower and upper bounds, the upper-bounded
      constraint comes first in the matrix. *)
      ( [ expanded_row; List.map expanded_row ~f:(fun x -> Float.neg x) ],
        [ row_upper; Float.neg row_lower ] )

let transform_rows_to_matrix (list_rows : row list) (list_vars : int list) :
    float list list * float list =
  let list_list_rows, list_list_bounds =
    List.unzip
      (List.map list_rows ~f:(fun r -> transform_row_to_vector r list_vars))
  in
  (List.concat list_list_rows, List.concat list_list_bounds)

(* Let LP1 be the linear program before equality constraint elimination and LP2
be the linear program after the elimination. In the following functions, we map
LP2's solutions back to LP1's solutions. *)

let get_value_of_LP_variable_row (list_vars : int list) (target_var : int)
    (x : 'a list) : 'a =
  let index = get_index_from_list_vars list_vars target_var in
  List.nth_exn x index

let get_value_of_LP_variable_equality_constraint (list_vars : int list)
    (equality_constraint : equality_constraint) (x : float list) : float =
  let { coefficient_list; constant } = equality_constraint in
  let coefficient_list_concrete =
    List.map coefficient_list ~f:(fun (var, coefficient) ->
        coefficient *. get_value_of_LP_variable_row list_vars var x)
  in
  constant
  +. List.fold coefficient_list_concrete ~init:0. ~f:(fun acc x -> acc +. x)

let get_value_of_LP_variable (list_vars : int list)
    (equality_constraints : equality_constraint list) (target_var : int)
    (x : float list) : float =
  match List.find list_vars ~f:(fun x -> x = target_var) with
  | Some _ -> get_value_of_LP_variable_row list_vars target_var x
  | None ->
      let equality_constraint =
        List.find_exn equality_constraints ~f:(fun equality ->
            let { var } = equality in
            var = target_var)
      in
      get_value_of_LP_variable_equality_constraint list_vars equality_constraint
        x

(* Create box constraints for cost data categorized by sizes *)

let create_box_constraints_for_cost_data_categorized_by_sizes runtime_data =
  let list_maximum_cost =
    List.map runtime_data ~f:(List.reduce_exn ~f:Float.max)
  in
  let create_box_constraint var maximum_cost =
    let row_lower = maximum_cost in
    (* Initially, I mistakenly set row_upper = maximum_cost *. 2.0. However,
    this caused an error when maximum_cost = 0 because it would give row_upper =
    0 as well, resulting in the zero Chebyshev radius. To prevent it, I add a
    positive constant to row_upper. *)
    let row_upper = (maximum_cost *. 2.0) +. 2.5 in
    { row_lower; row_upper; row_elements = [ (var, 1.) ] }
  in
  List.mapi list_maximum_cost ~f:create_box_constraint

(* Set the LP variables in the return type to zero *)

let set_return_type_vars_to_zero list_vars list_rows =
  (* It is safe to row_lower and set_upper to the exact zero. The function
    categorize_rows_into_inequality_and_equality in the Linear_programs_processing
    module will correctly detect this equality. *)
  let set_var_to_zero var =
    { row_lower = 0.; row_upper = 0.; row_elements = [ (var, 1.0) ] }
  in
  let list_rows_for_return_type = List.map list_vars ~f:set_var_to_zero in
  list_rows_for_return_type @ list_rows
