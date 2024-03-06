(* * * * * * * * * * *
 * Resource Aware ML *
 * * * * * * * * * * *
 *
 * File:
 *   simplify.ml
 *
 * Author:
 *   Jan Hoffmann, Shu-Chun Weng (2014)
 *
 * Description:
 *   LP solver interfaces.
 *
 *   Currently supported:
 *     - CLP via C bindings
 *
 *)

open Core


type result =
  | Feasible
  | Infeasible


module type SOLVER =
sig
  module VMap : Map.S
  exception E of string
  type var = VMap.Key.t [@@deriving sexp, compare]

  (* It stores three components: (i) a list of potential associated with each
  cindex's LP variable in the typing context, (ii) a list of potential
  associated with each index's LP variable in the return type, and (ii) a list
  of costs. *)
  type runtime_data_of_e_in_solver = (var * float list) list * (var * float list) list * float list
  type runtime_data_in_solver = runtime_data_of_e_in_solver list
  type runtime_data_in_solver_int_var = ((int * float list) list * (int * float list) list * float list) list
  
  val fresh_var : ?nonnegative:bool -> unit -> var
  val add_constr_list : ?lower:float -> ?upper:float -> (var*float) list -> unit
  val add_constr_array : ?lower:float -> ?upper:float -> (var*float) array -> unit
  val add_objective : var -> float -> unit
  val set_objective : float VMap.t  -> unit
  val reset_objective : unit -> unit
  val first_solve : unit -> result
  val resolve : unit -> result
  val get_objective_val : unit -> float
  val get_solution : var -> float
  val get_num_constraints : unit -> int
  val get_num_vars : unit -> int

  (* Resetting an LP solver is necessary for hybrid AARA. *)
  val reset_everything : unit -> unit

  (* Write the linear program to an MPS file for debugging *)
  val write_to_file : string -> unit
  
  (* Obtain the list of all rows in the linear program *)
  val get_list_rows : unit -> Clp.row list

  (* Obtain (i) LP variables of (c)indices, (ii) their associated lists of
  potential, and (iii) lists of costs *)
  val get_runtime_data_tick_metric : unit -> runtime_data_in_solver_int_var
  val get_runtime_data_cost_free_metric : unit -> runtime_data_in_solver_int_var
  val add_runtime_data_in_solver_tick_metric : runtime_data_of_e_in_solver -> unit
  val add_runtime_data_in_solver_cost_free_metric : runtime_data_of_e_in_solver -> unit
  val var_to_int : var -> int
  val int_to_var : int -> var

  (* Getter and setter functions for whether we have encountered Raml.stat(...)
  while collecting linear constraints *)
  val is_stat_aara_encontered : unit -> bool
  val stat_aara_is_encountered : unit -> unit

  (* Get the largest value in a solution *)
  val get_largest_value_in_solution : unit -> float option
end


(*The following is for debugging and performance tests*)
module Dummy_solver : SOLVER =
struct
  exception E of string
  module VMap = Unit.Map
  type var = unit [@@deriving sexp, compare]
  type runtime_data_of_e_in_solver = (var * float list) list * (var * float list) list * float list
  type runtime_data_in_solver = runtime_data_of_e_in_solver list
  type runtime_data_in_solver_int_var = ((int * float list) list * (int * float list) list * float list) list

  let fresh_var ?nonnegative () = ()
  let add_constr_list ?lower ?upper _  = ()
  let add_constr_array ?lower ?upper _  = ()
  let add_objective () _ = ()
  let set_objective _ = ()
  let reset_objective () = ()
  let first_solve () = Infeasible
  let resolve () = Infeasible
  let get_objective_val () = 0.0
  let get_solution () = 0.0
  let get_num_constraints () = 0
  let get_num_vars () = 0

  let reset_everything () = ()
  let write_to_file _ = ()
  let get_list_rows () = []
  let get_runtime_data_tick_metric () = []
  let get_runtime_data_cost_free_metric () = []
  let add_runtime_data_in_solver_tick_metric _ = ()
  let add_runtime_data_in_solver_cost_free_metric _ = ()
  let var_to_int _ = 0
  let int_to_var _ = ()
  let is_stat_aara_encontered () = false
  let stat_aara_is_encountered () = ()
  let get_largest_value_in_solution () = None

end


module type CLP_OPTIONS =
sig
  val row_buffer_size : int
  val col_buffer_size : int
  val log_level : int
  val direction : Clp.direction
end


module Clp_std_options : CLP_OPTIONS =
struct
  (* let row_buffer_size = 8000 *)
  let row_buffer_size = 1
  (* let col_buffer_size = 5000 *)
  let col_buffer_size = 1
  let log_level = 0
  let direction = Clp.Minimize
end

module Clp_std_maximize : CLP_OPTIONS =
struct
  (* let row_buffer_size = 8000 *)
  let row_buffer_size = 1
  (* let col_buffer_size = 5000 *)
  let col_buffer_size = 1
  let log_level = 0
  let direction = Clp.Maximize
end

module Clp =
  functor (Options: CLP_OPTIONS) ->
struct
  type var = int [@@deriving sexp, compare]
  type runtime_data_of_e_in_solver = (var * float list) list * (var * float list) list * float list
  type runtime_data_in_solver = runtime_data_of_e_in_solver list
  type runtime_data_in_solver_int_var = ((int * float list) list * (int * float list) list * float list) list

  let col_buffer_size = Options.col_buffer_size
  let row_buffer_size = Options.row_buffer_size

  let () = assert (col_buffer_size > 0)
  let () = assert (row_buffer_size > 0)

  exception E of string

  module VMap = Int.Map

  let clp_state = ref (Clp.create ())
  let objective = ref (Int.Map.empty : float Int.Map.t)
  let solution = ref [| |]
  let () = Clp.set_log_level !clp_state Options.log_level (* use n>0 for debugging *)
  let () = Clp.set_direction !clp_state Options.direction

  let var_count = ref 0
  let row_count = ref 0

  let num_rows = ref 0

  (* list_rows stores all rows of the linear program. I need to explicitly store
  them because the C interface of CLP does not allow users to retrieve rows
  (i..e coefficients of linear constraints, upper bounds, and lower bounds) in
  linear programs. So I should record the rows before they are fed to CLP. 
  
  Keep in mind that all variables in linear programs are non-negative (see the
  function many_cols). Therefore, if we want to represent a linear program in
  the form A x <= b, it is not enough to blindly translate list_rows to matrix
  A. Instead, A is obtained by combining list_rows with the non-negativity
  constraints of all LP variables. *)
  let list_rows : Clp.row list ref = ref []

  let get_list_rows () = !list_rows    

  let runtime_data_in_solver_tick_metric: runtime_data_in_solver ref = ref []
  let runtime_data_in_solver_cost_free_metric: runtime_data_in_solver ref = ref []

  let get_runtime_data_tick_metric () = !runtime_data_in_solver_tick_metric
  let get_runtime_data_cost_free_metric () = !runtime_data_in_solver_cost_free_metric

  let add_runtime_data_in_solver_tick_metric (runtime_data : runtime_data_of_e_in_solver) =
    runtime_data_in_solver_tick_metric := runtime_data :: !runtime_data_in_solver_tick_metric

  let add_runtime_data_in_solver_cost_free_metric (runtime_data : runtime_data_of_e_in_solver) =
    runtime_data_in_solver_cost_free_metric := runtime_data :: !runtime_data_in_solver_cost_free_metric

  let var_to_int x = x
  let int_to_var x = x

  let flag_stat_aara_is_encountered = ref false
  let is_stat_aara_encontered () = !flag_stat_aara_is_encountered
  let stat_aara_is_encountered () = flag_stat_aara_is_encountered := true

  (* Reset everything stored in the LP solver. It is used in hybrid AARA, where
  we want to repeatedly solve linear programs. After the solver is reset, we can
  no longer access the previous linear program's solution, which is stored in
  the ref solution. Consequently, at runtime, if the function get_solution is
  executed after we have reset the solver, we will get a wrong result. Such a
  situation arises when we call get_solution with no arguments (i.e. partial
  application). In this case, get_solution is only evaluated when its closure is
  actually applied at runtime. So the dereferencing statement !solution inside
  get_solution may unexpectedly return the solution to the next linear program.
  *)
  let reset_everything () =
    clp_state := Clp.create ();
    objective := Int.Map.empty;
    solution := [||];
    Clp.set_log_level !clp_state Options.log_level;
    Clp.set_direction !clp_state Options.direction;
    var_count := 0;
    row_count := 0;
    num_rows := 0;
    list_rows := [];
    runtime_data_in_solver_tick_metric := [];
    runtime_data_in_solver_cost_free_metric := [];
    flag_stat_aara_is_encountered := false

  let write_to_file file_name = 
    Clp.write_Mps !clp_state file_name

  let get_num_vars () = Clp.number_columns !clp_state

  let get_num_constraints () =
    !num_rows + !row_count

  (* By default, newly minted variables are lower-bounded by zero. *)
  let many_cols =
      let new_col =
	{ Clp.column_obj = 0.
	; Clp.column_lower = 0.
	; Clp.column_upper = Float.max_value
	; Clp.column_elements = [| |]
	}
      in
      Array.create ~len:col_buffer_size new_col

  let fresh_var () =
    let count = !var_count in
    let () =
      if count % col_buffer_size = 0 then
	Clp.add_columns !clp_state many_cols
      else
	()
    in
    let () = var_count := count + 1 in
    count

  (* override with a naive version for testing nonnegative potential *)
  let fresh_var ?(nonnegative=true) () =
    if false then
      let count = !var_count in
      let () = Clp.add_columns !clp_state [| {
        Clp.column_obj = 0.;
        Clp.column_lower = if nonnegative then 0.0 else Float.min_value;
        Clp.column_upper = Float.max_value;
        Clp.column_elements = [||]
      } |] in
      let () = var_count := count + 1 in
      count
    else
      fresh_var ()

  let row_buffer =
    let init_row =
      { Clp.row_lower = 0.
      ; Clp.row_upper = 0.
      ; Clp.row_elements = [| |]
      }
    in
    Array.create ~len:row_buffer_size init_row


  let flush_row_buffer () =
    let () = Clp.add_rows !clp_state row_buffer in
    list_rows := (Array.to_list row_buffer) @ !list_rows;
    num_rows := !num_rows + !row_count;
    row_count := 0


  let flush_buffers () =
    flush_row_buffer ()


  let add_constr_array ?(lower=(-.Float.max_value)) ?(upper=Float.max_value) row_array =
    (* let () =
      if !row_count = row_buffer_size then flush_row_buffer ()
      else ()
    in *)
    let () = row_count := !row_count+1 in
    let row =
      { Clp.row_lower = lower
      ; Clp.row_upper = upper
      ; Clp.row_elements = row_array
      }
    in
    Array.set row_buffer (!row_count-1) row;
    flush_row_buffer ()


  let add_constr_list ?(lower=(-.Float.max_value)) ?(upper=Float.max_value) row_list =
    (* Print out a linear constraint for debugging *)
    (* let () = 
      printf "Lower: %f; Upper: %f; Row array: " lower upper;
      List.iter row_list ~f:(fun (lp_var, coefficient) -> printf "(lp_var: %i, coefficient: %f); " lp_var coefficient);
      print_newline () 
    in *)
    let row_array = Array.of_list row_list in
    add_constr_array row_array ~lower ~upper

  let add_objective v q =
    objective := Map.set !objective v q

  let set_objective obj =
    objective := obj

  let reset_objective () =
    objective := Int.Map.empty

  let get_solution () =
    solution := Clp.primal_column_solution !clp_state

  let copy_objective () =
    let arr = Array.create (get_num_vars ()) 0.0 in
    let () = Int.Map.iteri !objective (fun ~key ~data -> Array.set arr key data) in
    Clp.change_objective_coefficients !clp_state arr

  let first_solve () =
    copy_objective ()
    ; flush_buffers ()
    ; Clp.initial_solve !clp_state
    ; get_solution ()
    ; match Clp.status !clp_state with
      | 0 -> Feasible
      | _ -> Infeasible

  let resolve () =
    copy_objective ()
    ; flush_buffers ()
    ; Clp.dual !clp_state
    ; get_solution ()
    ; match Clp.status !clp_state with
      | 0 -> Feasible
      | _ -> Infeasible

  let get_objective_val () =
    Clp.objective_value !clp_state

  let get_solution v =
    if -1 < v && v < (Array.length !solution) then
      !solution.(v)
    else
      raise (E ("Variable " ^ (string_of_int v) ^ " is not in the solution."))

  let get_largest_value_in_solution () = 
    Array.max_elt ~compare:Float.compare !solution

end

