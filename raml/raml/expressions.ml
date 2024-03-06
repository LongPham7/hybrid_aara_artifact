(* * * * * * * * * * *
 * Resource Aware ML *
 * * * * * * * * * * *
 *
 * File:
 *   expressions.ml
 *
 * Author:
 *   Jan Hoffmann, Shu-Chun Weng (2014)
 *
 * Description:
 *   RAML expressions.
 *)


open Rtypes
open Core
open Format


exception Emalformed of string

type expression_kind =
  | Efree
  | Enormal


type var_id = string [@@deriving sexp]

type call_name = string option

type 'a var_rbind = var_id * 'a rtype [@@deriving sexp]
type var_bind = unit var_rbind [@@deriving sexp]

type builtin_op =
  | Un_not
  | Un_iminus
  | Un_fminus
  | Bin_iadd
  | Bin_isub
  | Bin_imult
  | Bin_imod
  | Bin_idiv
  | Bin_fadd
  | Bin_fsub
  | Bin_fmult
  | Bin_fdiv
  | Bin_and
  | Bin_or
  | Bin_eq
  | Bin_iless_eq
  | Bin_igreater_eq
  | Bin_iless
  | Bin_igreater
  | Bin_fless_eq
  | Bin_fgreater_eq
  | Bin_fless
  | Bin_fgreater [@@deriving compare]

let is_binop op =
  match op with
    | Un_not
    | Un_iminus
    | Un_fminus -> false
    | _ -> true

let  string_of_builtin_op op =
  match op with
    | Un_not -> "not"
    | Un_iminus -> "-"
    | Un_fminus -> "-."
    | Bin_iadd -> "+"
    | Bin_isub -> "-"
    | Bin_imult -> "*"
    | Bin_imod -> "%"
    | Bin_idiv -> "/"
    | Bin_fadd -> "+."
    | Bin_fsub -> "-."
    | Bin_fmult -> "*."
    | Bin_fdiv -> "/."
    | Bin_and -> "&&"
    | Bin_or -> "||"
    | Bin_eq -> "="
    | Bin_iless_eq -> "<="
    | Bin_igreater_eq -> ">="
    | Bin_iless -> "<"
    | Bin_igreater -> ">"
    | Bin_fless_eq -> "<=."
    | Bin_fgreater_eq -> ">=."
    | Bin_fless -> "<."
    | Bin_fgreater -> ">."


type builtin_fun =
  (* natural numbers *)
  | Nat_succ
  | Nat_to_int
  | Nat_of_int
  | Nat_of_intc of int
  | Nat_add
  | Nat_mult
  | Nat_minus
  | Nat_minusc of int
  | Nat_div_mod

  (* arrays *)
  | Arr_make
  | Arr_set
  | Arr_get
  | Arr_length

  (* references *)
  | Ref_swap

  (* resource managemt *)
  | Res_consume of int

  (* probabilities *)
  | Prob_create
  | Prob_createc of int * int
  | Prob_flip
  | Prob_consume of float * float * float
  | Prob_inv
  | Prob_mult [@@deriving compare]

let string_of_builtin_fun f =

  let prob_module str =
    Rconfig.ocaml_prob_module ^ "." ^ str
  in

  let nat_module str =
    Rconfig.ocaml_nat_module ^ "." ^ str
  in


  let arr_module str =
    Rconfig.ocaml_array_module ^ "." ^ str
  in

  let raml_module str =
    Rconfig.ocaml_raml_module ^ "." ^ str
  in


  match f with
    | Nat_succ -> nat_module "Succ"
    | Nat_to_int -> nat_module "to_int"
    | Nat_of_int -> nat_module "of_int"
    | Nat_of_intc n -> nat_module ("of_intc " ^ string_of_int(n))
    | Nat_add -> nat_module "add"
    | Nat_mult -> nat_module "mult"
    | Nat_minus -> nat_module "minus"
    | Nat_minusc n -> nat_module ("minus_" ^ string_of_int(n))
    | Nat_div_mod -> nat_module "div_mod"
    | Arr_make -> arr_module "make"
    | Arr_set -> arr_module "set"
    | Arr_get -> arr_module "get"
    | Arr_length -> arr_module "length"
    | Ref_swap -> raml_module "swap"
    | Res_consume _ -> raml_module "consume"
    | Prob_create -> prob_module "create"
    | Prob_createc (a,b) -> prob_module ("createc(" ^ string_of_int a ^ "," ^ string_of_int b ^ ")")
    | Prob_flip -> prob_module "flip"
    | Prob_consume (qh,qt,qc) -> prob_module ("consume[" ^ string_of_float qh ^ ";" ^ string_of_float qt ^ ";" ^ string_of_float qc ^ "]")
    | Prob_inv -> prob_module "inv"
    | Prob_mult -> prob_module "mult"


type constant =
  | Cint of int
  | Cfloat of float
  | Cbool of bool
  | Czero  (* of Nat *)
  | Cunit [@@deriving compare]

let string_of_constant c =
  match c with
    | Cint n -> string_of_int n
    | Cfloat q -> Float.to_string q
    | Cbool b -> string_of_bool b
    | Czero -> Rconfig.ocaml_nat_module ^ ".Zero"
    | Cunit -> "()"





type ('a, 'b) expression =
    { exp_desc : ('a, 'b) expression_desc
    ; exp_type : 'b rtype
    ; exp_kind : expression_kind
    ; exp_info : 'a
    }


and ('a, 'b) expression_desc =
  (* constants / built-in functions *)
  (* there are no constants of type nat anymore *)
  | Ebase_const of constant
  | Ebase_fun of builtin_fun
  | Ebase_op of builtin_op

  (* variables *)
  | Evar of var_id

  (* functions *)
  | Eapp of call_name * ('a, 'b) expression * (('a, 'b) expression) list
  | Elambda of 'b var_rbind * ('a, 'b) expression

  (* let bindings / control flow / sharing *)
  | Elet of 'b var_rbind option * ('a, 'b) expression * ('a, 'b) expression
  | Eletrec of 'b var_rbind list * ('a, 'b) expression list * ('a, 'b) expression
  | Econd of ('a, 'b) expression * ('a, 'b) expression * ('a, 'b) expression
  | Eshare of ('a, 'b) expression * 'b var_rbind * 'b var_rbind * ('a, 'b) expression
  | Eflip of ('a, 'b) expression * ('a, 'b) expression * ('a, 'b) expression
  | Eflipc of (int * int) * ('a, 'b) expression * ('a, 'b) expression

  (* user defined data types *)
  | Econst of constr_id * ('a, 'b) expression
  | Ematch of ('a, 'b) expression * (constr_id * 'b var_rbind list * ('a, 'b) expression) list

  (* matching for natural numbers *)
  | Enat_match of ('a, 'b) expression * ('a, 'b) expression * 'b var_rbind * ('a, 'b) expression

  (* references *)
  | Eref of ('a, 'b) expression
  | Eref_deref of ('a, 'b) expression
  | Eref_assign of ('a, 'b) expression * ('a, 'b) expression

  (* tuples *)
  | Etuple of ('a, 'b) expression list
  | Etuple_match of ('a, 'b) expression * 'b var_rbind list * ('a, 'b) expression

  (* raml specific *)
  | Eundefined
  | Etick of float

  (* hybrid AARA *)
  | Estat of ('a, 'b) expression

let get_type : ('a, 'b) expression -> 'b rtype =
  fun e -> e.exp_type

let set_type: ('a, 'b) expression -> 'b rtype -> ('a, 'b) expression =
  fun e t -> {e with exp_type = t}

let get_kind : ('a, 'b) expression -> expression_kind =
  fun e -> e.exp_kind

let set_kind: ('a, 'b) expression -> expression_kind -> ('a, 'b) expression =
  fun e k -> {e with exp_kind = k}

let get_info : ('a, 'b) expression -> 'a =
  fun e -> e.exp_info

let set_info: ('a, 'b) expression -> 'a -> ('a, 'b) expression =
  fun e b -> {e with exp_info = b}


type typed = Location.t

type sln = Sln

type sln_expression = (sln, unit) expression

type typed_expression = (typed, unit) expression

let rec convert_typed_to_sln_expression (exp : typed_expression) : sln_expression =
  let { exp_desc = exp_desc } = exp in
  let updated_exp_desc = convert_typed_to_sln_expression_desc exp_desc in
  { exp with exp_desc = updated_exp_desc; exp_info = Sln}
and convert_typed_to_sln_expression_desc exp_desc =
  match exp_desc with
  | Ebase_const c -> Ebase_const c
  | Ebase_fun f -> Ebase_fun f
  | Ebase_op op -> Ebase_op op
  | Evar v -> Evar v
  | Eapp (call_name, f, args) -> Eapp (call_name, convert_typed_to_sln_expression f, List.map args ~f:convert_typed_to_sln_expression)
  | Elambda (x, e) -> Elambda (x, convert_typed_to_sln_expression e)
  | Elet (x, e1, e2) -> Elet (x, convert_typed_to_sln_expression e1, convert_typed_to_sln_expression e2)
  | Eletrec (list_x, list_e1, e2) -> Eletrec (list_x, List.map list_e1 ~f: convert_typed_to_sln_expression, convert_typed_to_sln_expression e2)
  | Econd (guard, e1, e2) -> Econd (convert_typed_to_sln_expression guard, convert_typed_to_sln_expression e1, convert_typed_to_sln_expression e2)
  | Eshare (e1, x1, x2, e2) -> Eshare (convert_typed_to_sln_expression e1, x1, x2, convert_typed_to_sln_expression e2)
  | Eflip (e1, e2, e3) -> Eflip (convert_typed_to_sln_expression e1, convert_typed_to_sln_expression e2, convert_typed_to_sln_expression e3)
  | Eflipc ((n1, n2), e1, e2) -> Eflipc ((n1, n2), convert_typed_to_sln_expression e1, convert_typed_to_sln_expression e2)
  | Econst (constr_id, e) -> Econst (constr_id, convert_typed_to_sln_expression e)
  | Ematch (e, list_constr_id_list_es) -> Ematch (convert_typed_to_sln_expression e, List.map list_constr_id_list_es ~f:(fun (constr_id, list, e) -> (constr_id, list, convert_typed_to_sln_expression e)))
  | Enat_match (e1, e2, x, e3) -> Enat_match (convert_typed_to_sln_expression e1, convert_typed_to_sln_expression e2, x, convert_typed_to_sln_expression e3)
  | Eref e -> Eref (convert_typed_to_sln_expression e)
  | Eref_deref e -> Eref_deref (convert_typed_to_sln_expression e)
  | Eref_assign (e1, e2) -> Eref_assign (convert_typed_to_sln_expression e1, convert_typed_to_sln_expression e2)
  | Etuple list_e -> Etuple (List.map list_e ~f:convert_typed_to_sln_expression)
  | Etuple_match (e1, list_x, e2) -> Etuple_match (convert_typed_to_sln_expression e1, list_x, convert_typed_to_sln_expression e2)
  | Eundefined -> Eundefined
  | Etick f -> Etick f
  | Estat e -> Estat (convert_typed_to_sln_expression e)

let is_lambda exp =
  match exp.exp_desc with
    | Elambda _ -> true
    | _ -> false

let substitute
    : ('a, 'b) expression -> var_id ->
      ('b rtype -> expression_kind -> 'a -> ('a, 'b) expression) ->
      ('a, 'b) expression =
  fun exp var update ->

    let rec subst exp =

      match exp.exp_desc with
	| Evar x ->
	  if x = var then
	    update (get_type exp) (get_kind exp) (get_info exp)
	  else
	    exp

        | desc -> {exp with exp_desc =
	    match desc with
	      | Ebase_const _ -> desc
              | Ebase_fun _ -> desc
              | Ebase_op _ -> desc

              | Evar _ -> raise (Emalformed "This is dead code.")

 	      | Eapp (name,e,es) -> Eapp (name, subst e, List.map es subst)
	      | Elambda (x,e) -> Elambda (x,sub_bound [x] e)

	      | Elet (x_opt,e1,e2) ->
                let xs = Option.to_list x_opt in
		Elet (x_opt,subst e1, sub_bound xs e2)

	      | Eletrec (xs,es,e) ->
                let e' = sub_bound xs e in
		let es' = List.map es (sub_bound xs) in
                Eletrec (xs,es', e')

	      | Econd (e,e1,e2) -> Econd (subst e, subst e1, subst e2)
        | Eflip (e,e1,e2) -> Eflip (subst e, subst e1, subst e2)
        | Eflipc (p,e1,e2) -> Eflipc (p, subst e1, subst e2)
	      | Eshare (e1,x1,x2,e2) -> Eshare (subst e1,x1,x2, sub_bound [x1;x2] e2)

	      | Econst (c,e) -> Econst (c, subst e)
	      | Ematch (e,matches) ->
		let matches' =
		  List.map matches (fun (c,xs,e) -> (c, xs, sub_bound xs e))
		in
		Ematch (subst e, matches')

	      | Enat_match (e,e1,x,e2) -> Enat_match (subst e, subst e1,x, sub_bound [x] e2)

	      | Eref e -> Eref (subst e)
	      | Eref_deref e -> Eref_deref (subst e)
	      | Eref_assign (e1,e2) -> Eref_assign (subst e1,subst e2)

	      | Etuple es -> Etuple (List.map es subst)
	      | Etuple_match (e1,xs,e2) -> Etuple_match (subst e1, xs, sub_bound xs e2)

	      | Eundefined -> desc
	      | Etick n -> desc
        | Estat e -> Estat (subst e)
		  }

    and	sub_bound xs e =
      match List.find xs (fun (x,t) -> x=var) with
	| Some _ -> e
	| None -> subst e
    in
    subst exp


(* substituting variables *)

let subst_var
    : ('a, 'b) expression -> var_id -> var_id -> ('a, 'b) expression =
  (* exp[x1 <- x2] *)
  fun exp x1 x2 ->
    let update t kind a  = { exp_desc = Evar x2
			   ; exp_type = t
			   ; exp_kind = kind
			   ; exp_info = a
                           } in
    substitute exp x1 update


(* apply a function to each subexpression *)

let apply_to_subexps
    : 'a 'b.
      (('a, 'c) expression -> ('b, 'c) expression) ->
      ('a, 'c) expression_desc -> ('b, 'c) expression_desc =
  fun f desc ->
    match desc with
      | Ebase_const c -> Ebase_const c
      | Ebase_fun base_fun -> Ebase_fun base_fun
      | Ebase_op op -> Ebase_op op

      | Evar v -> Evar v

      | Eapp (cname,e,es) -> Eapp (cname, f e, List.map es f)
      | Elambda (x,e) -> Elambda (x, f e)

      | Elet (x_opt,e1,e2) -> Elet (x_opt, f e1, f e2)
      | Eletrec (xs, es, e) -> Eletrec (xs, List.map es f, f e)
      | Econd (e,e1,e2) -> Econd (f e, f e1, f e2)
      | Eflip (e,e1,e2) -> Eflip (f e, f e1, f e2)
      | Eflipc (p,e1,e2) -> Eflipc (p, f e1, f e2)
      | Eshare (e1,x1,x2,e2) -> Eshare (f e1, x1, x2, f e2)

      | Econst (c,e) -> Econst (c, f e)
      | Ematch (e,matches) ->
	let matches' = List.map matches (fun (c,xs,e) -> (c, xs, f e)) in
	Ematch (f e, matches')

      | Enat_match (e,e1,x,e2) -> Enat_match (f e, f e1, x, f e2)

      | Eref e -> Eref (f e)
      | Eref_deref e -> Eref_deref (f e)
      | Eref_assign (e1,e2) -> Eref_assign (f e1,f e2)

      | Etuple es -> Etuple (List.map es f)
      | Etuple_match (e1,xs,e2) -> Etuple_match (f e1, xs, f e2)

      | Eundefined -> Eundefined
      | Etick n -> Etick n
      | Estat e -> Estat (f e)


(* map for expressions *)

let rec e_map
    : 'a 'b.
      ('c rtype -> expression_kind -> 'a -> 'c rtype * expression_kind * 'b) ->
      ('a, 'c) expression -> ('b, 'c) expression =
  fun f exp ->
    let e_desc = apply_to_subexps (e_map f) exp.exp_desc in
    let (e_type,e_kind,e_info) = f exp.exp_type exp.exp_kind exp.exp_info in

    { exp_desc = e_desc
    ; exp_type = e_type
    ; exp_kind = e_kind
    ; exp_info = e_info
    }

let e_map_type
    : 'b 'c. ('b rtype -> 'c rtype) -> ('a, 'b) expression -> ('a, 'c) expression =
  fun f ->
    let map_var_bind (v, t) = (v, f t) in
    let rec map_rec exp =
      { exp_type = f exp.exp_type
      ; exp_kind = exp.exp_kind
      ; exp_info = exp.exp_info
      ; exp_desc = match exp.exp_desc with
      | Ebase_const c -> Ebase_const c
      | Ebase_fun f -> Ebase_fun f
      | Ebase_op o -> Ebase_op o
      | Evar v -> Evar v
      | Eapp (cname, e, es) -> Eapp (cname, map_rec e, List.map es map_rec)
      | Elambda (x, e) -> Elambda (map_var_bind x, map_rec e)
      | Elet (x_opt, e1, e2) -> Elet (Option.map x_opt map_var_bind,
                                      map_rec e1, map_rec e2)
      | Eletrec (xs, es, e) -> Eletrec (List.map xs map_var_bind,
                                        List.map es map_rec, map_rec e)
      | Econd (e, e1, e2) -> Econd (map_rec e, map_rec e1, map_rec e2)
      | Eflip (e, e1, e2) -> Eflip (map_rec e, map_rec e1, map_rec e2)
      | Eflipc (p, e1, e2) -> Eflipc (p, map_rec e1, map_rec e2)
      | Eshare (e1, x1, x2, e2) ->
        Eshare (map_rec e1, map_var_bind x1, map_var_bind x2, map_rec e2)

      | Econst (c, e) -> Econst (c, map_rec e)
      | Ematch (e, matches) ->
	let matches' =
          List.map matches (fun (c, xs, e) ->
                                (c, List.map xs map_var_bind, map_rec e)) in
        Ematch (map_rec e, matches')

      | Enat_match (e, e1, x, e2) ->
        Enat_match (map_rec e, map_rec e1, map_var_bind x, map_rec e2)

      | Eref e -> Eref (map_rec e)
      | Eref_deref e -> Eref_deref (map_rec e)
      | Eref_assign (e1, e2) -> Eref_assign (map_rec e1, map_rec e2)

      | Etuple es -> Etuple (List.map es map_rec)
      | Etuple_match (e1, xs, e2) ->
        Etuple_match (map_rec e1, List.map xs map_var_bind, map_rec e2)
      | Eundefined -> Eundefined
      | Etick n -> Etick n
      | Estat e -> Estat (map_rec e)
      }
    in map_rec

(* apply a transformer to each subexpression, starting at the leaves *)

let rec e_transform
    : (('a, 'b) expression -> ('a, 'b) expression) ->
      ('a, 'b) expression -> ('a, 'b) expression =
  fun f exp ->
    let e_desc = apply_to_subexps (e_transform f) exp.exp_desc in
    f {exp with exp_desc = e_desc}


(* apply a transformer to each subexpression, starting at the root *)
let rec e_transform_outside_in
    : (('a, 'b) expression -> ('a, 'b) expression) ->
      ('a, 'b) expression -> ('a, 'b) expression =
  fun f exp ->
    let exp' = f exp in
    let e_desc = apply_to_subexps (e_transform_outside_in f) exp'.exp_desc in
    {exp' with exp_desc = e_desc}

(* free variables of an expression *)


module T = struct
  type t = var_bind [@@deriving sexp]

  let compare (x,t) (y,u) =
    let c = compare x y in c
  (* sanity check conflicts with analysis.ml in one place *)
(*    if c <> 0 then
      c
    else if t = u then
      c
    else
      raise (Emalformed "A free variable has different types in the same expression.") *)
end

module Vb_set = Set.Make(T)


let free_vars
    : ('a, 'b) expression -> Vb_set.t =

  let empty_set = Vb_set.empty in
  let union_list = Vb_set.union_list in
  let remove_list s xs = List.fold xs ~init:s ~f:Set.remove in

  let rec fvars exp =
    match exp.exp_desc with
      | Ebase_const _ -> empty_set
      | Ebase_fun _ -> empty_set
      | Ebase_op _ -> empty_set

      | Evar v -> Vb_set.singleton (v,exp.exp_type)

      | Eapp (cname,e,es) ->
	let fvs = Vb_set.union_list (List.map es fvars) in
	Set.union (fvars e) fvs

      | Elambda (x,e) -> Set.remove (fvars e) x

      | Elet (x_opt,e1,e2) ->
	let fv1 = fvars e1 in
        let fv2 = match x_opt with
	  	    | None -> fvars e2
		    | Some x -> Set.remove (fvars e2) x
	in
	Set.union fv1 fv2

      | Eletrec (xs, es, e) ->
	let fv_es = union_list (List.map es fvars) in
	let fvs = Set.union (fvars e) fv_es in
        remove_list fvs xs

      | Econd (e,e1,e2) -> union_list [fvars e; fvars e1; fvars e2]
      | Eflip (e,e1,e2) -> union_list [fvars e; fvars e1; fvars e2]
      | Eflipc (p,e1,e2) -> union_list [fvars e1; fvars e2]

      | Eshare (e1,x1,x2,e2) ->
	let fv2 = remove_list (fvars e2) [x1;x2] in
        Set.union (fvars e1) fv2

      | Econst (c,e) -> fvars e

      | Ematch (e,matches) ->
	let fvs = fvars e in
	let mvars = List.map matches
	  (fun (c,xs,e) -> remove_list (fvars e) xs )
	in
        union_list (fvs::mvars)

      | Enat_match (e,e1,x,e2) ->
	let fvs = Set.union (fvars e) (fvars e1) in
	let fv = Set.remove (fvars e2) x in
        Set.union fvs fv

      | Eref e -> fvars e
      | Eref_deref e -> fvars e
      | Eref_assign (e1,e2) -> Set.union (fvars e1) (fvars e2)

      | Etuple es -> union_list (List.map es fvars)

      | Etuple_match (e1,xs,e2) ->
	let fv2 = remove_list (fvars e2) xs in
	Set.union (fvars e1) fv2

      | Eundefined -> empty_set
      | Etick n -> empty_set
      | Estat e -> fvars e
  in

  fvars

(* The printing function for expressions is copied here for debugging. It allows
us to print out expressions in a module that "is used" by pprint.ml. Otherwise,
we would need to use the printing function in pprint.ml. But it cannot be used
in a module that is used by pprint.ml, since it would lead to a circular
dependency between modules, causing an error during compilation. *)

let out_fixed (*form f*) = Fn.flip fprintf (*f form*)

let rec fprint_list_sep f (xs, fprintx, sep) =
  match xs with
  | [] -> ()
  | [x] -> fprintx f x
  | x::xs -> fprintf f "%a%t%a" fprintx x sep fprint_list_sep (xs, fprintx, sep)

let prec_of_builtin_op = function
  | Un_not          -> 70  (* Pervasives.not is a function *)
  | Un_iminus       -> 65
  | Un_fminus       -> 65
  | Bin_iadd        -> 50
  | Bin_isub        -> 50
  | Bin_imult       -> 55
  | Bin_imod        -> 55
  | Bin_idiv        -> 55
  | Bin_fadd        -> 50
  | Bin_fsub        -> 50
  | Bin_fmult       -> 55
  | Bin_fdiv        -> 55
  | Bin_and         -> 30
  | Bin_or          -> 25
  | Bin_eq          -> 35
  | Bin_iless_eq    -> 35
  | Bin_igreater_eq -> 35
  | Bin_iless       -> 35
  | Bin_igreater    -> 35
  | Bin_fless_eq    -> 35
  | Bin_fgreater_eq -> 35
  | Bin_fless       -> 35
  | Bin_fgreater    -> 35

let builtin_op_is_left_assoc = function
  | Un_not          -> Some true  (* Pervasives.not is a function *)
  | Un_iminus       -> Some false  (* noassoc in spec, but totally right assoc *)
  | Un_fminus       -> Some false
  | Bin_iadd        -> Some true
  | Bin_isub        -> Some true
  | Bin_imult       -> Some true
  | Bin_imod        -> Some true
  | Bin_idiv        -> Some true
  | Bin_fadd        -> Some true
  | Bin_fsub        -> Some true
  | Bin_fmult       -> Some true
  | Bin_fdiv        -> Some true
  | Bin_and         -> Some false
  | Bin_or          -> Some false
  | Bin_eq          -> Some true
  | Bin_iless_eq    -> Some true
  | Bin_igreater_eq -> Some true
  | Bin_iless       -> Some true
  | Bin_igreater    -> Some true
  | Bin_fless_eq    -> Some true
  | Bin_fgreater_eq -> Some true
  | Bin_fless       -> Some true
  | Bin_fgreater    -> Some true

let fprint_expression ?(indent=2) ?(print_types=false) f exp =

  (* Use the default values of optional parameters for [fprint_raml_type]. *)
  let fprint_type f tp = Rtypes.fprint_raml_type f tp in

  let fprint_binding f (x, t) =
    fprintf f "%s" x;
    if print_types then
      fprintf f " :@ %a" fprint_type t
  in

  let fprint_bindings_sep f (xts, sep) =
    fprint_list_sep f (xts, fprint_binding, sep)
  in

  let fprint_binding_list f = function
    | []   -> ()
    | [xt] -> fprint_binding f xt
    | xts  -> pp_open_box f indent;
              fprintf f "(%a)@]" fprint_bindings_sep (xts, out_fixed ",@ ")

  in

  let fprint_binding_option f = function
    | None -> pp_print_string f "_"
    | Some xt -> fprint_binding f xt
  in

  let color = ref Enormal in
  let set_color f c =
    if !color <> c then
      begin
        color := c
      ; match c with
        | Efree   -> fprintf f "@<0>%s" Rconfig.ansi_esc_sequence_free
        | Enormal -> fprintf f "@<0>%s" Rconfig.ansi_esc_sequence_normal
      end in

  let rec fprint_exps_sep f (parents, exps, sep) =
    fprint_list_sep f (exps, (fun f e -> fprint_exp f (parents, e)), sep)

  and fprint_letrecs f xes =
    match xes with
      | [] -> ()
      | [(xt,e)] ->
        fprintf f "%a =@ %a" fprint_binding xt fprint_exp (0, e)
      | (xt,e)::xes ->
        fprintf f "%a =@ %a@]@ @[and@ %a"
          fprint_binding xt fprint_exp (0, e) fprint_letrecs xes

  and fprint_matches f matches =
    match matches with
      | [] -> ()
      | (constr, xts, e) :: matches ->
        printf "@]@ @[| %s %a@ ->@ %a%a"
          (if print_types then constr
                          else List.hd_exn (String.split constr '|'))
          fprint_binding_list xts
          fprint_exp (0, e)
          fprint_matches matches

  and fprint_exp f (d, exp) =
    let print_ann = print_types && match exp.exp_desc with
      | Elet (_, _, e) | Eletrec (_, _, e) | Etuple_match (_, _, e)
      | Eshare (_, _, _, e) ->
        (* Only print the result type of let again if it's marked different
           from the innor expression -- likely a faulty expression. *)
        e.exp_type <> exp.exp_type
      | _ -> true in
    let last_color = !color in
    begin
      pp_open_box f 0
    ; set_color f exp.exp_kind
    ; if print_ann then pp_print_string f "("
    ; fprint_exp_desc f ((if print_ann then 0 else d), exp.exp_desc)
    ; if print_ann then fprintf f " :@ %a)" fprint_type exp.exp_type
    ; set_color f last_color
    ; pp_close_box f ()
    end

  and fprint_exp_desc f (d, desc) =

    let open_paren f (indent, printing_prec) =
      pp_open_box f indent;
      if d > printing_prec then
        pp_print_string f "("
    in

    let close_paren f printing_prec =
      if d > printing_prec then
        pp_print_string f ")";
      pp_close_box f ()
    in
    match desc with
    | Ebase_const c ->   pp_print_string f (string_of_constant c)
    | Ebase_fun fn ->    pp_print_string f (string_of_builtin_fun fn)
    | Ebase_op Un_not -> pp_print_string f (string_of_builtin_op Un_not)
    | Ebase_op op ->     fprintf f "(%s)"  (string_of_builtin_op op)

    | Evar x -> pp_print_string f x

    | Eapp (name, e, es) -> begin
      match e.exp_desc, es with
      | Ebase_op op, [{ exp_desc = Etuple [e1; e2] }] when is_binop op ->
        let prec = prec_of_builtin_op op in
        let curr_color = !color in
        let (e1_prec, e2_prec) =
          match builtin_op_is_left_assoc op with
          | Some true  -> (prec, prec + 1)
          | Some false -> (prec + 1, prec)
          | None       -> (prec + 1, prec + 1) in
        fprintf f "%a%a@ %a%s%a@ %a%a"
          open_paren (0, prec)
          fprint_exp (e1_prec, e1)
          set_color e.exp_kind
          (string_of_builtin_op op)
          set_color curr_color
          fprint_exp (e2_prec, e2)
          close_paren prec
      | _, _ ->
        let app_prec = 85 in
        fprintf f "%a%a@ %a%a"
          open_paren (0, app_prec)
          fprint_exp (app_prec + 1, e)
          fprint_exps_sep (app_prec + 1, es, Fn.flip pp_print_space ())
          close_paren app_prec
      end

    | Elambda (xt, e) ->
      let lambda_prec = 0 in
      fprintf f "%afun %a ->@ %a%a"
        open_paren (0, lambda_prec)
        fprint_binding xt fprint_exp (lambda_prec, e)
        close_paren lambda_prec

    | Elet (x_opt, e1, e2) ->
      let let_prec = 0 in
      fprintf f "%alet@ %a =@ %a@ in@]@ @[%a%a"
        open_paren (0, let_prec)
        fprint_binding_option x_opt
        fprint_exp (0, e1)  (* inner exp has 0 precedence *)
        fprint_exp (let_prec, e2)
        close_paren let_prec

    | Eletrec (xs, es, e) ->
      let let_prec = 0 in
      fprintf f "%alet rec@ %a@ in@]@ @[%a%a"
        open_paren (0, let_prec)
        fprint_letrecs (List.zip_exn xs es) fprint_exp (let_prec, e)
        close_paren let_prec

    | Econd (e, e1, e2) ->
      let if_prec = 10 in
      fprintf f "%aif@ %a@ then@ %a@ @]%aelse@ %a%a"
        open_paren (indent, if_prec)
        fprint_exp (0, e)  fprint_exp (0, e1)
        pp_open_box indent fprint_exp (if_prec, e2)
        close_paren if_prec

    | Eflip (e, e1, e2) ->
      let if_prec = 10 in
      fprintf f "%aflip@ %a@ then@ %a@ @]%aelse@ %a%a"
        open_paren (indent, if_prec)
        fprint_exp (0, e) fprint_exp (0, e1)
        pp_open_box indent fprint_exp (if_prec, e2)
        close_paren if_prec

    | Eflipc ((a,b), e1, e2) ->
      let if_prec = 10 in
      fprintf f "%aflipc@ (%d,%d)@ then@ %a@ @]%aelse@ %a%a"
        open_paren (indent, if_prec)
        a b
        fprint_exp (0, e1)
        pp_open_box indent fprint_exp (if_prec, e2)
        close_paren if_prec

    | Eshare (e1, x1, x2, e2) ->
      let share_prec = 0 in
      fprintf f "%ashare@ %a@ as@ (%a,@ %a)@ in@]@ @[%a%a"
        open_paren (0, share_prec)
        fprint_exp (0, e1) fprint_binding x1 fprint_binding x2
        fprint_exp (share_prec, e2)
        close_paren share_prec

    | Econst (c, e) ->
      let constr_prec = 70 in
      fprintf f "%a%s@ %a%a"
        open_paren (indent, constr_prec)
        (if print_types then c else List.hd_exn (String.split c '|'))
        fprint_exp (constr_prec + 1, e)
        close_paren constr_prec

    | Ematch (e, matches) ->
      let match_prec = 0 in
      fprintf f "@[<2>%amatch@ %a@ with%a%a@]"
        open_paren (indent, match_prec)
        fprint_exp (0, e) fprint_matches matches
        close_paren match_prec

    | Enat_match (e, e1, xt, e2) ->
      let match_prec = 0 in
      fprintf f "@[<2>%amatch@ %a@ with@]@ @[| %s.%s ->@ %a@]@ @[| %s.%s %a ->@ %a%a@]"
        open_paren (indent, match_prec)
        fprint_exp (0, e)
        Rconfig.ocaml_nat_module Rconfig.ocaml_nat_zero
        fprint_exp (0, e1)
        Rconfig.ocaml_nat_module Rconfig.ocaml_nat_succ
        fprint_binding xt fprint_exp (0, e2)
        close_paren match_prec

    | Eref e ->
      let ref_prec = 70 in
      fprintf f "%aref@ %a%a" open_paren (indent, ref_prec)
        fprint_exp (ref_prec + 1, e) close_paren ref_prec

    | Eref_deref e ->
      let deref_prec = 85 in
      fprintf f "%a!%a%a" open_paren (0, deref_prec)
        fprint_exp (deref_prec, e) close_paren deref_prec

    | Eref_assign (e1, e2) ->
      let assign_prec = 15 in
      fprintf f "%a%a@ :=@ %a%a" open_paren (indent, assign_prec)
        fprint_exp (assign_prec + 1, e1) fprint_exp (assign_prec, e2)
        close_paren assign_prec

    | Etuple es ->
      let always_paren = -1 in
      fprintf f "%a%a%a" open_paren (indent, always_paren)
        fprint_exps_sep (0, es, out_fixed ",@ ")
        close_paren always_paren

    | Etuple_match (e1, xs, e2) ->
      let let_prec = 0 in
      fprintf f "%alet (%a) =@ %a@ in@]@ @[%a%a"
        open_paren (0, let_prec)
        fprint_bindings_sep (xs, out_fixed ",@ ")
        fprint_exp (0, e1)  (* inner exp has 0 precedence *)
        fprint_exp (let_prec, e2)
        close_paren let_prec

    | Eundefined ->
      fprintf f "%s.%s" Rconfig.ocaml_raml_module Rconfig.ocaml_raml_undefined

    | Etick q ->
      fprintf f "%s.%s(%f)" Rconfig.ocaml_raml_module Rconfig.ocaml_raml_tick q

    | Estat e -> fprintf f "%s.%s(%a)" Rconfig.ocaml_raml_module Rconfig.ocaml_raml_stat fprint_exp (70 + 1, e)

  in fprintf f "%a%a" fprint_exp (0, exp) set_color Enormal

let print_expression ?output:(formatter=std_formatter)
    ?(indent=2) ?(print_types=false) exp =
  let orig_max_indent = pp_get_max_indent formatter () in
  let orig_margin     = pp_get_margin     formatter () in
  let _ = pp_set_max_indent formatter 30 in
  let _ = pp_set_margin     formatter 120 in
    fprintf formatter "%a@." (fprint_expression ~indent ~print_types) exp
  ; pp_set_max_indent formatter orig_max_indent
  ; pp_set_margin     formatter orig_margin

let expression_to_string exp = 
  print_expression ~output:Format.str_formatter exp;
  Format.flush_str_formatter ()
