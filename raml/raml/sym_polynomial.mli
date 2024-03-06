(** polynomail.mli *)

(** Monomials *)
module Monom :
sig
  type t [@@deriving sexp, compare]

  val is_one : t -> bool

  val degree : t -> int

  val var_exists : (string -> bool) -> t -> bool

  val var_forall : (string -> bool) -> t -> bool

  val fold : t -> init:'a -> f:(var:string -> expo:int -> 'a -> 'a) -> 'a

  val one : t

  val of_var : ?expo:int -> string -> t

  val pow : int -> t -> t

  val mul_var : ?expo:int -> string -> t -> t

  val mul : t -> t -> t

  val split_on : string list -> t -> t * t

  val print : Format.formatter -> t -> unit
end

(** Polymorphic polynomials with abstract coefficients *)
module Poly :
sig
  type 'a t [@@deriving sexp, compare]

  val zero : 'a t

  val const : ?is_zero:('a -> bool) -> 'a -> 'a t

  val of_monom : ?is_zero:('a -> bool) -> Monom.t -> 'a -> 'a t

  val degree : 'a t -> int

  val var_exists : (string -> bool) -> 'a t -> bool

  val monom_exists : Monom.t -> 'a t -> bool

  val fold : 'a t -> init:'b -> f:(mono:Monom.t -> coef:'a -> 'b -> 'b) -> 'b

  val is_const : zer:'a -> 'a t -> 'a option

  val get_coeff : zer:'a -> Monom.t -> 'a t -> 'a

  val scale : ?is_zero:('a -> bool) -> mult:('b -> 'a -> 'a) -> 'b -> 'a t -> 'a t

  (** [mul_monom m c p] computes [c * m * p]. *)
  val mul_monom : ?is_zero:('a -> bool) -> mult:('a -> 'a -> 'a) -> Monom.t -> 'a -> 'a t -> 'a t

  (** [add_monom m c p] computes [c * m + p]. *)
  val add_monom : ?is_zero:('a -> bool) -> addi:('a -> 'a -> 'a) -> Monom.t -> 'a -> 'a t -> 'a t

  (** [add_scale scal p1 p2] computes [scal * p1 + p2]. *)
  val add_scale : ?is_zero:('a -> bool) -> addi:('a -> 'a -> 'a) -> mult:('b -> 'a -> 'a) -> 'b -> 'a t -> 'a t -> 'a t

  val add : ?is_zero:('a -> bool) -> addi:('a -> 'a -> 'a) -> 'a t -> 'a t -> 'a t

  val sub : ?is_zero:('a -> bool) -> subt:('a -> 'a -> 'a) -> zer:'a -> 'a t -> 'a t -> 'a t

  val mul : ?is_zero:('a -> bool) -> addi:('a -> 'a -> 'a) -> mult:('a -> 'a -> 'a) -> 'a t -> 'a t -> 'a t

  val pow : ?is_zero:('a -> bool) -> addi:('a -> 'a -> 'a) -> mult:('a -> 'a -> 'a) -> one:'a -> int -> 'a t -> 'a t

  val print : (Format.formatter -> 'a -> unit) -> Format.formatter -> 'a t -> unit

  val mul_var : ?expo:int -> string -> 'a t -> 'a t

  val map : ?is_zero:('b -> bool) -> ('a -> 'b) -> 'a t -> 'b t

  val void_merge : 'a t -> 'b t -> f:(mono:Monom.t -> [ `Both of 'a * 'b | `Left of 'a | `Right of 'b ] -> unit) -> unit
end

(** Polynomials with float-valued coefficients *)
module FloatPoly :
sig
  type t = float Poly.t [@@deriving compare]

  val zero : t

  val const : float -> t

  val of_monom : Monom.t -> float -> t

  val degree : t -> int

  val var_exists : (string -> bool) -> t -> bool

  val fold : t -> init:'a -> f:(mono:Monom.t -> coef:float -> 'a -> 'a) -> 'a

  val is_const : t -> float option

  val get_coeff : Monom.t -> t -> float

  val scale : float -> t -> t

  (** [mul_monom m c p] computes [c * m * p]. *)
  val mul_monom : Monom.t -> float -> t -> t

  (** [add_monom m c p] computes [c * m + p]. *)
  val add_monom : Monom.t -> float -> t -> t

  (** [add_scale scal p1 p2] computes [scal * p1 + p2]. *)
  val add_scale : float -> t -> t -> t

  val add : t -> t -> t

  val sub : t -> t -> t

  val mul : t -> t -> t

  val pow : int -> t -> t

  val print : Format.formatter -> t -> unit

  val mul_var : ?expo:int -> string -> t -> t

  val monom_subst : string -> t -> Monom.t -> t

  val monom_subst_expo : int -> string -> t -> Monom.t -> t
end
