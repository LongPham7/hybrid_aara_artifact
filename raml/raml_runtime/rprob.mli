type t = int * int

val create : int * int -> t

val flip : t -> bool

val consume : float * float * float -> t -> t

val inv : t -> t

val mult : t -> t -> t