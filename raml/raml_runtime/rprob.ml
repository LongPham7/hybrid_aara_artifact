type t = int * int

let create (a,b) = (a,b)

let flip (a,b) = Random.int (a + b) < a

let consume (_, _, _) (p : t) = p

let inv (a,b) = (b,a)

let mult (a1,b1) (a2,b2) = (* a1/(a1+b1) * a2/(a2+b2) = a1a2/(a1a2+a1b2+b1a2+b1b2) *) (a1 * a2, a1 * b2 + b1 * a2 + b1 * b2)