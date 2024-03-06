(** polynomial.ml *)

open Core


module Monom =
struct
  type t = int String.Map.t [@@deriving sexp, compare]

  let equal m1 m2 = compare m1 m2 = 0

  let is_one = String.Map.is_empty

  let fold m ~init ~f = String.Map.fold m ~init ~f:(fun ~key:var ~data:expo acc -> f ~var ~expo acc)

  let one = String.Map.empty

  let degree = fold ~init:0 ~f:(fun ~var:_ ~expo deg -> deg + expo)

  let var_exists p = String.Map.existsi ~f:(fun ~key:var ~data:_ -> p var)

  let var_forall p = String.Map.for_alli ~f:(fun ~key:var ~data:_ -> p var)

  let of_var ?(expo=1) x = String.Map.singleton x expo

  let get_pow x m = String.Map.find m x |> Option.value ~default:0

  let pow e = String.Map.map ~f:(( * ) e)

  let mul_var ?(expo=1) x m = if expo = 0 then m else String.Map.set m ~key:x ~data:(expo + get_pow x m)

  let mul =
    String.Map.merge ~f:(fun ~key:_ -> function
        | `Both (e1, e2) -> Some (e1 + e2)
        | `Left e1 -> Some e1
        | `Right e2 -> Some e2)

  let print fmt m =
    let superscript n = if n = 1 then "" else "^" ^ string_of_int n in
    Format.fprintf fmt "@[<h>";
    let is_one = fold m ~init:true ~f:(fun ~var ~expo first ->
        if expo = 0 then first else
          begin
            Format.fprintf fmt (if first then "%a%s" else " %a%s") String.pp var (superscript expo);
            false
          end)
    in
    if is_one then Format.fprintf fmt "1";
    Format.fprintf fmt "@]"

  let split_on vars m =
    fold m ~init:(one, one) ~f:(fun ~var ~expo (acc, other) ->
        if List.mem vars var ~equal:String.equal then
          mul_var ~expo var acc, other
        else
          acc, mul_var ~expo var other)
end

module MonomMap = Map.Make(struct type t = Monom.t [@@deriving sexp, compare] end)

module Poly =
struct
  type 'a t = 'a MonomMap.t [@@deriving sexp, compare]

  let zero = MonomMap.empty

  let fold p ~init ~f = MonomMap.fold p ~init ~f:(fun ~key:mono ~data:coef acc -> f ~mono ~coef acc)

  let degree p = fold p ~init:0 ~f:(fun ~mono ~coef:_ acc -> max acc (Monom.degree mono))

  let var_exists p = MonomMap.existsi ~f:(fun ~key:mono ~data:_ -> Monom.var_exists p mono)

  let monom_exists m p = MonomMap.mem p m

  let opt_iszero opt c = Option.value_map opt ~default:false ~f:(fun chk -> chk c)

  let of_monom ?is_zero m c = if opt_iszero is_zero c then zero else MonomMap.singleton m c

  let const ?is_zero c =
    if opt_iszero is_zero c then zero
    else MonomMap.singleton Monom.one c

  let get_coeff ~zer m p = MonomMap.find p m |> Option.value ~default:zer

  let is_const ~zer p =
    if MonomMap.is_empty p then Some zer
    else if MonomMap.length p <> 1 then None
    else MonomMap.find p Monom.one

  let scale ?is_zero ~mult scal p = MonomMap.filter_map p ~f:(fun c -> let c' = mult scal c in if opt_iszero is_zero c' then None else Some c')

  let add_monom ?is_zero ~addi m c p =
    let c' =
      match MonomMap.find p m with
      | None -> c
      | Some c_old -> addi c c_old
    in
    if opt_iszero is_zero c' then MonomMap.remove p m
    else MonomMap.set p ~key:m ~data:c'

  let add_scale ?is_zero ~addi ~mult scal =
    MonomMap.merge ~f:(fun ~key:_ -> function
        | `Both (c1, c2) ->
          let v = addi (mult scal c1) c2 in
          if opt_iszero is_zero v then None else Some v
        | `Left c1 ->
          let v = mult scal c1 in
          if opt_iszero is_zero v then None else Some v
        | `Right c2 -> Some c2)

  let add ?is_zero ~addi =
    MonomMap.merge ~f:(fun ~key:_ -> function
        | `Both (c1, c2) ->
          let c' = addi c1 c2 in
          if opt_iszero is_zero c' then None else Some c'
        | `Left c1 -> Some c1
        | `Right c2 -> Some c2)

  let sub ?is_zero ~subt ~zer =
    MonomMap.merge ~f:(fun ~key:_ -> function
        | `Both (c1, c2) ->
          let c' = subt c1 c2 in
          if opt_iszero is_zero c' then None else Some c'
        | `Left c1 -> Some c1
        | `Right c2 -> Some (subt zer c2))

  let mul_var ?(expo=1) x p = fold p ~init:zero ~f:(fun ~mono ~coef acc ->
      MonomMap.set acc ~key:(Monom.mul_var x ~expo mono) ~data:coef)

  let mul_monom ?is_zero ~mult m c p =
    if opt_iszero is_zero c then zero
    else fold p ~init:zero ~f:(fun ~mono ~coef acc ->
        let c' = mult coef c in
        if opt_iszero is_zero c' then acc else MonomMap.set acc ~key:(Monom.mul mono m) ~data:c')

  let mul ?is_zero ~addi ~mult p1 p2 = fold p1 ~init:zero ~f:(fun ~mono ~coef acc -> add ?is_zero ~addi acc (mul_monom ?is_zero ~mult mono coef p2))

  let rec pow ?is_zero ~addi ~mult ~one n p =
    if n = 0 then const one else mul ?is_zero ~addi ~mult p (pow ?is_zero ~addi ~mult ~one (n - 1) p)

  let print pp_coeff fmt p =
    Format.fprintf fmt "@[<hov>";
    Format.fprintf fmt "[";
    let _ : bool = fold p ~init:true ~f:(fun ~mono ~coef first ->
        if not first then Format.fprintf fmt "; ";
        Format.fprintf fmt "(";
        pp_coeff fmt coef;
        Format.fprintf fmt ", ";
        Format.fprintf fmt "@[<h>%a@]" Monom.print mono;
        Format.fprintf fmt ")"; false) in
    Format.fprintf fmt "]";
    Format.fprintf fmt "@]"

  let map ?is_zero f p = MonomMap.filter_map p ~f:(fun c -> let c' = f c in if opt_iszero is_zero c' then None else Some c')

  let void_merge p1 p2 ~f =
    let _ : 'a t = MonomMap.merge p1 p2 ~f:(fun ~key:mono op -> f ~mono op; None) in ()
end

module FloatPoly =
struct
  type t = float Poly.t [@@deriving sexp, compare]

  let zer = 0.0
  let one = 1.0
  let is_zero f = Float.(abs f < 1e-6)
  let addi = (+.)
  let mult = ( *. )
  let subt = (-.)

  let zero = Poly.zero
  let const = Poly.const ~is_zero
  let of_monom = Poly.of_monom ~is_zero
  let degree = Poly.degree
  let var_exists = Poly.var_exists
  let fold = Poly.fold
  let is_const = Poly.is_const ~zer
  let get_coeff = Poly.get_coeff ~zer
  let scale = Poly.scale ~is_zero ~mult
  let mul_monom = Poly.mul_monom ~is_zero ~mult
  let add_monom = Poly.add_monom ~is_zero ~addi
  let add_scale = Poly.add_scale ~is_zero ~addi ~mult
  let add = Poly.add ~is_zero ~addi
  let sub = Poly.sub ~is_zero ~subt ~zer
  let mul = Poly.mul ~is_zero ~addi ~mult
  let pow = Poly.pow ~is_zero ~addi ~mult ~one
  let mul_var = Poly.mul_var

  let print fmt p =
    Format.fprintf fmt "@[<hov>";
    let is_zero = fold p ~init:true ~f:(fun ~mono ~coef first ->
        let pref, flt =
          if Float.(coef < 0.) then "-", (-. coef) else (if first then "" else "+"), coef
        in
        if Monom.is_one mono then
          Format.fprintf fmt (if first then "%s%.10f" else "@ %s %.10f") pref flt
        else if Float.(abs (1. -. abs coef) < 1e-6) then
          Format.fprintf fmt (if first then "%s%a" else "@ %s %a") pref Monom.print mono
        else
          Format.fprintf fmt (if first then "%s@[<h>%.10f %a@]" else "@ %s @[<h>%.10f %a@]") pref flt Monom.print mono;
        false) in
    if is_zero then Format.fprintf fmt "0";
    Format.fprintf fmt "@]"

  let mul_fmp a b =
    match a, b with
    | `Poly pa, `Poly pb -> `Poly (mul pa pb)
    | `Poly pa, `Monom (mb, kb) -> `Poly (mul_monom mb kb pa)
    | `Monom (ma, ka), `Poly pb -> `Poly (mul_monom ma ka pb)
    | `Monom (ma, ka), `Monom (mb, kb) -> `Monom (Monom.mul ma mb, ka *. kb)

  let normalize = function
    | `Poly p -> p
    | `Monom (m, k) -> of_monom m k

  let monom_subst x p m =
    Monom.fold m ~init:(`Monom (Monom.one, 1.0)) ~f:(fun ~var:y ~expo acc ->
        let fe =
          if String.equal x y then `Poly (pow expo p)
          else `Monom (Monom.of_var ~expo y, 1.0)
        in mul_fmp fe acc) |> normalize

  let monom_subst_expo e x p m =
    Monom.fold m ~init:(`Monom (Monom.one, 1.0)) ~f:(fun ~var:y ~expo acc ->
        let fe =
          if e = expo && String.equal x y then `Poly p
          else `Monom (Monom.of_var ~expo y, 1.0)
        in mul_fmp fe acc) |> normalize
end
