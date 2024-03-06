open Lwt
open Awslib
open Dblib

(*let query student_sum course_sum =
       for i = 1 to student_sum do
       for j = 1 to course_sum do
       match db_query i j with
       | `OK n -> Printf.printf "(%d, %d) : %d\n" i j n
       | `ERROR -> Printf.printf "ERROR!"
       done
       done


let _ =
       db_update 4 1 89

let _ =
       query 3 3*)

exception Not_found of int * int

type 'a option = None | Some of 'a

(*let db_query student_id course_id =
  Raml.tick(1.0);  Some 1.0*)

let rec append l1 l2 =
  match l1 with
    | [] -> l2
    | x::xs -> x::(append xs l2)

let rec partition gt acc l =
  match l with
    | [] -> acc
    | x::xs ->
      let (cs,bs,acc) = acc in
      let (is_greater,acc') = gt x acc in
      if is_greater then
  	partition gt (cs,x::bs,acc') xs
      else
  	partition gt (x::cs,bs,acc') xs

let rec quicksort gt acc l = match l with
  | [] -> ([],acc)
  | x::xs ->
    let ys, zs, acc' = partition (gt x) ([],[],acc) xs in
    let (l1,acc'') = quicksort gt acc' ys in
    let (l2,acc''') = quicksort gt acc'' zs in
    (append  l1 (x :: l2), acc''')


let rec foldl f acc l =
  match l with
    | [] -> acc
    | x::xs -> foldl f (f acc x) xs

let average_grade student_id course_ids =
  let f acc cid =
    let (length,sum) = acc in
    let grade = match db_query student_id cid with
      | Some q -> q
      | None -> raise (Not_found (student_id,cid))
    in
    (length +. 1.0, sum +. grade)
  in
  let (length,sum) = foldl f (0.0,0.0) course_ids in
  sum /. length

let greater_eq sid1 sid2 course_ids =
  (average_grade sid1 course_ids >= average_grade sid2 course_ids, course_ids)

let sort_students student_ids course_ids =
  let (sorted_sids, acc) = quicksort greater_eq course_ids student_ids in
  sorted_sids


let rec map f l =
  match l with
    | [] -> []
    | x::xs -> (f x) :: (map f xs)

let make_table student_ids course_ids =

  let rec mk_table sids cids =
    match sids with
      | [] -> []
      | x::xs ->
	let cid_map =
	  let f cid =
	    let grade =
	      match db_query x cid with
	  	| Some q -> q
	  	| None -> raise (Not_found (x,cid))
	    in
	    (cid,grade)
	  in
	  map f cids
	in
	(x,cid_map)::(mk_table xs cids)
  in
  mk_table student_ids course_ids


let rec find f l =
  match l with
    | [] -> raise (Not_found (-1,-1))
    | x::xs ->
      let (key,value) = x in
      if f key then
	value
      else
	find f xs

let lookup sid cid table =
  let cid_map = find (fun (id:int) -> id = sid) table in
  find (fun (id:int) -> id = cid) cid_map


let average_grade' student_id course_ids table =
  let f acc cid =
    let (length,sum,table) = acc in
    let grade = lookup student_id cid table in
    (length +. 1.0, sum +. grade, table)
  in
  let (length,sum,table') = foldl f (0.0,0.0,table) course_ids in
  (sum /. length,table')


let greater_eq' course_ids sid1 sid2 table =
  let (grade1, table1) = average_grade' sid1 course_ids table in
  let (grade2, table2) = average_grade' sid2 course_ids table1 in
  (grade1 >= grade2, table2)


let sort_students_efficient student_ids course_ids =
  let (sorted_sids, acc) = quicksort (greater_eq' course_ids)  (make_table student_ids course_ids) student_ids in
  sorted_sids


let make_table_test student_ids course_ids =
  let cids = course_ids in
  let rec mk_table sids cids =
    match sids with
      | [] -> []
      | x::xs ->
	(cids)::(mk_table xs cids)
  in
  mk_table student_ids cids

let _ = let g = (average_grade 2 [2;1;3]) in
        Printf.printf "Ave grade: %f\n" g                       
    
let _ = let l = (make_table [2; 1; 3] [2;1;3]) in
         List.iter (fun (sid, l') ->   
                       Printf.printf "start %d\n" sid;
                       List.iter (fun (cid, grade) ->   
                                 Printf.printf "(cid: %d, grade: %f) " cid grade) l';
                       Printf.printf "end %d\n" sid                       
                       ) l

let _ = let l = (sort_students [2; 1; 3] [2;3;1]) in
        List.iter (fun x ->   
                       Printf.printf "%d \n" x;
                       ) l
  
(*let _ = let l = (sort_students_efficient [1;2;3] [2;1;3]) in
        List.iter (fun l' ->   
                       Printf.printf "start\n";
                       List.iter (fun x ->   
                                 Printf.printf "%d " x) l';
                       Printf.printf "end\n"                       
                       ) l*)



