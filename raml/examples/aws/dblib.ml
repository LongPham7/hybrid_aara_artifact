open Lwt
open Ocsigen_http_frame
open Creds
open Pa_lwt
open Awslib

(*type returnvalue =
  [ `OK of float
  | `ERROR ]*)

let rec get_grade l =
   match l with
   | [] -> None
   | hd :: tl -> 
   match hd with
   | (name, `Number n) -> 
     if name = "Grade" then Some (float_of_int (Int64.to_int n))
   else get_grade tl
   | _ -> None

let get_item_by_key ~token ~table ~key ~range =
    get_item ~token:token ~range:(string_of_int range, `Number) table (string_of_int key, `Number) ()
     >>= function
     | consum, value -> 
   let res = get_grade value in
   Lwt.return (res)

let ctoken = 
  Lwt_main.run 
    (
     create_token "AKIAI3UAGOYVEC522JJA" "DKXIU13d4593ApFcXNdQb3bJLNEk5L7vV8OSZnpV"
      )

let db_query sid cid = 
  Lwt_main.run 
    (
     (get_item_by_key ~token:ctoken ~table:"Course" ~key:sid ~range:cid)
      )

let db_update sid cid update = 
  Lwt_main.run 
    (
     update_item ~token:ctoken ~table:"Course" ~key:(`Number (Int64.of_int sid)) ~range:(`Number (Int64.of_int cid)) (("Grade", `Number (Int64.of_int update)) :: []) ()
      )

(*let query student_sum course_sum =
       for i = 1 to student_sum do
       for j = 1 to course_sum do
       match db_query i j with
       | `OK n -> Printf.printf "(%d, %d) : %d\n" i j n
       | `ERROR -> Printf.printf "ERROR!"
       done
       done

let _ =
       query 3 3*)



