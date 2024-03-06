(*try Topdirs.dir_directory (Sys.getenv "OCAML_TOPLEVEL_PATH") with _ -> ();;
#use "topfind";;
#thread;;
#camlp4o;;
#require "aws";;
#require "aws.ocsigen";;*)

open Lwt
open Ocsigen_http_frame
open Creds
open Pa_lwt

type headers = (string * string) list
type request_body =  [ `InChannel of int * Lwt_io.input_channel
                     | `None
                     | `String of string ]

exception Http_error of (int * headers * string)

let extract_headers frame =
  let headers = Ocsigen_http_frame.Http_header.get_headers frame.frame_header in
  Http_headers.fold (fun s l acc ->
    let s = Http_headers.name_to_string s in
    (List.map (fun v -> (s,v)) l) @ acc) headers []

let extract_content frame =
  let header = extract_headers frame in
  match frame.frame_content with
  | None -> return (header, "")
  | Some content ->
    let st = Ocsigen_stream.get content in
    Ocsigen_stream.string_of_stream 15000 st >>= fun s ->
    return (header, s)

let extract_content_to_chan chan frame =
  let header = extract_headers frame  in
  match frame.frame_content with
    None -> return header
  | Some content ->
    let st = Ocsigen_stream.get content in
    let rec loop st =
      Ocsigen_stream.next st >>= fun step ->
      match step with
      | Ocsigen_stream.Cont (s,st) -> Lwt_io.write chan s >>= fun () -> loop st
      | Ocsigen_stream.Finished None -> Lwt.return_unit
      | Ocsigen_stream.Finished (Some st) -> loop st
    in
    loop st >>= fun () -> return header

let call ?(headers=[]) ?(body=`None) ~http_method url =
  let (https, host, port, uri, _, _, _) = Ocsigen_lib.Url.parse url in
  let uri = match uri with (* WHY *)
    | "" -> "/"
    | s when s.[0] <> '/' -> "/"^s
    | s -> s in
  let host = match host with None -> "localhost" | Some h -> h in
  Ocsigen_lib.Ip_address.get_inet_addr host >>= fun inet_addr ->
  let content = match body with
    | `None -> Lwt.return_none
    | `String s -> Lwt.return (Some (Ocsigen_stream.of_string s))
    | `InChannel (count,chan) ->
      let rec read c () =
        Lwt_io.read ~count:(min 1024 c) chan >>= fun s ->
        match String.length s with
        | 0 -> Ocsigen_stream.empty None
        | l ->
          let c = c-l in
          if c < 0
          then Ocsigen_stream.cont s (fun () -> Ocsigen_stream.empty None)
          else Ocsigen_stream.cont s (read c)
      in
      return (Some (Ocsigen_stream.make (read count)))
  in
  let content_length = match body with
    | `None -> None
    | `String s -> Some (Int64.of_int (String.length s))
    | `InChannel (count,_) -> Some (Int64.of_int count) in
  let headers = match content_length with
    | Some l ->
      let headers = ("Content-Length",Int64.to_string l)::(List.remove_assoc "Content-Length" headers) in
      if List.mem_assoc "Content-Type" headers
      then headers
      else ("Content-Type","application/x-www-form-urlencoded")::headers
    | _ -> headers in
  let headers =
    List.fold_left
      (fun h (n,v) ->
        Http_headers.add (Http_headers.name n) v h)
      Http_headers.empty headers
    in
  content >>= fun content ->
  Ocsigen_http_client.raw_request
    ?https
    ?port
    ~http_method:http_method
    ~content
    ?content_length
    ~headers
    ~host:(match port with None -> host | Some p -> host^":"^string_of_int p)
    ~inet_addr
    ~uri
    ()
    ()

let get ?headers url =
  call ?headers ~http_method:Ocsigen_http_frame.Http_header.GET url
  >>= extract_content

let get_to_chan ?headers url chan =
  call ?headers ~http_method:Ocsigen_http_frame.Http_header.GET url
  >>= extract_content_to_chan chan

let post ?headers ?(body=`None) url =
  call ?headers ~body ~http_method:Ocsigen_http_frame.Http_header.POST url
  >>= extract_content

let put ?headers ?(body=`None) url =
  call ?headers ~body ~http_method:Ocsigen_http_frame.Http_header.PUT url
  >>= extract_content

let delete ?headers url =
  call ?headers ~http_method:Ocsigen_http_frame.Http_header.DELETE url
  >>= extract_content

let head ?headers url =
  call ?headers ~http_method:Ocsigen_http_frame.Http_header.HEAD url
  >>= extract_content

module Util = Aws_util

module C = CalendarLib.Calendar
module P = CalendarLib.Printer.CalendarPrinter

exception Service_down

(* Miscellaneous *****************************************************************)

let remove_newline =
  Pcre.replace ~rex:(Pcre.regexp "\n") ~templ:""

let base64 str =
  (* the encoder is consumed by its use, so we have to recreated *)
  let b64_encoder = Cryptokit.Base64.encode_multiline () in
  let encoded = Cryptokit.transform_string b64_encoder str in

  (* we want to retain the trailing '=' characters, but eliminate the
     newlines.  Unfortunately, [encode_compact] has neither. *)
  remove_newline encoded

let encode_url ?(safe=false) str =
(*  if not safe then
    Netencoding.Url.encode ~plus:false str
  else *)
    begin
      let strlist = ref [] in
      let code = Char.code in
      for i = 0 to String.length str - 1 do
        let c = code (str.[i]) in
        if
          (65 <= c && c <= 90)
          || (48 <= c && c <= 57 )
          || (97 <= c && c <= 122)
          || (c = 126)
          || (c = 95)
          || (c = 46)
          || (c = 45)
          (* || (c = 47) *)
        then
	  strlist := Printf.sprintf "%c" str.[i] :: !strlist
	else if (c = 32)
	then
	  strlist := Printf.sprintf "%%20" :: !strlist
        else
	  strlist :=  Printf.sprintf "%%%X" c :: !strlist
      done ;
      String.concat "" (List.rev !strlist)
    end

let encode_key_equal_value ?(safe=false) (k,v) =
  (encode_url ~safe k) ^ "=" ^ (encode_url ~safe v)

let encode_key_equals_value ?safe kvs =
  List.map (encode_key_equal_value ?safe) kvs

let sort_assoc_list kv_list =
  List.sort (fun (k1,_) (k2,_) -> String.compare k1 k2) kv_list

let host = "sts.amazonaws.com"

let sign_request aws_access_key_id aws_secret_access_key params =
  let signature =
    let sorted_params = sort_assoc_list params in
    let key_equals_value = encode_key_equals_value sorted_params in
    let uri_query_component = String.concat "&" key_equals_value in
    let string_to_sign = String.concat "\n" [
      "POST" ;
      String.lowercase host ;
      "/" ;
      uri_query_component
    ]
    in
    let hmac_sha256_encoder = Cryptokit.MAC.hmac_sha256 aws_secret_access_key in
    let signed_string = Cryptokit.hash_string hmac_sha256_encoder string_to_sign in
    let signed_string = String.sub signed_string 0 ((String.length signed_string) - 0) in
    let b64 = base64 signed_string in
    String.sub b64 0 ((String.length b64) - 0)
  in
  ("Signature", signature)::params


type node =
  | E of string * attr2 list * node list
  | P of string
and attr2 = (string * string) * string

let xml_of_string s =
  (* we drop the namespace part of the element here *)
  let el ((ns, name), atts) kids = E (name, atts, kids) in
  let data d = P d in
  let input = Xmlm.make_input ~strip:true (`String (0,s)) in
  let _, node = Xmlm.input_doc_tree ~el ~data input in
  node

exception Invalid_credential

let xml_get_content_of name xml : string option =
  let rec aux = function
    | E (n,_,[P v]) when n=name -> Some v
    | E (_,_,l) ->
      let rec loop = function
        | [] -> None
	| x::xs -> match aux x with
	    | Some x -> Some x
	    | None -> loop xs
      in loop l
    | _ -> None
  in aux xml

type session_token = {
  session_token : string;
  secret_access_key : string;
  access_key_id : string;
  expiration : int64;
}

let token_is_valid token = token.expiration > (Int64.of_float (C.to_unixfloat (C.now ())))

let get_session_token ?duration ?(version="2011-06-15") aws_access_key_id aws_secret_access_key =
  let content =
    let s (n,v)= Some (n,v) in
    let n (n,v)= Util.option_map (fun v -> (n,string_of_int v)) v in
    let now = P.sprint "%FT%T" (C.from_unixfloat (Unix.gettimeofday ())) in
    s("AWSAccessKeyId",aws_access_key_id)
    ::s("Action","GetSessionToken")
    ::n("DurationSeconds",duration)
    ::s("SignatureMethod","HmacSHA256")
    ::s("SignatureVersion","2")
    ::s("Timestamp",now )
    ::s("Version",version)
    ::[] in
  let params = Util.filter_map (fun x -> x) content in
  let params = sign_request aws_access_key_id aws_secret_access_key params in
  let key_equals_value = encode_key_equals_value params in
  let content = String.concat "&" key_equals_value in
  (post ~headers:["Content-Type","application/x-www-form-urlencoded"] ~body:(`String content) (Printf.sprintf "https://%s/" host)
>>=
(function
| _,s ->
  let xml = xml_of_string s in
  let session_token = xml_get_content_of "SessionToken" xml
  and secret_access_key = xml_get_content_of "SecretAccessKey" xml
  and access_key_id = xml_get_content_of "AccessKeyId" xml
  and expiration = xml_get_content_of "Expiration" xml
  in match session_token,secret_access_key,access_key_id,expiration with
    | Some session_token,Some secret_access_key,Some access_key_id,Some expiration ->
      Printf.printf "Expiration: %s\n" expiration;
      let expiration = Scanf.sscanf (expiration ^ "0") "%d-%d-%dT%d:%d:%dZ%d" Util.make in
      return {session_token; secret_access_key; access_key_id; expiration}
    | _ -> raise Invalid_credential))

exception Service_down
exception Bad_response

type action =
  [ `BatchGetItem
  | `CreateTable
  | `DeleteItem
  | `DeleteTable
  | `DescribeTable
  | `GetItem
  | `ListTables
  | `PutItem
  | `Query
  | `Scan
  | `UpdateItem
  | `UpdateTable ]

let string_of_action = function
  | `BatchGetItem -> "BatchGetItem"
  | `CreateTable -> "CreateTable"
  | `DeleteItem -> "DeleteItem"
  | `DeleteTable -> "DeleteTable"
  | `DescribeTable -> "DescribeTable"
  | `GetItem -> "GetItem"
  | `ListTables -> "ListTables"
  | `PutItem -> "PutItem"
  | `Query -> "Query"
  | `Scan -> "Scan"
  | `UpdateItem -> "UpdateItem"
  | `UpdateTable -> "UpdateTable"

type attr =
  [ `Number
  | `String
  | `NumberSet
  | `StringSet ]

let string_of_attr = function
  | `Number -> "N" (*number*)
  | `NumberSet -> "NS" (*number set*)
  | `String -> "S" (*string*)
  | `StringSet -> "SS" (*string set*)

let attr_of_string = function
  | "N" -> `Number
  | "NS" -> `NumberSet
  | "S" -> `String
  | "SS" -> `StringSet
  | _ -> raise Not_found

type value =
  [ `Number of int64
  | `NumberSet of int64 list
  | `String of string
  | `StringSet of string list ]

let json_of_value = function
  | `Number n -> `Assoc ["N",`String (Int64.to_string n)]
  | `NumberSet ns -> `Assoc ["NS",`List (List.map (fun n -> `String (Int64.to_string n)) ns)]
  | `String s -> `Assoc ["S",`String s]
  | `StringSet ss -> `Assoc ["SS", `List (List.map (fun s -> `String s) ss)]

type key = string * attr

type update_action =
  [ `Put
  | `Delete
  | `Add ]

let string_of_update_action = function
  | `Put -> "PUT"
  | `Delete -> "DELETE"
  | `Add -> "ADD"

type token_state =
    Valid
  | Updating of unit Lwt.u list

type token = {
  mutable token_state: token_state;
  mutable token : session_token;
  aws_access_key_id : string;
  aws_secret_access_key : string;
}

let host2 = "dynamodb.us-east-1.amazonaws.com"

let create_token aws_access_key_id aws_secret_access_key =
  get_session_token aws_access_key_id aws_secret_access_key
>>=
(function
| token ->
  return {
    token;
    token_state=Valid;
    aws_access_key_id;
    aws_secret_access_key
  })

let get_token_value token = token.token

let make_headers ~token ?(version="20111205") action body =
  let rec loop () =
    (*if not(Iam.token_is_valid token.token)
    then
      begin
	match token.token_state with
	  | Updating l ->
	    let t,u = Lwt.wait () in
	    token.token_state <- Updating (u::l);
	    t >>= loop
	  | Valid ->
	    token.token_state <- Updating [];
	    try_lwt
	      begin
		(* debug "Dynamo: getting new session token"; *)
		lwt token_ = Iam.get_session_token token.aws_access_key_id token.aws_secret_access_key in
		let l = match token.token_state with | Updating l -> l | _ -> [] in
		token.token_state <- Valid;
		List.iter (fun x -> Lwt.wakeup x ()) l;
		loop ()
	      end
	    with _ ->
	      ((* debug "Dynamo: fail to get valid session token, retry later"; *)
	       Lwt_unix.sleep 0.1 >>= loop)
      end
    else*)
      let date = P.sprint "%a, %d %b %Y %H:%M:%S GMT" (C.now ()) in
      let target = Printf.sprintf "DynamoDB_%s.%s" version (string_of_action action) in
      let headers_amz =
	[
	  "host",host2;
	  "x-amz-date",date;
	  "x-amz-security-token",token.token.session_token;
	  "x-amz-target",target]
      in
      let cano =
	String.concat "" (List.map (fun (a,b) -> a^":"^b^"\n") headers_amz) in
      let string_to_sign =
	Printf.sprintf "POST\n/\n\n%s\n%s" cano body in
      let encoder = Cryptokit.Base64.encode_multiline () in
      let digester = Cryptokit.Hash.sha1 () in
      let hasher = Cryptokit.MAC.hmac_sha1 token.token.secret_access_key in
      let signature = Cryptokit.transform_string encoder (Cryptokit.hash_string hasher (Cryptokit.hash_string digester string_to_sign)) in
      let signature = String.sub signature 0 ((String.length signature) - 1) in
      let value = Printf.sprintf "AWS3 AWSAccessKeyId=%s,Algorithm=HmacSHA1,SignedHeaders=%s,Signature=%s"
	token.token.access_key_id (String.concat ";" (List.map fst headers_amz)) signature in
      let headers = ("x-amzn-authorization",value)::headers_amz in
      let headers = ("Content-Type","application/x-amz-json-1.0")::headers in
      return headers
 in loop ()

let post2 ~token action json =
  let content =Yojson.Safe.to_string json in
  make_headers ~token action content
>>= function
| headers ->
post ~headers ~body:(`String content) (Printf.sprintf "http://%s/" host2)
>>= function
| _,s ->
  Lwt.return (Yojson.Safe.from_string s)
>>= function
| s -> return s
(*| _ -> return (`String "")*)

let assoc_none_if_empty name :(string * Yojson.Safe.json)list -> (string * Yojson.Safe.json) option  = function
  | [] -> None
  | l -> Some (name,`Assoc l)

let list_none_if_empty name = function
  | [] -> None
  | l -> Some (name,`List l)

let filter_map_opt name f l =
  match Util.filter_map f l with
    | [] -> None
    | l -> Some (name,`Assoc l)

let string_of_json_string : Yojson.Safe.json -> string = function
  | `String s -> s
  | _ -> raise Bad_response

let string_list_of_json_list : Yojson.Safe.json -> string list = function
  | `List l -> List.map string_of_json_string l
  | _ -> raise Bad_response

let ident x = x


type table_status = [`Creating | `Active | `Deleting | `Updating ]

type table_description = {
  name : string;
  count : int;
  key :(string * attr);
  range : (string * attr) option;
  size : int;
  status : table_status;
  write_limit : int;
  read_limit : int;
  creation : int64;
}
type scan_result = {
  res_count : int;
  res_items : (string * value) list list;
  res_last : (key * key option) option;
  res_consumed : float;
  res_scanned_count : int option;
}

let table_status_of_string = function
  | "CREATING" -> `Creating
  | "ACTIVE" -> `Active
  | "DELETING" -> `Deleting
  | "UPDATING" -> `Updating
  | _ -> raise Not_found

let string_of_table_status = function
  | `Creating -> "CREATING"
  | `Active -> "ACTIVE"
  | `Deleting -> "DELETING"
  | `Updating -> "UPDATING"

let get_string f = function `String s -> f s | _ -> raise Not_found

let get_float = function `Float f -> f | _ -> raise Not_found

let get_list_of_assoc l name =
  try
    match List.assoc name l with
      | `Assoc l -> l
      | _ -> []
  with _ -> []

let get_list l name =
  try
    match List.assoc name l with
      | `List l -> l
      | _ -> []
  with _ -> []


let get_int = function `Int i -> i | _ -> raise Not_found

let json_to_key = function
  | `Assoc l -> let name = List.assoc "AttributeName" l in
		let typ = List.assoc "AttributeType" l in
		get_string ident name, (get_string attr_of_string typ)
  | _ -> raise Not_found

let table_description_of_json = function
  | `Assoc l ->
    let opt l name f default = try f (List.assoc name l) with _ -> match default with | Some x -> x | _ -> raise Not_found in

    let limits = get_list_of_assoc l "ProvisionedThroughput" in
    let schema = get_list_of_assoc l "KeySchema" in
    begin
      try
	return {
	  name  = opt l "TableName" (get_string ident) None;
	  count = opt l "ItemCount" get_int (Some 0) ;
	  size  = opt l "TableSizeBytes" get_int (Some 0);
	  write_limit = opt limits "ReadCapacityUnits" get_int (Some 0);
	  read_limit  = opt limits "WriteCapacityUnits" get_int (Some 0);
	  status = opt l "TableStatus" (get_string table_status_of_string) None;
	  creation = opt l "CreationDateTime" (get_string (fun s -> Int64.of_float (float_of_string s))) (Some 0L);
	  key = opt schema "HashKeyElement" json_to_key None;
	  range = try Some (opt schema "RangeKeyElement" json_to_key None) with _ -> None;
	}
      with _ -> fail Bad_response
    end
  | _ -> fail Bad_response

let get_string = function
  | `String s-> s
  | _ -> raise Not_found

let get_string_list = function
  | `List l -> List.map get_string l
  | _ -> raise Not_found

let value_of_json = function
  | `Assoc ["SS",l] -> `StringSet (get_string_list l)
  | `Assoc ["S",l] -> `String (get_string l)
  | `Assoc ["NS",l] -> `NumberSet (List.map Int64.of_string (get_string_list l))
  | `Assoc ["N",l] -> `Number (Int64.of_string (get_string l))
  | _ -> raise Not_found

let item_of_json = function
  | `Assoc l -> List.map (fun (name,json) -> name,value_of_json json) l
  | _ -> raise Not_found

open Yojson.Safe

(** Api **)

(** Error **)
exception Error of (string * string)
exception UnknownError of json
let error_of_json = function
  | `Assoc l as json -> (
    try
      let t = match List.assoc "__type" l with
	| `String s ->
	  begin
	    try
	      let i = String.index s '#' in
	      String.sub s (i+1) ((String.length s) - i - 1)
	    with _ -> s
	  end
	| _ -> "" in
      let m = match List.assoc "message" l with
	| `String s -> s
	| _ -> ""
      in fail (Error (t,m))
    with _ -> fail (UnknownError json))
  | json -> fail (UnknownError json)


let describe_table ~token ~table =
  post2 ~token `DescribeTable (`Assoc ["TableName",`String table])
>>= function
| json ->
  match json with
    | `Assoc ["Table",json] -> table_description_of_json json
    | _ -> error_of_json json

let create_table ~token ~table ~key ?range ~readlimit ~writelimit () =
  let make_key name (n,t) =
    name,`Assoc [
      "AttributeName",`String n;
      "AttributeType",`String (string_of_attr t)
    ] in
  let schema =
    let key = Some (make_key "HashKeyElement" key) in
    let range = Util.option_map (make_key "RangeKeyElement") range in
    filter_map_opt "KeySchema" ident [key;range] in
  let make_limit name i = Some(name,`Int i) in
  let limits =
    filter_map_opt
      "ProvisionedThroughput"
      ident
      [make_limit "ReadCapacityUnits" readlimit;
       make_limit "WriteCapacityUnits" writelimit] in
  let json = `Assoc (Util.filter_map ident [Some("TableName", `String table);schema;limits]) in
  post2 ~token `CreateTable json
>>= function
| json ->
  match json with
    | `Assoc ["TableDescription",json] -> table_description_of_json json
    | _ -> error_of_json json

let put_item ~token ~table ~item ?(expect=[]) ?(return=false) () =
  let make_item (name,v) : string * json =
    name,`Assoc [
      "Value",json_of_value v;
    ] in
  let item : (string * json) option = assoc_none_if_empty "Item" [make_item item] in
  let expect =
    List.map (fun (e,p) -> e,`Assoc (match p with
      | `Exists ->        ["Exists",`Bool true]
      | `StringValue s -> ["Value",`Assoc ["S",`String s]]
      | `IntValue s ->    ["Value",`Assoc ["I",`String s]])) expect in
  let expect = assoc_none_if_empty "Expected" expect in
  let return = if return then Some ("ReturnValues",`String "ALL_OLD") else None in
  let json = `Assoc (Util.filter_map ident [Some("TableName", `String table);item;expect;return]) in
  post2 ~token `PutItem json
  >>= function
    | `Assoc l ->
      let value = try Some (item_of_json (List.assoc "Attributes" l)) with _ -> None in
      let consum = get_float (List.assoc "ConsumedCapacityUnits" l) in
      Lwt.return (consum,value)
    | json -> error_of_json json

let get_item ~token table key ?range ?(select=[]) ?consistent () =
  let make_key name (n,t) =
    name,`Assoc [
      string_of_attr t,`String n;
    ] in
  let keys =
    let key = Some (make_key "HashKeyElement" key) in
    let range = Util.option_map (make_key "RangeKeyElement") range in
    filter_map_opt "Key" ident [key;range] in
  let get = list_none_if_empty "AttributesToGet" (List.map (fun x -> `String x) select) in
  let consistent = Util.option_map (fun b -> "ConsistentRead", `Bool b) consistent in
  let json = `Assoc (Util.filter_map ident [Some("TableName", `String table);keys;get;consistent]) in
  post2 ~token `GetItem json
>>= function
    | `Assoc l ->
      let value = item_of_json (List.assoc "Item" l) in
      let consum = get_float (List.assoc "ConsumedCapacityUnits" l) in
      Lwt.return (consum,value)
    | json -> error_of_json json


let update_item ~token ~table ~key ?range update ?(expect=[]) ?(return=false) () =
  match update with
    | [] -> Lwt.return (0.,None)
    | update ->
      let make_key name v =
	name, json_of_value v in
      let keys =
	let key = Some (make_key "HashKeyElement" key) in
	let range = Util.option_map (make_key "RangeKeyElement") range in
	filter_map_opt "Key" ident [key;range] in
      let expect =
	List.map (fun (e,p) -> e,`Assoc (match p with
	  | `Exists ->        ["Exists",`Bool true]
	  | `StringValue s -> ["Value",`Assoc ["S",`String s]]
	  | `IntValue s ->    ["Value",`Assoc ["I",`String s]])) expect in
      let expect = assoc_none_if_empty "Expected" expect in
      let return = if return then Some ("ReturnValues",`String "ALL_OLD") else None in
      let make_update (attr_name,value) =
	attr_name,
	`Assoc [
	  match value with
	    | `Delete -> "Action", `String (string_of_update_action `Delete)
	    | `Number _ | `NumberSet _ | `String _ | `StringSet _ as value -> "Value",json_of_value value;
    (* "Action",`String (string_of_update_action action); *)
	] in
      let update = assoc_none_if_empty "AttributeUpdates" (List.map make_update update) in
      let json = `Assoc (Util.filter_map ident [Some ("TableName",`String table);keys;update;expect;return]) in
      post2 ~token `UpdateItem json
         >>= function
	| `Assoc l ->
	  let value = try Some (item_of_json (List.assoc "Attributes" l)) with _ -> None in
	  let consum = get_float (List.assoc "ConsumedCapacityUnits" l) in
	  Lwt.return (consum,value)
	| json -> error_of_json json