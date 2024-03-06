open Ocamlbuild_plugin
open Command

let clp_inc_dir = Sys.getenv "INCDIRS"
let clp_lib_dir = Sys.getenv "LIBDIRS"

let clp_stubs = Filename.concat "clp" "clp_stubs.o"

let flag_clp_include = [A "-ccopt"; A ("-O2" ^ " -Wall" ^ " -I" ^ clp_inc_dir)]
let flag_clp_link =
  [A clp_stubs] @
  [A "-cclib"; A ("-L" ^ clp_lib_dir ^ " -lClp" ^ " -lCoinUtils")]

(* lpsolve library used in volesti *)
let lpsolve_inc_dir = "/home/hybrid_aara/volesti/external/_deps/lpsolve-src"
let lpsolve_lib_dir = "/usr/lib/lp_solve"

(* Eigen library used in volesti *)
let eigen_inc_dir = "/usr/include/eigen3"

(* Intel's MKL library used in volesti *)
let mkl_root = "/opt/intel/oneapi/mkl/latest"
let mkl_inc_dir = Filename.concat mkl_root "include"
let mkl_lib_dir = Filename.concat mkl_root "lib/intel64"

(* Custom-made C++ library for invoking sampling algorithms (i.e. reflective HMC
and hit-and-run samplers) from volesti. This library can be called from C, so it
acts as an interface between volesti and OCaml (which has an FFI for C). *)
let my_volesti_inc_dir = "/home/hybrid_aara/volesti_raml_interface/sampling_algorithms"
let my_volesti_lib_dir = "/home/hybrid_aara/volesti_raml_interface/build/sampling_algorithms"

(* Custom-made C++ library for invoking automatic differentiation from autodiff,
which is a C++ library for automatic differentiation. This library can be called
from C, so it acts as an interface between autodiff and OCaml (which has an FFI
for C). *)
let my_autodiff_inc_dir = "/home/hybrid_aara/volesti_raml_interface/automatic_differentiation"
let my_autodiff_lib_dir = "/home/hybrid_aara/volesti_raml_interface/build/automatic_differentiation"

(* C stub file for volesti's conventional interface. I have implemented two
interfaces for volesti using different OCaml-C bindings. The first one uses the
conventional OCaml-C binding that requires C stub functions. The second one uses
OCaml's Ctypes library. The first interface does not allow us to pass functions
from OCaml to C as function pointers, while the second interface does. To keep
the source code clean, I have removed the source files for the conventional
OCaml-C binding. But if you are interested in this interface, you can dig into
the Git history. *)
(* let volesti_conventional_c_binding_stubs = Filename.concat "volesti" "volesti_conventional_c_binding_stubs.o" *)
let flag_volesti_include = 
  [A "-ccopt"; A ("-O2" ^ " -Wall" ^ " -I" ^ my_volesti_inc_dir ^ " -I" ^ my_autodiff_inc_dir ^ " -I" ^ lpsolve_inc_dir ^ " -I" ^ mkl_inc_dir ^ " -I" ^ eigen_inc_dir)]
let flag_volesti_link =
  (* [A volesti_conventional_c_binding_stubs] @ *)
  [
  A "-cclib"; A ("-L" ^ my_volesti_lib_dir ^ " -Wl,-rpath=" ^ my_volesti_lib_dir ^ " -llogconcave_hmc -lhit_and_run");
  A "-cclib"; A ("-L" ^ my_autodiff_lib_dir ^ " -Wl,-rpath=" ^ my_autodiff_lib_dir ^ " -lruntime_data");
  A "-cclib"; A ("-L" ^ lpsolve_lib_dir ^ " -Wl,-rpath=" ^ lpsolve_lib_dir ^ " -llpsolve55");
  A "-cclib"; A ("-L" ^ mkl_lib_dir ^ " -Wl,-rpath=" ^ mkl_lib_dir ^ " -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl");
  A "-cclib"; A ("-lstdc++")
  ]

let () =
  dispatch begin function
    | After_rules ->
      dep ["compile"; "ocaml"; "use_clp"] [clp_stubs];
      (* dep ["compile"; "ocaml"; "use_volesti"] [volesti_conventional_c_binding_stubs]; *)

      flag ["compile"; "c"; "use_clp"] (S flag_clp_include);
      flag ["compile"; "c"; "use_volesti"] (S flag_volesti_include);
      flag ["link"; "ocaml"; "native"; "use_clp"] (S flag_clp_link);
      flag ["link"; "ocaml"; "native"; "use_volesti"] (S flag_volesti_link);
      flag ["link"; "ocaml"; "byte"; "use_clp"] (S (A "-custom" :: flag_clp_link));
      flag ["link"; "ocaml"; "byte"; "use_volesti"] (S (A "-custom" :: flag_volesti_link));

    | _ -> ()
  end
