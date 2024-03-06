# Hybrid Resource-Aware ML (RaML)

Hybrid Resource-Aware ML (RaML) is a program analysis tool that takes in an
OCaml program and infers a worst-case polynomial cost bound using the technique
Hybrid AARA.

Hybrid RaML is built on RaML. The README file for the original RaML is available
[here](./README-original).

# Hybrid AARA

Hybrid AARA incorporates data-driven resource analysis (specifically Bayesian
inference) into conventional AARA, which is a type-based static resource
analysis technique. Purely static analysis (e.g., conventional AARA) examines
the source code of an input program and reasons about all possible behaviors of
the program, including the worst-case one. Static analysis guarantees the
soundness of inferred cost bounds. But it suffers incompleteness because
resource analysis is undecidable in general. On the other hand, purely
data-driven analysis runs the program on many inputs to collect runtime cost
data and then infers a cost bound by analyzing the data. Data-driven analysis
can infer a polynomial cost bound for any program. However, data-driven analysis
offers no soundness guarantees of the inferred cost bounds. By integrating these
two complementary resource analysis techniques, Hybrid AARA partially retains
their strengths while mitigating their weaknesses.

The user interacts with Hybrid AARA as follows. First, given an OCaml program,
the user annotates the code to specify which code fragment is to be analyzed by
static analysis or data-driven analysis. Second, the user provides a collection
of inputs that are used to generate runtime cost data. Third, Hybrid AARA runs
the program on these inputs, recording the inputs, outputs, and costs of the
code fragments to be analyzed statistically (by data-driven analysis). Finally,
Hybrid AARA performs static analysis and data-driven analysis on different code
fragments of the input OCaml program, combining their inference results into an
overall cost bound for the entire program.

# Installation

We assume the user has Ubuntu 22.04 LTS.

## Requirements

The original RaML requires:
- OCaml compiler version 4.06.0
- Findlib OCaml library manager (ocamlfind):
  http://projects.camlcity.org/projects/findlib.html
- Jane Street's Core library: https://github.com/janestreet/core
- Coin-or Linear Programming solver (CLP): https://github.com/coin-or/Clp

In addition, Hybrid RaML requires:
- Various OCaml libraries: pyml (for OCaml-Python binding), owl (for scientific
  computing), yojson (for JSON), ctypes (for OCaml-C++ binding), and
  ctypes-foreign (for OCaml-C++ binding)
- C++ compiler gcc
- CMake
- C++ library volesti for reflective HMC: https://github.com/GeomScale/volesti
- C++ code for the interface between volesti and Hybrid RaML:
  https://github.com/LongPham7/volesti_raml_interface
- Python
- Pystan: https://pystan.readthedocs.io/en/latest/index.html

## Instructions for the requirements of the original RaML

We install the packages required by the original RaML as follows.

1. Install the OCaml package manager opam by first running
```
sudo apt install -y opam
```

2. initialize opam by running
```
opam init
eval $(opam env)
```

3. Install a compiler for OCaml version 4.06.0 by running
```
opam switch create 4.06.0
opam switch 4.06.0
eval $(opam env)
```

4. Install ocamlfind and core by running
```
opam install ocamlbuild.0.14.0 core.v0.11.3
```

5. To install CLP, first create a directory for CLP on your computer. For
   example, create a new directory on Desktop by running
```
cd /home/<user_name>/Desktop
mkdir clp
```
where `<user_name>` is your username. Next, inside the newly created directory
`/home/<user_name>/Desktop/clp`, download and extract a prebuilt release of CLP
available on its GitHub page. For example, as of March 2024, the latest release
of CLP has version 1.17.9. To download and extract the release for Ubuntu 22.04,
run
```
cd /home/<user_name>/Desktop/clp
wget https://github.com/coin-or/Clp/releases/download/releases%2F1.17.9/Clp-releases.1.17.9-x86_64-ubuntu22-gcc1140.tar.gz
tar xvzf Clp-releases.1.17.9-x86_64-ubuntu22-gcc1140.tar.gz
```

## Instructions for installing volesti and its interface with Hybrid RaML

We now install volesti and its interface with Hybrid RaML as follows.

1. Before installing volesti, we install its dependencies. First, install the
   library lp-solve by running:
```
sudo apt install -y lp-solve
```

2. Install Intel's OneAPI
   (https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)
   MKL by running
```
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
sudo apt update
sudo apt install -y intel-oneapi-mkl-devel
```

3. Install more libraries (BLAS, Eigen, and Boost) by running
```
sudo apt install -y libblas-dev
sudo apt install -y libeigen3-dev
sudo apt install -y libboost-dev
```

4. Clone volesti from GitHub by running
```
cd /home/<user_name>/Desktop
git clone https://github.com/GeomScale/volesti
```

5. Clone volesti's interface with Hybrid RaML from GitHub by running
```
cd /home/<user_name>/Desktop
git clone https://github.com/LongPham7/volesti_raml_interface
```

6. In the file `volesti_raml_interface/CMakeLists.txt`, set the variable
`VOLESTIROOT` to the correct location of volesti (from Step 5). Specifically, we
should set
```
set(VOLESTIROOT /home/<user_name>/Desktop/volesti)
```
on line 15 of the file `volesti_raml_interface/CMakeLists.txt`. In the version
of `volesti_raml_interface` hosted on GitHub, `<user_name>` is set to my own
username `longpham`, and this should be replaced with your username.

7. Build the volesti-Hybrid RaML interface by running
```
cd /home/<user_name>/Desktop/volesti_raml_interface
mkdir build && cd build
cmake ..
cmake --build .
```

## Instructions for the requirements of Hybrid RaML

We next install the additional packages required by Hybrid RaML as follows.

1. Install various OCaml packages (and their dependencies) by running
```
sudo apt install -y autoconf liblapacke-dev libopenblas-dev libplplot-dev libshp-dev
sudo opam install -y pyml owl yojson
sudo apt install -y libffi-dev
sudo opam install -y ctypes ctypes-foreign
```

2. Clone Hybrid RaML (i.e., the hybrid_aara branch of the forked RaML GitHub
    repository) from GitHub by running
```
cd /home/<user_name>/Desktop
git clone https://github.com/LongPham7/raml/tree/hybrid_aara
```

3. Inside `hybrid_aara`, specify the locations of volesti-related libraries. Go
to the file `hybrid_aara/myocamlbuild.ml`. Change every occurrence of
`/home/longpham/Desktop` in `myocamlbuild.ml` with `/home/<user_name>/Desktop`,
where `<user_name>` is your username.

4. Specify the location of CLP for `hybrid_aara`. Run
```
cd /home/<user_name>/Desktop/hybrid_aara
./configure --with-coin-clp /home/<user_name>/Desktop/clp
```

5. Set the environment variable `LD_LIBRARY_PATH` to `/home/hybrid_aara/Clp/lib`
   permanently. This can be done by inserting the following line to the file
   `~/.bashrc`.
```
export LD_LIBRARY_PATH=/home/<user_name>/Desktop/clp/dist/lib
```
After changing the file `~/.bashrc`, refresh the environment variables in the
current shell session by running `source ~/.bashrc`.

6. Build Hybrid AARA by running
```
cd /home/<user_name>/Desktop/hybrid_aara
make
```

7. To test if Hybrid AARA works properly, run
```
cd /home/<user_name>/Desktop/hybrid_aara
./main usage
```
This should display a list of commands supported by (the original and Hybrid)
RaML.

# Usage

Hybrid AARA offers two commands: `generate` and `stat_analyze`. The command
`generate` generates a runtime cost dataset and displays it on the standard
output. The command `stat_analyze` generates a runtime cost dataset and then
performs Hybrid AARA.

## Example OCaml program

For illustration, consider an OCaml file `append_hybrid.raml` (inside the
directory `/home/<user_name>/Desktop/benchmark`) with the following content:
```ocaml
let incur_cost (hd : int) =
  let modulo = 5 in
  if (hd mod 100) = 0 then Raml.tick 1.0
  else (if (hd mod modulo) = 1 then Raml.tick 0.85
        else (if (hd mod modulo) = 2 then Raml.tick 0.65 else Raml.tick 0.5))

let step_function (x : int) (xs : int list) (ys : int list) =
  let _ = incur_cost x in (xs, ys)

let rec append (xs : int list) (ys : int list) =
  match xs with
  | [] -> ys
  | hd :: tl ->
    let rec_xs, rec_ys = Raml.stat (step_function hd tl ys) in
    hd :: append rec_xs rec_ys

let rec map list f = match list with [] -> [] | x :: xs -> f x :: map xs f

;;

let input_dataset = [
   ([31; 11; 26; 25], [10; 36; 20; 24]);
   ([26; 34; 1; 25], [3; 40; 14; 22]);
   ([5; 32; 22; 11], [40; 20; 32; 28; 13; 18]);
   ([10; 7; 1; 16], [39; 27; 12; 17; 20; 30]);
   ([3; 17; 21; 39], [38; 12; 11; 21; 14; 7; 12; 17; 34]);
   ([30; 26; 22; 32], [35; 36; 30; 36; 5; 9; 26; 15; 17]);
   ([15; 28; 21; 9], [23; 12; 27; 32; 31; 12; 37; 36; 24; 31; 40; 35; 25]);
]
in map input_dataset (fun (x, y) -> append x y)
```

This file contains the function `append` that takes in two input lists and
appends the first one to the second one. Every time we process an element in the
first input list, we invoke the function `incur_cost`, which has a cost ranging
from 0.5 to 1.0. In (the original and Hybrid) RaML, the cost is denoted by the
function `tick q` for some number `q`. The worst-case cost bound of the function
`append` is therefore equal to the length of the first input list.

Inside the function `append`, it contains `Raml.stat (step_function hd tl ys)`.
The annotation `Raml.stat (e)` for expression `e` means we analyze the resource
usage of expression `e` statistically (i.e., by data-driven analysis). The rest
of the source code will be analyzed by static analysis (i.e., conventional
AARA).

The file also contains a list of inputs after `;;`. Any expression after `;;` in
the file is evaluated, and during its evaluation, we collect runtime cost
measurements of `Raml.stat(e)`. The runtime cost dataset is a finite set
triples: input, output, and cost.

To generate a runtime cost dataset and display it, run
```
cd /home/<user_name>/Desktop/benchmark
/home/<user_name>/Desktop/raml/main generate append_hybrid.raml
```
Here, the file `/home/<user_name>/Desktop/raml/main` is an executable file of
Hybrid RaML.

## Perform Hybrid AARA

### Hybrid Opt

First, we will consider Hybrid Opt (i.e., conventional AARA + optimization).
Create a file `config_opt.json` (inside the directory
`/home/<user_name>/Desktop/benchmark`) with the following content:
```json
{
  "mode": "opt",
  "lp_objective": {
    "cost_gaps_optimization": true,
    "coefficients_optimization": "Equal_weights"
  }
}
```

To run Hybrid Opt, run
```
cd /home/<user_name>/Desktop/benchmark
/home/<user_name>/Desktop/raml/main stat_analyze ticks 1 -m append_hybrid.raml append -config config_opt.json
```

In the command line, `ticks` means we consider the tick metric (instead of heap
or evaluation steps). The degree `1` means we care about degree-one polynomial
cost bounds. The flag `-m` means we analyze the input file in the module mode
(i.e., we analyze the cost bounds of individual functions in the file, instead
of the cost bound of the expression after `;;`). The input `append_hybrid.raml`
is the input file. The input `append` specifies the function whose cost bound we
want to infer. Finally, `-config config_opt.json` means we use the file
`config_opt.json` to set the configuration for Hybrid AARA.

Additionally, if we want to store the output of the inference, we can extend the
above command with `-o output_file`, where `output_file` is the pathname of the
file storing the inference result.

The inferred cost bound for the above example is `0.85 * M`, where `M` is the
length of the first input list. Thus, Hybrid Opt fails to infer the correct
worst-case bound. This is because the worst-case behavior of the function
`append` does not arise in the finite dataset provided to Hybrid AARA.

## Hybrid BayesWC

To perform Hybrid BayesWC (i.e., conventional AARA + BayesWC), create a
configuration file `config_bayeswc.json` with the following content:
```json
{
  "mode": "bayeswc",
  "stan_params": {
    "scale_beta": 5.0,
    "scale_s": 5.0,
    "num_chains": 2,
    "num_stan_samples": 500
  },
  "lp_objective": {
    "cost_gaps_optimization": true,
    "coefficients_optimization": "Equal_weights"
  }
}
```

To perform Hybrid BayesWC, we run the same command as Hybrid Opt, except that we
replace `config_opt.json` with `config_bayeswc.json`.

## Hybrid BayesPC

To perform Hybrid BayesPC, create a configuration file `config_bayespc.json` with
the following content:
```json
{
  "mode": "bayespc",
  "lp_params": {
    "box_constraint": {
      "upper_bound": 10.0
    },
    "implicit_equality_removal": false,
    "output_potential_set_to_zero": true
  },
  "warmup_params": {
    "algorithm": "Gaussian_rdhr",
    "variance": 36.0,
    "num_samples": 100,
    "walk_length": 100
  },
  "hmc_params": {
    "coefficient_distribution_with_target": {
      "distribution": {
        "distribution_type": "Gaussian",
        "mu": 0.0,
        "sigma": 1.0
      },
      "target": "Individual_coefficients"
    },
    "cost_model_with_target": {
      "distribution": {
        "distribution_type": "Weibull",
        "alpha": 1.0,
        "sigma": 15.0
      },
      "target": "Individual_coefficients"
    },
    "num_samples": 300,
    "walk_length": 300,
    "step_size": 0.0007
  }
}
```

# Structure of the Hybrid RaML project directory

The extra source files added by Hybrid RaML (compared to the original RaML)
reside in the following directories: `automatic_differentiation`, `clp`,
`hybrid_aara`, `probabilistic_programming`, and `volesti`. If you want to
understand how Hybrid RaML extended the original RaML, you can take a look at
the files in these directories.
