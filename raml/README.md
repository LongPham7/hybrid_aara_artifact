# Hybrid Resource-Aware ML (RaML)

Hybrid Resource-Aware ML (RaML) is a program analysis tool that takes in an
OCaml program and infers a worst-case polynomial cost bound using the technique
Hybrid AARA.

Hybrid RaML has been developed by Long Pham and builds on RaML. The README file
for the original RaML is available in the file README-original.

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
cd /home/user_name/Desktop
mkdir clp
```
where `user_name` is your username. Next, inside the newly created directory
`/home/user_name/Desktop/clp`, download and extract a prebuilt release of CLP
available on its GitHub page. For example, as of March 2024, the latest release of CLP has version 1.17.9. To download and extract the release for Ubuntu 22.04, run
```
cd /home/user_name/Desktop/clp
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
cd /home/user_name/Desktop
git clone https://github.com/GeomScale/volesti
```

5. Clone volesti's interface with Hybrid RaML from GitHub by running
```
cd /home/user_name/Desktop
git clone https://github.com/LongPham7/volesti_raml_interface
```

6. In the file `volesti_raml_interface/CMakeLists.txt`, set the variable
`VOLESTIROOT` to the correct location of volesti (from Step 5). Specifically, we
should set
```
set(VOLESTIROOT /home/user_name/Desktop/volesti)
```
on line 15 of the file `volesti_raml_interface/CMakeLists.txt`. In the version
of `volesti_raml_interface` hosted on GitHub, `user_name` is set to my own
username `longpham`, and this should be replaced with your username.

7. Build the volesti-Hybrid RaML interface by running
```
cd /home/user_name/Desktop/volesti_raml_interface
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
cd /home/user_name/Desktop
git clone https://github.com/LongPham7/raml/tree/hybrid_aara
```

3. Inside `hybrid_aara`, specify the locations of volesti-related libraries. Go
to the file `hybrid_aara/myocamlbuild.ml`. Change every occurrence of
`/home/longpham/Desktop` in `myocamlbuild.ml` with `/home/user_name/Desktop`,
where `user_name` is your username. 

4. Specify the location of CLP for `hybrid_aara`. Run
```
cd /home/user_name/Desktop/hybrid_aara
./configure --with-coin-clp /home/user_name/Desktop/clp
```

5. Set the environment variable `LD_LIBRARY_PATH` to `/home/hybrid_aara/Clp/lib`
   permanently. This can be done by inserting the following line to the file
   `~/.bashrc`.
```
export LD_LIBRARY_PATH=/home/user_name/Desktop/clp/dist/lib
```
After changing the file `~/.bashrc`, refresh the environment variables in the
current shell session by running `source ~/.bashrc`.

6. Build Hybrid AARA by running
```
cd /home/user_name/Desktop/hybrid_aara
make
```

7. To test if Hybrid AARA works properly, run
```
cd /home/user_name/Desktop/hybrid_aara
./main usage
```
This should display a list of commands supported by (the original and Hybrid)
RaML.

# Usage



# Structure of the Hybrid RaML project directory


