FROM ubuntu:22.04
RUN apt update && apt -y upgrade

# Install a compiler for C++ and CMake
RUN apt install -y g++ cmake

# Install Python and pip
RUN apt install -y python3 pip

# Install OCaml 4.06.0
RUN apt install -y opam
RUN opam init --comp 4.06.0 --disable-sandboxing -y
RUN eval $(opam env)

# Install the library lp-solve
RUN apt install -y lp-solve

# Install wget (to be used by subseqeunt commands in this Dockerfile) and vim
# (so that the user can edit files inside the Docker container)
RUN apt install -y wget vim

# Install the library Intel OneAPI MKL used by volesti
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list
RUN apt update && apt install -y intel-oneapi-mkl-devel

# Install the libraries BLAS, Eigen, and Boost used by volesti
RUN apt install -y libblas-dev libeigen3-dev libboost-dev

# Install OCaml libraries
RUN opam install -y ocamlbuild.0.14.0 core.v0.11.3
RUN apt install -y autoconf liblapacke-dev libopenblas-dev libplplot-dev libshp-dev
RUN opam install -y pyml owl yojson
RUN apt install -y libffi-dev
RUN opam install -y ctypes ctypes-foreign

# Install CLP
WORKDIR /home/hybrid_aara/clp
RUN wget https://github.com/coin-or/Clp/releases/download/releases%2F1.17.9/Clp-releases.1.17.9-x86_64-ubuntu22-gcc1140.tar.gz
RUN tar xvzf Clp-releases.1.17.9-x86_64-ubuntu22-gcc1140.tar.gz

# Install the Python-Stan binding and other Python libraries
RUN pip install pystan numpy matplotlib joblib

# Copy volesti from a local machine to the Docker container
ADD ./volesti /home/hybrid_aara/volesti

# Copy the volesti-RaML interface from a local machine to the Docker container
ADD ./volesti_raml_interface /home/hybrid_aara/volesti_raml_interface

# Build the volesti-RaML interface
WORKDIR /home/hybrid_aara/volesti_raml_interface/build
RUN cmake .. && cmake --build .

# Copy Hybrid AARA from a local machine to the Docker container
ADD ./raml /home/hybrid_aara/raml

# Build Hybrid AARA
WORKDIR /home/hybrid_aara/raml
RUN ./configure --with-coin-clp /home/hybrid_aara/clp
ENV LD_LIBRARY_PATH "/home/hybrid_aara/clp/lib"
RUN eval $(opam env) && make

# Add Hybrid RaML's executbale to the PATH so that we can call main anywhere in
# the filesystem to run Hybrid RaML
ENV PATH "$PATH:/home/hybrid_aara/raml"

# Copy the benchmark suite of Hybrid AARA from a local machine to the Docker
# container
ADD ./benchmark_suite /home/hybrid_aara/benchmark_suite

# Go to the playground directory where we have an example OCaml program
WORKDIR /home/hybrid_aara/benchmark_suite/playground
